"""The molecular (Transition1x) R + P -> TS workflow.

This is the orchestration behind ``basinflow train-transition1x-ts``. Two
backends share it: the ReactOT bridge (``leftnet_sb``) and the PaiNN/EGNN
velocity-flow baseline. Both endpoints are conditioning inputs and the recorded
transition state is the target.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from basinflow.data import BridgeFlowDataset, EventCatalog, TransitionStateDataset
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD
from basinflow.workflows.dataset import load_configured_dataset
from basinflow.evaluation import run_bridge_trials, run_transition_state_trials
from basinflow.models import FlowLossWeights
from basinflow.models.sb import (
    DEFAULT_BETA_MAX,
    DEFAULT_INV_POWER,
    DEFAULT_POWER,
    DEFAULT_T_EPS,
    DEFAULT_TIMESTEPS,
    SBSchedule,
)

#: The reference evaluates with ``--atol/--rtol 1e-2`` (``evaluation.py``).
_BRIDGE_SOLVER_TOL = 1e-2
from basinflow.models.factory import build_model, model_cutoff, model_spec
from basinflow.training import train_loop, write_loss_artifacts
from basinflow.training.init import load_initial_weights
from basinflow.workflows.config import optional_section, parse_bool, read_config


REQUIRED = {
    "data": ("events_dir",),
    "output": ("output_dir", "log_to_stdout"),
    "model": (),
    # ``[init]`` is intentionally absent: the bridge backend has no seed, prior or
    # pinned flow time, so requiring the section would force bridge configs to
    # carry a meaningless block.
    "training": ("epochs", "lr", "batch_size"),
    "loss": ("velocity_weight",),
    "runtime": ("device", "cpu_threads"),
    "evaluation": ("eval_num_steps", "trials_per_event", "sampling_seed"),
}


def _device(torch, config):
    requested = config["runtime"]["device"].strip().lower()
    if config["runtime"].getint("cpu_threads") > 0:
        torch.set_num_threads(config["runtime"].getint("cpu_threads"))
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps requested but MPS is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("runner supports device=auto, cpu, cuda, or mps")
    return torch.device(requested)


def _select(catalog: EventCatalog, split, name: str) -> EventCatalog:
    """One partition, allowing an empty one: ReactOT has no validation set."""
    identifiers = split.partition(name)
    if not identifiers:
        return EventCatalog(events={})
    return catalog.select(identifiers, name=name)


def _record_solver(config, *, integrator: str, atol: float, rtol: float) -> None:
    """Write the *effective* solver into the config that gets saved as the record.

    The config only carries an ``integrator`` key when a run overrides the
    default, so without this the resolved config could not say which solver
    produced the numbers.
    """
    config["evaluation"]["integrator"] = str(integrator)
    config["evaluation"]["solver_atol"] = str(atol)
    config["evaluation"]["solver_rtol"] = str(rtol)


def run_transition1x_ts(config_path: Path | str) -> dict[str, Any]:
    config = read_config(Path(config_path), REQUIRED)

    import torch

    device = _device(torch, config)
    split_config = config["split"] if config.has_section("split") else {}
    seed = int(config["training"].get("seed", split_config.get("seed", 42)))
    torch.manual_seed(seed)
    output_dir = Path(config["output"]["output_dir"])
    resume_setting = config["training"].get("resume_from", "").strip()
    resume_from = None
    if resume_setting == "auto":
        # HPC jobs come back after a wall-time kill: pick the newest complete
        # checkpoint so a resubmission continues instead of restarting.
        candidates = [output_dir / "training_state.pt"]
        epoch_dir = output_dir / "epoch_checkpoints"
        if epoch_dir.is_dir():
            candidates.extend(sorted(epoch_dir.glob("epoch_*.pt"), reverse=True))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if torch.load(candidate, map_location="cpu", weights_only=False).get("epoch_complete"):
                resume_from = candidate
                break
        if resume_from is None and any(
            (output_dir / name).exists() for name in ("training.log", "checkpoint.pt")
        ):
            # Fresh output directory: start normally. Prior artifacts without a
            # complete checkpoint mean the run died mid-epoch; refuse to clobber.
            raise FileNotFoundError(
                f"resume_from=auto found no complete checkpoint under {output_dir}"
            )
    elif resume_setting:
        resume_from = Path(resume_setting)
        if not resume_from.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
    if resume_from is None and any(
        (output_dir / name).exists() for name in ("training.log", "checkpoint.pt", "config.resolved.ini")
    ):
        raise FileExistsError("output contains a prior run; use a new output directory, including for resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog, split, data_sources = load_configured_dataset(config, seed=seed)
    train_catalog = _select(catalog, split, "train")
    val_catalog = _select(catalog, split, "val")
    evaluation_split = config["evaluation"].get("split", "test").strip().lower()
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    # Only the partition actually used is selected.
    evaluation_catalog = val_catalog if evaluation_split == "val" else _select(catalog, split, "test")
    if not evaluation_catalog.event_ids:
        raise ValueError(f"evaluation partition {evaluation_split!r} is empty")
    (output_dir / "data_sources.json").write_text(
        json.dumps({**data_sources,
                    "num_events_by_partition": {
                        name: len(getattr(split, f"{name}_ids")) for name in ("train", "val", "test")
                    }}, indent=2) + "\n", encoding="utf-8")

    loss_weights = FlowLossWeights(
        velocity=config["loss"].getfloat("velocity_weight"),
        norm=config["loss"].get("norm", "mse").strip().lower(),
    )
    init = optional_section(config, "init")
    source_scale = float(init.get("source_scale", 0.0))
    time_distribution = str(init.get("time_distribution", "fixed")).strip().lower()
    beta_alpha = float(init.get("beta_alpha", 0.8))
    time_epsilon = float(init.get("time_epsilon", 0.0))
    # ``flow_time`` is only read for the deterministic-bridge parameterisation.
    flow_time = float(init.get("flow_time", 0.0)) if time_distribution == "fixed" else None
    backend, model_config = model_spec(config["model"], default_backend="painn")
    model = build_model(backend, model_config).to(device)

    # ``leftnet_sb`` is the Schrodinger-bridge backend: training uses the bridge
    # pairs and scoring integrates the bridge ODE, with no velocity field.
    bridge = backend == "leftnet_sb"
    # The reference batches by ``sum(n_atoms ** 2) <= 2800`` and clips adaptively;
    # the PaiNN/EGNN baselines keep fixed-batch/fixed-clip semantics.
    sampler_mode = config["training"].get(
        "batch_sampler", "node2" if bridge else "samples"
    ).strip().lower()
    if sampler_mode not in {"node2", "samples"}:
        raise ValueError("batch_sampler must be 'node2' or 'samples'")
    clip_mode = config["training"].get(
        "grad_clip", "adaptive" if bridge else "fixed"
    ).strip().lower()
    if clip_mode not in {"adaptive", "fixed"}:
        raise ValueError("grad_clip must be 'adaptive' or 'fixed'")
    max_num = config["training"].getint("max_num", fallback=2800)
    bridge_section = config["bridge"] if config.has_section("bridge") else {}
    bridge_params = {
        "timesteps": int(bridge_section.get("timesteps", DEFAULT_TIMESTEPS)),
        "beta_max": float(bridge_section.get("beta_max", DEFAULT_BETA_MAX)),
        "power": float(bridge_section.get("power", DEFAULT_POWER)),
        "inv_power": float(bridge_section.get("inv_power", DEFAULT_INV_POWER)),
    }
    bridge_t_eps = float(bridge_section.get("t_eps", DEFAULT_T_EPS))
    schedule = SBSchedule(**bridge_params) if bridge else None

    def dataset_for(part, trial_seed, *, swap: bool = False, lowest_step: int = 0):
        if bridge:
            return BridgeFlowDataset(
                part,
                seed=trial_seed,
                lowest_step=lowest_step,
                swap_reactant_product=swap,
                **bridge_params,
            )
        return TransitionStateDataset(part, flow_time=flow_time, seed=trial_seed, source_scale=source_scale,
                                      time_distribution=time_distribution, beta_alpha=beta_alpha,
                                      time_epsilon=time_epsilon,
                                      path=str(init.get("path", "linear")).strip().lower())

    validate_each_epoch = config["training"].getboolean("validate_each_epoch", fallback=bridge)
    bridge_validation_dataset = None
    validation_dataset = None
    if validate_each_epoch:
        if bridge:
            # ReactOT's ``valid_rpsb_all`` is this project's test split; when a
            # real val partition exists, prefer it.
            validation_split = config["training"].get("validation_split", "").strip().lower()
            if not validation_split:
                validation_split = "val" if val_catalog.event_ids else "test"
            if validation_split not in {"train", "val", "test"}:
                raise ValueError("validation_split must be train, val or test")
            validation_catalog = _select(catalog, split, validation_split)
            if not validation_catalog.event_ids:
                raise ValueError(f"validation partition {validation_split!r} is empty")
            bridge_validation_dataset = dataset_for(
                validation_catalog, seed, swap=False, lowest_step=1
            )
        else:
            validation_dataset = dataset_for(val_catalog, seed)

    ema_decay = config["training"].getfloat("ema_decay", fallback=None)
    swap_reactant_product = config["training"].getboolean(
        "swap_reactant_product", fallback=False
    )
    train_dataset = dataset_for(train_catalog, seed, swap=swap_reactant_product)
    init_checkpoint = config["training"].get("init_checkpoint", "").strip()
    if init_checkpoint:
        format_name = load_initial_weights(model, init_checkpoint)
        print(f"[train] initialised weights from {init_checkpoint} ({format_name})")
    configured_batch_size = config["training"]["batch_size"]
    batch_size = len(train_dataset) if configured_batch_size == "full" else int(configured_batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or 'full'")
    _record_solver(
        config,
        integrator=config["evaluation"].get("integrator", DEFAULT_METHOD).strip().lower(),
        atol=config["evaluation"].getfloat("solver_atol", fallback=DEFAULT_ATOL),
        rtol=config["evaluation"].getfloat("solver_rtol", fallback=DEFAULT_RTOL),
    )
    with (output_dir / "config.resolved.ini").open("w", encoding="utf-8") as file:
        config.write(file)

    history = train_loop(
        model,
        train_dataset,
        epochs=config["training"].getint("epochs"),
        lr=config["training"].getfloat("lr"),
        log_path=output_dir / "training.log",
        log_to_stdout=parse_bool(config["output"]["log_to_stdout"]),
        device=device,
        batch_size=batch_size,
        loss_weights=loss_weights,
        max_steps=config["training"].getint("max_steps", fallback=None),
        max_grad_norm=config["training"].getfloat("max_grad_norm", fallback=10.0),
        batch_sampler=None if sampler_mode == "samples" else "node2",
        max_num=max_num,
        adaptive_clip=clip_mode == "adaptive",
        # The reference runs 3000 epochs x 200 batches; the bridge arm keeps that
        # cadence unless the config overrides it.
        max_batches_per_epoch=config["training"].getint(
            "limit_train_batches", fallback=200 if bridge else None
        ),
        validation_dataset=validation_dataset,
        bridge_validation_dataset=bridge_validation_dataset,
        validation_max_num=config["training"].getint("val_max_num", fallback=8400),
        validation_max_batches=config["training"].getint("limit_val_batches", fallback=20),
        early_stopping_patience=config["training"].getint(
            "early_stopping_patience", fallback=None
        ),
        best_checkpoint_path=output_dir / "best.pt" if bridge_validation_dataset is not None else None,
        epoch_checkpoint_dir=output_dir / "epoch_checkpoints" if config["training"].getboolean(
            "save_each_epoch", fallback=False) else None,
        optimizer_name=config["training"].get("optimizer", "adam"),
        scheduler_name=config["training"].get("scheduler", "constant").strip().lower(),
        warmup_fraction=config["training"].getfloat("warmup_fraction", fallback=0.0),
        min_lr_fraction=config["training"].getfloat("min_lr_fraction", fallback=0.0),
        amsgrad=config["training"].getboolean("amsgrad", fallback=False),
        weight_decay=config["training"].getfloat("weight_decay", fallback=0.0),
        resume_from=resume_from,
        ema_decay=ema_decay,
        # The full training state carries the EMA copy; the slimmer ``checkpoint.pt``
        # stays the scoring artifact, so bridge runs keep both.
        checkpoint_path=output_dir / "training_state.pt" if bridge else None,
    )
    write_loss_artifacts(output_dir / "training.log", output_dir)
    evaluated_weights = "raw"
    if bridge and ema_decay is not None:
        saved = torch.load(output_dir / "training_state.pt", map_location="cpu", weights_only=False)
        ema_state = saved.get("ema_model_state_dict")
        if ema_state is None:
            raise ValueError("ema_decay is set but the training state carries no EMA weights")
        # ReactOT reports the averaged model (train_rpsb_ts1x.py:74-76), so the
        # scored weights must be the EMA ones.
        model.load_state_dict(ema_state)
        evaluated_weights = "ema"
    checkpoint = {
        "model_backend": backend,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "evaluated_weights": evaluated_weights,
        "history": history,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    if bridge:
        summary = run_bridge_trials(
            model=model,
            catalog=evaluation_catalog,
            output_dir=output_dir,
            cutoff=model_cutoff(model),
            num_steps=config["evaluation"].getint("eval_num_steps"),
            schedule=schedule,
            trials_per_event=config["evaluation"].getint("trials_per_event"),
            num_events=config["evaluation"].getint("num_events", fallback=None),
            sampling_seed=config["evaluation"].getint("sampling_seed"),
            integrator=config["evaluation"].get("integrator", DEFAULT_METHOD).strip().lower(),
            solver_atol=config["evaluation"].getfloat("solver_atol", fallback=_BRIDGE_SOLVER_TOL),
            solver_rtol=config["evaluation"].getfloat("solver_rtol", fallback=_BRIDGE_SOLVER_TOL),
            t_eps=bridge_t_eps,
            device=device,
        )
    else:
        summary = run_transition_state_trials(
        model=model,
        catalog=evaluation_catalog,
        output_dir=output_dir,
        cutoff=model_cutoff(model),
        num_steps=config["evaluation"].getint("eval_num_steps"),
        source_scale=source_scale,
        time_epsilon=time_epsilon,
        integrator=config["evaluation"].get("integrator", DEFAULT_METHOD).strip().lower(),
        solver_atol=config["evaluation"].getfloat("solver_atol", fallback=DEFAULT_ATOL),
        solver_rtol=config["evaluation"].getfloat("solver_rtol", fallback=DEFAULT_RTOL),
        trials_per_event=config["evaluation"].getint("trials_per_event"),
        num_events=config["evaluation"].getint("num_events", fallback=None),
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        # Sampling uses the same time law as training.
        time_distribution=time_distribution,
        beta_alpha=beta_alpha,
        write_trajectories=config["evaluation"].getboolean("write_trajectories", fallback=False),
        device=device,
    )
    best_path = output_dir / "best.pt"
    validation_summary = None
    if best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        validation_summary = {
            "best_epoch": int(best["epoch"]),
            "val_scaled_err": float(best["val_scaled_err"]),
            "weights": best.get("weights", "raw"),
        }

    report = {
        "task": "reactant_plus_product_to_transition_state",
        "training_contract": (
            "schrodinger_bridge_rp_to_transition_state" if bridge else "pairwise_rp_to_transition_state"
        ),
        "validation": validation_summary,
        "backend": backend,
        "evaluated_weights": evaluated_weights,
        "ema_decay": ema_decay,
        "bridge": None if schedule is None else {
            "timesteps": int(schedule.timesteps),
            "beta": float(schedule.betas[0]) * float(schedule.timesteps),
            "t_eps": bridge_t_eps,
        },
        "split": {"train": len(train_catalog.event_ids), "val": len(val_catalog.event_ids),
                  "test": len(split.test_ids)},
        "endpoint_condition": bool(model.endpoint_condition),
        "endpoint_equivariant": bool(model.endpoint_equivariant),
        "flow_time": flow_time,
        "source_scale": source_scale,
        "time_epsilon": time_epsilon,
        "time_distribution": time_distribution,
        "final_loss": history[-1]["loss"] if history else None,
        "evaluation": summary,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return report
