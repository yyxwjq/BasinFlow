"""The molecular (Transition1x) R + P -> TS workflow.

This is the orchestration behind ``basinflow train-transition1x-ts``. It shares
the backbone, the loss and the training loop with the R -> P workflow and
differs only in the dataset: both endpoints are conditioning inputs and the
target is the recorded transition state, which is the task ReactOT and MolGEN
report their 0.1-0.2 A numbers on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from basinflow.data import BridgeFlowDataset, TransitionStateDataset, single_event_scope
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD
from basinflow.workflows.dataset import load_configured_dataset
from basinflow.evaluation import run_bridge_trials, run_transition_state_trials, write_loss_artifacts
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
from basinflow.training import train_product_flow
from basinflow.workflows.config import optional_section, parse_bool, read_config
from basinflow.workflows.transition1x import _device


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
    if any((output_dir / name).exists() for name in ("training.log", "checkpoint.pt", "config.resolved.ini")):
        raise FileExistsError("output contains a prior run; use a new output directory, including for resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog, split, data_sources = load_configured_dataset(config, seed=seed)
    scope = config["data"].get("training_scope", "all").strip().lower()
    if scope not in {"all", "single_event"}:
        raise ValueError("training_scope must be 'all' or 'single_event'")
    scope_filter = single_event_scope if scope == "single_event" else (lambda part: part)
    train_catalog = scope_filter(split.select(catalog, "train"))
    val_catalog = scope_filter(split.select(catalog, "val"))
    evaluation_split = config["evaluation"].get("split", "test").strip().lower()
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    # Only the partition actually used is selected. An empty partition is legal
    # data, and selecting it must raise rather than evaluate on nothing.
    evaluation_catalog = (
        val_catalog if evaluation_split == "val" else scope_filter(split.select(catalog, "test"))
    )
    (output_dir / "data_sources.json").write_text(
        json.dumps({**data_sources, "training_scope": scope,
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
    # A pinned time is only legal for the deterministic-bridge parameterisation.
    # With a sampled time distribution the flow has to be supervised over the
    # whole path, so ``flow_time`` must stay unset (see docs/25).
    flow_time = float(init.get("flow_time", 0.0)) if time_distribution == "fixed" else None
    backend, model_config = model_spec(config["model"], default_backend="painn")
    model = build_model(backend, model_config).to(device)

    # ``leftnet_sb`` is the Schrodinger-bridge backend. It has no velocity field:
    # training uses the bridge pairs and scoring integrates the bridge ODE, so the
    # whole data/sampler pair switches together with the backend.
    bridge = backend == "leftnet_sb"
    bridge_section = config["bridge"] if config.has_section("bridge") else {}
    bridge_params = {
        "timesteps": int(bridge_section.get("timesteps", DEFAULT_TIMESTEPS)),
        "beta_max": float(bridge_section.get("beta_max", DEFAULT_BETA_MAX)),
        "power": float(bridge_section.get("power", DEFAULT_POWER)),
        "inv_power": float(bridge_section.get("inv_power", DEFAULT_INV_POWER)),
    }
    bridge_t_eps = float(bridge_section.get("t_eps", DEFAULT_T_EPS))
    schedule = SBSchedule(**bridge_params) if bridge else None

    def dataset_for(part, trial_seed):
        if bridge:
            return BridgeFlowDataset(part, seed=trial_seed, **bridge_params)
        return TransitionStateDataset(part, flow_time=flow_time, seed=trial_seed, source_scale=source_scale,
                                      time_distribution=time_distribution, beta_alpha=beta_alpha,
                                      time_epsilon=time_epsilon,
                                      path=str(init.get("path", "linear")).strip().lower())

    validate_each_epoch = config["training"].getboolean("validate_each_epoch", fallback=False)
    if bridge and validate_each_epoch:
        raise ValueError(
            "validate_each_epoch runs the velocity-MSE probe, which the bridge backend does not "
            "produce; score bridge runs with the bridge sampler instead"
        )
    ema_decay = config["training"].getfloat("ema_decay", fallback=None)
    train_dataset = dataset_for(train_catalog, seed)
    validation_dataset = dataset_for(val_catalog, seed) if validate_each_epoch else None
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

    history = train_product_flow(
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
        validation_dataset=validation_dataset,
        epoch_checkpoint_dir=output_dir / "epoch_checkpoints" if config["training"].getboolean(
            "save_each_epoch", fallback=False) else None,
        optimizer_name=config["training"].get("optimizer", "adam"),
        scheduler_name=config["training"].get("scheduler", "constant").strip().lower(),
        warmup_fraction=config["training"].getfloat("warmup_fraction", fallback=0.0),
        min_lr_fraction=config["training"].getfloat("min_lr_fraction", fallback=0.0),
        amsgrad=config["training"].getboolean("amsgrad", fallback=False),
        weight_decay=config["training"].getfloat("weight_decay", fallback=0.0),
        resume_from=config["training"].get("resume_from", "").strip() or None,
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
        # Sampling must use the *training* time law, otherwise the sampler
        # integrates a field outside the regime it was fitted on.
        time_distribution=time_distribution,
        beta_alpha=beta_alpha,
        write_trajectories=config["evaluation"].getboolean("write_trajectories", fallback=False),
        device=device,
    )
    report = {
        "task": "reactant_plus_product_to_transition_state",
        "training_contract": (
            "schrodinger_bridge_rp_to_transition_state" if bridge else "pairwise_rp_to_transition_state"
        ),
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
