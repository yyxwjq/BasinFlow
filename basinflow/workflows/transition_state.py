"""The molecular (Transition1x) R + P -> TS workflow.

This is the orchestration behind ``basinflow train-transition1x-ts``. Every
backend (``leftnet_sb``, ``painn``, ``egnn``) shares the same bridge dataset,
objective, training loop and sampler; only the message-passing architecture
differs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from basinflow.data import BridgeFlowDataset, EventCatalog, Split, partition
from basinflow.evaluation import run_bridge_trials
from basinflow.models.factory import build_model, model_spec
from basinflow.models.sb import (
    DEFAULT_BETA_MAX,
    DEFAULT_INV_POWER,
    DEFAULT_POWER,
    DEFAULT_T_EPS,
    DEFAULT_TIMESTEPS,
    SBSchedule,
)
from basinflow.sampling.ode import DEFAULT_METHOD
from basinflow.training import train_loop, write_loss_artifacts
from basinflow.training.init import load_initial_weights
from basinflow.workflows.config import optional_section, parse_bool, read_config
from basinflow.workflows.dataset import load_configured_dataset

#: The reference evaluates with ``--atol/--rtol 1e-2`` (``evaluation.py``).
_BRIDGE_SOLVER_TOL = 1e-2


REQUIRED = {
    "data": ("events_dir",),
    "output": ("output_dir", "log_to_stdout"),
    "model": (),
    "training": ("epochs", "lr", "batch_size"),
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


def _select(catalog: EventCatalog, split: Split, name: str) -> EventCatalog:
    """One partition, allowing an empty one: ReactOT has no validation set."""
    identifiers = partition(split, name)
    if not identifiers:
        return EventCatalog(events={})
    return catalog.select(identifiers, name=name)


def _record_solver(config, *, integrator: str, atol: float, rtol: float) -> None:
    """Write the effective solver into the config saved as the run record."""
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
            raise FileNotFoundError(
                f"resume_from=auto found no complete checkpoint under {output_dir}"
            )
    elif resume_setting:
        resume_from = Path(resume_setting)
        if not resume_from.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
    if resume_from is None and any(
        (output_dir / name).exists()
        for name in ("training.log", "checkpoint.pt", "config.resolved.ini")
    ):
        raise FileExistsError(
            "output contains a prior run; use a new output directory, including for resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog, split, data_sources = load_configured_dataset(config, seed=seed)
    train_catalog = _select(catalog, split, "train")
    val_catalog = _select(catalog, split, "val")
    evaluation_split = config["evaluation"].get("split", "test").strip().lower()
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    evaluation_catalog = (
        val_catalog if evaluation_split == "val" else _select(catalog, split, "test")
    )
    if not evaluation_catalog.event_ids:
        raise ValueError(f"evaluation partition {evaluation_split!r} is empty")
    (output_dir / "data_sources.json").write_text(
        json.dumps(
            {
                **data_sources,
                "num_events_by_partition": {
                    name: len(split[name]) for name in ("train", "val", "test")
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    loss_section = optional_section(config, "loss")
    loss_norm = str(loss_section.get("norm", "mse")).strip().lower()
    if loss_norm not in {"mse", "l1"}:
        raise ValueError("loss norm must be 'mse' or 'l1'")

    backend, model_config = model_spec(config["model"], default_backend="leftnet_sb")
    model = build_model(backend, model_config).to(device)

    sampler_mode = config["training"].get("batch_sampler", "node2").strip().lower()
    if sampler_mode not in {"node2", "samples"}:
        raise ValueError("batch_sampler must be 'node2' or 'samples'")
    clip_mode = config["training"].get("grad_clip", "adaptive").strip().lower()
    if clip_mode not in {"adaptive", "fixed"}:
        raise ValueError("grad_clip must be 'adaptive' or 'fixed'")
    max_num = config["training"].getint("max_num", fallback=2800)

    bridge_section = optional_section(config, "bridge")
    bridge_params = {
        "timesteps": int(bridge_section.get("timesteps", DEFAULT_TIMESTEPS)),
        "beta_max": float(bridge_section.get("beta_max", DEFAULT_BETA_MAX)),
        "power": float(bridge_section.get("power", DEFAULT_POWER)),
        "inv_power": float(bridge_section.get("inv_power", DEFAULT_INV_POWER)),
    }
    bridge_t_eps = float(bridge_section.get("t_eps", DEFAULT_T_EPS))
    schedule = SBSchedule(**bridge_params)

    def dataset_for(part, dataset_seed, *, swap: bool = False, lowest_step: int = 0):
        return BridgeFlowDataset(
            part,
            schedule,
            seed=dataset_seed,
            lowest_step=lowest_step,
            swap_reactant_product=swap,
        )

    bridge_validation_dataset = None
    if config["training"].getboolean("validate_each_epoch", fallback=True):
        # ReactOT's ``valid_rpsb_all`` is this project's test split; a real val
        # partition wins when it exists.
        validation_split = config["training"].get("validation_split", "").strip().lower()
        if not validation_split:
            validation_split = "val" if val_catalog.event_ids else "test"
        if validation_split not in {"train", "val", "test"}:
            raise ValueError("validation_split must be train, val or test")
        validation_catalog = _select(catalog, split, validation_split)
        if not validation_catalog.event_ids:
            raise ValueError(f"validation partition {validation_split!r} is empty")
        bridge_validation_dataset = dataset_for(
            validation_catalog, seed, lowest_step=1
        )

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
    batch_size = (
        len(train_dataset) if configured_batch_size == "full" else int(configured_batch_size)
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or 'full'")
    integrator = config["evaluation"].get("integrator", DEFAULT_METHOD).strip().lower()
    solver_atol = config["evaluation"].getfloat(
        "solver_atol", fallback=_BRIDGE_SOLVER_TOL
    )
    solver_rtol = config["evaluation"].getfloat(
        "solver_rtol", fallback=_BRIDGE_SOLVER_TOL
    )
    _record_solver(config, integrator=integrator, atol=solver_atol, rtol=solver_rtol)
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
        loss_norm=loss_norm,
        max_steps=config["training"].getint("max_steps", fallback=None),
        max_grad_norm=config["training"].getfloat("max_grad_norm", fallback=10.0),
        batch_sampler=None if sampler_mode == "samples" else "node2",
        max_num=max_num,
        adaptive_clip=clip_mode == "adaptive",
        max_batches_per_epoch=config["training"].getint(
            "limit_train_batches", fallback=200
        ),
        bridge_validation_dataset=bridge_validation_dataset,
        validation_max_num=config["training"].getint("val_max_num", fallback=8400),
        validation_max_batches=config["training"].getint("limit_val_batches", fallback=20),
        early_stopping_patience=config["training"].getint(
            "early_stopping_patience", fallback=None
        ),
        best_checkpoint_path=(
            output_dir / "best.pt" if bridge_validation_dataset is not None else None
        ),
        save_every_n_epochs=config["training"].getint("save_every_n_epochs", fallback=1),
        epoch_checkpoint_dir=(
            output_dir / "epoch_checkpoints"
            if config["training"].getboolean("save_each_epoch", fallback=False)
            else None
        ),
        optimizer_name=config["training"].get("optimizer", "adam"),
        scheduler_name=config["training"].get("scheduler", "constant").strip().lower(),
        warmup_fraction=config["training"].getfloat("warmup_fraction", fallback=0.0),
        min_lr_fraction=config["training"].getfloat("min_lr_fraction", fallback=0.0),
        amsgrad=config["training"].getboolean("amsgrad", fallback=False),
        weight_decay=config["training"].getfloat("weight_decay", fallback=0.0),
        resume_from=resume_from,
        ema_decay=ema_decay,
        checkpoint_path=output_dir / "training_state.pt",
    )
    write_loss_artifacts(output_dir / "training.log", output_dir)

    evaluated_weights = "raw"
    if ema_decay is not None:
        saved = torch.load(output_dir / "training_state.pt", map_location="cpu", weights_only=False)
        ema_state = saved.get("ema_model_state_dict")
        if ema_state is None:
            raise ValueError("ema_decay is set but the training state carries no EMA weights")
        # ReactOT reports the averaged model, so the scored weights must be EMA.
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

    summary = run_bridge_trials(
        model=model,
        catalog=evaluation_catalog,
        output_dir=output_dir,
        num_steps=config["evaluation"].getint("eval_num_steps"),
        schedule=schedule,
        trials_per_event=config["evaluation"].getint("trials_per_event"),
        num_events=config["evaluation"].getint("num_events", fallback=None),
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        integrator=integrator,
        solver_atol=solver_atol,
        solver_rtol=solver_rtol,
        t_eps=bridge_t_eps,
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
        "training_contract": "schrodinger_bridge_rp_to_transition_state",
        "validation": validation_summary,
        "backend": backend,
        "evaluated_weights": evaluated_weights,
        "ema_decay": ema_decay,
        "loss_norm": loss_norm,
        "bridge": {
            "timesteps": int(schedule.timesteps),
            "beta": float(schedule.betas[0]) * float(schedule.timesteps),
            "t_eps": bridge_t_eps,
        },
        "split": {
            "train": len(train_catalog.event_ids),
            "val": len(val_catalog.event_ids),
            "test": len(split["test"]),
        },
        "evaluation_split": evaluation_split,
        "endpoint_condition": bool(getattr(model, "endpoint_condition", False)),
        "final_loss": history[-1]["loss"] if history else None,
        "evaluation": summary,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return report
