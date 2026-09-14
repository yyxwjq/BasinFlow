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

from basinflow.data.partitions import load_data_partitions
from basinflow.data.pyg import TransitionStateDataset
from basinflow.evaluation import run_transition_state_trials, write_loss_artifacts
from basinflow.models import FlowLossWeights
from basinflow.models.factory import build_model, model_cutoff, model_spec
from basinflow.training import train_product_flow
from basinflow.workflows.config import parse_bool, read_config
from basinflow.workflows.transition1x import _device


REQUIRED = {
    "data": (),
    "output": ("output_dir", "log_to_stdout"),
    "model": (),
    "init": (),
    "training": ("epochs", "lr", "batch_size"),
    "loss": ("velocity_weight", "active_weight", "direction_weight", "active_pos_weight"),
    "runtime": ("device", "cpu_threads"),
    "evaluation": ("eval_num_steps", "trials_per_basin", "sampling_seed"),
}


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

    catalog, split, data_sources = load_data_partitions(config["data"], split_config)
    scope = config["data"].get("training_scope", "all").strip().lower()
    if scope not in {"all", "single_event"}:
        raise ValueError("training_scope must be 'all' or 'single_event'")
    scope_filter = (lambda part: part.single_event_subset()) if scope == "single_event" else (lambda part: part)
    train_catalog = scope_filter(split.select(catalog, "train"))
    val_catalog = scope_filter(split.select(catalog, "val"))
    test_catalog = scope_filter(split.select(catalog, "test"))
    evaluation_split = config["evaluation"].get("split", "test").strip().lower()
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    evaluation_catalog = val_catalog if evaluation_split == "val" else test_catalog
    if not evaluation_catalog.event_ids:
        raise ValueError(f"evaluation partition {evaluation_split} is empty")
    (output_dir / "data_sources.json").write_text(
        json.dumps({**data_sources, "training_scope": scope,
                    "num_basins_by_partition": {
                        name: len(part.basin_ids)
                        for name, part in (("train", train_catalog), ("val", val_catalog), ("test", test_catalog))
                    }}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "split_manifest.json").write_text(json.dumps({
        "train": list(train_catalog.basin_ids),
        "val": list(val_catalog.basin_ids),
        "test": list(test_catalog.basin_ids),
        "seed": split.seed,
        "training_scope": scope,
    }, indent=2) + "\n", encoding="utf-8")

    loss_weights = FlowLossWeights(
        velocity=config["loss"].getfloat("velocity_weight"),
        active=config["loss"].getfloat("active_weight"),
        direction=config["loss"].getfloat("direction_weight"),
        active_pos_weight=config["loss"].getfloat("active_pos_weight"),
        norm=config["loss"].get("norm", "mse").strip().lower(),
    )
    source_scale = config["init"].getfloat("source_scale", fallback=0.0)
    time_distribution = config["init"].get("time_distribution", "fixed").strip().lower()
    beta_alpha = config["init"].getfloat("beta_alpha", fallback=0.8)
    # A pinned time is only legal for the deterministic-bridge parameterisation.
    # With a sampled time distribution the flow has to be supervised over the
    # whole path, so ``flow_time`` must stay unset (see docs/25).
    flow_time = config["init"].getfloat("flow_time", fallback=0.0) if time_distribution == "fixed" else None
    backend, model_config = model_spec(config["model"], default_backend="painn")
    model = build_model(backend, model_config).to(device)

    def dataset_for(part, trial_seed):
        return TransitionStateDataset(part, flow_time=flow_time, seed=trial_seed, source_scale=source_scale,
                                      time_distribution=time_distribution, beta_alpha=beta_alpha)

    train_dataset = dataset_for(train_catalog, seed)
    validation_dataset = dataset_for(val_catalog, seed) if config["training"].getboolean(
        "validate_each_epoch", fallback=False) else None
    configured_batch_size = config["training"]["batch_size"]
    batch_size = len(train_dataset) if configured_batch_size == "full" else int(configured_batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or 'full'")
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
    )
    write_loss_artifacts(output_dir / "training.log", output_dir)
    checkpoint = {
        "model_backend": backend,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    summary = run_transition_state_trials(
        model=model,
        catalog=evaluation_catalog,
        output_dir=output_dir,
        cutoff=model_cutoff(model),
        num_steps=config["evaluation"].getint("eval_num_steps"),
        source_scale=source_scale,
        trials_per_basin=config["evaluation"].getint("trials_per_basin"),
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
        "training_contract": "pairwise_rp_to_transition_state",
        "split": {"train": len(train_catalog.event_ids), "val": len(val_catalog.event_ids),
                  "test": len(test_catalog.event_ids)},
        "endpoint_condition": bool(model.endpoint_condition),
        "endpoint_equivariant": bool(model.endpoint_equivariant),
        "flow_time": flow_time,
        "source_scale": source_scale,
        "time_distribution": time_distribution,
        "final_loss": history[-1]["loss"] if history else None,
        "evaluation": summary,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return report
