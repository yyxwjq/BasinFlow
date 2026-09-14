"""The molecular (Transition1x) R-to-P workflow.

This is the orchestration behind ``basinflow train-transition1x``; the example
script of the same name is a thin wrapper so that there is exactly one
implementation of the pipeline.
"""
from __future__ import annotations

from typing import Any

from basinflow.seeds.from_config import init_generators
from basinflow.workflows.config import parse_bool, read_config

import json
from pathlib import Path


from basinflow.evaluation import (
    run_single_event_molecule_trials,
    write_loss_artifacts,
)
from basinflow.data.pyg import EventFlowDataset
from basinflow.data.partitions import load_data_partitions
from basinflow.models import FlowLossWeights
from basinflow.models.factory import build_model, model_cutoff, model_spec
from basinflow.training import active_pos_weight, train_product_flow


REQUIRED = {
    "data": ("active_threshold",),
    "output": ("output_dir", "log_to_stdout"),
    "model": (),
    "init": ("types", "gaussian_scale"),
    "training": ("epochs", "lr", "batch_size"),
    "loss": ("velocity_weight", "active_weight", "direction_weight", "active_pos_weight"),
    "runtime": ("device", "cpu_threads"),
    "evaluation": ("eval_num_steps", "eval_graph_update_interval", "trials_per_basin", "sampling_seed", "selection_seed"),
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
        raise ValueError("Transition1x runner supports device=auto, cpu, cuda, or mps")
    return torch.device(requested)



def run_transition1x(config_path: Path | str) -> dict[str, Any]:
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
    # ``single_event`` drops the dissociations whose product is a separated
    # fragment pair, so training and evaluation cover the same event class
    # (docs/22 section 二之二). It is recorded in data_sources either way.
    training_scope = config["data"].get("training_scope", "all").strip().lower()
    if training_scope not in {"all", "single_event"}:
        raise ValueError("training_scope must be 'all' or 'single_event'")
    scope_filter = (lambda part: part.single_event_subset()) if training_scope == "single_event" else (lambda part: part)
    manifest_path = data_sources.get("manifest", data_sources["mode"])
    train_catalog = scope_filter(split.select(catalog, "train"))
    val_catalog = scope_filter(split.select(catalog, "val"))
    test_catalog = scope_filter(split.select(catalog, "test"))
    evaluation_split = config["evaluation"].get("split", "test").strip().lower()
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    evaluation_catalog = val_catalog if evaluation_split == "val" else test_catalog
    if not evaluation_catalog.basin_ids:
        raise ValueError(f"evaluation partition {evaluation_split} is empty")
    data_sources = {
        **data_sources,
        "training_scope": training_scope,
        "num_basins_by_partition": {
            name: len(part.basin_ids)
            for name, part in (("train", train_catalog), ("val", val_catalog), ("test", test_catalog))
        },
    }
    (output_dir / "data_sources.json").write_text(json.dumps(data_sources, indent=2) + "\n")
    manifest = {
        "train": list(train_catalog.basin_ids),
        "val": list(val_catalog.basin_ids),
        "test": list(test_catalog.basin_ids),
        "seed": split.seed,
        "training_scope": training_scope,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    active_threshold = config["data"].getfloat("active_threshold")
    pos_weight_text = config["loss"]["active_pos_weight"].strip().lower()
    pos_weight = active_pos_weight(train_catalog, active_threshold=active_threshold) if pos_weight_text == "auto" else float(pos_weight_text)
    loss_weights = FlowLossWeights(
        velocity=config["loss"].getfloat("velocity_weight"),
        active=config["loss"].getfloat("active_weight"),
        direction=config["loss"].getfloat("direction_weight"),
        active_pos_weight=pos_weight,
        norm=config["loss"].get("norm", "mse").strip().lower(),
    )
    # Time sampling and prior scale are the two knobs that decide whether the
    # generative end (t~0) carries information at all; see docs/20.
    time_distribution = config["init"].get("time_distribution", "uniform").strip().lower()
    beta_alpha = config["init"].getfloat("beta_alpha", fallback=0.8)
    # A fixed flow time turns the flow objective into direct regression at that
    # time. It exists because the interpolated position leaks the target for
    # every t > 0 whenever the prior is a near-copy of the reactant; see
    # docs/22. Absent, the time is sampled per item.
    flow_time = config["init"].getfloat("flow_time", fallback=None)
    backend, model_config = model_spec(config["model"], default_backend="egnn")
    # The bond-change channel is an input condition, so the dataset must expose
    # exactly the labels the model was built to read; see docs/19 section 6.
    bond_change_condition = bool(model_config.get("bond_change_condition", False))
    bond_change_source = config["model"].get("bond_change_source", "oracle").strip().lower()
    model = build_model(backend, model_config).to(device)
    train_dataset = EventFlowDataset(
        train_catalog,
        init_generators(config, seed),
        active_threshold=active_threshold,
        seed=seed,
        time_distribution=time_distribution,
        beta_alpha=beta_alpha,
        flow_time=flow_time,
        bond_change_condition=bond_change_condition,
        bond_change_source=bond_change_source,
    )
    validation_dataset = EventFlowDataset(
        val_catalog, init_generators(config, seed), active_threshold=active_threshold, seed=seed,
        time_distribution=time_distribution, beta_alpha=beta_alpha,
        flow_time=flow_time,
        bond_change_condition=bond_change_condition,
        bond_change_source=bond_change_source,
    ) if config["training"].getboolean("validate_each_epoch", fallback=False) else None
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
        epoch_checkpoint_dir=output_dir / "epoch_checkpoints" if config["training"].getboolean("save_each_epoch", fallback=False) else None,
        optimizer_name=config["training"].get("optimizer", "adam"),
        scheduler_name=config["training"].get("scheduler", "constant").strip().lower(),
        warmup_fraction=config["training"].getfloat("warmup_fraction", fallback=0.0),
        min_lr_fraction=config["training"].getfloat("min_lr_fraction", fallback=0.0),
        amsgrad=config["training"].getboolean("amsgrad", fallback=False),
        weight_decay=config["training"].getfloat("weight_decay", fallback=0.0),
        resume_from=config["training"].get("resume_from", "").strip() or None,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    config["loss"]["active_pos_weight_resolved"] = str(pos_weight)
    config["evaluation"]["evaluation_protocol"] = "single_event_pairwise_no_relaxation"
    with (output_dir / "config.resolved.ini").open("w", encoding="utf-8") as file:
        config.write(file)
    write_loss_artifacts(output_dir / "training.log", output_dir)
    checkpoint = {
        "model_backend": backend,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "split_manifest": manifest,
        "history": history,
        "training_contract": "transition1x_pairwise_r_to_p_no_oracle",
        "evaluation_split": evaluation_split,
        "split_source": manifest_path or "generated",
        "training_config": {**dict(config["training"]), "active_threshold": active_threshold},
        "init_config": {"generators": [{"type": initializer.seed_type, "scale": config["init"].getfloat("gaussian_scale"), "random_seed": seed} for initializer in init_generators(config, seed)]},
        "loss_weights": {
            "velocity": loss_weights.velocity,
            "active": loss_weights.active,
            "direction": loss_weights.direction,
            "active_pos_weight": loss_weights.active_pos_weight,
        },
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    sampling_summary = run_single_event_molecule_trials(
        model=model,
        test_catalog=evaluation_catalog,
        output_dir=output_dir,
        cutoff=model_cutoff(model),
        num_steps=config["evaluation"].getint("eval_num_steps"),
        graph_update_interval=config["evaluation"].getint("eval_graph_update_interval"),
        gaussian_scale=config["init"].getfloat("gaussian_scale"),
        trials_per_basin=config["evaluation"].getint("trials_per_basin"),
        num_test_basins=config["evaluation"].getint("num_test_basins", fallback=None),
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        selection_seed=config["evaluation"].getint("selection_seed"),
        active_threshold=active_threshold,
        device=device,
        bond_change_source=config["evaluation"].get("bond_change_source", "seed").strip().lower(),
    )
    report = {
        "dataset": {
            "kind": "transition1x_single_event_molecule",
            "num_events": len(catalog.event_ids),
            "num_pseudo_basins": len(catalog.basin_ids),
        },
        "split": {"train": len(train_catalog.basin_ids), "val": len(val_catalog.basin_ids), "test": len(test_catalog.basin_ids)},
        "training_contract": "pairwise_r_to_p_no_oracle",
        "training_scope": training_scope,
        "flow_time": flow_time,
        "bond_change": {
            "enabled": bond_change_condition,
            "training_source": bond_change_source,
            "sampling_source": config["evaluation"].get("bond_change_source", "seed").strip().lower(),
        },
        "evaluation_split": evaluation_split,
        "split_source": manifest_path or "generated",
        "model_backend": backend,
        "model_config": model_config,
        "sampling": sampling_summary,
        "relaxation": "not_run",
        "saddle_validation": "not_run",
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset: {len(catalog.event_ids)} Transition1x events / pseudo-basins")
    print(f"Split: {len(train_catalog.basin_ids)} train / {len(val_catalog.basin_ids)} val / {len(test_catalog.basin_ids)} test")
    print(f"Final loss: {history[-1]['loss']:.6f}")
    print(f"Sampling trials: {sampling_summary['num_trials']}")
    print(f"Raw RMSD mean: {sampling_summary['raw_rmsd_angstrom']['mean']}")
    print(f"Checkpoint: {output_dir / 'checkpoint.pt'}")
    print(f"Sampling summary: {output_dir / 'sampling' / 'summary.json'}")
    return report
