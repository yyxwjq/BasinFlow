"""The periodic-system (Au/Pt) basin benchmark workflow.

This is the orchestration behind ``basinflow benchmark``; the example script of
the same name is a thin wrapper so there is exactly one implementation.
"""
from __future__ import annotations

from typing import Any

from basinflow.seeds.from_config import init_generators
from basinflow.workflows.config import parse_bool, read_config

import json
from pathlib import Path


from basinflow.evaluation.semantic_flow import run_gaussian_trials, write_loss_artifacts
from basinflow.data.partitions import load_data_partitions
from basinflow.data.pyg import EventFlowDataset
from basinflow.models import FlowLossWeights
from basinflow.models.factory import build_model, model_cutoff, model_spec
from basinflow.relaxation import RelaxationConfig, eam_calculator, prepare_eam_potential
from basinflow.training import active_pos_weight, train_product_flow


REQUIRED = {
    "data": ("pbc_override", "active_source", "active_threshold"),
    "output": ("output_dir", "log_to_stdout"),
    "model": (),
    "init": ("types", "gaussian_scale"),
    "training": ("epochs", "lr", "batch_size"),
    "loss": ("velocity_weight", "active_weight", "direction_weight", "active_pos_weight"),
    "runtime": ("device", "cpu_threads", "gpu_ids", "gpu_count"),
    "evaluation": ("eval_num_steps", "eval_graph_update_interval", "trials_per_test_basin", "sampling_seed"),
    "relaxation": ("eam_potential", "eam_form", "relax_fmax", "relax_steps"),
}


def _parse_pbc_override(text: str):
    value = str(text).strip().lower()
    if value in {"", "auto", "none"}:
        return None
    tokens = [token.strip() for token in value.split(",")]
    if len(tokens) != 3 or any(token not in {"true", "false"} for token in tokens):
        raise ValueError("pbc_override must be auto or three comma-separated true/false values")
    return [token == "true" for token in tokens]




def _runtime(torch, config) -> object:
    requested = config["runtime"].get("device", "cpu").strip().lower()
    if requested != "cpu":
        raise RuntimeError("semantic benchmark currently supports CPU execution only")
    threads = config["runtime"].getint("cpu_threads")
    if threads > 0:
        torch.set_num_threads(threads)
    return torch.device("cpu")



def run_semantic_benchmark(config_path: Path | str, *, system: str | None = None) -> dict[str, Any]:
    config = read_config(Path(config_path), REQUIRED)
    import torch

    device = _runtime(torch, config)
    split_config = config["split"] if config.has_section("split") else {}
    seed = int(config["training"].get("seed", split_config.get("seed", 42)))
    torch.manual_seed(seed)
    events_dir = Path(config["data"].get("events_dir", config["data"].get("train_events_dir", "")))
    output_dir = Path(config["output"]["output_dir"])
    if any((output_dir / name).exists() for name in ("training.log", "checkpoint.pt", "config.resolved.ini")):
        raise FileExistsError("output contains a prior run; use a new output directory, including for resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, split, data_sources = load_data_partitions(
        config["data"], split_config,
        pbc_override=_parse_pbc_override(config["data"]["pbc_override"]),
    )
    manifest_path = data_sources.get("manifest", data_sources["mode"])
    split.validate(dataset)
    split.save(output_dir / "split_manifest.json")
    (output_dir / "data_sources.json").write_text(json.dumps(data_sources, indent=2) + "\n")
    train_ds = split.select(dataset, "train")
    val_ds = split.select(dataset, "val")
    test_ds = split.select(dataset, "test")
    if not test_ds.basin_ids:
        raise ValueError("evaluation partition test is empty")
    manifest = {"train": list(split.train_ids), "val": list(split.val_ids), "test": list(split.test_ids), "seed": split.seed}

    active_threshold = config["data"].getfloat("active_threshold")
    pos_weight_text = config["loss"]["active_pos_weight"].strip().lower()
    resolved_pos_weight = active_pos_weight(train_ds, active_threshold=active_threshold) if pos_weight_text == "auto" else float(pos_weight_text)
    if resolved_pos_weight < 0.0:
        raise ValueError("active_pos_weight must be non-negative or 'auto'")
    loss_weights = FlowLossWeights(
        velocity=config["loss"].getfloat("velocity_weight"),
        active=config["loss"].getfloat("active_weight"),
        direction=config["loss"].getfloat("direction_weight"),
        active_pos_weight=resolved_pos_weight,
    )
    backend, model_config = model_spec(config["model"], default_backend="egnn")
    model = build_model(backend, model_config).to(device)
    generators = init_generators(config, seed)
    validation_dataset = EventFlowDataset(
        val_ds, init_generators(config, seed), active_threshold=active_threshold, seed=seed,
    ) if config["training"].getboolean("validate_each_epoch", fallback=False) else None
    with (output_dir / "config.resolved.ini").open("w", encoding="utf-8") as file:
        config.write(file)
    history = train_product_flow(
        model,
        train_ds,
        generators,
        epochs=config["training"].getint("epochs"),
        lr=config["training"].getfloat("lr"),
        log_path=output_dir / "training.log",
        log_to_stdout=parse_bool(config["output"]["log_to_stdout"]),
        device=device,
        batch_size=config["training"]["batch_size"],
        active_threshold=active_threshold,
        loss_weights=loss_weights,
        max_steps=config["training"].getint("max_steps", fallback=None),
        max_grad_norm=config["training"].getfloat("max_grad_norm", fallback=10.0),
        validation_dataset=validation_dataset,
        epoch_checkpoint_dir=output_dir / "epoch_checkpoints" if config["training"].getboolean("save_each_epoch", fallback=False) else None,
        optimizer_name=config["training"].get("optimizer", "adam"),
        amsgrad=config["training"].getboolean("amsgrad", fallback=False),
        weight_decay=config["training"].getfloat("weight_decay", fallback=0.0),
        resume_from=config["training"].get("resume_from", "").strip() or None,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    config["loss"]["active_pos_weight_resolved"] = str(resolved_pos_weight)
    with (output_dir / "config.resolved.ini").open("w", encoding="utf-8") as file:
        config.write(file)
    write_loss_artifacts(output_dir / "training.log", output_dir)

    checkpoint = {
        "model_backend": backend,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "split_manifest": manifest,
        "history": history,
        "training_contract": "no_oracle_direct_basin",
        "training_config": {**dict(config["training"]), "active_threshold": active_threshold},
        "init_config": {"generators": [{"type": initializer.seed_type, "scale": config["init"].getfloat("gaussian_scale"), "random_seed": seed} for initializer in generators]},
        "loss_weights": {
            "velocity": loss_weights.velocity,
            "active": loss_weights.active,
            "direction": loss_weights.direction,
            "active_pos_weight": loss_weights.active_pos_weight,
            "active_pos_weight_source": pos_weight_text,
        },
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    calculator_factory = None
    relaxation_config = None
    if config["relaxation"].getboolean("enabled", fallback=True):
        # The potential is copied into the run directory because ASE's EAM reader
        # rejects inline comments. Any other ASE calculator needs no such step:
        # pass its own factory here instead.
        source = Path(config["relaxation"]["eam_potential"])
        form = config["relaxation"]["eam_form"].strip().lower()

        def calculator_factory():
            # Prepared lazily, so a missing or malformed potential is reported as
            # a per-trial failure instead of aborting the whole benchmark.
            prepared = prepare_eam_potential(source, output_dir / "sampling" / "eam")
            return eam_calculator(prepared, form)
        relaxation_config = RelaxationConfig(
            fmax=config["relaxation"].getfloat("relax_fmax"),
            steps=config["relaxation"].getint("relax_steps"),
        )
    system = system or events_dir.parent.name
    summary = run_gaussian_trials(
        system=system,
        model=model,
        test_catalog=test_ds,
        output_dir=output_dir,
        cutoff=model_cutoff(model),
        num_steps=config["evaluation"].getint("eval_num_steps"),
        graph_update_interval=config["evaluation"].getint("eval_graph_update_interval"),
        gaussian_scale=config["init"].getfloat("gaussian_scale"),
        trials_per_basin=config["evaluation"].getint("trials_per_test_basin"),
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        active_threshold=active_threshold,
        device=device,
        calculator_factory=calculator_factory,
        relaxation_config=relaxation_config,
    )
    report = {
        "dataset": {"num_basins": len(dataset.basin_ids), "num_events": len(dataset.event_ids)},
        "model_backend": backend,
        "model_config": model_config,
        "split": manifest,
        "training_contract": "pairwise_flow_matching_no_oracle",
        "sampling_contract": "basin_level_no_oracle",
        "sampling": summary,
        "relaxation": "eam" if calculator_factory is not None else "not_run",
        "saddle_validation": "not_run",
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset: {len(dataset.basin_ids)} basins, {len(dataset.event_ids)} events")
    print(f"Split: {len(train_ds.basin_ids)} train / {len(val_ds.basin_ids)} val / {len(test_ds.basin_ids)} test basins")
    print(f"Trials: {summary['num_trials']} ({summary['trials_per_test_basin']} per test basin)")
    print(f"Raw RMSD mean: {summary['raw_rmsd_angstrom']['mean']}")
    print(f"EAM-relaxed RMSD mean: {summary['eam_relaxed_rmsd_angstrom']['mean']}")
    print(f"Sampling summary: {output_dir / 'sampling' / 'summary.json'}")
    return report
