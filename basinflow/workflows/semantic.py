"""The periodic-system (Au/Pt) event benchmark workflow.

This is the orchestration behind ``basinflow benchmark``; the example script of
the same name is a thin wrapper so there is exactly one implementation.
"""
from __future__ import annotations

from typing import Any

from basinflow.seeds.from_config import init_generators
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD
from basinflow.workflows.config import parse_bool, parse_pbc_override, read_config

import json
from pathlib import Path


from basinflow.evaluation.semantic_flow import run_gaussian_trials, write_loss_artifacts
from basinflow.data import EventFlowDataset, comparison_groups, load_groups
from basinflow.workflows.dataset import load_configured_dataset
from basinflow.models import FlowLossWeights
from basinflow.models.factory import build_model, model_cutoff, model_spec
from basinflow.relaxation import RelaxationConfig, eam_calculator, prepare_eam_potential
from basinflow.training import train_product_flow


REQUIRED = {
    "data": ("events_dir",),
    "output": ("output_dir", "log_to_stdout"),
    "model": (),
    "init": ("types", "gaussian_scale"),
    "training": ("epochs", "lr", "batch_size"),
    "loss": ("velocity_weight",),
    "runtime": ("device", "cpu_threads", "gpu_ids", "gpu_count"),
    "evaluation": ("eval_num_steps", "trials_per_test_event", "sampling_seed"),
    "relaxation": ("eam_potential", "eam_form", "relax_fmax", "relax_steps"),
}


def _runtime(torch, config) -> object:
    requested = config["runtime"].get("device", "cpu").strip().lower()
    if requested != "cpu":
        raise RuntimeError("semantic benchmark currently supports CPU execution only")
    threads = config["runtime"].getint("cpu_threads")
    if threads > 0:
        torch.set_num_threads(threads)
    return torch.device("cpu")



def _record_solver(config, *, integrator: str, atol: float, rtol: float) -> None:
    """Write the *effective* solver into the config that gets saved as the record.

    The config only carries an ``integrator`` key when a run overrides the
    default, so without this the resolved config could not say which solver
    produced the numbers.
    """
    config["evaluation"]["integrator"] = str(integrator)
    config["evaluation"]["solver_atol"] = str(atol)
    config["evaluation"]["solver_rtol"] = str(rtol)


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
    dataset, split, data_sources = load_configured_dataset(config, seed=seed)
    # A converted dataset may ship basin_table.csv next to the events. When it
    # does, candidates are scored against every known product of the same basin;
    # when it does not, scoring is per event. There is no config switch: the
    # table's presence is the switch.
    groups_path = Path(config["data"]["events_dir"].strip()) / "basin_table.csv"
    groups = comparison_groups(dataset, load_groups(groups_path, dataset)) if groups_path.is_file() else None
    if groups is not None:
        data_sources["groups_file"] = str(groups_path)
        print(f"[data] scoring against basins from {groups_path}")
    (output_dir / "data_sources.json").write_text(json.dumps(data_sources, indent=2) + "\n")
    # Only the partitions actually used are selected: an empty partition is legal
    # data, and selecting one must raise rather than silently train or evaluate on
    # nothing.
    train_ds = split.select(dataset, "train")
    test_ds = split.select(dataset, "test")
    eval_split = {"train": list(split.train_ids), "val": list(split.val_ids), "test": list(split.test_ids)}

    loss_weights = FlowLossWeights(
        velocity=config["loss"].getfloat("velocity_weight"),
        norm=config["loss"].get("norm", "mse").strip().lower(),
    )
    backend, model_config = model_spec(config["model"], default_backend="egnn")
    model = build_model(backend, model_config).to(device)
    generators = init_generators(config, seed)
    validate_each_epoch = config["training"].getboolean("validate_each_epoch", fallback=False)
    validation_dataset = EventFlowDataset(
        split.select(dataset, "val"), init_generators(config, seed), seed=seed,
    ) if validate_each_epoch else None
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
        train_ds,
        generators,
        epochs=config["training"].getint("epochs"),
        lr=config["training"].getfloat("lr"),
        log_path=output_dir / "training.log",
        log_to_stdout=parse_bool(config["output"]["log_to_stdout"]),
        device=device,
        batch_size=config["training"]["batch_size"],
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
    with (output_dir / "config.resolved.ini").open("w", encoding="utf-8") as file:
        config.write(file)
    write_loss_artifacts(output_dir / "training.log", output_dir)

    checkpoint = {
        "model_backend": backend,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "split": eval_split,
        "history": history,
        "training_contract": "pairwise_flow_matching_no_oracle",
        "training_config": dict(config["training"]),
        "init_config": {"generators": [{"type": initializer.seed_type, "scale": config["init"].getfloat("gaussian_scale"), "random_seed": seed} for initializer in generators]},
        "loss_weights": {"velocity": loss_weights.velocity, "norm": loss_weights.norm},
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
        integrator=config["evaluation"].get("integrator", DEFAULT_METHOD).strip().lower(),
        solver_atol=config["evaluation"].getfloat("solver_atol", fallback=DEFAULT_ATOL),
        solver_rtol=config["evaluation"].getfloat("solver_rtol", fallback=DEFAULT_RTOL),
        gaussian_scale=config["init"].getfloat("gaussian_scale"),
        trials_per_event=config["evaluation"].getint("trials_per_test_event"),
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        device=device,
        calculator_factory=calculator_factory,
        relaxation_config=relaxation_config,
        groups=groups,
    )
    report = {
        "dataset": {"num_events": len(dataset.event_ids)},
        "model_backend": backend,
        "model_config": model_config,
        "split": eval_split,
        "training_contract": "pairwise_flow_matching_no_oracle",
        "sampling_contract": "basin_level_no_oracle" if groups else "event_level_no_oracle",
        "sampling": summary,
        "relaxation": "eam" if calculator_factory is not None else "not_run",
        "saddle_validation": "not_run",
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset: {len(dataset.event_ids)} events")
    print(f"Split: {len(train_ds.event_ids)} train / {len(split.val_ids)} val / {len(test_ds.event_ids)} test events")
    print(f"Trials: {summary['num_trials']} ({summary['trials_per_test_event']} per test event)")
    print(f"Raw RMSD mean: {summary['raw_rmsd_angstrom']['mean']}")
    print(f"EAM-relaxed RMSD mean: {summary['eam_relaxed_rmsd_angstrom']['mean']}")
    print(f"Sampling summary: {output_dir / 'sampling' / 'summary.json'}")
    return report
