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
from basinflow.data import EventFlowDataset, single_event_scope
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD
from basinflow.workflows.dataset import load_configured_dataset
from basinflow.models import FlowLossWeights
from basinflow.models.factory import build_model, model_cutoff, model_spec
from basinflow.training import train_product_flow


REQUIRED = {
    "data": ("events_dir",),
    "output": ("output_dir", "log_to_stdout"),
    "model": (),
    "init": ("types", "gaussian_scale"),
    "training": ("epochs", "lr", "batch_size"),
    "loss": ("velocity_weight",),
    "runtime": ("device", "cpu_threads"),
    "evaluation": ("eval_num_steps", "trials_per_event", "sampling_seed", "selection_seed"),
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
    catalog, split, data_sources = load_configured_dataset(config, seed=seed)
    # ``single_event`` drops the dissociations whose product is a separated
    # fragment pair, so training and evaluation cover the same event class
    # (docs/22 section 二之二). It is recorded in data_sources either way.
    training_scope = config["data"].get("training_scope", "all").strip().lower()
    if training_scope not in {"all", "single_event"}:
        raise ValueError("training_scope must be 'all' or 'single_event'")
    scope_filter = single_event_scope if training_scope == "single_event" else (lambda part: part)
    train_catalog = scope_filter(split.select(catalog, "train"))
    val_catalog = scope_filter(split.select(catalog, "val"))
    evaluation_split = config["evaluation"].get("split", "test").strip().lower()
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    # Only the partition actually used is selected. An empty partition is legal
    # data; selecting it would raise, and it must, rather than silently evaluating
    # on nothing.
    evaluation_catalog = (
        val_catalog if evaluation_split == "val" else scope_filter(split.select(catalog, "test"))
    )
    data_sources = {
        **data_sources,
        "training_scope": training_scope,
        "num_events_by_partition": {
            name: len(getattr(split, f"{name}_ids")) for name in ("train", "val", "test")
        },
    }
    (output_dir / "data_sources.json").write_text(json.dumps(data_sources, indent=2) + "\n")
    loss_weights = FlowLossWeights(
        velocity=config["loss"].getfloat("velocity_weight"),
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
        seed=seed,
        time_distribution=time_distribution,
        beta_alpha=beta_alpha,
        flow_time=flow_time,
        bond_change_condition=bond_change_condition,
        bond_change_source=bond_change_source,
    )
    validation_dataset = EventFlowDataset(
        val_catalog, init_generators(config, seed), seed=seed,
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
    config["evaluation"]["evaluation_protocol"] = "single_event_pairwise_no_relaxation"
    _record_solver(
        config,
        integrator=config["evaluation"].get("integrator", DEFAULT_METHOD).strip().lower(),
        atol=config["evaluation"].getfloat("solver_atol", fallback=DEFAULT_ATOL),
        rtol=config["evaluation"].getfloat("solver_rtol", fallback=DEFAULT_RTOL),
    )
    with (output_dir / "config.resolved.ini").open("w", encoding="utf-8") as file:
        config.write(file)
    write_loss_artifacts(output_dir / "training.log", output_dir)
    checkpoint = {
        "model_backend": backend,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "split": {
            "train": list(train_catalog.event_ids),
            "val": list(val_catalog.event_ids),
            "test": list(split.test_ids),
            "training_scope": training_scope,
        },
        "history": history,
        "training_contract": "transition1x_pairwise_r_to_p_no_oracle",
        "evaluation_split": evaluation_split,
        "split_source": data_sources.get("split_file") or "all_train",
        "training_config": dict(config["training"]),
        "init_config": {"generators": [{"type": initializer.seed_type, "scale": config["init"].getfloat("gaussian_scale"), "random_seed": seed} for initializer in init_generators(config, seed)]},
        "loss_weights": {"velocity": loss_weights.velocity, "norm": loss_weights.norm},
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    sampling_summary = run_single_event_molecule_trials(
        model=model,
        test_catalog=evaluation_catalog,
        output_dir=output_dir,
        cutoff=model_cutoff(model),
        num_steps=config["evaluation"].getint("eval_num_steps"),
        integrator=config["evaluation"].get("integrator", DEFAULT_METHOD).strip().lower(),
        solver_atol=config["evaluation"].getfloat("solver_atol", fallback=DEFAULT_ATOL),
        solver_rtol=config["evaluation"].getfloat("solver_rtol", fallback=DEFAULT_RTOL),
        gaussian_scale=config["init"].getfloat("gaussian_scale"),
        trials_per_event=config["evaluation"].getint("trials_per_event"),
        num_test_events=config["evaluation"].getint("num_test_events", fallback=None),
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        selection_seed=config["evaluation"].getint("selection_seed"),
        device=device,
        bond_change_source=config["evaluation"].get("bond_change_source", "seed").strip().lower(),
    )
    report = {
        "dataset": {
            "kind": "transition1x_single_event_molecule",
            "num_events": len(catalog.event_ids),
        },
        "split": {
            "train": len(train_catalog.event_ids),
            "val": len(val_catalog.event_ids),
            "test": len(split.test_ids),
        },
        "training_contract": "pairwise_r_to_p_no_oracle",
        "training_scope": training_scope,
        "flow_time": flow_time,
        "bond_change": {
            "enabled": bond_change_condition,
            "training_source": bond_change_source,
            "sampling_source": config["evaluation"].get("bond_change_source", "seed").strip().lower(),
        },
        "evaluation_split": evaluation_split,
        "split_source": data_sources.get("split_file") or "all_train",
        "model_backend": backend,
        "model_config": model_config,
        "sampling": sampling_summary,
        "relaxation": "not_run",
        "saddle_validation": "not_run",
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset: {len(catalog.event_ids)} Transition1x events")
    print(f"Split: {len(train_catalog.event_ids)} train / {len(val_catalog.event_ids)} val / {len(split.test_ids)} test")
    print(f"Final loss: {history[-1]['loss']:.6f}")
    print(f"Sampling trials: {sampling_summary['num_trials']}")
    print(f"Raw RMSD mean: {sampling_summary['raw_rmsd_angstrom']['mean']}")
    print(f"Checkpoint: {output_dir / 'checkpoint.pt'}")
    print(f"Sampling summary: {output_dir / 'sampling' / 'summary.json'}")
    return report
