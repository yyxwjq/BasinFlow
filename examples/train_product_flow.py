"""
Train the BasinFlow product/event flow on an events folder.

Default dataset: /Users/wx/Desktop/benchmark/au/events.
Default outputs: runs/au_product_flow/checkpoint.pt, training.log.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.data import (EventFlowDataset, Structure, event_target, load_events,
                            split_events)
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD, sample_ode
from basinflow.evaluation import movable_mic_rmsd
from basinflow.models import FlowLossWeights
from basinflow.models.factory import build_model, model_spec
from basinflow.seeds import (
    GaussianInit,
    ProductInit,
    ZeroInit,
)
from basinflow.training import train_product_flow


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"num_items": 0, "mean_rmsd": None}
    return {"num_items": len(values), "mean_rmsd": float(sum(values) / len(values))}


def _initialization_rmsd(catalog, generators) -> dict:
    """Report diagnostic seed quality without exposing targets to normal seeds."""
    from basinflow.seeds import SeedContext

    report = {}
    for name, initializer in generators.items():
        values = []
        for event_id in catalog.event_ids:
            target = event_target(catalog.events[event_id])
            context = SeedContext(event_id, target.reactant)
            kwargs = {"seed_id": f"{event_id}:{name}"}
            if getattr(initializer, "requires_target", False):
                kwargs["target_displacement"] = target.displacement
            seed = initializer.generate(context, **kwargs)
            generated = Structure(
                list(target.reactant.species),
                seed.initial_positions(target.reactant.positions),
                cell=target.reactant.cell,
                pbc=target.reactant.pbc,
                movable_mask=target.reactant.movable_mask,
            )
            values.append(movable_mic_rmsd(generated, target.product, target.reactant))
        report[name] = {"overall": _summary(values)}
    return report


def _rollout_rmsd(
    model,
    catalog,
    generators,
    *,
    num_steps: int,
    device,
    seed: int,
    integrator: str = DEFAULT_METHOD,
    solver_atol: float = DEFAULT_ATOL,
    solver_rtol: float = DEFAULT_RTOL,
) -> dict:
    """Pairwise no-relaxation RMSD, using labels only after rollout.

    The rollout goes through the shared ``torchdiffeq`` solver, the same one the
    sampler and MolGEN use.
    """
    import torch
    from torch_geometric.loader import DataLoader

    was_training = model.training
    model.eval()
    report = {}
    try:
        for initializer in generators:
            dataset = EventFlowDataset(catalog, [initializer], flow_time=0.0, seed=seed)
            values = []
            for batch in DataLoader(dataset, batch_size=1, shuffle=False):
                batch = batch.to(device)
                frozen = ~batch.movable_mask

                def field(state, time):
                    batch.pos = state
                    batch.flow_time = time.reshape(1, 1).expand(batch.num_graphs, 1)
                    velocity = model(batch, t=batch.flow_time.squeeze(-1))["velocity"]
                    return torch.where(frozen[:, None], torch.zeros_like(velocity), velocity)

                positions, _ = sample_ode(field, batch.pos.clone(), num_steps=num_steps,
                                          atol=solver_atol, rtol=solver_rtol, method=integrator)
                event_id = batch.event_id[0]
                target = event_target(catalog.events[event_id])
                generated = Structure(
                    list(target.reactant.species),
                    positions.detach().cpu().numpy(),
                    cell=target.reactant.cell,
                    pbc=target.reactant.pbc,
                    movable_mask=target.reactant.movable_mask,
                )
                values.append(movable_mic_rmsd(generated, target.product, target.reactant))
            report[initializer.seed_type] = {"overall": _summary(values)}
    finally:
        if was_training:
            model.train()
    return report


def _default_events_dir() -> str:
    return os.environ.get(
        "BASINFLOW_EVENTS_DIR",
        "/Users/wx/Desktop/benchmark/au/events",
    )


def _parse_pbc_override(text: str):
    value = text.strip().lower()
    if value in {"", "none", "auto"}:
        return None
    tokens = [item.strip() for item in value.split(",")]
    if len(tokens) != 3:
        raise argparse.ArgumentTypeError("pbc override must have three comma-separated booleans")
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n"}
    parsed = []
    for token in tokens:
        if token in truthy:
            parsed.append(True)
        elif token in falsy:
            parsed.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid pbc boolean: {token!r}")
    return parsed


DEFAULT_CONFIG = {
    "data": {
        "events_dir": _default_events_dir(),
        "pbc_override": "auto",
    },
    "output": {
        "output_dir": "runs/au_product_flow",
        "log_to_stdout": "true",
    },
    "split": {
        "train": "0.7",
        "val": "0.15",
        "test": "0.15",
        "seed": "42",
    },
    "model": {
        "backend": "painn",
        "hidden_dim": "128",
        "num_layers": "4",
        "cutoff": "5.0",
        "num_rbf": "32",
        "time_dim": "32",
        "radial_basis": "bessel",
        "envelope": "polynomial",
    },
    "init": {
        "types": "zero, gaussian",
        "gaussian_scale": "0.05",
    },
    "training": {
        "epochs": "20",
        "lr": "0.001",
        "batch_size": "1",
    },
    "loss": {
        "velocity_weight": "1.0",
        "norm": "mse",
    },
    "runtime": {
        "device": "auto",
        "cpu_threads": "0",
        "gpu_ids": "",
        "gpu_count": "1",
    },
    "evaluation": {
        "eval_num_steps": "50",
        "integrator": "dopri5",
    },
}


def _config_with_defaults() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    return config


def _load_config(path: str | None) -> configparser.ConfigParser:
    config = _config_with_defaults()
    if path is None:
        return config
    read_files = config.read(path)
    if not read_files:
        raise FileNotFoundError(f"config file not found: {path}")
    explicit = configparser.ConfigParser()
    explicit.read(path)
    if not explicit.has_option("model", "backend"):
        config["model"]["backend"] = "egnn"
    for canonical, alias in (("num_features", "hidden_dim"), ("r_max", "cutoff"), ("num_radial_basis", "num_rbf")):
        if explicit.has_option("model", canonical):
            config["model"][alias] = explicit["model"][canonical]
    return config


def _get_config_value(config: configparser.ConfigParser, section: str, key: str):
    if not config.has_section(section) and section not in config.defaults():
        raise KeyError(f"missing config section [{section}]")
    return config[section][key]


def _parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {text!r}")


def _parse_init_types(text: str) -> list[str]:
    values = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("init types must include at least one generator")
    allowed = {"zero", "gaussian"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unsupported training init type(s): {', '.join(unknown)}")
    return values


def _parse_gpu_ids(text: str) -> list[int]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    return [int(item) for item in values]


def _select_runtime(torch, *, requested_device: str, cpu_threads: int, gpu_ids: list[int], gpu_count: int) -> dict:
    if cpu_threads < 0:
        raise ValueError("cpu_threads must be non-negative")
    if gpu_count < 0:
        raise ValueError("gpu_count must be non-negative")
    if gpu_count > 1:
        raise ValueError(
            "gpu_count > 1 is not supported by the current dynamic-graph EGNN trainer; "
            "choose one GPU with gpu_ids or set gpu_count = 1"
        )
    if cpu_threads > 0:
        torch.set_num_threads(int(cpu_threads))

    requested = str(requested_device).strip().lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if requested == "cpu":
        device_text = "cpu"
    elif requested == "cuda":
        if gpu_count == 0:
            raise ValueError("device=cuda requires gpu_count > 0")
        if not cuda_available:
            raise RuntimeError("device=cuda requested, but CUDA is not available")
        gpu_id = gpu_ids[0] if gpu_ids else 0
        torch.cuda.set_device(gpu_id)
        device_text = f"cuda:{gpu_id}"
    elif requested == "mps":
        if not mps_available:
            raise RuntimeError("device=mps requested, but MPS is not available")
        device_text = "mps"
    else:
        if gpu_count > 0 and cuda_available:
            gpu_id = gpu_ids[0] if gpu_ids else 0
            torch.cuda.set_device(gpu_id)
            device_text = f"cuda:{gpu_id}"
        elif mps_available:
            device_text = "mps"
        else:
            device_text = "cpu"

    return {
        "requested_device": requested,
        "device": device_text,
        "cpu_threads": int(cpu_threads),
        "gpu_ids": list(gpu_ids),
        "gpu_count": int(gpu_count),
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "torch_num_threads": int(torch.get_num_threads()),
    }


def _resolved_config(args: argparse.Namespace) -> configparser.ConfigParser:
    config = _config_with_defaults()
    config["data"] = {
        "events_dir": str(args.events_dir),
        "pbc_override": "auto" if args.pbc_override is None else ",".join(str(v).lower() for v in args.pbc_override),
    }
    config["output"] = {
        "output_dir": str(args.output_dir),
        "log_to_stdout": str(bool(args.log_to_stdout)).lower(),
    }
    config["split"] = {
        "train": str(args.train),
        "val": str(args.val),
        "test": str(args.test),
        "seed": str(args.seed),
    }
    config["model"] = {
        "backend": args.backend,
        "hidden_dim": str(args.hidden_dim),
        "num_layers": str(args.num_layers),
        "cutoff": str(args.cutoff),
        "num_rbf": str(args.num_rbf),
        "time_dim": str(args.time_dim),
        "radial_basis": args.radial_basis,
        "envelope": args.envelope,
    }
    config["init"] = {
        "types": ", ".join(args.init_types),
        "gaussian_scale": str(args.gaussian_scale),
    }
    config["training"] = {
        "epochs": str(args.epochs),
        "lr": str(args.lr),
        "batch_size": str(args.batch_size),
    }
    config["loss"] = {
        "velocity_weight": str(args.velocity_weight),
        "norm": str(args.norm),
    }
    config["runtime"] = {
        "device": str(args.device),
        "cpu_threads": str(args.cpu_threads),
        "gpu_ids": ",".join(str(item) for item in args.gpu_ids),
        "gpu_count": str(args.gpu_count),
    }
    config["evaluation"] = {
        "eval_num_steps": str(args.eval_num_steps),
        "integrator": str(args.integrator),
        "solver_atol": str(args.solver_atol),
        "solver_rtol": str(args.solver_rtol),
    }
    return config


def _write_resolved_config(args: argparse.Namespace, path: Path) -> None:
    config = _resolved_config(args)
    with path.open("w", encoding="utf-8") as file:
        config.write(file)


def _config_or_cli(raw_args: argparse.Namespace, config: configparser.ConfigParser, attr: str,
                   section: str, key: str, cast, default=None):
    value = getattr(raw_args, attr)
    if value is not None:
        return value
    if default is not None and not config.has_option(section, key):
        return default
    return cast(_get_config_value(config, section, key))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BasinFlow product/event flow on an events directory.",
    )
    parser.add_argument("--config", default=None, help="Path to an INI training config.")
    parser.add_argument("--events-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--backend", choices=["painn", "egnn"], default=None)
    parser.add_argument("--num-rbf", type=int, default=None)
    parser.add_argument("--time-dim", type=int, default=None)
    parser.add_argument("--radial-basis", default=None)
    parser.add_argument("--envelope", default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--train", type=float, default=None)
    parser.add_argument("--val", type=float, default=None)
    parser.add_argument("--test", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pbc-override", type=_parse_pbc_override, default=None)
    parser.add_argument("--gaussian-scale", type=float, default=None)
    parser.add_argument(
        "--init-types",
        type=_parse_init_types,
        default=None,
        help="Comma-separated training init generators: zero,gaussian.",
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        help="Training batch size as a positive integer, or 'full' for one update per epoch.",
    )
    parser.add_argument("--velocity-weight", type=float, default=None)
    parser.add_argument("--norm", choices=["mse", "l1"], default=None)
    parser.add_argument(
        "--active-pos-weight",
        default=None,
        help="Positive active-label BCE weight, or 'auto' for train-split balancing.",
    )
    parser.add_argument("--eval-num-steps", type=int, default=None,
                        help="saved points on the integration grid (MolGEN --inference_steps)")
    parser.add_argument("--integrator", choices=["dopri5", "euler"], default=None)
    parser.add_argument("--solver-atol", type=float, default=None)
    parser.add_argument("--solver-rtol", type=float, default=None)
    parser.add_argument("--log-to-stdout", action="store_true", default=None)
    parser.add_argument("--no-log-to-stdout", action="store_false", dest="log_to_stdout")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--gpu-ids", type=_parse_gpu_ids, default=None)
    parser.add_argument("--gpu-count", type=int, default=None)
    raw_args = parser.parse_args()
    config = _load_config(raw_args.config)
    return argparse.Namespace(
        config=raw_args.config,
        events_dir=_config_or_cli(raw_args, config, "events_dir", "data", "events_dir", str),
        output_dir=_config_or_cli(raw_args, config, "output_dir", "output", "output_dir", str),
        epochs=_config_or_cli(raw_args, config, "epochs", "training", "epochs", int),
        lr=_config_or_cli(raw_args, config, "lr", "training", "lr", float),
        hidden_dim=_config_or_cli(raw_args, config, "hidden_dim", "model", "hidden_dim", int),
        backend=_config_or_cli(raw_args, config, "backend", "model", "backend", str),
        num_rbf=_config_or_cli(raw_args, config, "num_rbf", "model", "num_rbf", int),
        time_dim=_config_or_cli(raw_args, config, "time_dim", "model", "time_dim", int),
        radial_basis=_config_or_cli(raw_args, config, "radial_basis", "model", "radial_basis", str),
        envelope=_config_or_cli(raw_args, config, "envelope", "model", "envelope", str),
        num_layers=_config_or_cli(raw_args, config, "num_layers", "model", "num_layers", int),
        cutoff=_config_or_cli(raw_args, config, "cutoff", "model", "cutoff", float),
        train=_config_or_cli(raw_args, config, "train", "split", "train", float),
        val=_config_or_cli(raw_args, config, "val", "split", "val", float),
        test=_config_or_cli(raw_args, config, "test", "split", "test", float),
        seed=_config_or_cli(raw_args, config, "seed", "split", "seed", int),
        pbc_override=_config_or_cli(
            raw_args,
            config,
            "pbc_override",
            "data",
            "pbc_override",
            _parse_pbc_override,
        ),
        gaussian_scale=_config_or_cli(raw_args, config, "gaussian_scale", "init", "gaussian_scale", float),
        init_types=_config_or_cli(raw_args, config, "init_types", "init", "types", _parse_init_types),
        batch_size=_config_or_cli(raw_args, config, "batch_size", "training", "batch_size", str),
        velocity_weight=_config_or_cli(raw_args, config, "velocity_weight", "loss", "velocity_weight", float),
        norm=_config_or_cli(raw_args, config, "norm", "loss", "norm", str),
        eval_num_steps=_config_or_cli(raw_args, config, "eval_num_steps", "evaluation", "eval_num_steps", int),
        # The solver controls default to MolGEN's, so a config written before them
        # still runs instead of failing on a missing option.
        integrator=_config_or_cli(raw_args, config, "integrator", "evaluation", "integrator", str,
                                  default=DEFAULT_METHOD),
        solver_atol=_config_or_cli(raw_args, config, "solver_atol", "evaluation", "solver_atol", float,
                                   default=DEFAULT_ATOL),
        solver_rtol=_config_or_cli(raw_args, config, "solver_rtol", "evaluation", "solver_rtol", float,
                                   default=DEFAULT_RTOL),
        log_to_stdout=_config_or_cli(raw_args, config, "log_to_stdout", "output", "log_to_stdout", _parse_bool),
        device=_config_or_cli(raw_args, config, "device", "runtime", "device", str),
        cpu_threads=_config_or_cli(raw_args, config, "cpu_threads", "runtime", "cpu_threads", int),
        gpu_ids=_config_or_cli(raw_args, config, "gpu_ids", "runtime", "gpu_ids", _parse_gpu_ids),
        gpu_count=_config_or_cli(raw_args, config, "gpu_count", "runtime", "gpu_count", int),
    )


def main() -> None:
    args = parse_args()
    import torch

    runtime_config = _select_runtime(
        torch,
        requested_device=args.device,
        cpu_threads=args.cpu_threads,
        gpu_ids=args.gpu_ids,
        gpu_count=args.gpu_count,
    )
    device = torch.device(runtime_config["device"])
    torch.manual_seed(args.seed)
    events_dir = Path(args.events_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(args, output_dir / "config.resolved.ini")

    catalog = load_events(events_dir, pbc_override=args.pbc_override)
    split = split_events(catalog, train=args.train, val=args.val, test=args.test, seed=args.seed)
    # Only the partitions actually used are selected: an empty partition is legal
    # data, and selecting one must raise rather than run on nothing.
    train_catalog = split.select(catalog, "train")
    val_catalog = (
        split.select(catalog, "val")
        if any(value > 0.0 for value in (args.val,)) and split.val_ids
        else train_catalog
    )
    test_catalog = split.select(catalog, "test") if split.test_ids else train_catalog
    split_record = {
        "train": list(split.train_ids),
        "val": list(split.val_ids),
        "test": list(split.test_ids),
    }

    backend, model_config = model_spec(vars(args))
    model = build_model(backend, model_config).to(device)
    init_generators = []
    init_config_generators = []
    for init_type in args.init_types:
        if init_type == "zero":
            init_generators.append(ZeroInit())
            init_config_generators.append({"type": "zero"})
        elif init_type == "gaussian":
            init_generators.append(
                GaussianInit(
                    scale=args.gaussian_scale,
                    random_seed=args.seed,
                )
            )
            init_config_generators.append(
                {
                    "type": "gaussian_movable",
                    "scale": args.gaussian_scale,
                    "random_seed": args.seed,
                }
            )
    init_config = {"generators": init_config_generators}
    loss_weights = FlowLossWeights(velocity=args.velocity_weight, norm=args.norm)

    train_dataset = EventFlowDataset(train_catalog, init_generators, seed=args.seed)
    batch_size = len(train_dataset) if args.batch_size == "full" else int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or 'full'")
    if args.log_to_stdout:
        print("epoch step total_loss velocity_loss active_loss direction_loss")
    history = train_product_flow(
        model,
        train_dataset,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=None,
        log_path=output_dir / "training.log",
        log_to_stdout=args.log_to_stdout,
        device=device,
        batch_size=batch_size,
        loss_weights=loss_weights,
    )
    split_catalogs = {
        "train": train_catalog,
        "val": val_catalog,
        "test": test_catalog,
    }
    baseline_generators = {
        "zero": ZeroInit(),
        "gaussian_movable": GaussianInit(
            scale=args.gaussian_scale,
            random_seed=args.seed,
        ),
        "oracle_product_displacement": ProductInit(),
    }
    eval_metrics = {
        "model_rollout": {
            "splits": {
                split_name: _rollout_rmsd(
                    model,
                    split_catalog,
                    init_generators,
                    num_steps=args.eval_num_steps,
                    integrator=args.integrator,
                    solver_atol=args.solver_atol,
                    solver_rtol=args.solver_rtol,
                    device=device,
                    seed=args.seed,
                )
                for split_name, split_catalog in split_catalogs.items()
            }
        },
        "init_baseline": _initialization_rmsd(catalog, baseline_generators),
        "init_baseline_splits": {
            split_name: _initialization_rmsd(
                split_catalog,
                baseline_generators,
            )
            for split_name, split_catalog in split_catalogs.items()
        },
        "config": {
            "num_steps": args.eval_num_steps,
            "integrator": args.integrator,
            "solver_atol": args.solver_atol,
            "solver_rtol": args.solver_rtol,
            "cutoff": args.cutoff,
        },
    }

    checkpoint = {
        "model_backend": backend,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "init_config": init_config,
        "split": split_record,
        "runtime_config": runtime_config,
        "training_config": {
            "events_dir": str(events_dir),
            "config": args.config,
            "epochs": args.epochs,
            "lr": args.lr,
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "seed": args.seed,
            "pbc_override": args.pbc_override,
            "batch_size": args.batch_size,
            "loss_weights": {"velocity": loss_weights.velocity, "norm": loss_weights.norm},
            "flow_time_sampling": "uniform",
            "init_types": list(args.init_types),
            "gaussian_scale": args.gaussian_scale,
            "eval_num_steps": args.eval_num_steps,
            "integrator": args.integrator,
            "solver_atol": args.solver_atol,
            "solver_rtol": args.solver_rtol,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "gpu_ids": args.gpu_ids,
            "gpu_count": args.gpu_count,
        },
        "eval_metrics": eval_metrics,
        "history": history,
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Loaded: {len(catalog.event_ids)} events")
    print(
        "Split: "
        f"{len(train_catalog.event_ids)} train / "
        f"{len(split.val_ids)} val / "
        f"{len(test_catalog.event_ids)} test events"
    )
    print(f"Final loss: {history[-1]['loss']:.6f}")
    print(f"Runtime device: {runtime_config['device']}")
    test_rollout = eval_metrics["model_rollout"]["splits"]["test"]
    test_init_metrics = test_rollout.get("zero") or next(iter(test_rollout.values()), None)
    test_rmsd = test_init_metrics["overall"]["mean_rmsd"] if test_init_metrics else None
    oracle_test = eval_metrics["init_baseline_splits"]["test"]["oracle_product_displacement"]["overall"]["mean_rmsd"]
    print("Test rollout RMSD: n/a" if test_rmsd is None else f"Test rollout RMSD: {test_rmsd:.6f} Å")
    print("Oracle product init RMSD: n/a" if oracle_test is None else f"Oracle product init RMSD: {oracle_test:.6f} Å")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {len(split.train_ids)}/{len(split.val_ids)}/{len(split.test_ids)} (train/val/test)")
    print(f"Training log: {output_dir / 'training.log'}")
    print(f"Eval metrics: {output_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
