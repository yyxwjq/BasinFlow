"""Separate on-path velocity fit from off-path rollout drift for a trained flow.

Diagnostic only. Nothing here produces basin-level proposals or reads a target
during normal inference; every target-dependent number is labelled as such.

Three questions are answered for one saved checkpoint:

1. on-path fit     - with ``pos = (1-t)*x0 + t*P`` exactly on the training
                     straight line, how well does the model reproduce
                     ``target_velocity``? Compared against the zero-velocity
                     baseline, which is the trivial "do not move" predictor.
2. rollout from x0 - the ordinary inference path, started from the seed.
3. rollout from P  - an ORACLE start at the true product. A model whose field is
                     benign at the answer stays there; a model that destroys the
                     answer cannot be rescued by better conditioning.
"""
from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.data.partitions import load_data_partitions
from basinflow.data.pyg import EventFlowDataset
from basinflow.seeds import GaussianInit, ProductInit
from basinflow.models.factory import load_model_checkpoint, model_cutoff


def _catalog_and_split(run_dir):
    config = configparser.ConfigParser()
    if not config.read(run_dir / "config.resolved.ini"):
        raise FileNotFoundError(f"no resolved configuration in {run_dir}")
    split_config = dict(config["split"]) if config.has_section("split") else {}
    pbc_text = config["data"].get("pbc_override", "auto").lower()
    pbc = None if pbc_text in {"auto", "none", ""} else [
        value.strip() == "true" for value in pbc_text.split(",")]
    catalog, split, _ = load_data_partitions(config["data"], split_config, pbc_override=pbc)
    return config, catalog, split


def _mse(prediction, target, movable):
    residual = (prediction - target)[movable]
    return float((residual ** 2).sum() / max(int(movable.sum()) * 3, 1))


def on_path_fit(model, catalog, times, *, gaussian_scale, active_threshold, seed, batch_size,
                bond_change_condition=False):
    """Velocity fit on the exact training line, against the zero-velocity baseline.

    ``norm_ratio`` and ``cosine`` separate two very different failure modes that
    a single MSE hides: shrinking a correct direction (small ratio, high cosine)
    is a calibration problem, while a near-zero ratio with a near-zero cosine
    means the field carries no direction at that time at all.
    """
    rows = []
    for index, time in enumerate(times):
        dataset = EventFlowDataset(
            catalog,
            [GaussianInit(scale=gaussian_scale, random_seed=seed + index)],
            active_threshold=active_threshold, flow_time=float(time), seed=seed,
            bond_change_condition=bond_change_condition)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model_error = 0.0
        zero_error = 0.0
        atoms = 0
        predicted_norm = 0.0
        target_norm = 0.0
        inner = 0.0
        predicted_square = 0.0
        target_square = 0.0
        for batch in loader:
            with torch.no_grad():
                output = model(batch)
            movable = batch.movable_mask
            count = int(movable.sum())
            predicted = output["velocity"][movable]
            target = batch.target_velocity[movable]
            model_error += _mse(output["velocity"], batch.target_velocity, movable) * count
            zero_error += _mse(torch.zeros_like(batch.target_velocity), batch.target_velocity, movable) * count
            atoms += count
            predicted_norm += float(predicted.norm(dim=1).sum())
            target_norm += float(target.norm(dim=1).sum())
            inner += float((predicted * target).sum())
            predicted_square += float(predicted.pow(2).sum())
            target_square += float(target.pow(2).sum())
        model_mse = model_error / max(atoms, 1)
        zero_mse = zero_error / max(atoms, 1)
        rows.append({
            "flow_time": float(time),
            "model_velocity_mse": model_mse,
            "zero_velocity_mse": zero_mse,
            "variance_explained": 1.0 - (model_mse / zero_mse if zero_mse else float("nan")),
            "mean_predicted_norm": predicted_norm / max(atoms, 1),
            "mean_target_norm": target_norm / max(atoms, 1),
            "norm_ratio": predicted_norm / target_norm if target_norm else float("nan"),
            "cosine": inner / np.sqrt(predicted_square * target_square)
            if predicted_square and target_square else float("nan"),
        })
    return rows


def rollout(model, catalog, initializer, *, num_steps, active_threshold, seed,
            batch_size, oracle, bond_change_condition=False):
    """Euler rollout of the learned field from the seed's own starting geometry."""
    dataset = EventFlowDataset(
        catalog, [initializer], active_threshold=active_threshold, flow_time=0.0,
        seed=seed, diagnostic_oracle=oracle, bond_change_condition=bond_change_condition)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            start = batch.pos.clone()
            positions = batch.pos.clone()
            for step in range(num_steps):
                batch.pos = positions
                output = model(batch, t=step / num_steps)
                positions = positions + output["velocity"] / float(num_steps)
                positions[~batch.movable_mask] = batch.reactant_pos[~batch.movable_mask]
            movable = batch.movable_mask
            count = max(int(movable.sum()), 1)
            rmsd = lambda a, b: float(np.sqrt(
                ((a - b)[movable] ** 2).sum().item() / count))
            rows.append({
                "start_rmsd_vs_target": rmsd(start, batch.target_pos),
                "final_rmsd_vs_target": rmsd(positions, batch.target_pos),
                "total_motion": rmsd(positions, start),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new diagnostic file (json)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-events", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--times", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="defaults to <run-dir>/checkpoint.pt")
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic: {args.output}")

    config, catalog, split = _catalog_and_split(args.run_dir)
    checkpoint_path = args.checkpoint or args.run_dir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" in checkpoint:
        model = load_model_checkpoint(checkpoint, device="cpu")
    else:
        # Epoch checkpoints carry weights only, so the backbone is rebuilt from
        # the run's resolved config, exactly as tools/evaluate_checkpoint.py does.
        from basinflow.models.factory import build_model, model_spec

        backend, model_config = model_spec(config["model"])
        model = build_model(backend, model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    bond_change_condition = bool(getattr(model, "bond_change_condition", False))
    cutoff = model_cutoff(model)

    ids = list({"train": split.train_ids, "val": split.val_ids,
                "test": split.test_ids}[args.split])
    chosen = sorted(np.random.default_rng(args.seed).choice(
        ids, size=min(args.num_events, len(ids)), replace=False).tolist())
    subset_catalog = catalog.subset(chosen)

    gaussian_scale = config["init"].getfloat("gaussian_scale")
    active_threshold = config["data"].getfloat("active_threshold")
    training_seed = int(config["training"].getint("seed", 42))

    fit = on_path_fit(model, subset_catalog, args.times, gaussian_scale=gaussian_scale,
                      active_threshold=active_threshold, seed=training_seed,
                      batch_size=args.batch_size,
                      bond_change_condition=bond_change_condition)
    seed_rollout = rollout(
        model, subset_catalog,
        GaussianInit(scale=gaussian_scale, random_seed=args.seed),
        num_steps=args.rollout_steps, active_threshold=active_threshold,
        seed=training_seed, batch_size=args.batch_size, oracle=False,
        bond_change_condition=bond_change_condition)
    oracle_rollout = rollout(
        model, subset_catalog, ProductInit(), num_steps=args.rollout_steps,
        active_threshold=active_threshold, seed=training_seed,
        batch_size=args.batch_size, oracle=True,
        bond_change_condition=bond_change_condition)

    def summarize(rows, key):
        values = np.array([row[key] for row in rows])
        return {"mean": float(values.mean()), "median": float(np.median(values))}

    report = {
        "diagnostic": "on_path_fit_and_rollout_provenance",
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "split": args.split,
        "num_events": len(chosen),
        "cutoff": cutoff,
        "gaussian_scale": gaussian_scale,
        "active_threshold": active_threshold,
        "oracle_inputs": True,
        "warning": "the oracle rollout conditions on the true product and is a diagnostic, "
                   "never a basin-level proposal capability",
        "on_path_velocity_fit": fit,
        "rollout_from_seed": {
            "start_rmsd_vs_target": summarize(seed_rollout, "start_rmsd_vs_target"),
            "final_rmsd_vs_target": summarize(seed_rollout, "final_rmsd_vs_target"),
            "total_motion": summarize(seed_rollout, "total_motion"),
        },
        "rollout_from_true_product": {
            "start_rmsd_vs_target": summarize(oracle_rollout, "start_rmsd_vs_target"),
            "final_rmsd_vs_target": summarize(oracle_rollout, "final_rmsd_vs_target"),
            "total_motion": summarize(oracle_rollout, "total_motion"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
