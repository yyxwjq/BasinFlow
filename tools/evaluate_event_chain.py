"""End-to-end ``R_basin -> {P, TS}`` chain: does a predicted product still give a TS?

The project's contract is that a reactant basin alone yields a product *and* a
transition state. The two models are trained separately, so the open question is
how much the product error costs the downstream transition state. This tool
answers it on one split with four arms, all scored against the recorded
transition state with the same aligned RMSD:

``midpoint of true endpoints``
    the trivial baseline ``(R + P) / 2``, no learning at all.
``TS from true endpoints``
    the upper bound: the transition-state model gets the real product.
``midpoint of predicted endpoints``
    ``(R + P̂) / 2``, the trivial baseline once the product is predicted.
``TS from predicted endpoints``
    the deployable chain: R -> P̂ -> T̂S with no oracle input after the first step.

The bond-change condition is an oracle in the first step, exactly as in the
R-to-P benchmark, so the chain number is an upper bound on a seed-conditioned
system rather than a claim of a fully autonomous one.
"""
from __future__ import annotations

import argparse
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
from basinflow.data.pyg import EventFlowDataset, TransitionStateDataset
from basinflow.data.records import StructureRecord
from basinflow.evaluation.basin_recall import kabsch_aligned_rmsd
from basinflow.geometry.bonds import bond_adjacency, preserves_fragments
from basinflow.models import FlowLossWeights, flow_loss
from basinflow.models.factory import build_model, model_spec
from basinflow.seeds import ZeroInit
from basinflow.workflows.config import read_config
from basinflow.workflows.transition1x import REQUIRED as RP_REQUIRED
from basinflow.workflows.transition_state import REQUIRED as TS_REQUIRED


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "clipped_mean": float(np.minimum(array, 1.0).mean()),
        "below_0.2": float((array < 0.2).mean()),
        "below_0.5": float((array < 0.5).mean()),
    }


def _load_model(run_dir: Path, checkpoint: Path, required):
    config = read_config(run_dir / "config.resolved.ini", required)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_config" in saved:
        backend, model_config = saved.get("model_backend", "painn"), saved["model_config"]
    else:
        backend, model_config = model_spec(config["model"])
    model = build_model(backend, model_config)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rp-run-dir", required=True, type=Path)
    parser.add_argument("--rp-checkpoint", type=Path, default=None)
    parser.add_argument("--ts-run-dir", required=True, type=Path)
    parser.add_argument("--ts-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", required=True, type=Path, help="aligned export with transition states")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-events", type=int, default=128)
    parser.add_argument("--single-event-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rp_ckpt = args.rp_checkpoint or sorted((args.rp_run_dir / "epoch_checkpoints").glob("epoch_*.pt"))[-1]
    ts_ckpt = args.ts_checkpoint or sorted((args.ts_run_dir / "epoch_checkpoints").glob("epoch_*.pt"))[-1]
    rp_model, rp_config = _load_model(args.rp_run_dir, rp_ckpt, RP_REQUIRED)
    ts_model, ts_config = _load_model(args.ts_run_dir, ts_ckpt, TS_REQUIRED)

    catalog, split, _ = load_data_partitions(
        {"train_events_dir": str(args.data_dir / "train"),
         "val_events_dir": str(args.data_dir / "val"),
         "test_events_dir": str(args.data_dir / "test")}, {})
    part = split.select(catalog, args.split)
    ids = list(part.basin_ids)
    if args.single_event_only:
        ids = [bid for bid in ids if preserves_fragments(
            bond_adjacency(part.structures[part.basins[bid].reactant_structure_id].positions,
                           part.structures[part.basins[bid].reactant_structure_id].atomic_numbers,
                           part.structures[part.basins[bid].reactant_structure_id].cell,
                           part.structures[part.basins[bid].reactant_structure_id].pbc),
            bond_adjacency(part.event_target(part.basins[bid].known_event_ids[0]).target_positions,
                           part.structures[part.basins[bid].reactant_structure_id].atomic_numbers,
                           part.structures[part.basins[bid].reactant_structure_id].cell,
                           part.structures[part.basins[bid].reactant_structure_id].pbc))]
    ids = ids[: args.num_events]
    scoped = catalog.subset(ids)

    # Stage one: reactant -> predicted product, oracle bond-change condition.
    rp_dataset = EventFlowDataset(scoped, [ZeroInit()], flow_time=0.0, seed=0,
                                  bond_change_condition=True)
    predicted = {}
    with torch.no_grad():
        for batch in DataLoader(rp_dataset, batch_size=8, shuffle=False):
            pos = (batch.pos + rp_model(batch)["velocity"]).clone()
            pos[~batch.movable_mask] = batch.reactant_pos[~batch.movable_mask]
            start = 0
            for graph in range(batch.num_graphs):
                count = int((batch.batch == graph).sum())
                sl = slice(start, start + count); start += count
                predicted[batch.event_id[graph]] = pos[sl].numpy()
    rp_target = {event_id: scoped.event_target(event_id).target_positions for event_id in scoped.event_ids}

    # Stage two: the recorded transition-state model, fed either endpoint pair.
    ts_dataset = TransitionStateDataset(scoped, flow_time=0.0)
    reference = {}
    with torch.no_grad():
        for batch in DataLoader(ts_dataset, batch_size=8, shuffle=False):
            reference[batch.event_id[0]] = (batch.reactant_pos.numpy(), batch.target_pos.numpy())

    def ts_rollout(endpoints: dict) -> dict:
        dataset = TransitionStateDataset(scoped, flow_time=0.0)
        out = {}
        with torch.no_grad():
            for index, event_id in enumerate(scoped.event_ids):
                item = dataset[index]
                product = endpoints[event_id]
                item.endpoint_pos = torch.as_tensor(product, dtype=torch.float32)
                item.pos = torch.as_tensor(0.5 * (item.reactant_pos.numpy() + product), dtype=torch.float32)
                item.source_pos = item.pos.clone()
                batch = next(iter(DataLoader([item], batch_size=1, shuffle=False)))
                positions = batch.pos
                steps = ts_config["evaluation"].getint("eval_num_steps")
                for step in range(steps):
                    batch.pos = positions
                    positions = positions + ts_model(batch, t=step / float(steps))["velocity"] / float(steps)
                out[event_id] = positions.numpy()
        return out

    true_products = {event_id: rp_target[event_id] for event_id in scoped.event_ids}
    ts_from_true = ts_rollout(true_products)
    ts_from_pred = ts_rollout(predicted)

    def frame(name, positions, reference_structure):
        return StructureRecord(name, list(reference_structure.species), positions,
                               reference_structure.cell.copy(), reference_structure.pbc.copy(),
                               reference_structure.movable_mask.copy())

    arms = {key: [] for key in ("midpoint_true", "midpoint_predicted", "ts_true_endpoints", "ts_predicted_endpoints")}
    product_errors = []
    for event_id in scoped.event_ids:
        target = scoped.event_target(event_id)
        reactant = target.reactant
        state = target.transition_state.positions
        ref = frame("r", reactant.positions, reactant)
        truth = frame("t", state, reactant)
        product_errors.append(kabsch_aligned_rmsd(
            frame("p", predicted[event_id], reactant), frame("q", true_products[event_id], reactant), ref))
        for key, positions in (
            ("midpoint_true", 0.5 * (reactant.positions + true_products[event_id])),
            ("midpoint_predicted", 0.5 * (reactant.positions + predicted[event_id])),
            ("ts_true_endpoints", ts_from_true[event_id]),
            ("ts_predicted_endpoints", ts_from_pred[event_id]),
        ):
            arms[key].append(kabsch_aligned_rmsd(frame("c", positions, reactant), truth, ref))

    report = {
        "task": "reactant_basin_to_product_and_transition_state",
        "split": args.split,
        "num_events": len(scoped.event_ids),
        "single_event_only": bool(args.single_event_only),
        "rp_run_dir": str(args.rp_run_dir.resolve()),
        "rp_checkpoint": str(rp_ckpt.resolve()),
        "ts_run_dir": str(args.ts_run_dir.resolve()),
        "ts_checkpoint": str(ts_ckpt.resolve()),
        "arms": {key: _summary(values) for key, values in arms.items()},
        "product_aligned_rmsd": _summary(product_errors),
        "notice": (
            "Stage one uses the oracle bond-change condition, so every arm is an upper "
            "bound for a seed-conditioned system, not a fully autonomous one."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["arms"], indent=2))
    print("product:", json.dumps(report["product_aligned_rmsd"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
