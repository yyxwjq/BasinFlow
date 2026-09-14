"""How much can the bond-change condition buy, measured without a network?

Two nearest-neighbour oracles transfer a training reaction's displacement onto
a validation reactant and score it with the benchmark's own RMSD:

``reactant only``
    nearest training reactant with the same composition.  This is the ceiling
    of an R-only map: it is exactly what a perfectly fitted R-to-P regressor
    would do if the product depended on the reactant geometry alone.
``reactant + topology``
    same, restricted to training reactions whose formed/broken bond sets match
    the validation reaction.  The difference between the two rows is the
    information the condition actually adds on this dataset.

Frames are compared with ``basinflow.evaluation.metrics.movable_mic_rmsd``, the
same function the benchmark uses, so the numbers are directly comparable with a
reported candidate RMSD.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import StructureRecord
from basinflow.evaluation.basin_recall import movable_mic_rmsd
from basinflow.geometry.bonds import BROKEN, FORMED, bond_adjacency, bond_change_labels

ARMS = ("reactant_only", "reactant_and_topology")


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_centred = source - source.mean(axis=0)
    target_centred = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(source_centred.T @ target_centred)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, target.mean(axis=0) - source.mean(axis=0) @ rotation.T


def _aligned_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    rotation, translation = _kabsch(first, second)
    return float(np.sqrt(np.mean(np.sum((first @ rotation.T + translation - second) ** 2, axis=1))))


def _signature(labels: np.ndarray, species) -> frozenset:
    entries = []
    for first, second in zip(*np.nonzero(np.triu(labels, k=1))):
        label = int(labels[first, second])
        if label in (FORMED, BROKEN):
            left, right = sorted((str(species[first]), str(species[second])))
            entries.append(f"{left}-{right}:{'F' if label == FORMED else 'B'}")
    return frozenset(entries)


def _composition(species) -> str:
    counts: dict[str, int] = defaultdict(int)
    for symbol in species:
        counts[str(symbol)] += 1
    return "".join(f"{name}{counts[name]}" for name in sorted(counts))


def _load(events_dir: str, max_atoms: int) -> list[dict]:
    catalog = EventCatalog.from_eon_directory(events_dir)
    records = []
    for event_id in catalog.event_ids:
        target = catalog.event_target(event_id)
        reactant = target.reactant
        if reactant.n_atoms > max_atoms:
            continue
        labels = bond_change_labels(
            bond_adjacency(reactant.positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
            bond_adjacency(target.target_positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
        )
        records.append({
            "event_id": event_id,
            "composition": _composition(reactant.species),
            "signature": _signature(labels, reactant.species),
            "reactant": reactant,
            "product": target.product,
            "target_positions": target.target_positions,
        })
    return records


def _transfer(source: dict, query: dict) -> np.ndarray:
    """Apply the source reaction's displacement to the query reactant frame."""
    rotation, translation = _kabsch(source["reactant"].positions, query["reactant"].positions)
    source_reactant = source["reactant"].positions @ rotation.T + translation
    source_product = source["target_positions"] @ rotation.T + translation
    return source_product - source_reactant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-events-dir", required=True)
    parser.add_argument("--test-events-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument("--neighbours", type=int, default=3, help="how many candidates to average")
    args = parser.parse_args()

    train = _load(args.train_events_dir, args.max_atoms)
    test = _load(args.test_events_dir, args.max_atoms)

    by_composition: dict[str, list[dict]] = defaultdict(list)
    for record in train:
        by_composition[record["composition"]].append(record)

    results = {arm: {"rmsd": [], "matched": 0, "unmatched": 0} for arm in ARMS}
    for query in test:
        pool = [item for item in by_composition.get(query["composition"], [])
                if item["reactant"].n_atoms == query["reactant"].n_atoms]
        for arm in ARMS:
            candidates = pool
            if arm == "reactant_and_topology":
                candidates = [item for item in pool if item["signature"] == query["signature"]]
            if not candidates:
                results[arm]["unmatched"] += 1
                continue
            ranked = sorted(
                candidates,
                key=lambda item: _aligned_rmsd(item["reactant"].positions, query["reactant"].positions),
            )[: args.neighbours]
            predictions = [_transfer(item, query) for item in ranked]
            prediction = query["reactant"].positions + np.mean(predictions, axis=0)
            predicted = StructureRecord(
                structure_id="nn",
                species=list(query["product"].species),
                positions=prediction,
                cell=query["product"].cell.copy(),
                pbc=query["product"].pbc.copy(),
                movable_mask=query["product"].movable_mask.copy(),
            )
            results[arm]["matched"] += 1
            # Score the transferred frame through the benchmark metric itself.
            results[arm]["rmsd"].append(
                movable_mic_rmsd(predicted, query["product"], query["reactant"])
            )

    report = {
        "train_events_dir": str(args.train_events_dir),
        "test_events_dir": str(args.test_events_dir),
        "num_train_events": len(train),
        "num_test_events": len(test),
        "neighbours": args.neighbours,
        "arms": {},
    }
    for arm in ARMS:
        values = np.asarray(results[arm]["rmsd"], dtype=float)
        report["arms"][arm] = {
            "matched": results[arm]["matched"],
            "unmatched": results[arm]["unmatched"],
            "rmsd_angstrom": {
                "count": int(values.size),
                "mean": float(values.mean()) if values.size else None,
                "median": float(np.median(values)) if values.size else None,
                "min": float(values.min()) if values.size else None,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
