"""Is the bond-change condition enough to identify the event?

Doc 18 section 8 shows that near-identical reactants in Transition1x ask for
very different displacements, which is why an R-only model cannot beat the
conditional mean.  The bond-change condition is the proposed fix, so this tool
measures the property that makes the fix work: among event pairs that share a
nearly identical reactant, how much disagreement in the required displacement
survives once both events are required to share the same topology change?

Two divergences are reported per bucket, in Angstrom:

``raw``
    RMS difference between the two required displacement fields.
``chemical``
    The same after each field has had its own best rigid motion removed, which
    is the part a model could in principle learn from internal geometry.

Read the result as a bound on how well any map from ``(reactant geometry,
topology change)`` to a product can do on this dataset.
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
from basinflow.geometry.bonds import BROKEN, FORMED, bond_adjacency, bond_change_labels
from basinflow.geometry.mic import minimum_image_displacement

BUCKETS = ("same_topology", "different_topology")


def _kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation that best maps centred ``source`` onto centred ``target``."""
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ correction @ u.T


def _aligned_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    a = first - first.mean(axis=0)
    b = second - second.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((a @ _kabsch_rotation(a, b).T - b) ** 2, axis=1))))


def _chemical_displacement(positions: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Displacement with its best rigid motion (rotation + translation) removed.

    Exact rather than linearised, because the relative orientation of a
    Transition1x product is arbitrary and can be a large rotation.
    """
    moved = positions + displacement
    source = positions - positions.mean(axis=0)
    target = moved - moved.mean(axis=0)
    rotation = _kabsch_rotation(source, target)
    return displacement - ((source @ rotation.T + moved.mean(axis=0)) - positions)


def _divergence(first: np.ndarray, second: np.ndarray, *, chemical: bool,
                positions: np.ndarray | None = None) -> float:
    if chemical:
        first = _chemical_displacement(positions, first)
        second = _chemical_displacement(positions, second)
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def _topology_signature(labels: np.ndarray, species) -> frozenset:
    """Order-independent description of which bonds form and break."""
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


def _load(events_dir: str, max_atoms: int) -> tuple[list[dict], int]:
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
            "positions": reactant.positions,
            "displacement": minimum_image_displacement(
                target.target_positions, reactant.positions, cell=reactant.cell, pbc=reactant.pbc
            ),
            "topology": _topology_signature(labels, reactant.species),
        })
    return records, len(catalog.event_ids)


def _pairs(records: list[dict], *, tolerance: float, max_per_composition: int):
    groups: dict[str, list] = defaultdict(list)
    for record in records:
        groups[record["composition"]].append(record)
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item["event_id"])[:max_per_composition]
        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                if first["positions"].shape != second["positions"].shape:
                    continue
                if _aligned_rmsd(first["positions"], second["positions"]) >= tolerance:
                    continue
                yield first, second


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--events-dir", required=True, help="converted Transition1x events directory")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--max-atoms", type=int, default=32, help="skip larger systems")
    parser.add_argument("--reactant-tolerance", type=float, default=0.2,
                        help="aligned reactant RMSD below which two events count as the same reactant")
    parser.add_argument("--max-per-composition", type=int, default=24,
                        help="cap on records compared per composition group")
    args = parser.parse_args()

    records, total_events = _load(args.events_dir, args.max_atoms)
    collected = {bucket: {"raw": [], "chemical": []} for bucket in BUCKETS}
    for first, second in _pairs(
        records, tolerance=args.reactant_tolerance, max_per_composition=args.max_per_composition
    ):
        bucket = "same_topology" if first["topology"] == second["topology"] else "different_topology"
        collected[bucket]["raw"].append(_divergence(first["displacement"], second["displacement"], chemical=False))
        collected[bucket]["chemical"].append(
            _divergence(first["displacement"], second["displacement"], chemical=True,
                        positions=first["positions"])
        )

    paired = sum(len(collected[bucket]["raw"]) for bucket in BUCKETS)
    report = {
        "events_dir": str(args.events_dir),
        "num_events": total_events,
        "num_events_analyzed": len(records),
        "reactant_tolerance_angstrom": float(args.reactant_tolerance),
        "num_near_identical_reactant_pairs": paired,
        "pairs": {bucket: len(collected[bucket]["raw"]) for bucket in BUCKETS},
        "fraction_sharing_topology": (
            len(collected["same_topology"]["raw"]) / paired if paired else None
        ),
        "divergence_angstrom": {
            f"{bucket}_{kind}": _summarize(values)
            for bucket in BUCKETS
            for kind, values in collected[bucket].items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
