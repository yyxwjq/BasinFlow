"""
Sample product/event candidates from a trained BasinFlow checkpoint.

Default dataset: /Users/wx/Desktop/benchmark/au/events.
Default checkpoint: runs/au_product_flow/checkpoint.pt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import ase.io

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.evaluation import cluster_candidates, movable_mic_rmsd
from basinflow.data.catalog import BasinSplit, EventCatalog
from basinflow.models.factory import load_model_checkpoint, model_cutoff
from basinflow.sampling import CandidateSampler
from basinflow.seeds import DirectionalInit, GaussianInit, ZeroInit


def _directional_generators(count: int, *, scale: float) -> list[DirectionalInit]:
    base_directions = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (-1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0),
        (0.0, -1.0, 1.0),
        (0.0, 1.0, -1.0),
        (0.0, -1.0, -1.0),
    ]
    generators = []
    for index in range(max(int(count), 0)):
        generators.append(
            DirectionalInit(
                direction=base_directions[index % len(base_directions)],
                scale=scale,
                movable_rank=index // len(base_directions),
            )
        )
    return generators


def _default_events_dir() -> str:
    return os.environ.get(
        "BASINFLOW_EVENTS_DIR",
        "/Users/wx/Desktop/benchmark/au/events",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample BasinFlow product/event candidates from a checkpoint.",
    )
    parser.add_argument("--events-dir", default=_default_events_dir())
    parser.add_argument("--checkpoint", default="runs/au_product_flow/checkpoint.pt")
    parser.add_argument("--split-manifest", default="runs/au_product_flow/split_manifest.json")
    parser.add_argument("--output-dir", default="runs/au_product_flow/sampling")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--graph-update-interval", type=int, default=2)
    parser.add_argument("--cluster-threshold", type=float, default=0.25)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _candidate_to_dict(candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "basin_id": candidate.basin_id,
        "reactant_structure_id": candidate.reactant_structure_id,
        "initial_seed_id": candidate.initial_seed_id,
        "generated_structure_id": candidate.generated_structure_id,
        "model_checkpoint": candidate.model_checkpoint,
        "sampling_config": dict(candidate.sampling_config),
        "active_atom_scores": _jsonable(candidate.active_atom_scores),
        "predicted_active_atoms": candidate.predicted_active_atoms,
        "event_direction": _jsonable(candidate.event_direction),
        "status": candidate.status,
        "metadata": dict(candidate.metadata),
    }


def _test_event_rows(catalog) -> list[dict[str, Any]]:
    rows = []
    for basin_id in catalog.basin_ids:
        basin = catalog.basins[basin_id]
        for event_id in basin.known_event_ids:
            event = catalog.events[event_id]
            metadata = dict(event.metadata)
            source_file = metadata.get("source_file")
            filename = metadata.get("file")
            if filename is None and source_file:
                filename = Path(source_file).name
            rows.append(
                {
                    "basin_id": basin_id,
                    "event_id": event.event_id,
                    "file": filename,
                    "source_file": source_file,
                }
            )
    return rows


def _evaluate_catalog_basin(catalog, basin_id: str, candidates, structures, *, cluster_threshold: float, match_threshold: float) -> dict[str, Any]:
    """No-relaxation matching that keeps known products outside sampler inputs."""
    basin = catalog.basins[basin_id]
    reactant = catalog.structures[basin.reactant_structure_id]
    targets = [catalog.event_target(event_id) for event_id in basin.known_event_ids]
    clusters = cluster_candidates(candidates, structures, reactant, threshold=cluster_threshold)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    recalled = set()
    nearest_rmsds = []
    for cluster in clusters:
        candidate_id = cluster.representative_candidate_id or cluster.candidate_ids[0]
        generated = structures[candidate_by_id[candidate_id].generated_structure_id]
        best_target = min(targets, key=lambda target: movable_mic_rmsd(generated, target.product, reactant))
        best_rmsd = movable_mic_rmsd(generated, best_target.product, reactant)
        nearest_rmsds.append(best_rmsd)
        if best_rmsd <= match_threshold:
            recalled.add(best_target.event.event_id)
    return {
        "basin_id": basin_id,
        "num_known_events": len(targets),
        "num_candidates": len(candidates),
        "num_clusters": len(clusters),
        "best_candidate_rmsd": min(nearest_rmsds) if nearest_rmsds else None,
        "recalled_event_ids": [event_id for event_id in basin.known_event_ids if event_id in recalled],
        "recall": len(recalled) / len(targets) if targets else 0.0,
        "duplicate_rate": (len(candidates) - len(clusters)) / len(candidates) if candidates else 0.0,
    }


def _init_generators_from_checkpoint(checkpoint: dict[str, Any]):
    generators = []
    init_config = checkpoint.get("init_config") or checkpoint.get("seed_config", {})
    for item in init_config.get("generators", [{"type": "zero"}]):
        if item["type"] == "zero":
            generators.append(ZeroInit())
        elif item["type"] == "gaussian_movable":
            generators.append(
                GaussianInit(
                    scale=float(item["scale"]),
                    random_seed=int(item["random_seed"]),
                )
            )
        elif item["type"] == "directional":
            generators.extend(
                _directional_generators(
                    int(item.get("count", 0)),
                    scale=float(item["scale"]),
                )
            )
    return generators or [ZeroInit()]


def main() -> None:
    args = parse_args()
    import torch

    events_dir = Path(args.events_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_model_checkpoint(checkpoint)

    catalog = EventCatalog.from_eon_directory(events_dir)
    split_path = Path(args.split_manifest)
    if split_path.is_file():
        split = BasinSplit.load(split_path)
    else:
        split_manifest = checkpoint["split_manifest"]
        split = BasinSplit(
            train_ids=tuple(split_manifest["train"]),
            val_ids=tuple(split_manifest["val"]),
            test_ids=tuple(split_manifest["test"]),
            seed=int(split_manifest["seed"]),
        )
    test_catalog = split.select(catalog, "test")
    test_events = _test_event_rows(test_catalog)
    init_generators = _init_generators_from_checkpoint(checkpoint)

    sampler = CandidateSampler(
        model=model,
        init_generators=init_generators,
        cutoff=model_cutoff(model),
        num_steps=args.num_steps,
        graph_update_interval=args.graph_update_interval,
        checkpoint_id=str(checkpoint_path),
        active_threshold=float(checkpoint.get("training_config", checkpoint.get("config", {})).get("active_threshold", 0.1)),
    )

    basin_reports = []
    candidate_rows = []
    generated_atoms = []
    total_recall = 0.0
    basin_best_rmsds = []
    for basin_id in test_catalog.basin_ids:
        result = sampler.sample(test_catalog, basin_id)
        report = _evaluate_catalog_basin(
            test_catalog,
            basin_id,
            result.candidates,
            result.generated_structures,
            cluster_threshold=args.cluster_threshold,
            match_threshold=args.match_threshold,
        )
        basin_reports.append(report)
        total_recall += report["recall"]
        if report["best_candidate_rmsd"] is not None:
            basin_best_rmsds.append(float(report["best_candidate_rmsd"]))

        for candidate in result.candidates:
            candidate_rows.append(_candidate_to_dict(candidate))
            atoms = result.generated_structures[candidate.generated_structure_id].to_ase()
            atoms.info["candidate_id"] = candidate.candidate_id
            atoms.info["basin_id"] = candidate.basin_id
            atoms.info["seed_id"] = candidate.initial_seed_id
            generated_atoms.append(atoms)

    metrics = {
        "events_dir": str(events_dir),
        "checkpoint": str(checkpoint_path),
        "num_test_basins": len(test_catalog.basin_ids),
        "num_candidates": len(candidate_rows),
        "test_events": test_events,
        "average_recall": total_recall / max(len(test_catalog.basin_ids), 1),
        "average_test_best_rmsd": (
            sum(basin_best_rmsds) / len(basin_best_rmsds)
            if basin_best_rmsds
            else None
        ),
        "basins": basin_reports,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "candidates.json").write_text(
        json.dumps(candidate_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    if generated_atoms:
        ase.io.write(output_dir / "generated_candidates.extxyz", generated_atoms)

    print(f"Sampled basins: {len(test_catalog.basin_ids)}")
    if test_events:
        print("Test events:")
        for row in test_events:
            print(f"  basin {row['basin_id']} / {row['event_id']}: {row['file']}")
    print(f"Candidates: {len(candidate_rows)}")
    print(f"Average recall: {metrics['average_recall']:.4f}")
    if metrics["average_test_best_rmsd"] is None:
        print("Average test best RMSD: n/a")
    else:
        print(f"Average test best RMSD: {metrics['average_test_best_rmsd']:.6f} Å")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print(f"Candidates: {output_dir / 'candidates.json'}")
    print(f"Structures: {output_dir / 'generated_candidates.extxyz'}")


if __name__ == "__main__":
    main()
