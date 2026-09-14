"""Compare Euler discretizations on fixed held-out molecular inputs and noise."""
from __future__ import annotations

import argparse
import configparser
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.evaluation import movable_mic_rmsd
from basinflow.data.catalog import BasinSplit, EventCatalog
from basinflow.data.pyg import BasinDataset
from basinflow.seeds import GaussianInit
from basinflow.models.factory import load_model_checkpoint, model_cutoff
from basinflow.sampling import CandidateSampler

_SUMMARY_SPEC = importlib.util.spec_from_file_location("sampling_diagnostic_metrics", REPO_ROOT / "tools/summarize_benchmark.py")
_SUMMARY = importlib.util.module_from_spec(_SUMMARY_SPEC)
_SUMMARY_SPEC.loader.exec_module(_SUMMARY)


def make_sampling_plan(basin_ids, *, basin_count=8, trials_per_basin=4, seed=20260910):
    available = sorted(basin_ids)
    if not available or basin_count <= 0 or trials_per_basin <= 0:
        raise ValueError("sampling requires nonempty basins and positive budgets")
    selected = np.random.default_rng(seed).choice(available, size=min(basin_count, len(available)), replace=False)
    return [
        {"basin_id": str(basin_id), "trial_index": trial_index,
         "init_random_seed": seed + basin_index * trials_per_basin + trial_index}
        for basin_index, basin_id in enumerate(selected)
        for trial_index in range(trials_per_basin)
    ]


def diagnose_sampling(model, catalog, plan, *, steps=(16, 32, 64), gaussian_scale,
                      cutoff, checkpoint_id, device, active_threshold=0.1):
    if not steps or any(value <= 0 for value in steps) or len(set(steps)) != len(steps):
        raise ValueError("integration steps must be distinct positive integers")
    rows = []
    started = time.perf_counter()
    for trial in plan:
        basin_id = trial["basin_id"]
        basin = catalog.basins[basin_id]
        if len(basin.known_event_ids) != 1:
            raise ValueError("molecular integration diagnostic requires one known event per pseudo-basin")
        event = catalog.events[basin.known_event_ids[0]]
        reactant = catalog.structures[basin.reactant_structure_id]
        product = catalog.structures[event.product_structure_id]
        if event.metadata.get("dataset_kind") != "transition1x_single_event_molecule" or np.any(reactant.pbc):
            raise ValueError("integration diagnostic requires nonperiodic Transition1x pseudo-basins")
        initializer = GaussianInit(scale=gaussian_scale, random_seed=trial["init_random_seed"])
        initial = BasinDataset(catalog.subset([basin_id]), [initializer])[0].pos.cpu().numpy()
        initial_hash = hashlib.sha256(initial.astype("<f4").tobytes()).hexdigest()
        for count in steps:
            row = {**trial, "event_id": event.event_id, "num_steps": int(count),
                   "initial_positions_sha256": initial_hash, "finite": False,
                   "raw_rmsd_angstrom": None, "kabsch_rmsd_angstrom": None,
                   "pair_distance_mae_angstrom": None, "collision": None}
            sampler = CandidateSampler(model, [initializer], cutoff=cutoff, num_steps=count,
                                       graph_update_interval=1, checkpoint_id=checkpoint_id,
                                       device=device, active_threshold=active_threshold)
            trial_started = time.perf_counter()
            try:
                result = sampler.sample(catalog, basin_id)
                row["sampling_seconds"] = time.perf_counter() - trial_started
                generated = result.generated_structures[result.candidates[0].generated_structure_id]
                diagnostic = _SUMMARY.molecular_alignment_diagnostics([generated], [product])
                row.update({
                    "finite": True,
                    "raw_rmsd_angstrom": movable_mic_rmsd(generated, product, reactant),
                    "kabsch_rmsd_angstrom": diagnostic["nearest_aligned_rmsd_angstrom"][0],
                    "pair_distance_mae_angstrom": diagnostic["nearest_pair_distance_mae_angstrom"][0],
                    "collision": _SUMMARY._collision(generated, reactant),
                })
            except FloatingPointError as error:
                row["sampling_seconds"] = time.perf_counter() - trial_started
                row["error"] = str(error)
            row["total_trial_seconds"] = time.perf_counter() - trial_started
            rows.append(row)
    aggregates = {}
    for count in steps:
        selected = [row for row in rows if row["num_steps"] == count]
        finite = [row for row in selected if row["finite"]]
        aggregates[str(count)] = {
            "num_trials": len(selected), "finite_trials": len(finite),
            "nonfinite_fraction": 1.0 - len(finite) / len(selected) if selected else None,
            "sampling_seconds": sum(row["sampling_seconds"] for row in selected),
            **{f"mean_{key}": float(np.mean([row[key] for row in finite])) if finite else None
               for key in ("raw_rmsd_angstrom", "kabsch_rmsd_angstrom", "pair_distance_mae_angstrom", "collision")},
        }
    return {"rows": rows, "by_num_steps": aggregates, "integrator": "euler_left_endpoint",
            "graph_update_interval": 1, "total_seconds": time.perf_counter() - started}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True, help="new diagnostic directory")
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--num-basins", type=int, default=8)
    parser.add_argument("--trials-per-basin", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    import torch

    if args.cpu_threads <= 0:
        raise ValueError("cpu threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    config = configparser.ConfigParser()
    config_path = run_dir / "config.resolved.ini"
    if not config.read(config_path):
        raise FileNotFoundError(config_path)
    split_path = run_dir / "split_manifest.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    split = BasinSplit.load(split_path)
    evaluation_split = config["evaluation"].get("split", "test")
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    available = split.val_ids if evaluation_split == "val" else split.test_ids
    plan = make_sampling_plan(available, basin_count=args.num_basins,
                              trials_per_basin=args.trials_per_basin, seed=args.seed)
    provenance = {
        "run_dir": str(run_dir), "evaluation_split": evaluation_split,
        "selection_method": "numpy_default_rng_choice_from_sorted_split_ids_before_outcomes",
        "selection_seed": args.seed, "sampling_seed": args.seed,
        "requested_num_basins": args.num_basins, "trials_per_basin": args.trials_per_basin,
        "steps": args.steps, "cpu_threads": args.cpu_threads,
        "source_sha256": {path.name: _sha256(path) for path in (checkpoint_path, config_path, split_path)},
        "plan": plan,
        "evaluation_protocol": "fixed_checkpoint_single_event_molecule_no_oracle_no_relaxation_integration_diagnostic",
        "definitions": {key: _SUMMARY.DEFINITIONS[key] for key in ("rmsd", "molecular_alignment_diagnostic", "pair_distance_mae", "collision_fraction")},
        "limitations": "Paired Euler step-count diagnostic; finite-only metric averages report failure counts separately. No physical validation or proof of learned-field correctness. Timing includes no device transfer synchronization because execution is CPU only.",
    }
    catalog = EventCatalog.from_eon_directory(config["data"]["events_dir"])
    split.validate(catalog)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_split = {"train": list(split.train_ids), "val": list(split.val_ids), "test": list(split.test_ids), "seed": split.seed}
    if checkpoint.get("split_manifest") != expected_split:
        raise ValueError("checkpoint split manifest differs from run split manifest")
    model = load_model_checkpoint(checkpoint, device="cpu")
    output.mkdir(parents=True)
    (output / "sampling_plan.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    report = diagnose_sampling(model, catalog, plan, steps=tuple(args.steps),
                               gaussian_scale=config["init"].getfloat("gaussian_scale"),
                               cutoff=model_cutoff(model), checkpoint_id=str(checkpoint_path),
                               device="cpu", active_threshold=config["data"].getfloat("active_threshold", fallback=0.1))
    temporary = output / "diagnostic.json.tmp"
    temporary.write_text(json.dumps({**provenance, **report}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "diagnostic.json")
    print(json.dumps(report["by_num_steps"], indent=2))


if __name__ == "__main__":
    main()
