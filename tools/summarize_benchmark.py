"""Compare saved proposals against paired reactant and Gaussian baselines."""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import pathlib
from pathlib import Path
import sys

import ase.io
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.evaluation import movable_mic_rmsd
from basinflow.data import Structure, load_events, load_split
from basinflow.workflows.config import parse_pbc_override
from basinflow.geometry.mic import pairwise_displacements
from basinflow.seeds import GaussianInit, SeedContext


THRESHOLDS = (0.1, 0.2, 0.5)
DEFINITIONS = {
    "evaluation": "Saved unrelaxed coordinates only; no relaxation, saddle validation, barriers, rates, or validated KMC event recall.",
    "rmsd": "sqrt(mean_i ||MIC(candidate_i-product_i)||^2) over reactant movable atoms; existing movable_mic_rmsd falls back to all atoms if none are movable. Fixed atom mapping; no rotation/translation alignment or permutation matching.",
    "nearest_product_rmsd": "Per candidate movable-atom RMSD to the known product of the event it was sampled for. This candidate-quality distance is distinct from event recall.",
    "geometric_event_recall": "For each known event product, independently test whether at least one candidate is within threshold (angstrom). Divide matched event records by all known event records. One candidate may geometrically match multiple nearby products; no nearest-only assignment, uniqueness, or physical connectivity is imposed.",
    "event_weighted_recall": "Sum matched event records across evaluated events divided by total known event records in those events.",
    "macro_recall": "Arithmetic mean of per-event geometric event recall; each evaluated event has equal weight.",
    "beyond_zero_baseline_recall": "Recall restricted at each threshold to known products whose movable RMSD from the reactant exceeds that threshold. Null when no such known products exist.",
    "zero_motion": "Reactant coordinates repeated to equal the model candidate budget, using the same events and products.",
    "gaussian_only": "Pre-integration Gaussian initialization reconstructed from saved per-trial init_random_seed, configured scale, EventProposalDataset context/seed identifiers, and float32 input-coordinate conversion. No model or product-derived conditioning.",
    "paired_improvement": "For matching event/trial pairs, baseline nearest-product RMSD minus model nearest-product RMSD; positive means the model improved. Strict improvement fraction excludes ties.",
    "duplicate_fraction": "Greedy first-representative clustering with existing movable MIC RMSD <= 0.1 angstrom, in trial order; (candidate_count-cluster_count)/candidate_count. Event recall uses all candidates, not only cluster representatives.",
    "collision_fraction": "Fraction of candidates containing an atom-image pair closer than 0.6*(covalent_radius_i+covalent_radius_j), with at least one movable atom, using ASE periodic neighbor lists. This is a geometric overlap warning, not a physical validity classification.",
    "frozen_drift": "Direct Cartesian candidate-minus-reactant displacement on fixed atoms, without MIC. Mean frozen-atom RMSD and maximum individual frozen-atom drift; null when no atoms are fixed. Float32 input conversion can produce numerical differences near 1e-6 angstrom; these values alone are not evidence of constraint violation.",
    "aggregation": "Per-event metrics plus macro means over events, candidate-weighted means over all proposals, and event-weighted recall. Molecular rows cover only the saved held-out event subset.",
    "molecular_alignment_diagnostic": "Additional posthoc evaluation only for Transition1x-tagged events with nonperiodic geometry. All-atom, equal-weight, atom-mapped RMSD after centering and proper Kabsch rotation (determinant +1). No reflections, atom permutations, mass weighting, or periodic-image changes. Source coordinates, training inputs, primary unaligned metrics, and primary recall are unchanged. Never applied to Au/Pt scaffold benchmarks.",
    "aligned_geometric_reference_recall": "Separate molecular diagnostic: each known product reference is hit when at least one candidate has proper-Kabsch aligned all-atom RMSD <= threshold. Reported at 0.1, 0.2, 0.5 angstrom; this is not raw-coordinate recall or validated reaction/event recall.",
    "pair_distance_mae": "Molecular diagnostic: mean absolute difference of all unordered mapped atom-pair Euclidean distances, excluding the diagonal; nearest reference chosen separately for this metric. Translation/rotation invariant but also reflection invariant, so it cannot distinguish mirror images and does not replace proper-Kabsch RMSD. Zero for a single-atom structure.",
}


def gaussian_baseline(reactant, event_id, *, random_seed, scale):
    initializer = GaussianInit(scale=scale, random_seed=random_seed)
    context = SeedContext(f"{event_id}:proposal:0", reactant)
    seed = initializer.generate(context, seed_id=f"{event_id}:seed:0:{initializer.seed_type}")
    return Structure(list(reactant.species), seed.initial_positions(reactant.positions).astype(np.float32),
                     cell=reactant.cell, pbc=reactant.pbc, movable_mask=reactant.movable_mask)


def _collision(structure, reactant):
    atoms = structure.to_ase()
    atoms.set_cell(reactant.cell)
    atoms.set_pbc(reactant.pbc)
    source, target = neighbor_list("ij", atoms, 0.6 * covalent_radii[atoms.numbers], self_interaction=False)
    movable = np.asarray(reactant.movable_mask, dtype=bool)
    return bool(np.any(movable[source] | movable[target]))


def score_candidates(candidates, products, reactant):
    if not candidates or not products:
        raise ValueError("scoring requires candidates and known products")
    for candidate in candidates:
        if candidate.species != reactant.species or candidate.n_atoms != reactant.n_atoms:
            raise ValueError("candidate atom order/species must match reactant")
        if not np.all(np.isfinite(candidate.positions)):
            raise ValueError("candidate coordinates must be finite")
    distances = np.asarray([[movable_mic_rmsd(candidate, product, reactant) for product in products] for candidate in candidates])
    nearest = distances.min(axis=1)
    event_best = distances.min(axis=0)
    zero_distances = np.asarray([movable_mic_rmsd(reactant, product, reactant) for product in products])
    representatives = []
    for candidate in candidates:
        if not any(movable_mic_rmsd(candidate, representative, reactant) <= 0.1 for representative in representatives):
            representatives.append(candidate)
    fixed = ~np.asarray(reactant.movable_mask, dtype=bool)
    fixed_rmsds = []
    fixed_maxima = []
    all_atom_nearest = []
    for candidate in candidates:
        if fixed.any():
            fixed_norms = np.linalg.norm(candidate.positions[fixed] - reactant.positions[fixed], axis=1)
            fixed_rmsds.append(float(np.sqrt(np.mean(fixed_norms ** 2))))
            fixed_maxima.append(float(fixed_norms.max()))
        all_atom_nearest.append(min(float(np.sqrt(np.mean(np.sum(pairwise_displacements(candidate.positions, product.positions, reactant.cell, reactant.pbc) ** 2, axis=1)))) for product in products))
    return {
        "num_candidates": len(candidates),
        "num_known_events": len(products),
        "nearest_product_rmsd_angstrom": nearest.tolist(),
        "event_best_candidate_rmsd_angstrom": event_best.tolist(),
        "mean_nearest_product_rmsd_angstrom": float(nearest.mean()),
        "best_nearest_product_rmsd_angstrom": float(nearest.min()),
        "mean_all_atom_nearest_product_rmsd_angstrom": float(np.mean(all_atom_nearest)),
        "mean_movable_displacement_from_reactant_angstrom": float(np.mean([movable_mic_rmsd(candidate, reactant, reactant) for candidate in candidates])),
        "mean_frozen_atom_rmsd_angstrom": float(np.mean(fixed_rmsds)) if fixed_rmsds else None,
        "max_frozen_atom_drift_angstrom": max(fixed_maxima) if fixed_maxima else None,
        "geometric_event_recall": {str(threshold): float(np.mean(event_best <= threshold)) for threshold in THRESHOLDS},
        "matched_event_counts": {str(threshold): int(np.sum(event_best <= threshold)) for threshold in THRESHOLDS},
        "beyond_zero_baseline_recall": {str(threshold): float(np.mean(event_best[zero_distances > threshold] <= threshold)) if np.any(zero_distances > threshold) else None for threshold in THRESHOLDS},
        "num_clusters": len(representatives),
        "duplicate_fraction": 1.0 - len(representatives) / len(candidates),
        "collision_fraction": float(np.mean([_collision(candidate, reactant) for candidate in candidates])),
    }


def molecular_alignment_diagnostics(candidates, products):
    if not candidates or not products:
        raise ValueError("molecular diagnostics require candidates and products")
    aligned_distances = []
    pair_distances = []
    for candidate in candidates:
        aligned_row = []
        pair_row = []
        for product in products:
            if candidate.species != product.species or candidate.n_atoms != product.n_atoms:
                raise ValueError("molecular diagnostics require mapped atoms with identical species order")
            if np.any(candidate.pbc) or np.any(product.pbc):
                raise ValueError("molecular alignment requires nonperiodic structures")
            source = candidate.positions - candidate.positions.mean(axis=0)
            target = product.positions - product.positions.mean(axis=0)
            left, _, right = np.linalg.svd(source.T @ target)
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(left @ right)
            rotation = left @ correction @ right
            aligned_row.append(float(np.sqrt(np.mean(np.sum((source @ rotation - target) ** 2, axis=1)))))
            source_pairs = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=-1)
            target_pairs = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=-1)
            upper = np.triu_indices(candidate.n_atoms, k=1)
            pair_row.append(float(np.mean(np.abs(source_pairs[upper] - target_pairs[upper]))) if candidate.n_atoms > 1 else 0.0)
        aligned_distances.append(aligned_row)
        pair_distances.append(pair_row)
    aligned_distances = np.asarray(aligned_distances)
    nearest = aligned_distances.min(axis=1)
    event_best = aligned_distances.min(axis=0)
    nearest_pair_mae = np.asarray(pair_distances).min(axis=1)
    return {
        "nearest_aligned_rmsd_angstrom": nearest.tolist(),
        "mean_nearest_aligned_rmsd_angstrom": float(nearest.mean()),
        "event_best_aligned_rmsd_angstrom": event_best.tolist(),
        "aligned_geometric_reference_recall": {str(threshold): float(np.mean(event_best <= threshold)) for threshold in THRESHOLDS},
        "aligned_matched_reference_counts": {str(threshold): int(np.sum(event_best <= threshold)) for threshold in THRESHOLDS},
        "nearest_pair_distance_mae_angstrom": nearest_pair_mae.tolist(),
        "mean_nearest_pair_distance_mae_angstrom": float(nearest_pair_mae.mean()),
    }


def _aggregate(reports):
    count = sum(report["num_candidates"] for report in reports)
    events = sum(report["num_known_events"] for report in reports)
    result = {
        "num_events": len(reports),
        "num_candidates": count,
        "num_known_events": events,
        "candidate_weighted_nearest_product_rmsd_angstrom": sum(report["mean_nearest_product_rmsd_angstrom"] * report["num_candidates"] for report in reports) / count,
        "macro_nearest_product_rmsd_angstrom": float(np.mean([report["mean_nearest_product_rmsd_angstrom"] for report in reports])),
        "macro_all_atom_nearest_product_rmsd_angstrom": float(np.mean([report["mean_all_atom_nearest_product_rmsd_angstrom"] for report in reports])),
        "macro_movable_displacement_from_reactant_angstrom": float(np.mean([report["mean_movable_displacement_from_reactant_angstrom"] for report in reports])),
        "macro_geometric_event_recall": {str(threshold): float(np.mean([report["geometric_event_recall"][str(threshold)] for report in reports])) for threshold in THRESHOLDS},
        "event_weighted_geometric_recall": {str(threshold): sum(report["matched_event_counts"][str(threshold)] for report in reports) / events for threshold in THRESHOLDS},
        "candidate_weighted_collision_fraction": sum(report["collision_fraction"] * report["num_candidates"] for report in reports) / count,
        "macro_duplicate_fraction": float(np.mean([report["duplicate_fraction"] for report in reports])),
        "max_frozen_atom_drift_angstrom": max((report["max_frozen_atom_drift_angstrom"] for report in reports if report["max_frozen_atom_drift_angstrom"] is not None), default=None),
    }
    if all("molecular_alignment_diagnostic" in report for report in reports):
        aligned = [report["molecular_alignment_diagnostic"] for report in reports]
        result["molecular_alignment_diagnostic"] = {
            "candidate_weighted_nearest_aligned_rmsd_angstrom": sum(diagnostic["mean_nearest_aligned_rmsd_angstrom"] * report["num_candidates"] for report, diagnostic in zip(reports, aligned)) / count,
            "macro_nearest_aligned_rmsd_angstrom": float(np.mean([diagnostic["mean_nearest_aligned_rmsd_angstrom"] for diagnostic in aligned])),
            "macro_nearest_pair_distance_mae_angstrom": float(np.mean([diagnostic["mean_nearest_pair_distance_mae_angstrom"] for diagnostic in aligned])),
            "macro_aligned_geometric_reference_recall": {str(threshold): float(np.mean([diagnostic["aligned_geometric_reference_recall"][str(threshold)] for diagnostic in aligned])) for threshold in THRESHOLDS},
            "reference_weighted_aligned_geometric_recall": {str(threshold): sum(diagnostic["aligned_matched_reference_counts"][str(threshold)] for diagnostic in aligned) / events for threshold in THRESHOLDS},
        }
    return result


def summarize_run(run_dir, *, catalog_cache=None):
    run_dir = Path(run_dir)
    config = configparser.ConfigParser()
    if not config.read(run_dir / "config.resolved.ini"):
        raise FileNotFoundError(f"no resolved configuration in {run_dir}")
    pbc_text = config["data"].get("pbc_override", "auto").lower()
    cache = {} if catalog_cache is None else catalog_cache
    cache_key = (config["data"]["events_dir"], pbc_text)
    if cache_key not in cache:
        cache[cache_key] = load_events(
            config["data"]["events_dir"], pbc_override=parse_pbc_override(pbc_text)
        )
    catalog = cache[cache_key]
    events_dir = config["data"]["events_dir"]
    data_sources = {"events_dir": events_dir, "split_file": str(pathlib.Path(events_dir) / "split.csv")}
    # The dataset's own split.csv is authoritative for what the run evaluated: a
    # later resplit of the same directory must not silently change the basis.
    split = load_split(pathlib.Path(events_dir) / "split.csv", catalog)
    unknown = (
        set(split.train_ids) | set(split.val_ids) | set(split.test_ids)
    ) - set(catalog.events)
    if unknown:
        raise ValueError(f"saved split names event ids absent from the dataset: {sorted(unknown)[:3]}")
    evaluation_split = config["evaluation"].get("split", "test")
    allowed = set(split.val_ids if evaluation_split == "val" else split.test_ids)
    sampling_dir = run_dir / "sampling"
    with (sampling_dir / "trial_metrics.csv").open() as source:
        rows = list(csv.DictReader(source))
    molecular = "dataset_kind" in catalog.events[next(iter(allowed))].metadata
    frames_path = sampling_dir / ("generated_candidates.extxyz" if molecular else "generated_before_eam.traj")
    if not frames_path.is_file():
        fallback = sampling_dir / ("generated_before_eam.traj" if molecular else "generated_candidates.extxyz")
        if not fallback.is_file():
            raise FileNotFoundError(f"no candidate frames in {sampling_dir}")
        frames_path = fallback
    frames = ase.io.read(frames_path, index=":")
    if not rows or len(frames) != len(rows):
        raise ValueError("completed candidate frames and trial rows must have equal nonzero counts")
    grouped = {}
    seen_trials = set()
    for row, frame in zip(rows, frames):
        event_id = str(row["event_id"])
        if event_id not in allowed:
            raise ValueError(f"candidate event {event_id} is outside evaluation split")
        trial_key = (event_id, int(row["trial_index"]))
        if trial_key in seen_trials:
            raise ValueError("duplicate event/trial row in saved sampling results")
        seen_trials.add(trial_key)
        if str(frame.info.get("event_id")) != event_id or int(frame.info["trial_index"]) != int(row["trial_index"]):
            raise ValueError("candidate frame order does not match trial metrics")
        grouped.setdefault(event_id, []).append((row, frame))
    reports = []
    paired_rows = []
    scale = config["init"].getfloat("gaussian_scale")
    for event_id, trials in grouped.items():
        trials.sort(key=lambda item: int(item[0]["trial_index"]))
        event = catalog.events[event_id]
        reactant = event.reactant
        products = [event.product]
        methods = {
            "model": [Structure.from_ase(frame) for _, frame in trials],
            "zero_motion": [reactant] * len(trials),
            "gaussian_only": [gaussian_baseline(reactant, event_id, random_seed=int(row["init_random_seed"]), scale=scale) for row, _ in trials],
        }
        scores = {name: score_candidates(candidates, products, reactant) for name, candidates in methods.items()}
        transition1x = event.metadata.get("dataset_kind") == "transition1x_single_event_molecule"
        if transition1x:
            if np.any(reactant.pbc):
                raise ValueError("Transition1x molecular diagnostics require a nonperiodic reactant")
            for name, candidates in methods.items():
                scores[name]["molecular_alignment_diagnostic"] = molecular_alignment_diagnostics(candidates, products)
        reports.append({"event_id": event_id, "known_event_ids": [event_id], "methods": scores})
        for index, (row, _) in enumerate(trials):
            paired_rows.append({"event_id": event_id, "trial_index": int(row["trial_index"]), "init_random_seed": int(row["init_random_seed"]), **{name: score["nearest_product_rmsd_angstrom"][index] for name, score in scores.items()}})
            if transition1x:
                paired_rows[-1]["molecular_alignment_diagnostic"] = {name: score["molecular_alignment_diagnostic"]["nearest_aligned_rmsd_angstrom"][index] for name, score in scores.items()}
    comparisons = {}
    for baseline in ("zero_motion", "gaussian_only"):
        differences = np.asarray([row[baseline] - row["model"] for row in paired_rows])
        comparisons[baseline] = {"candidate_weighted_mean_rmsd_improvement_angstrom": float(differences.mean()), "strict_improvement_fraction": float(np.mean(differences > 0)), "macro_mean_rmsd_improvement_angstrom": float(np.mean([report["methods"][baseline]["mean_nearest_product_rmsd_angstrom"] - report["methods"]["model"]["mean_nearest_product_rmsd_angstrom"] for report in reports]))}
        if all("molecular_alignment_diagnostic" in row for row in paired_rows):
            aligned_differences = np.asarray([row["molecular_alignment_diagnostic"][baseline] - row["molecular_alignment_diagnostic"]["model"] for row in paired_rows])
            comparisons[baseline]["molecular_alignment_diagnostic"] = {"candidate_weighted_mean_aligned_rmsd_improvement_angstrom": float(aligned_differences.mean()), "strict_aligned_improvement_fraction": float(np.mean(aligned_differences > 0)), "macro_mean_aligned_rmsd_improvement_angstrom": float(np.mean([report["methods"][baseline]["molecular_alignment_diagnostic"]["mean_nearest_aligned_rmsd_angstrom"] - report["methods"]["model"]["molecular_alignment_diagnostic"]["mean_nearest_aligned_rmsd_angstrom"] for report in reports]))}
    pbc_patterns = sorted({tuple(bool(value) for value in catalog.events[event_id].reactant.pbc) for event_id in grouped})
    geometry = "periodic" if all(any(pattern) for pattern in pbc_patterns) else "nonperiodic" if not any(any(pattern) for pattern in pbc_patterns) else "mixed_pbc"
    return {
        "run_dir": str(run_dir.resolve()),
        "dataset": config["data"]["events_dir"],
        "data_sources": data_sources,
        "evaluation_split": evaluation_split,
        "evaluation_scope": "heldout_single_event_molecular_subset" if molecular else f"heldout_{geometry}_reactant_events",
        "evaluated_pbc_patterns": [list(pattern) for pattern in pbc_patterns],
        "available_evaluation_events": len(allowed),
        "evaluated_events": list(grouped),
        "model_backend": config["model"].get("backend", "egnn"),
        "training_budget": dict(config["training"]),
        "methods": {name: _aggregate([report["methods"][name] for report in reports]) for name in ("model", "zero_motion", "gaussian_only")},
        "paired_comparisons": comparisons,
        "per_event": reports,
        "paired_trials": paired_rows,
        "relaxation": "not_used",
        "saddle_validation": "not_run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cache = {}
    result = {"metric_definitions": DEFINITIONS, "runs": [summarize_run(path, catalog_cache=cache) for path in args.run_dir]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
