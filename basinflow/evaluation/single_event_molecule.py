"""No-relaxation sampling diagnostics for one-event molecular datasets."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import ase.io
import numpy as np

from basinflow.evaluation.basin_recall import kabsch_aligned_rmsd, movable_mic_rmsd
from basinflow.data.catalog import EventCatalog
from basinflow.seeds import GaussianInit
from basinflow.sampling import CandidateSampler


_TRIAL_COLUMNS = [
    "evaluation_protocol",
    "pseudo_basin_id",
    "event_id",
    "source_index",
    "trial_index",
    "global_trial_index",
    "init_random_seed",
    "raw_rmsd_angstrom",
    "aligned_rmsd_angstrom",
    "target_active_atoms",
    "predicted_active_atoms",
    "active_precision",
    "active_recall",
    "active_f1",
]


def _active_metrics(target: np.ndarray, predicted: list[int] | None) -> dict[str, float]:
    target_set = set(np.flatnonzero(target).tolist())
    predicted_set = set(predicted or [])
    true_positive = len(target_set & predicted_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(target_set) if target_set else 0.0
    return {
        "active_precision": precision,
        "active_recall": recall,
        "active_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


# ReactOT and MolGEN both clip a proposal at 1.0 A before averaging, which is
# what makes their reported means comparable across molecules.
_RMSD_CLIP = 1.0


def _best_per_basin(rows: list[dict[str, Any]], key: str) -> list[float]:
    best: dict[str, float] = {}
    for row in rows:
        basin_id = row["pseudo_basin_id"]
        value = float(row[key])
        best[basin_id] = min(best[basin_id], value) if basin_id in best else value
    return list(best.values())


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/basinflow-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_rmsd_plots(rows: list[dict[str, Any]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()
    by_basin: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_basin.setdefault(str(row["pseudo_basin_id"]), []).append(row)

    basin_means = []
    for basin_id, basin_rows in by_basin.items():
        basin_rows.sort(key=lambda row: int(row["trial_index"]))
        trials = [int(row["trial_index"]) for row in basin_rows]
        values = [float(row["raw_rmsd_angstrom"]) for row in basin_rows]
        basin_mean = float(np.mean(values))
        basin_means.append((basin_id, basin_mean))
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(trials, values, "o-", markersize=2, linewidth=0.8)
        axes[0].axhline(basin_mean, color="tab:red", linestyle="--", linewidth=1, label=f"mean={basin_mean:.3f} Å")
        axes[0].set_xlabel("Trial index")
        axes[0].set_ylabel("Raw product RMSD (Å)")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[1].hist(values, bins=min(25, max(5, len(values) // 4)), alpha=0.75)
        axes[1].axvline(basin_mean, color="tab:red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Raw product RMSD (Å)")
        axes[1].set_ylabel("Trial count")
        axes[1].grid(alpha=0.25)
        figure.suptitle(f"Basin {basin_id} · {len(values)} trials")
        figure.tight_layout()
        figure.savefig(plots_dir / f"{basin_id}_rmsd_100_trials.png", dpi=140)
        plt.close(figure)

    ordered = sorted(basin_means, key=lambda item: item[0])
    figure, axis = plt.subplots(figsize=(max(7, len(ordered) * 0.32), 4.5))
    axis.plot(np.arange(len(ordered)), [value for _, value in ordered], "o", markersize=3)
    axis.set_xlabel("Test pseudo-basin")
    axis.set_ylabel("Mean raw product RMSD (Å)")
    axis.set_xticks(np.arange(len(ordered)), [basin_id for basin_id, _ in ordered], rotation=60, ha="right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_dir / "rmsd_by_basin.png", dpi=140)
    plt.close(figure)

    values = np.asarray([float(row["raw_rmsd_angstrom"]) for row in rows], dtype=float)
    mean_value = float(np.mean(values)) if values.size else 0.0
    below = values < mean_value
    above = values > mean_value
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(values, bins=min(80, max(10, int(np.sqrt(max(values.size, 1))))), alpha=0.75)
    axis.axvline(mean_value, color="tab:red", linestyle="--", linewidth=1.5, label=f"global mean={mean_value:.3f} Å")
    axis.set_xlabel("Raw product RMSD (Å)")
    axis.set_ylabel("Total trial count")
    axis.set_title(f"All test-basin samples · below mean={below.mean():.1%}, above mean={above.mean():.1%}")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_dir / "rmsd_distribution.png", dpi=160)
    plt.close(figure)


def run_single_event_molecule_trials(
    *,
    model: Any,
    test_catalog: EventCatalog,
    output_dir: str | Path,
    cutoff: float,
    num_steps: int,
    graph_update_interval: int,
    gaussian_scale: float,
    num_trials: int | None = None,
    trials_per_basin: int = 1,
    num_test_basins: int | None = None,
    sampling_seed: int,
    selection_seed: int,
    active_threshold: float = 0.1,
    device=None,
    bond_change_source: str = "seed",
) -> dict[str, Any]:
    """Run a fixed number of one-shot, no-oracle molecular product samples.

    Transition1x has no multi-event basin label.  The input dataset therefore
    uses one pseudo-basin per reaction, and this routine samples a reproducible
    subset of those reactions once each.  It deliberately does not report
    basin recall, relaxation, or saddle-validation quantities.
    """
    if not isinstance(test_catalog, EventCatalog):
        raise TypeError("test_catalog must be an EventCatalog")
    if trials_per_basin <= 0:
        raise ValueError("trials_per_basin must be positive")
    available = list(test_catalog.basin_ids)
    if not available:
        raise ValueError("test_catalog has no pseudo-basins to sample")
    if num_test_basins is None:
        chosen_count = len(available) if num_trials is None else min(int(num_trials), len(available))
    else:
        if num_test_basins <= 0:
            raise ValueError("num_test_basins must be positive")
        chosen_count = min(int(num_test_basins), len(available))
    rng = np.random.default_rng(selection_seed)
    selected_ids = [str(item) for item in rng.choice(available, size=chosen_count, replace=False)]

    sampling_dir = Path(output_dir) / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    generated_atoms = []
    reference_atoms = []
    rmsds: list[float] = []
    aligned_rmsds: list[float] = []
    f1_scores: list[float] = []
    for basin_index, basin_id in enumerate(selected_ids):
        basin = test_catalog.basins[basin_id]
        if len(basin.known_event_ids) != 1:
            raise ValueError(
                "single-event molecular diagnostics require one known event per pseudo-basin"
            )
        event_id = basin.known_event_ids[0]
        event = test_catalog.events[event_id]
        if event.metadata.get("dataset_kind") != "transition1x_single_event_molecule":
            raise ValueError(
                "single-event molecular diagnostics require Transition1x pseudo-basins"
            )
        target = test_catalog.event_target(
            event_id,
            active_threshold=active_threshold,
        )
        product = target.product
        for trial_index in range(int(trials_per_basin)):
            global_trial_index = basin_index * int(trials_per_basin) + trial_index
            init_random_seed = int(sampling_seed) + global_trial_index
            sampler = CandidateSampler(
                model=model,
                init_generators=[GaussianInit(scale=gaussian_scale, random_seed=init_random_seed)],
                cutoff=float(cutoff),
                num_steps=int(num_steps),
                graph_update_interval=int(graph_update_interval),
                checkpoint_id=str(Path(output_dir) / "checkpoint.pt"),
                device=device,
                active_threshold=active_threshold,
                bond_change_source=bond_change_source,
            )
            result = sampler.sample(test_catalog, basin_id)
            candidate = result.candidates[0]
            generated = result.generated_structures[candidate.generated_structure_id]
            rmsd = movable_mic_rmsd(generated, product, target.reactant)
            aligned_rmsd = kabsch_aligned_rmsd(generated, product, target.reactant)
            active = _active_metrics(target.active_mask, candidate.predicted_active_atoms)
            row = {
                "evaluation_protocol": "single_event_pairwise_no_relaxation",
                "pseudo_basin_id": basin_id,
                "event_id": event_id,
                "source_index": event.metadata.get("source_index"),
                "trial_index": trial_index,
                "global_trial_index": global_trial_index,
                "init_random_seed": init_random_seed,
                "raw_rmsd_angstrom": rmsd,
                "aligned_rmsd_angstrom": aligned_rmsd,
                "target_active_atoms": int(np.sum(target.active_mask)),
                "predicted_active_atoms": len(candidate.predicted_active_atoms or []),
                **active,
            }
            rows.append(row)
            rmsds.append(rmsd)
            aligned_rmsds.append(aligned_rmsd)
            f1_scores.append(active["active_f1"])

            generated_frame = generated.to_ase()
            generated_frame.info.update({"role": "generated_candidate", **row})
            reference_frame = product.to_ase()
            reference_frame.info.update({"role": "reference_product", **row})
            generated_atoms.append(generated_frame)
            reference_atoms.append(reference_frame)

    metrics_path = sampling_dir / "trial_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_TRIAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    ase.io.write(sampling_dir / "generated_candidates.extxyz", generated_atoms)
    ase.io.write(sampling_dir / "reference_products.extxyz", reference_atoms)
    _write_rmsd_plots(rows, sampling_dir / "plots")
    values = np.asarray(rmsds, dtype=float)
    global_mean = float(np.mean(values)) if values.size else None
    below_count = int(np.count_nonzero(values < global_mean)) if global_mean is not None else 0
    above_count = int(np.count_nonzero(values > global_mean)) if global_mean is not None else 0
    equal_count = int(values.size - below_count - above_count) if global_mean is not None else 0
    summary = {
        "evaluation_protocol": "single_event_pairwise_no_relaxation",
        "dataset_kind": "transition1x_single_event_molecule",
        "available_test_pseudo_basins": len(available),
        "selected_test_pseudo_basins": selected_ids,
        "num_test_basins": len(selected_ids),
        "trials_per_test_basin": int(trials_per_basin),
        "num_trials": len(rows),
        "sampling_seed": int(sampling_seed),
        "selection_seed": int(selection_seed),
        "raw_rmsd_angstrom": _summary(rmsds),
        "aligned_rmsd_angstrom": _summary(aligned_rmsds),
        "clipped_raw_rmsd_angstrom": _summary([min(value, _RMSD_CLIP) for value in rmsds]),
        "best_of_trials": {
            "selector": "oracle_min_rmsd",
            "notice": (
                "Lower bound of the per-basin best-candidate arm. Selection uses the "
                "reference product, so it is a diagnostic ceiling and not a KMC result."
            ),
            "raw_rmsd_angstrom": _summary(_best_per_basin(rows, "raw_rmsd_angstrom")),
            "aligned_rmsd_angstrom": _summary(_best_per_basin(rows, "aligned_rmsd_angstrom")),
        },
        "raw_rmsd_mean_split": {
            "mean_angstrom": global_mean,
            "below_mean_count": below_count,
            "below_mean_fraction": below_count / len(rows) if rows else None,
            "equal_mean_count": equal_count,
            "equal_mean_fraction": equal_count / len(rows) if rows else None,
            "above_mean_count": above_count,
            "above_mean_fraction": above_count / len(rows) if rows else None,
        },
        "active_f1": _summary(f1_scores),
        "relaxation": "not_run",
        "saddle_validation": "not_run",
    }
    (sampling_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
