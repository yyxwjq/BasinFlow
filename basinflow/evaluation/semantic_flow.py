from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import ase.io
import numpy as np

from basinflow.evaluation.basin_recall import movable_mic_rmsd, nearest_product_match
from basinflow.data.catalog import EventCatalog
from basinflow.data.records import StructureRecord
from basinflow.geometry.mic import pairwise_displacements
from basinflow.seeds import GaussianInit
from basinflow.relaxation import RelaxationConfig, relax
from basinflow.sampling import CandidateSampler


TRIAL_COLUMNS = [
    "system",
    "basin_id",
    "trial_index",
    "candidate_id",
    "init_random_seed",
    "nearest_event_before",
    "source_file_before",
    "raw_rmsd_angstrom",
    "relax_status",
    "relax_converged",
    "relax_steps",
    "relax_error",
    "nearest_event_after",
    "source_file_after",
    "relaxed_rmsd_angstrom",
    "relaxed_rmsd_to_before_event",
    "event_changed_after_relax",
]


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/basinflow-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _as_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    return float(value)


def _source_name(source_file: str | None) -> str | None:
    return Path(source_file).name if source_file else None


def _rmsd_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "quantiles": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "quantiles": {
            "min": float(np.min(array)),
            "p25": float(np.quantile(array, 0.25)),
            "median": float(np.median(array)),
            "p75": float(np.quantile(array, 0.75)),
            "max": float(np.max(array)),
        },
    }


def _write_atoms(path: Path, atoms_list: list) -> None:
    if atoms_list:
        ase.io.write(path, atoms_list)
    else:
        path.touch()


def _candidate_atoms(
    candidate: StructureRecord,
    product: StructureRecord,
    *,
    role: str,
    basin_id: str,
    trial_index: int,
    event_id: str,
    source_file: str | None = None,
    representative_rank: str | None = None,
    row: dict[str, Any] | None = None,
):
    atoms = candidate.to_ase()
    error = pairwise_displacements(
        candidate.positions,
        product.positions,
        cell=product.cell,
        pbc=product.pbc,
    )
    mic_error = np.linalg.norm(error, axis=1)
    atoms.set_array("mic_error", mic_error)
    atoms.info.update(_frame_info(
        role=role,
        basin_id=basin_id,
        trial_index=trial_index,
        event_id=event_id,
        source_file=source_file,
        representative_rank=representative_rank,
        row=row,
        mic_error=mic_error.tolist(),
    ))
    return atoms


def _reference_atoms(
    product: StructureRecord,
    *,
    role: str,
    basin_id: str,
    trial_index: int,
    event_id: str,
    source_file: str | None = None,
    representative_rank: str | None = None,
    row: dict[str, Any] | None = None,
):
    atoms = product.to_ase()
    atoms.set_array("mic_error", np.zeros(product.n_atoms, dtype=float))
    atoms.info.update(_frame_info(
        role=role,
        basin_id=basin_id,
        trial_index=trial_index,
        event_id=event_id,
        source_file=source_file,
        representative_rank=representative_rank,
        row=row,
        mic_error=[0.0] * product.n_atoms,
    ))
    return atoms


def _frame_info(
    *,
    role: str,
    basin_id: str,
    trial_index: int,
    event_id: str,
    source_file: str | None,
    representative_rank: str | None,
    row: dict[str, Any] | None,
    mic_error: list[float],
) -> dict[str, Any]:
    """Build ASE-serializable metadata shared by candidate and reference frames."""
    info: dict[str, Any] = {
        "role": role,
        "basin_id": str(basin_id),
        "trial_index": int(trial_index),
        "nearest_event_id": str(event_id),
        "mic_error": mic_error,
    }
    if source_file is not None:
        info["source_file"] = _source_name(source_file)
    if representative_rank is not None:
        info["representative_rank"] = representative_rank
    if row is not None:
        for key in (
            "candidate_id",
            "init_random_seed",
            "raw_rmsd_angstrom",
            "relaxed_rmsd_angstrom",
            "relaxed_rmsd_to_before_event",
            "event_changed_after_relax",
        ):
            value = row.get(key)
            if value is not None:
                info[key] = value
    return info


def write_loss_artifacts(training_log: Path, output_dir: Path) -> None:
    """Aggregate step log records by epoch and write a four-loss plot."""
    rows: list[dict[str, float]] = []
    with Path(training_log).open(encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=" ", skipinitialspace=True)
        for row in reader:
            if not row or row.get("epoch") in {None, ""}:
                continue
            rows.append({key: float(row[key]) for key in row if row[key] not in {None, ""}})

    by_epoch: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_epoch[int(row["epoch"])].append(row)
    aggregate_rows = []
    for epoch, epoch_rows in sorted(by_epoch.items()):
        aggregate_rows.append(
            {
                "epoch": epoch,
                "total_loss": float(np.mean([row["total_loss"] for row in epoch_rows])),
                "velocity_loss": float(np.mean([row["velocity_loss"] for row in epoch_rows])),
                "active_loss": float(np.mean([row["active_loss"] for row in epoch_rows])),
                "direction_loss": float(np.mean([row["direction_loss"] for row in epoch_rows])),
            }
        )
    csv_path = output_dir / "loss_by_epoch.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "total_loss", "velocity_loss", "active_loss", "direction_loss"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for column, label in [
        ("total_loss", "total"),
        ("velocity_loss", "velocity"),
        ("active_loss", "active"),
        ("direction_loss", "direction"),
    ]:
        axis.plot([row["epoch"] for row in aggregate_rows], [row[column] for row in aggregate_rows], label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean training loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close(figure)


def write_trial_plots(rows: list[dict[str, Any]], plots_dir: Path) -> None:
    """Write per-basin trial diagnostics and global raw/EAM comparisons."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()
    by_basin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_basin[str(row["basin_id"])].append(row)

    for basin_id, basin_rows in by_basin.items():
        basin_rows.sort(key=lambda row: int(row["trial_index"]))
        trial = [int(row["trial_index"]) for row in basin_rows]
        raw = [float(row["raw_rmsd_angstrom"]) for row in basin_rows]
        relaxed = [_as_float(row["relaxed_rmsd_angstrom"]) for row in basin_rows]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(trial, raw, "o-", label="raw", markersize=3)
        valid_trial = [value for value, rmsd in zip(trial, relaxed) if rmsd is not None]
        valid_relaxed = [rmsd for rmsd in relaxed if rmsd is not None]
        if valid_relaxed:
            axes[0].plot(valid_trial, valid_relaxed, "o-", label="EAM relaxed", markersize=3)
        axes[0].set_xlabel("Trial index")
        axes[0].set_ylabel("Nearest-product RMSD (Å)")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].hist(raw, bins=min(20, max(5, len(raw))), alpha=0.65, label="raw")
        if valid_relaxed:
            axes[1].hist(valid_relaxed, bins=min(20, max(5, len(valid_relaxed))), alpha=0.65, label="EAM relaxed")
        axes[1].set_xlabel("Nearest-product RMSD (Å)")
        axes[1].set_ylabel("Trial count")
        axes[1].legend()
        figure.suptitle(f"Basin {basin_id}")
        figure.tight_layout()
        figure.savefig(plots_dir / f"{basin_id}_rmsd_100_trials.png", dpi=160)
        plt.close(figure)

    basin_ids = list(by_basin)
    raw_medians = [float(np.median([float(row["raw_rmsd_angstrom"]) for row in by_basin[basin_id]])) for basin_id in basin_ids]
    relaxed_medians = []
    for basin_id in basin_ids:
        values = [_as_float(row["relaxed_rmsd_angstrom"]) for row in by_basin[basin_id]]
        usable = [value for value in values if value is not None]
        relaxed_medians.append(float(np.median(usable)) if usable else np.nan)
    figure, axis = plt.subplots(figsize=(max(7, len(basin_ids) * 0.32), 4.5))
    x = np.arange(len(basin_ids))
    axis.plot(x, raw_medians, "o", label="raw median")
    if np.isfinite(relaxed_medians).any():
        axis.plot(x, relaxed_medians, "o", label="EAM-relaxed median")
    axis.set_xticks(x, basin_ids, rotation=60, ha="right")
    axis.set_ylabel("Median nearest-product RMSD (Å)")
    axis.set_xlabel("Test basin")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_dir / "rmsd_by_basin.png", dpi=160)
    plt.close(figure)

    valid_rows = [row for row in rows if _as_float(row["relaxed_rmsd_angstrom"]) is not None]
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    if valid_rows:
        raw = np.asarray([float(row["raw_rmsd_angstrom"]) for row in valid_rows])
        relaxed = np.asarray([float(row["relaxed_rmsd_angstrom"]) for row in valid_rows])
        switched = np.asarray([str(row["event_changed_after_relax"]).lower() == "true" for row in valid_rows])
        axis.scatter(raw[~switched], relaxed[~switched], label="same nearest event", alpha=0.7, s=18)
        if switched.any():
            axis.scatter(raw[switched], relaxed[switched], label="nearest event changed", alpha=0.8, s=18)
        limit = max(float(np.max(raw)), float(np.max(relaxed)), 1e-6) * 1.05
        axis.plot([0.0, limit], [0.0, limit], "k--", linewidth=1, label="y = x")
        improved = float(np.mean(relaxed < raw))
        switched_fraction = float(np.mean(switched))
        legend_title = f"valid={len(valid_rows)}\nimproved={improved:.1%}\nswitch={switched_fraction:.1%}"
    else:
        axis.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1, label="y = x")
        legend_title = "valid=0"
    axis.set_xlabel("Raw nearest-product RMSD (Å)")
    axis.set_ylabel("EAM-relaxed nearest-product RMSD (Å)")
    axis.legend(title=legend_title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_dir / "eam_rmsd_before_after.png", dpi=160)
    plt.close(figure)


def _write_representatives(representatives: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    for basin_id, records in representatives.items():
        ordered = sorted(records, key=lambda record: float(record["row"]["raw_rmsd_angstrom"]))
        selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]] if ordered else []
        frames = []
        manifest_records = []
        for label, record in zip(["best", "median", "worst"], selected):
            row = record["row"]
            raw_match = record["raw_match"]
            raw_source = _source_name(raw_match.get("source_file")) or row.get("source_file_before")
            raw_candidate_role = "raw_candidate"
            frames.append(_candidate_atoms(
                record["raw"],
                raw_match["product"],
                role=raw_candidate_role,
                basin_id=basin_id,
                trial_index=int(row["trial_index"]),
                event_id=raw_match["event_id"],
                source_file=raw_source,
                representative_rank=label,
                row=row,
            ))
            frame_roles = [raw_candidate_role, "raw_reference_product"]
            relaxed_match = record.get("relaxed_match")
            if record["relaxed"] is not None:
                relaxed_source = _source_name(relaxed_match.get("source_file")) or row.get("source_file_after")
                frames.append(_reference_atoms(
                    raw_match["product"],
                    role="raw_reference_product",
                    basin_id=basin_id,
                    trial_index=int(row["trial_index"]),
                    event_id=raw_match["event_id"],
                    source_file=raw_source,
                    representative_rank=label,
                    row=row,
                ))
                frames.append(_candidate_atoms(
                    record["relaxed"],
                    relaxed_match["product"],
                    role="eam_relaxed_candidate",
                    basin_id=basin_id,
                    trial_index=int(row["trial_index"]),
                    event_id=relaxed_match["event_id"],
                    source_file=relaxed_source,
                    representative_rank=label,
                    row=row,
                ))
                frames.append(_reference_atoms(
                    relaxed_match["product"],
                    role="eam_reference_product",
                    basin_id=basin_id,
                    trial_index=int(row["trial_index"]),
                    event_id=relaxed_match["event_id"],
                    source_file=relaxed_source,
                    representative_rank=label,
                    row=row,
                ))
                frame_roles = [
                    "raw_candidate",
                    "raw_reference_product",
                    "eam_relaxed_candidate",
                    "eam_reference_product",
                ]
            else:
                frames.append(_reference_atoms(
                    raw_match["product"],
                    role="raw_reference_product",
                    basin_id=basin_id,
                    trial_index=int(row["trial_index"]),
                    event_id=raw_match["event_id"],
                    source_file=raw_source,
                    representative_rank=label,
                    row=row,
                ))
            manifest_records.append(
                {
                    "rank": label,
                    "trial_index": int(row["trial_index"]),
                    "candidate_id": row.get("candidate_id"),
                    "init_random_seed": int(row["init_random_seed"]),
                    "raw_rmsd_angstrom": float(row["raw_rmsd_angstrom"]),
                    "relaxed_rmsd_angstrom": _as_float(row.get("relaxed_rmsd_angstrom")),
                    "relaxed_rmsd_to_before_event": _as_float(row.get("relaxed_rmsd_to_before_event")),
                    "relax_status": row.get("relax_status"),
                    "relax_converged": row.get("relax_converged"),
                    "relax_steps": row.get("relax_steps"),
                    "relax_error": row.get("relax_error"),
                    "nearest_event_before": row.get("nearest_event_before"),
                    "source_file_before": row.get("source_file_before"),
                    "nearest_event_after": row.get("nearest_event_after"),
                    "source_file_after": row.get("source_file_after"),
                    "event_changed_after_relax": row.get("event_changed_after_relax"),
                    "frames": frame_roles,
                }
            )
        _write_atoms(output_dir / f"{basin_id}.traj", frames)
        transitions: dict[str, int] = defaultdict(int)
        raw_events: dict[str, int] = defaultdict(int)
        relaxed_events: dict[str, int] = defaultdict(int)
        for record in records:
            row = record["row"]
            raw_event = row.get("nearest_event_before")
            relaxed_event = row.get("nearest_event_after")
            if raw_event:
                raw_events[str(raw_event)] += 1
            if relaxed_event:
                relaxed_events[str(relaxed_event)] += 1
                transitions[f"{raw_event} -> {relaxed_event}"] += 1
        manifest = {
            "basin_id": str(basin_id),
            "num_trials": len(records),
            "raw_nearest_event_counts": dict(raw_events),
            "relaxed_nearest_event_counts": dict(relaxed_events),
            "event_transition_counts": dict(transitions),
            "representatives": manifest_records,
        }
        (output_dir / f"{basin_id}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_gaussian_trials(
    *,
    system: str,
    model,
    test_catalog: EventCatalog,
    output_dir: Path,
    cutoff: float,
    num_steps: int,
    graph_update_interval: int,
    gaussian_scale: float,
    trials_per_basin: int,
    sampling_seed: int,
    active_threshold: float = 0.1,
    device=None,
    calculator_factory=None,
    relaxation_config: RelaxationConfig | None = None,
) -> dict[str, Any]:
    """Generate direct-basin Gaussian trials and optionally relax each one.

    ``calculator_factory`` returns a fresh ASE calculator per trial; any
    calculator works (EAM via ``basinflow.relaxation.eam_calculator``, EMT, a
    DFT code, or a machine-learned potential). ``None`` disables relaxation.
    """
    sampling_dir = Path(output_dir) / "sampling"
    plots_dir = sampling_dir / "plots"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    raw_atoms = []
    relaxed_atoms = []
    rows: list[dict[str, Any]] = []
    representatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    basin_wall_times: dict[str, float] = {}
    if calculator_factory is not None and not callable(calculator_factory):
        raise TypeError("calculator_factory must be callable")
    prepared_config = None
    preparation_error = None
    if calculator_factory is not None:
        prepared_config = relaxation_config or RelaxationConfig()

    metrics_path = sampling_dir / "trial_metrics.csv"
    progress_path = sampling_dir / "progress.log"
    started_at = time.perf_counter()
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file, progress_path.open("w", encoding="utf-8") as progress_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=TRIAL_COLUMNS)
        metrics_writer.writeheader()
        metrics_file.flush()
        for basin_id in test_catalog.basin_ids:
            basin_started_at = time.perf_counter()
            for trial_index in range(int(trials_per_basin)):
                init_random_seed = int(sampling_seed) + trial_index
                sampler = CandidateSampler(
                    model=model,
                    init_generators=[GaussianInit(scale=gaussian_scale, random_seed=init_random_seed)],
                    cutoff=float(cutoff),
                    num_steps=int(num_steps),
                    graph_update_interval=int(graph_update_interval),
                    checkpoint_id=str(Path(output_dir) / "checkpoint.pt"),
                    device=device,
                    active_threshold=active_threshold,
                )
                result = sampler.sample(test_catalog, basin_id)
                candidate = result.candidates[0]
                raw = result.generated_structures[candidate.generated_structure_id]
                raw_match = nearest_product_match(raw, test_catalog, basin_id)
                candidate_id = f"{basin_id}:trial:{trial_index}"
                row: dict[str, Any] = {
                    "system": system,
                    "basin_id": basin_id,
                    "trial_index": trial_index,
                    "candidate_id": candidate_id,
                    "init_random_seed": init_random_seed,
                    "nearest_event_before": raw_match["event_id"],
                    "source_file_before": _source_name(raw_match["source_file"]),
                    "raw_rmsd_angstrom": raw_match["rmsd_angstrom"],
                    "relax_status": "not_requested" if calculator_factory is None else "failed",
                    "relax_converged": None,
                    "relax_steps": None,
                    "relax_error": preparation_error,
                    "nearest_event_after": None,
                    "source_file_after": None,
                    "relaxed_rmsd_angstrom": None,
                    "relaxed_rmsd_to_before_event": None,
                    "event_changed_after_relax": None,
                }
                relaxed = None
                relaxed_match = None
                if prepared_config is not None and preparation_error is None:
                    try:
                        relaxed, converged, relax_steps = relax(raw, calculator_factory(), prepared_config)
                        relaxed_match = nearest_product_match(
                            relaxed,
                            test_catalog,
                            basin_id,
                        )
                        row.update(
                            {
                                "relax_status": "converged" if converged else "max_steps",
                                "relax_converged": converged,
                                "relax_steps": relax_steps,
                                "relax_error": None,
                                "nearest_event_after": relaxed_match["event_id"],
                                "source_file_after": _source_name(relaxed_match["source_file"]),
                                "relaxed_rmsd_angstrom": relaxed_match["rmsd_angstrom"],
                                "relaxed_rmsd_to_before_event": movable_mic_rmsd(
                                    relaxed,
                                    raw_match["product"],
                                    test_catalog.structures[
                                        test_catalog.basins[basin_id].reactant_structure_id
                                    ],
                                ),
                                "event_changed_after_relax": relaxed_match["event_id"] != raw_match["event_id"],
                            }
                        )
                    except Exception as exc:
                        row.update({"relax_status": "failed", "relax_error": f"{type(exc).__name__}: {exc}"})
                raw_atoms.append(_candidate_atoms(
                    raw,
                    raw_match["product"],
                    role="raw_candidate",
                    basin_id=basin_id,
                    trial_index=trial_index,
                    event_id=raw_match["event_id"],
                    source_file=raw_match.get("source_file"),
                    row=row,
                ))
                if relaxed is not None and relaxed_match is not None:
                    relaxed_atoms.append(_candidate_atoms(
                        relaxed,
                        relaxed_match["product"],
                        role="eam_relaxed_candidate",
                        basin_id=basin_id,
                        trial_index=trial_index,
                        event_id=relaxed_match["event_id"],
                        source_file=relaxed_match.get("source_file"),
                        row=row,
                    ))
                rows.append(row)
                metrics_writer.writerow(row)
                metrics_file.flush()
                representatives[basin_id].append({"row": row, "raw": raw, "raw_match": raw_match, "relaxed": relaxed, "relaxed_match": relaxed_match})
                progress_file.write(f"basin={basin_id} trial={trial_index} relax_status={row['relax_status']}\n")
                progress_file.flush()
            basin_wall_times[str(basin_id)] = time.perf_counter() - basin_started_at
            progress_file.write(f"basin={basin_id} complete wall_time_seconds={basin_wall_times[str(basin_id)]:.3f}\n")
            progress_file.flush()
    _write_atoms(sampling_dir / "generated_before_eam.traj", raw_atoms)
    _write_atoms(sampling_dir / "generated_after_eam.traj", relaxed_atoms)
    _write_representatives(representatives, sampling_dir / "representatives")
    write_trial_plots(rows, plots_dir)

    raw_values = [float(row["raw_rmsd_angstrom"]) for row in rows]
    relaxed_values = [value for row in rows if (value := _as_float(row["relaxed_rmsd_angstrom"])) is not None]
    valid_rows = [row for row in rows if _as_float(row["relaxed_rmsd_angstrom"]) is not None]
    raw_event_counts: dict[str, int] = defaultdict(int)
    relaxed_event_counts: dict[str, int] = defaultdict(int)
    event_transition_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        before = row.get("nearest_event_before")
        after = row.get("nearest_event_after")
        if before:
            raw_event_counts[str(before)] += 1
        if after:
            relaxed_event_counts[str(after)] += 1
            event_transition_counts[f"{before} -> {after}"] += 1
    summary = {
        "system": system,
        "num_test_basins": len(test_catalog.basin_ids),
        "trials_per_test_basin": int(trials_per_basin),
        "num_trials": len(rows),
        "raw_rmsd_angstrom": _rmsd_summary(raw_values),
        "eam_relaxed_rmsd_angstrom": _rmsd_summary(relaxed_values),
        "relaxation_failure_count": sum(row["relax_status"] == "failed" for row in rows),
        "relaxation_failure_fraction": float(sum(row["relax_status"] == "failed" for row in rows) / len(rows)) if rows else 0.0,
        "improvement_fraction": float(np.mean([float(row["relaxed_rmsd_angstrom"]) < float(row["raw_rmsd_angstrom"]) for row in valid_rows])) if valid_rows else None,
        "event_switch_fraction": float(np.mean([bool(row["event_changed_after_relax"]) for row in valid_rows])) if valid_rows else None,
        "raw_nearest_event_counts": dict(raw_event_counts),
        "relaxed_nearest_event_counts": dict(relaxed_event_counts),
        "event_transition_counts": dict(event_transition_counts),
        "wall_time_seconds": time.perf_counter() - started_at,
        "per_basin_wall_time_seconds": basin_wall_times,
    }
    import json

    (sampling_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
