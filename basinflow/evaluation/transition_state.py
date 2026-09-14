"""Basin-level-free evaluation of the R+P -> TS task.

Both endpoints are inputs here, exactly as in ReactOT and MolGEN, so no seed or
active-region machinery is involved: the model starts from the midpoint bridge
and is scored against the recorded transition state. Metrics follow the same
shape as the R -> P report (aligned, clipped, best-of) so the two tasks can be
put side by side.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import StructureRecord
from basinflow.evaluation.basin_recall import kabsch_aligned_rmsd

_RMSD_CLIP = 1.0
_TRIAL_COLUMNS = [
    "evaluation_protocol",
    "pseudo_basin_id",
    "event_id",
    "trial_index",
    "aligned_rmsd_angstrom",
    "clipped_aligned_rmsd_angstrom",
    "source_rmsd_angstrom",
    "total_motion_angstrom",
    "trajectory",
]

def _write_trajectory(path: Path, positions: np.ndarray, species: list[str], info: dict) -> None:
    """Write one trial's integration path as a multi-frame extended XYZ file.

    Frames are ``x0, x1, ..., x_num_steps``: every position the sampler held,
    starting at the drawn source and ending at the proposal. ``extxyz`` is
    self-describing, so a path can be re-analysed without the code that wrote it.
    """
    from ase import Atoms
    from ase.io import write

    if len(species) != positions.shape[1]:
        raise ValueError(
            f"species/positions mismatch: {len(species)} species vs {positions.shape[1]} atoms"
        )
    frames = [
        Atoms(symbols=species, positions=positions[step].astype(float), pbc=False)
        for step in range(positions.shape[0])
    ]
    for step, atoms in enumerate(frames):
        atoms.info = {"event_id": str(info["event_id"]), "trial_index": int(info["trial_index"]),
                      "step": step, "num_steps": int(info["num_steps"])}
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, frames, format="extxyz")


def _torch():
    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("transition-state sampling requires basinflow[models]") from exc
    return torch, DataLoader


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_transition_state_trials(
    *,
    model: Any,
    catalog: EventCatalog,
    output_dir: str | Path,
    cutoff: float,
    num_steps: int,
    source_scale: float,
    trials_per_basin: int = 1,
    num_events: int | None = None,
    sampling_seed: int = 20260910,
    time_distribution: str = "beta",
    beta_alpha: float = 0.8,
    write_trajectories: bool = False,
    device=None,
) -> dict[str, Any]:
    """Integrate the learned R+P -> TS field from a sampled source and score it.

    Sampling is the ODE of the trained field, so it must match training on three
    points (``docs/25``):

    * the source is drawn from the *same* distribution as training
      (``source_scale``), not fixed at the bridge, otherwise the trajectory
      starts outside the region the field was fitted on;
    * each step is advanced to a *new* time drawn from the training time law,
      so ``num_steps`` is an actual integration count rather than a repeated
      query at one time;
    * the time fed to the network is the current one, not ``step / num_steps``.

    A single step from a fixed source is direct regression, not flow sampling.

    ``write_trajectories`` additionally stores every trial's full integration
    path under ``sampling/trajectories/`` (one extended-XYZ file per trial, with
    frames ``x0 ... x_num_steps``) and records the relative path in the
    ``trajectory`` column of ``sampling/trial_metrics.csv``. It is off by
    default: with ``trials_per_basin`` samples per event the file count grows as
    ``num_events * trials_per_basin``.
    """
    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    if trials_per_basin <= 0:
        raise ValueError("trials_per_basin must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if time_distribution not in {"fixed", "beta", "uniform"}:
        raise ValueError(
            "sampling time_distribution must be 'fixed', 'beta' or 'uniform'"
        )

    from basinflow.data.pyg import TransitionStateDataset

    # A pinned-time (deterministic-bridge) model has no time law to draw from:
    # the sampler's uniform 0 -> 1 grid *is* its grid. Map it so the legacy arm
    # stays runnable, while the reported law keeps the configured name.
    sampling_law = "uniform" if time_distribution == "fixed" else time_distribution

    torch, DataLoader = _torch()
    available = list(catalog.event_ids)
    if not available:
        raise ValueError("catalog has no events")
    if num_events is not None:
        available = available[: int(num_events)]
    basin_ids = sorted({catalog.events[event_id].basin_id for event_id in available})
    scoped = catalog.subset(basin_ids)

    output_dir = Path(output_dir)
    sampling_dir = output_dir / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Species are read from the catalog rather than carried through the graph
    # batch; they are constant per event and only needed when writing.
    species_by_event = {
        event_id: list(catalog.structures[catalog.events[event_id].reactant_structure_id].species)
        for event_id in available
    }

    rows: list[dict[str, Any]] = []
    for trial_index in range(int(trials_per_basin)):
        # ``flow_time=None`` lets the dataset draw t from the training law, which
        # is what makes the per-step advance a genuine ODE step.
        dataset = TransitionStateDataset(
            scoped,
            flow_time=None,
            seed=int(sampling_seed) + trial_index,
            source_scale=float(source_scale),
            time_distribution=sampling_law,
            beta_alpha=float(beta_alpha),
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                if device is not None:
                    batch = batch.to(device)
                # Start from the *source* draw, not from the interpolant the
                # dataset returns: at t = 0 the flow begins at x0.
                positions = batch.source_pos.clone()
                batch.flow_time = torch.zeros_like(batch.flow_time)
                if write_trajectories:
                    # x0 first, then one entry per step, so the path has
                    # num_steps + 1 frames.
                    path = [positions.detach().cpu().numpy()]
                for step in range(int(num_steps)):
                    batch.pos = positions
                    # Uniform integration grid 0 -> 1. The field is queried at
                    # the current time, so `num_steps` is a real step count.
                    output = model(batch, t=batch.flow_time.squeeze(-1))
                    velocity = output["velocity"]
                    if not torch.isfinite(velocity).all():
                        raise FloatingPointError("non-finite transition-state velocity")
                    step_size = 1.0 / float(int(num_steps))
                    positions = positions + step_size * velocity
                    positions[~batch.movable_mask] = batch.source_pos[~batch.movable_mask]
                    batch.flow_time = batch.flow_time + step_size
                    if write_trajectories:
                        path.append(positions.detach().cpu().numpy())
                start = 0
                for graph in range(batch.num_graphs):
                    count = int((batch.batch == graph).sum())
                    sl = slice(start, start + count)
                    start += count
                    frame = lambda name, pos: StructureRecord(  # noqa: E731
                        name, ["H"] * count, pos[sl].detach().cpu().numpy(),
                        np.eye(3) * 100.0, np.zeros(3, dtype=bool),
                    )
                    reference = frame("r", batch.reactant_pos)
                    aligned = kabsch_aligned_rmsd(frame("p", positions), frame("t", batch.target_pos), reference)
                    source = kabsch_aligned_rmsd(frame("s", batch.source_pos), frame("t", batch.target_pos), reference)
                    motion = kabsch_aligned_rmsd(frame("m", positions), frame("s", batch.source_pos), reference)
                    trajectory = ""
                    if write_trajectories:
                        event_id = str(batch.event_id[graph])
                        relative = Path("trajectories") / f"{event_id.replace(':', '_')}_trial{trial_index}.extxyz"
                        _write_trajectory(
                            sampling_dir / relative,
                            np.stack([step[sl] for step in path]),
                            species_by_event[event_id],
                            {"event_id": event_id, "trial_index": trial_index, "num_steps": int(num_steps)},
                        )
                        trajectory = relative.as_posix()
                    rows.append({
                        "evaluation_protocol": "transition_state_flow_ode",
                        "pseudo_basin_id": batch.basin_id[graph],
                        "event_id": batch.event_id[graph],
                        "trial_index": trial_index,
                        "aligned_rmsd_angstrom": aligned,
                        "clipped_aligned_rmsd_angstrom": min(aligned, _RMSD_CLIP),
                        "source_rmsd_angstrom": source,
                        "total_motion_angstrom": motion,
                        "trajectory": trajectory,
                    })

    with (sampling_dir / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_TRIAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    aligned = [row["aligned_rmsd_angstrom"] for row in rows]
    best: dict[str, float] = {}
    for row in rows:
        key = row["pseudo_basin_id"]
        best[key] = min(best.get(key, float("inf")), row["aligned_rmsd_angstrom"])
    summary = {
        "evaluation_protocol": "transition_state_flow_ode",
        "task": "reactant_plus_product_to_transition_state",
        "num_events": len(scoped.event_ids),
        "trials_per_event": int(trials_per_basin),
        "num_trials": len(rows),
        "num_steps": int(num_steps),
        "source_scale": float(source_scale),
        "sampling_seed": int(sampling_seed),
        "time_distribution": time_distribution,
        "beta_alpha": float(beta_alpha),
        "trajectories": {
            "written": bool(write_trajectories),
            "directory": "sampling/trajectories" if write_trajectories else None,
            "frames_per_trial": int(num_steps) + 1 if write_trajectories else None,
            "format": "extxyz" if write_trajectories else None,
        },
        "aligned_rmsd_angstrom": _summary(aligned),
        "clipped_aligned_rmsd_angstrom": _summary([min(v, _RMSD_CLIP) for v in aligned]),
        "source_rmsd_angstrom": _summary([row["source_rmsd_angstrom"] for row in rows]),
        "best_of_trials": {
            "selector": "oracle_min_rmsd",
            "notice": "Diagnostic ceiling; selection uses the reference transition state.",
            "aligned_rmsd_angstrom": _summary(list(best.values())),
        },
        "relaxation": "not_run",
        "saddle_validation": "not_run",
    }
    (sampling_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
