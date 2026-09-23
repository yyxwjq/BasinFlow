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

from basinflow.data import EventCatalog, Structure, event_target
from basinflow.evaluation.basin_recall import kabsch_aligned_rmsd
from basinflow.evaluation.reactot_metrics import REACTOT_REFERENCE_PROTOCOL, dual_protocol_rmsd
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD, sample_ode

_RMSD_CLIP = 1.0
_TRIAL_COLUMNS = [
    "evaluation_protocol",
    "event_id",
    "trial_index",
    "aligned_rmsd_angstrom",
    "clipped_aligned_rmsd_angstrom",
    "reactot_reference_rmsd_angstrom",
    "reactot_reference_uncapped_rmsd_angstrom",
    "chirality_gain_angstrom",
    "source_rmsd_angstrom",
    "total_motion_angstrom",
    "trajectory",
]

def _write_trajectory(path: Path, positions: np.ndarray, species: list[str], info: dict) -> None:
    """Write one trial's integration path as a multi-frame extended XYZ file.

    Frames are the solver's `saved` points on the integration grid (``num_steps``
    of them, ``x0`` at t=0 through the proposal at t=1). The adaptive solver takes
    its own internal steps between them; the saved grid is dense interpolation, the
    same convention as MolGEN's ``--inference_steps``. ``extxyz`` is
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
                      "step": step, "saved_points": int(info["saved_points"])}
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
    trials_per_event: int = 1,
    num_events: int | None = None,
    sampling_seed: int = 20260910,
    time_distribution: str = "beta",
    beta_alpha: float = 0.8,
    time_epsilon: float = 0.0,
    solver_atol: float = DEFAULT_ATOL,
    solver_rtol: float = DEFAULT_RTOL,
    integrator: str = DEFAULT_METHOD,
    write_trajectories: bool = False,
    device=None,
) -> dict[str, Any]:
    """Integrate the learned R+P -> TS field from a sampled source and score it.

    Sampling solves the ODE of the trained field with MolGEN's solver --
    ``torchdiffeq.odeint(method="dopri5")`` over a ``linspace(0, 1, num_steps)``
    output grid (``basinflow.sampling.ode``) -- so it must match training on three
    points (``docs/25``):

    * the source is drawn from the *same* distribution as training
      (``source_scale``), not fixed at the bridge, otherwise the trajectory
      starts outside the region the field was fitted on;
    * the field is queried at the *current* time on every evaluation;
    * the step size follows the local error instead of a fixed ``1 / num_steps``
      grid, which is what stopped the earlier fixed-step runs from walking off
      the fitted manifold.

    ``num_steps`` is the number of saved points on that grid, as in MolGEN's
    ``--inference_steps``; the solver chooses the actual step sizes. The head is
    read as a velocity, which is the only thing the objective fits.

    ``write_trajectories`` additionally stores every trial's integration path
    under ``sampling/trajectories/`` (one extended-XYZ file per trial) and
    records the relative path in the ``trajectory`` column of
    ``sampling/trial_metrics.csv``.
    """
    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    if trials_per_event <= 0:
        raise ValueError("trials_per_event must be positive")
    if num_steps < 2:
        raise ValueError("num_steps is the number of saved integration points and must be >= 2")
    if not 0.0 <= float(time_epsilon) < 1.0:
        raise ValueError("time_epsilon must be in [0, 1)")
    if time_distribution not in {"fixed", "beta", "uniform"}:
        raise ValueError(
            "sampling time_distribution must be 'fixed', 'beta' or 'uniform'"
        )

    from basinflow.data import TransitionStateDataset

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
    scoped = catalog.select(available)

    output_dir = Path(output_dir)
    sampling_dir = output_dir / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Species are read from the catalog rather than carried through the graph
    # batch; they are constant per event and only needed when writing.
    species_by_event = {
        event_id: list(catalog.events[event_id].reactant.species)
        for event_id in available
    }

    rows: list[dict[str, Any]] = []
    integrator_stats: list[dict[str, Any]] = []
    for trial_index in range(int(trials_per_event)):
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
                frozen = ~batch.movable_mask

                def field(state, current_time):
                    """dx/dt for the current batch: one network query per call."""
                    batch.pos = state
                    scalar_time = current_time.reshape(())
                    batch.flow_time = scalar_time.reshape(1, 1).expand(batch.num_graphs, 1)
                    output = model(batch, t=batch.flow_time.squeeze(-1))
                    velocity = output["velocity"]
                    if not torch.isfinite(velocity).all():
                        raise FloatingPointError("non-finite transition-state velocity")
                    # Fixed atoms must not move; zeroing their velocity keeps the
                    # flow's fixed point consistent with the constraint.
                    return torch.where(frozen[:, None], torch.zeros_like(velocity), velocity)

                final, report = sample_ode(
                    field,
                    positions,
                    t0=0.0,
                    t1=1.0 - float(time_epsilon),
                    num_steps=int(num_steps),
                    atol=float(solver_atol),
                    rtol=float(solver_rtol),
                    method=integrator,
                    record=write_trajectories,
                )
                positions = torch.where(frozen[:, None], batch.source_pos, final)
                batch.flow_time = torch.full_like(batch.flow_time, 1.0 - float(time_epsilon))
                integrator_stats.append({key: value for key, value in report.items() if key != "states"})
                if write_trajectories:
                    path = [entry.detach().cpu().numpy() for entry in report["states"]]
                start = 0
                for graph in range(batch.num_graphs):
                    count = int((batch.batch == graph).sum())
                    sl = slice(start, start + count)
                    start += count
                    frame = lambda pos: Structure(  # noqa: E731
                        ["H"] * count, pos[sl].detach().cpu().numpy(),
                        np.eye(3) * 100.0, np.zeros(3, dtype=bool),
                    )
                    proposal = frame(positions)
                    target_frame = frame(batch.target_pos)
                    source_frame = frame(batch.source_pos)
                    reference = frame(batch.reactant_pos)
                    aligned = kabsch_aligned_rmsd(proposal, target_frame, reference)
                    source = kabsch_aligned_rmsd(source_frame, target_frame, reference)
                    motion = kabsch_aligned_rmsd(proposal, source_frame, reference)
                    dual = dual_protocol_rmsd(proposal, target_frame, reference, cap=_RMSD_CLIP)
                    trajectory = ""
                    if write_trajectories:
                        event_id = str(batch.event_id[graph])
                        relative = Path("trajectories") / f"{event_id.replace(':', '_')}_trial{trial_index}.extxyz"
                        _write_trajectory(
                            sampling_dir / relative,
                            np.stack([step[sl] for step in path]),
                            species_by_event[event_id],
                            {"event_id": event_id, "trial_index": trial_index,
                         "saved_points": int(report["num_steps"])},
                        )
                        trajectory = relative.as_posix()
                    rows.append({
                        "evaluation_protocol": "transition_state_flow_ode",
                        "event_id": batch.event_id[graph],
                        "trial_index": trial_index,
                        "aligned_rmsd_angstrom": aligned,
                        "clipped_aligned_rmsd_angstrom": min(aligned, _RMSD_CLIP),
                        "reactot_reference_rmsd_angstrom": dual["reactot_reference_rmsd_angstrom"],
                        "reactot_reference_uncapped_rmsd_angstrom": dual[
                            "reactot_reference_uncapped_rmsd_angstrom"
                        ],
                        "chirality_gain_angstrom": dual["chirality_gain_angstrom"],
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
        key = row["event_id"]
        best[key] = min(best.get(key, float("inf")), row["aligned_rmsd_angstrom"])
    summary = {
        "evaluation_protocol": "transition_state_flow_ode",
        "task": "reactant_plus_product_to_transition_state",
        "num_events": len(scoped.event_ids),
        "trials_per_event": int(trials_per_event),
        "num_trials": len(rows),
        "num_steps": int(num_steps),
        "time_epsilon": float(time_epsilon),
        "integrator": integrator,
        "solver": {"atol": float(solver_atol), "rtol": float(solver_rtol)},
        "saved_points": _summary([float(entry["num_steps"]) for entry in integrator_stats]),
        "field_evaluations": _summary([float(entry["field_evaluations"]) for entry in integrator_stats]),
        "source_scale": float(source_scale),
        "sampling_seed": int(sampling_seed),
        "time_distribution": time_distribution,
        "beta_alpha": float(beta_alpha),
        "trajectories": {
            "written": bool(write_trajectories),
            "directory": "sampling/trajectories" if write_trajectories else None,
            "frames_per_trial": "num_steps (saved grid)" if write_trajectories else None,
            "format": "extxyz" if write_trajectories else None,
        },
        "aligned_rmsd_angstrom": _summary(aligned),
        "clipped_aligned_rmsd_angstrom": _summary([min(v, _RMSD_CLIP) for v in aligned]),
        "reactot_reference_protocol": {
            "label": REACTOT_REFERENCE_PROTOCOL,
            "notice": (
                "Mirror allowed (ReactOT ignore_chirality=True) and capped at "
                f"{_RMSD_CLIP} A; the strict aligned_rmsd_angstrom above is unchanged."
            ),
            "rmsd_angstrom": _summary([row["reactot_reference_rmsd_angstrom"] for row in rows]),
            "uncapped_rmsd_angstrom": _summary(
                [row["reactot_reference_uncapped_rmsd_angstrom"] for row in rows]
            ),
            "chirality_gain_angstrom": _summary([row["chirality_gain_angstrom"] for row in rows]),
        },
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
