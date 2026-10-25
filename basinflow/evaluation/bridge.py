"""Sampling and scoring for the ``R + P -> TS`` bridge.

Generation starts at the midpoint prior ``(R + P) / 2`` at bridge time
``t_sb = 1`` and walks down to ``t_sb -> 0``, where the field is
``dx/dt_sb = net_out * sqrt(beta / t_sb)``
(``reactot/diffusion/en_sb.py:415-424``, ``net_out = (x_t - x_0) / sigma_t``).

:func:`basinflow.models.sb.make_bridge_ode_field` performs the change of variable to
``progress = 1 - t_sb`` so :func:`basinflow.sampling.ode.sample_ode` can integrate an
increasing interval. This module drives that field for every backend, applies the
movement constraint, and reports the strict RMSD next to the reference protocol.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from basinflow.data import BridgeFlowDataset, EventCatalog, Structure
from basinflow.evaluation.reactot_metrics import (
    REACTOT_REFERENCE_PROTOCOL,
    REACTOT_RMSD_CAP,
    dual_protocol_rmsd,
    kabsch_aligned_rmsd,
    summarize,
)
from basinflow.models.sb import DEFAULT_T_EPS, SBSchedule, make_bridge_ode_field
from basinflow.sampling.ode import DEFAULT_METHOD, sample_ode

_PROTOCOL = "reactot_bridge_ode"
_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def _write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(symbols)), comment]
    for symbol, (x, y, z) in zip(symbols, np.asarray(coords, dtype=float)):
        lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
]


def _torch_loader():
    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("bridge sampling requires basinflow[models]") from exc
    return torch, DataLoader


def run_bridge_trials(
    *,
    model: Any,
    catalog: EventCatalog,
    output_dir: str | Path,
    num_steps: int,
    schedule: SBSchedule,
    trials_per_event: int = 1,
    num_events: int | None = None,
    sampling_seed: int = 20260910,
    integrator: str = DEFAULT_METHOD,
    solver_atol: float = 1e-2,
    solver_rtol: float = 1e-2,
    t_eps: float = DEFAULT_T_EPS,
    device=None,
    write_structures: bool = False,
) -> dict[str, Any]:
    """Integrate the bridge field from the midpoint prior and score the proposal.

    ``num_steps`` is the number of saved points on ``torchdiffeq``'s output grid;
    the solver chooses its own step sizes. ``solver_atol``/``solver_rtol`` default
    to the reference's ``1e-2``.
    """
    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    if not isinstance(schedule, SBSchedule):
        raise TypeError("schedule must be an SBSchedule")
    if trials_per_event <= 0:
        raise ValueError("trials_per_event must be positive")
    if num_steps < 2:
        raise ValueError("num_steps is the number of saved integration points and must be >= 2")
    if not 0.0 < float(t_eps) < 1.0:
        raise ValueError("t_eps must be in (0, 1)")

    torch, DataLoader = _torch_loader()
    available = list(catalog.event_ids)
    if not available:
        raise ValueError("catalog has no events")
    if num_events is not None:
        available = available[: int(num_events)]
    scoped = catalog.select(available)

    output_dir = Path(output_dir)
    sampling_dir = output_dir / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = sampling_dir / "structures" if write_structures else None
    if structure_dir is not None:
        structure_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # A constant per-step variance makes ``beta = betas[0] * T`` the scalar that
    # appears in the reference's ODE field (en_sb.py:412).
    beta = float(schedule.betas[0]) * float(schedule.timesteps)

    rows: list[dict[str, Any]] = []
    fixed_motion = 0.0
    integrator_stats: list[dict[str, Any]] = []
    for trial_index in range(int(trials_per_event)):
        dataset = BridgeFlowDataset(
            scoped,
            schedule,
            seed=int(sampling_seed) + trial_index,
            lowest_step=0,
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                if device is not None:
                    batch = batch.to(device)
                initial = batch.source_pos.clone()
                frozen = ~batch.movable_mask
                field = make_bridge_ode_field(model, batch, beta=torch.tensor(beta), t_eps=t_eps)

                def constrained_field(state, progress):
                    velocity = field(state, progress)
                    if not torch.isfinite(velocity).all():
                        raise FloatingPointError("non-finite bridge velocity")
                    return torch.where(frozen[:, None], torch.zeros_like(velocity), velocity)

                final, report = sample_ode(
                    constrained_field,
                    initial,
                    t0=0.0,
                    t1=1.0 - float(t_eps),
                    num_steps=int(num_steps),
                    atol=float(solver_atol),
                    rtol=float(solver_rtol),
                    method=integrator,
                )
                positions = torch.where(frozen[:, None], initial, final)
                integrator_stats.append(report)
                if bool(frozen.any()):
                    drift = (positions - initial)[frozen].abs().max()
                    fixed_motion = max(fixed_motion, float(drift.detach().cpu()))

                start = 0
                for graph in range(batch.num_graphs):
                    count = int((batch.batch == graph).sum())
                    sl = slice(start, start + count)
                    start += count
                    frame = lambda pos: Structure(  # noqa: E731
                        ["H"] * count, pos[sl].detach().cpu().numpy()
                    )
                    proposal = frame(positions)
                    target_frame = frame(batch.target_pos)
                    source_frame = frame(batch.source_pos)
                    reference = frame(batch.reactant_pos)
                    aligned = kabsch_aligned_rmsd(proposal, target_frame, reference)
                    source = kabsch_aligned_rmsd(source_frame, target_frame, reference)
                    motion = kabsch_aligned_rmsd(proposal, source_frame, reference)
                    dual = dual_protocol_rmsd(proposal, target_frame, reference, cap=REACTOT_RMSD_CAP)
                    event_id = str(batch.event_id[graph])
                    symbols = [_SYMBOLS.get(int(value), "X") for value in batch.z[sl].tolist()]
                    if structure_dir is not None:
                        prefix = structure_dir / f"{event_id}_trial{trial_index}"
                        _write_xyz(
                            Path(f"{prefix}_pred.xyz"),
                            symbols,
                            positions[sl].detach().cpu().numpy(),
                            f"event={event_id} trial={trial_index} kind=pred reference_rmsd={dual['reactot_reference_rmsd_angstrom']:.6f}",
                        )
                        _write_xyz(
                            Path(f"{prefix}_target.xyz"),
                            symbols,
                            batch.target_pos[sl].detach().cpu().numpy(),
                            f"event={event_id} trial={trial_index} kind=target_ts",
                        )
                        _write_xyz(
                            Path(f"{prefix}_reactant.xyz"),
                            symbols,
                            batch.reactant_pos[sl].detach().cpu().numpy(),
                            f"event={event_id} trial={trial_index} kind=reactant",
                        )
                        _write_xyz(
                            Path(f"{prefix}_product.xyz"),
                            symbols,
                            batch.endpoint_pos[sl].detach().cpu().numpy(),
                            f"event={event_id} trial={trial_index} kind=product",
                        )
                        _write_xyz(
                            Path(f"{prefix}_midpoint.xyz"),
                            symbols,
                            batch.source_pos[sl].detach().cpu().numpy(),
                            f"event={event_id} trial={trial_index} kind=midpoint_prior",
                        )
                    rows.append(
                        {
                            "evaluation_protocol": _PROTOCOL,
                            "event_id": batch.event_id[graph],
                            "trial_index": trial_index,
                            "aligned_rmsd_angstrom": aligned,
                            "clipped_aligned_rmsd_angstrom": min(aligned, REACTOT_RMSD_CAP),
                            "reactot_reference_rmsd_angstrom": dual["reactot_reference_rmsd_angstrom"],
                            "reactot_reference_uncapped_rmsd_angstrom": dual[
                                "reactot_reference_uncapped_rmsd_angstrom"
                            ],
                            "chirality_gain_angstrom": dual["chirality_gain_angstrom"],
                            "source_rmsd_angstrom": source,
                            "total_motion_angstrom": motion,
                        }
                    )

    with (sampling_dir / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_TRIAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if trials_per_event > 1:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(str(row["event_id"]), []).append(float(row["reactot_reference_rmsd_angstrom"]))
        with (sampling_dir / "trial_stats.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["event_id", "trials", "mean_reference_rmsd", "std_reference_rmsd", "min_reference_rmsd", "max_reference_rmsd"])
            for event_id, values in grouped.items():
                writer.writerow([
                    event_id,
                    len(values),
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.min(values)),
                    float(np.max(values)),
                ])

    aligned = [row["aligned_rmsd_angstrom"] for row in rows]
    best: dict[str, float] = {}
    for row in rows:
        key = row["event_id"]
        best[key] = min(best.get(key, float("inf")), row["aligned_rmsd_angstrom"])
    summary = {
        "evaluation_protocol": _PROTOCOL,
        "task": "reactant_plus_product_to_transition_state",
        "method": "schrodinger_bridge_ode",
        "num_events": len(scoped.event_ids),
        "trials_per_event": int(trials_per_event),
        "num_trials": len(rows),
        "num_steps": int(num_steps),
        "integrator": integrator,
        "solver": {"atol": float(solver_atol), "rtol": float(solver_rtol)},
        "saved_points": summarize([float(entry["num_steps"]) for entry in integrator_stats]),
        "field_evaluations": summarize(
            [float(entry["field_evaluations"]) for entry in integrator_stats]
        ),
        "sampling_seed": int(sampling_seed),
        "bridge": {
            "timesteps": int(schedule.timesteps),
            "beta": beta,
            "t_eps": float(t_eps),
            "direction": "midpoint_prior_to_transition_state",
        },
        "fixed_atom_max_motion_angstrom": float(fixed_motion),
        "aligned_rmsd_angstrom": summarize(aligned),
        "clipped_aligned_rmsd_angstrom": summarize([min(value, REACTOT_RMSD_CAP) for value in aligned]),
        "reactot_reference_protocol": {
            "label": REACTOT_REFERENCE_PROTOCOL,
            "notice": (
                "Mirror allowed (ReactOT ignore_chirality=True) and capped at "
                f"{REACTOT_RMSD_CAP} A; the strict aligned_rmsd_angstrom above is unchanged."
            ),
            "rmsd_angstrom": summarize([row["reactot_reference_rmsd_angstrom"] for row in rows]),
            "uncapped_rmsd_angstrom": summarize(
                [row["reactot_reference_uncapped_rmsd_angstrom"] for row in rows]
            ),
            "chirality_gain_angstrom": summarize([row["chirality_gain_angstrom"] for row in rows]),
        },
        "source_rmsd_angstrom": summarize([row["source_rmsd_angstrom"] for row in rows]),
        "total_motion_angstrom": summarize([row["total_motion_angstrom"] for row in rows]),
        "per_event_trial_mean_rmsd_angstrom": summarize(
            [
                float(np.mean(values))
                for values in (
                    [
                        float(row["reactot_reference_rmsd_angstrom"])
                        for row in rows
                        if str(row["event_id"]) == str(event_id)
                    ]
                    for event_id in dict.fromkeys(str(row["event_id"]) for row in rows)
                )
            ]
        ),
        "best_of_trials": {
            "selector": "min_rmsd",
            "notice": "Diagnostic ceiling; selection uses the reference transition state.",
            "aligned_rmsd_angstrom": summarize(list(best.values())),
        },
    }
    (sampling_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
