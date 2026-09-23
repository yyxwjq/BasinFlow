"""ReactOT reference-protocol RMSD, reported next to BasinFlow's strict one.

ReactOT scores a generated transition state against the reference with
pymatgen's ``KabschMatcher`` (``same_order=True``), keeps the lower of the two
fits after reflecting one structure (``ignore_chirality=True``), and caps the
value at ``1.0`` A. That is ``batch_rmsd_sb`` ->
``reactot/analyze/rmsd.py:105-129``; ``reactot-pretrained``'s ``eval_rmsd``
reaches it through ``eval_sample_batch``.

BasinFlow's existing ``aligned_rmsd_angstrom`` is a *proper-rotation* fit and
must not be redefined: ``docs/17_reference_protocol_audit.md`` requires the
strict and reference conventions to be reported side by side, because the
reference numbers (``min(rmsd, 1)`` plus mirror) are systematically lower and a
capped value carries no information once an error exceeds the cap.

This module therefore adds a second, clearly labelled protocol instead of
changing the first. It reuses :func:`basinflow.evaluation.basin_recall.kabsch_aligned_rmsd`,
whose ``mirror_allowed`` branch is already the reference reflection rule, so
there is exactly one Kabsch implementation in the package.

Permutation search (``same_order=False`` in pymatgen) is intentionally not
implemented: the reference TS metric runs with ``same_order=True``
(``pl_trainer.py:1033``), so the atoms keep their file order here too.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from basinflow.data import Structure
from basinflow.evaluation.basin_recall import kabsch_aligned_rmsd

#: Protocol label for the reference (mirror + cap) convention.
REACTOT_REFERENCE_PROTOCOL = "reactot_reference_protocol"
#: Protocol label for BasinFlow's existing proper-rotation convention.
BASINFLOW_STRICT_PROTOCOL = "basinflow_strict_protocol"
#: ReactOT's ``min(rmsd, 1.0)`` cap, in angstrom.
REACTOT_RMSD_CAP = 1.0


def reactot_reference_rmsd(
    a: Structure,
    b: Structure,
    reference: Structure,
    *,
    cap: float | None = REACTOT_RMSD_CAP,
) -> float:
    """Reference-convention RMSD over the movable atoms.

    ``min`` of the proper fit and the fit after reflecting ``b`` (ReactOT's
    ``ignore_chirality=True``), optionally capped. ``reference`` supplies the
    movable mask and the cell/PBC used for minimum-image wrapping, exactly as in
    :func:`kabsch_aligned_rmsd`.
    """
    value = kabsch_aligned_rmsd(a, b, reference, mirror_allowed=True)
    if cap is None:
        return float(value)
    return float(min(value, float(cap)))


def dual_protocol_rmsd(
    a: Structure,
    b: Structure,
    reference: Structure,
    *,
    cap: float = REACTOT_RMSD_CAP,
) -> dict[str, float]:
    """Both conventions at once, so a report never has to choose between them.

    Keys:

    * ``strict_aligned_rmsd_angstrom`` -- proper rotation, uncapped (the current
      BasinFlow primary).
    * ``strict_clipped_aligned_rmsd_angstrom`` -- the same, capped at ``cap``.
    * ``reactot_reference_rmsd_angstrom`` -- mirror allowed, capped (ReactOT).
    * ``reactot_reference_uncapped_rmsd_angstrom`` -- mirror allowed, uncapped;
      the diagnostic that does *not* saturate at the cap.
    * ``chirality_gain_angstrom`` -- how much the mirror term alone buys,
      ``min(strict, cap) - reference``; non-negative by construction.
    """
    strict = float(kabsch_aligned_rmsd(a, b, reference))
    mirrored = float(kabsch_aligned_rmsd(a, b, reference, mirror_allowed=True))
    strict_capped = float(min(strict, cap))
    mirrored_capped = float(min(mirrored, cap))
    return {
        "strict_aligned_rmsd_angstrom": strict,
        "strict_clipped_aligned_rmsd_angstrom": strict_capped,
        "reactot_reference_rmsd_angstrom": mirrored_capped,
        "reactot_reference_uncapped_rmsd_angstrom": mirrored,
        "chirality_gain_angstrom": float(max(strict_capped - mirrored_capped, 0.0)),
    }


def summarize_rmsds(values: list[float]) -> dict[str, Any]:
    """Mean/median/min/max for a column, with an explicit ``None`` for empties.

    Mirrors ``basinflow.evaluation.transition_state._summary`` so a dual-protocol
    block has the same shape as the existing summary entries.
    """
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
