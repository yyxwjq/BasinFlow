"""ReactOT reference-protocol RMSD, reported next to BasinFlow's strict one.

ReactOT scores a generated transition state against the reference with pymatgen's
``KabschMatcher`` (``same_order=True``), keeps the lower of the two fits after
reflecting one structure (``ignore_chirality=True``), and caps the value at ``1.0``
A (``reactot/analyze/rmsd.py:105-129``).

``aligned_rmsd_angstrom`` stays a proper-rotation fit; this module adds the
reference convention as a second, separately labelled protocol. It reuses
:func:`kabsch_aligned_rmsd` below, whose
``mirror_allowed`` branch is the reference reflection rule.

Permutation search (``same_order=False`` in pymatgen) is not implemented: the
reference TS metric runs with ``same_order=True`` (``pl_trainer.py:1033``), so the
atoms keep their file order.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from basinflow.data import Structure

def kabsch_aligned_rmsd(
    a: Structure,
    b: Structure,
    reference: Structure,
    *,
    mirror_allowed: bool = False,
) -> float:
    """Movable-atom RMSD after optimal rigid alignment.

    ``mirror_allowed=False`` is a proper-rotation fit (``det = +1``);
    ``mirror_allowed=True`` also fits the reflected structure and keeps the lower
    value, which is ReactOT's ``ignore_chirality=True``.
    """
    if a.n_atoms != b.n_atoms or a.n_atoms != reference.n_atoms:
        raise ValueError("structures must have the same atom count")
    movable = np.asarray(reference.movable_mask, dtype=bool)
    if not movable.any():
        movable = np.ones((reference.n_atoms,), dtype=bool)
    first = np.asarray(a.positions, dtype=float)[movable]
    second = np.asarray(b.positions, dtype=float)[movable]
    first = first - first.mean(axis=0)
    second = second - second.mean(axis=0)
    mirrored = second.copy()
    mirrored[:, -1] = -mirrored[:, -1]
    u, _, vt = np.linalg.svd(first.T @ second)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    rmsd = float(np.sqrt(np.mean(np.sum((first @ rotation.T - second) ** 2, axis=1))))
    if not mirror_allowed:
        return rmsd
    u, _, vt = np.linalg.svd(first.T @ mirrored)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    reflected = float(np.sqrt(np.mean(np.sum((first @ rotation.T - mirrored) ** 2, axis=1))))
    return min(rmsd, reflected)


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
    """Both conventions at once.

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

    Keys match ``basinflow.evaluation.transition_state._summary``.
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
