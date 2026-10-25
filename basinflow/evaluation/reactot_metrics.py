"""ReactOT reference-protocol RMSD, reported next to BasinFlow's strict one.

ReactOT scores a generated transition state against the reference with pymatgen's
``KabschMatcher`` (``same_order=True``), keeps the lower of the two fits after
reflecting one structure (``ignore_chirality=True``), and caps the value at ``1.0``
A (``reactot/analyze/rmsd.py:105-129``).

``aligned_rmsd_angstrom`` stays a proper-rotation fit; :func:`dual_protocol_rmsd`
adds the reference convention as a second, separately labelled protocol. Atom
order is preserved; permutation search is not needed because the reference runs
with ``same_order=True`` (``pl_trainer.py:1033``).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from basinflow.data import Structure

#: Protocol label for the reference (mirror + cap) convention.
REACTOT_REFERENCE_PROTOCOL = "reactot_reference_protocol"
#: ReactOT's ``min(rmsd, 1.0)`` cap, in angstrom.
REACTOT_RMSD_CAP = 1.0


def _movable(first: Structure, second: Structure, reference: Structure) -> tuple[np.ndarray, np.ndarray]:
    if first.n_atoms != second.n_atoms or first.n_atoms != reference.n_atoms:
        raise ValueError("structures must have the same atom count")
    movable = np.asarray(reference.movable_mask, dtype=bool)
    if not movable.any():
        movable = np.ones((reference.n_atoms,), dtype=bool)
    a = np.asarray(first.positions, dtype=float)[movable]
    b = np.asarray(second.positions, dtype=float)[movable]
    return a - a.mean(axis=0), b - b.mean(axis=0)


def _fit(first: np.ndarray, second: np.ndarray) -> float:
    """Proper-rotation Kabsch RMSD of two centred coordinate sets."""
    u, _, vt = np.linalg.svd(first.T @ second)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return float(np.sqrt(np.mean(np.sum((first @ rotation.T - second) ** 2, axis=1))))


def kabsch_aligned_rmsd(
    a: Structure,
    b: Structure,
    reference: Structure,
    *,
    mirror_allowed: bool = False,
) -> float:
    """Movable-atom RMSD after optimal rigid alignment.

    ``mirror_allowed=True`` also fits the reflected structure and keeps the lower
    value, which is ReactOT's ``ignore_chirality=True``.
    """
    first, second = _movable(a, b, reference)
    strict = _fit(first, second)
    if not mirror_allowed:
        return strict
    mirrored = second.copy()
    mirrored[:, -1] = -mirrored[:, -1]
    return min(strict, _fit(first, mirrored))


def dual_protocol_rmsd(
    a: Structure,
    b: Structure,
    reference: Structure,
    *,
    cap: float = REACTOT_RMSD_CAP,
) -> dict[str, float]:
    """Both conventions from one alignment.

    Keys: strict (uncapped/capped), reference (mirror allowed, capped/uncapped)
    and ``chirality_gain_angstrom``, the non-negative amount the mirror buys.
    """
    first, second = _movable(a, b, reference)
    strict = _fit(first, second)
    mirrored = second.copy()
    mirrored[:, -1] = -mirrored[:, -1]
    # The reference keeps the lower of the proper and reflected fits.
    reference_value = float(min(strict, _fit(first, mirrored)))
    strict_capped = float(min(strict, cap))
    mirrored_capped = float(min(reference_value, cap))
    return {
        "strict_aligned_rmsd_angstrom": strict,
        "strict_clipped_aligned_rmsd_angstrom": strict_capped,
        "reactot_reference_rmsd_angstrom": mirrored_capped,
        "reactot_reference_uncapped_rmsd_angstrom": reference_value,
        "chirality_gain_angstrom": float(max(strict_capped - mirrored_capped, 0.0)),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    """Mean/median/min/max for a column, with explicit ``None`` for empties."""
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
