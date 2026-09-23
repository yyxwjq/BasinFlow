"""The ReactOT reference protocol must be reproducible from first principles.

These tests pin the *definition* rather than a run: mirror allowed, capped at
1.0 A, reported next to (never instead of) BasinFlow's strict proper-rotation
value. They also guard the arithmetic that a dual-protocol table depends on, so
a silent change of the cap or of the mirror branch cannot slip through.
"""
from __future__ import annotations

import numpy as np
import pytest

from basinflow.data import Structure
from basinflow.evaluation.reactot_metrics import (
    REACTOT_RMSD_CAP,
    dual_protocol_rmsd,
    reactot_reference_rmsd,
    summarize_rmsds,
)
from basinflow.evaluation.basin_recall import kabsch_aligned_rmsd

# A chiral four-point set: no proper rotation maps it onto its z-mirror, which is
# what makes the mirror branch observable.
_CHIRAL = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def _structure(positions: np.ndarray, movable: np.ndarray | None = None) -> Structure:
    positions = np.asarray(positions, dtype=float)
    return Structure(
        species=["C"] * len(positions),
        positions=positions,
        movable_mask=movable,
    )


def _mirror(positions: np.ndarray) -> np.ndarray:
    mirrored = np.array(positions, dtype=float)
    mirrored[:, -1] *= -1.0
    return mirrored


def test_reference_rmsd_is_zero_for_identical_structures() -> None:
    reference = _structure(_CHIRAL)
    assert reactot_reference_rmsd(reference, reference, reference) == pytest.approx(0.0)


def test_mirror_branch_sees_an_inverted_structure_as_a_match() -> None:
    target = _structure(_CHIRAL)
    inverted = _structure(_mirror(_CHIRAL))
    strict = kabsch_aligned_rmsd(inverted, target, target)
    reference = reactot_reference_rmsd(inverted, target, target)
    assert strict > 1e-6, "the chiral set must not be reachable by a proper rotation"
    assert reference == pytest.approx(0.0, abs=1e-6)


def test_cap_saturates_a_large_error() -> None:
    target = _structure(_CHIRAL)
    # A rigid translation would be removed by the Kabsch fit, so the error has to
    # be a genuine internal deformation: move a single atom far away.
    far_positions = _CHIRAL.copy()
    far_positions[3] += np.array([0.0, 0.0, 20.0])
    far = _structure(far_positions)
    assert reactot_reference_rmsd(far, target, target) == pytest.approx(REACTOT_RMSD_CAP)
    uncapped = reactot_reference_rmsd(far, target, target, cap=None)
    assert uncapped > REACTOT_RMSD_CAP


def test_dual_protocol_reports_both_and_a_nonnegative_gain() -> None:
    target = _structure(_CHIRAL)
    inverted = _structure(_mirror(_CHIRAL))
    dual = dual_protocol_rmsd(inverted, target, target)
    assert set(dual) == {
        "strict_aligned_rmsd_angstrom",
        "strict_clipped_aligned_rmsd_angstrom",
        "reactot_reference_rmsd_angstrom",
        "reactot_reference_uncapped_rmsd_angstrom",
        "chirality_gain_angstrom",
    }
    assert dual["strict_aligned_rmsd_angstrom"] > 1e-6
    assert dual["reactot_reference_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-6)
    assert dual["chirality_gain_angstrom"] == pytest.approx(
        dual["strict_clipped_aligned_rmsd_angstrom"] - dual["reactot_reference_rmsd_angstrom"]
    )
    assert dual["chirality_gain_angstrom"] >= 0.0


def test_gain_is_zero_when_the_strict_value_already_matches() -> None:
    target = _structure(_CHIRAL)
    shifted = _CHIRAL + np.array([0.0, 0.0, 0.25])
    dual = dual_protocol_rmsd(_structure(shifted), target, target)
    assert dual["chirality_gain_angstrom"] == pytest.approx(0.0, abs=1e-12)
    assert dual["reactot_reference_rmsd_angstrom"] == pytest.approx(
        dual["strict_aligned_rmsd_angstrom"]
    )


def test_movable_mask_restricts_the_fit() -> None:
    movable = np.array([True, True, False, False])
    target = _structure(_CHIRAL)
    moved_fixed_atom = _CHIRAL.copy()
    moved_fixed_atom[3] += np.array([0.0, 0.0, 9.0])
    assert reactot_reference_rmsd(
        _structure(moved_fixed_atom), target, _structure(_CHIRAL, movable)
    ) == pytest.approx(0.0)


def test_atom_count_mismatch_is_rejected() -> None:
    reference = _structure(_CHIRAL)
    shorter = _structure(_CHIRAL[:3])
    with pytest.raises(ValueError):
        reactot_reference_rmsd(shorter, reference, reference)


def test_summarize_rmsds_handles_empty_and_filled_inputs() -> None:
    assert summarize_rmsds([]) == {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    summary = summarize_rmsds([1.0, 0.0, 2.0])
    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["median"] == pytest.approx(1.0)
    assert summary["min"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(2.0)
