from __future__ import annotations

import numpy as np


def get_displacements(
    positions1: np.ndarray,
    positions2: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """
    Return positions1 minus positions2 with MIC along the periodic axes, for one
    (3,) vector or a batch of (N, 3) positions.
    """
    if positions1.shape != positions2.shape:
        raise ValueError(
            f"positions1 and positions2 must have the same shape, got {positions1.shape} and {positions2.shape}"
        )

    displacement = positions1 - positions2
    if not np.any(pbc):
        return displacement

    if np.linalg.matrix_rank(cell) < 3:
        raise ValueError("a 3D cell is required when any pbc flag is true")

    fractional = displacement @ np.linalg.inv(cell)
    fractional[..., pbc] -= np.round(fractional[..., pbc])
    return fractional @ cell
