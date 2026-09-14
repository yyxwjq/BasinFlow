from __future__ import annotations

from typing import Iterable

import numpy as np


def normalize_cell(cell: Iterable[Iterable[float]] | None) -> np.ndarray:
    """Return a normalized 3x3 cell matrix.

    Cell vectors follow ASE's row-vector convention. A missing cell is allowed
    for non-periodic records and is normalized to a zero 3x3 matrix.
    """
    if cell is None:
        return np.zeros((3, 3), dtype=float)
    array = np.asarray(cell, dtype=float)
    if array.shape == (3,):
        array = np.diag(array)
    if array.shape != (3, 3):
        raise ValueError(f"cell must have shape (3, 3) or (3,), got {array.shape}")
    return array


def normalize_pbc(pbc: bool | Iterable[bool] | None) -> np.ndarray:
    """Return a length-3 boolean PBC vector."""
    if pbc is None:
        return np.array([False, False, False], dtype=bool)
    if isinstance(pbc, (bool, np.bool_)):
        return np.array([bool(pbc)] * 3, dtype=bool)
    array = np.asarray(pbc, dtype=bool)
    if array.shape != (3,):
        raise ValueError(f"pbc must be a bool or length-3 sequence, got {array.shape}")
    return array


def minimum_image_displacement(
    product_position: Iterable[float] | np.ndarray,
    reactant_position: Iterable[float] | np.ndarray,
    cell: Iterable[Iterable[float]] | None,
    pbc: bool | Iterable[bool] | None,
) -> np.ndarray:
    """Return product minus reactant using MIC only along periodic axes."""
    displacement = np.asarray(product_position, dtype=float) - np.asarray(
        reactant_position, dtype=float
    )
    pbc_array = normalize_pbc(pbc)
    if not np.any(pbc_array):
        return displacement

    cell_array = normalize_cell(cell)
    if np.linalg.matrix_rank(cell_array) < 3:
        raise ValueError("a full-rank cell is required when any pbc flag is true")

    fractional = displacement @ np.linalg.inv(cell_array)
    fractional[..., pbc_array] -= np.round(fractional[..., pbc_array])
    return fractional @ cell_array

def pairwise_displacements(
    product_positions: Iterable[Iterable[float]] | np.ndarray,
    reactant_positions: Iterable[Iterable[float]] | np.ndarray,
    cell: Iterable[Iterable[float]] | None,
    pbc: bool | Iterable[bool] | None,
) -> np.ndarray:
    """Return atom-wise product minus reactant displacements."""
    product = np.asarray(product_positions, dtype=float)
    reactant = np.asarray(reactant_positions, dtype=float)
    if product.shape != reactant.shape:
        raise ValueError(
            f"product and reactant positions must have the same shape, got {product.shape} and {reactant.shape}"
        )
    if product.ndim != 2 or product.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {product.shape}")
    return minimum_image_displacement(product, reactant, cell=cell, pbc=pbc)


def displacement_norms(displacements: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    """Return Euclidean norm per displacement vector."""
    array = np.asarray(displacements, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"displacements must have shape (N, 3), got {array.shape}")
    return np.linalg.norm(array, axis=1)


def derive_active_atoms(
    displacements: Iterable[Iterable[float]] | np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Derive active atoms from displacement norms."""
    return displacement_norms(displacements) > float(threshold)


def derive_event_direction(
    displacements: Iterable[Iterable[float]] | np.ndarray,
    active_mask: Iterable[bool] | np.ndarray | None = None,
    threshold: float = 0.1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return unit displacement directions for active atoms and zero elsewhere."""
    array = np.asarray(displacements, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"displacements must have shape (N, 3), got {array.shape}")
    active = derive_active_atoms(array, threshold=threshold) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active.shape != (array.shape[0],):
        raise ValueError(f"active_mask must have shape ({array.shape[0]},), got {active.shape}")

    direction = np.zeros_like(array, dtype=float)
    norms = displacement_norms(array)
    valid = active & (norms > eps)
    direction[valid] = array[valid] / norms[valid, None]
    return direction
