"""Bond graphs and the bond-change label.

The bond-change condition tells the model *which bonds break or form*, without
handing it any product coordinate. It is the discrete part of a reaction: the
geometry follows from it, so it carries the information that an R-only model is
otherwise missing (see ``docs/18`` section 11 and ``docs/19`` section 6).

Labels are defined per atom pair:

===  ==========
 0   unchanged
 1   formed  (bond in the product, not in the reactant)
 2   broken  (bond in the reactant, not in the product)
===  ==========

At training time the product graph is supervision; at inference the same labels
must come from the seed as a proposed topology change. Passing a product graph
into a basin-level proposal is leakage and is not what this module is for.
"""
from __future__ import annotations

import numpy as np

UNCHANGED, FORMED, BROKEN = 0, 1, 2
BOND_CHANGE_CLASSES = 3
COVALENT_SCALE = 1.3


def bond_adjacency(
    positions,
    atomic_numbers,
    cell=None,
    pbc=None,
    *,
    scale: float = COVALENT_SCALE,
) -> np.ndarray:
    """Boolean ``[N, N]`` bond graph from covalent radii, obeying minimum image.

    A pair is bonded when its distance is below ``scale`` times the sum of the
    two covalent radii, which is the same criterion MolGEN uses for its fragment
    partition (``prep_data.ipynb``, ``mult=1.50``) and is deliberately stricter
    here so that a bond means a chemical bond rather than a near contact.
    """
    from ase.data import covalent_radii

    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be [N, 3]")
    if numbers.shape != (len(positions),):
        raise ValueError("atomic_numbers must be [N]")

    deltas = positions[:, None, :] - positions[None, :, :]
    if cell is not None and pbc is not None and np.any(pbc):
        cell = np.asarray(cell, dtype=float)
        if cell.shape != (3, 3):
            raise ValueError("cell must be [3, 3]")
        fractional = deltas @ np.linalg.inv(cell)
        for axis, periodic in enumerate(np.asarray(pbc, dtype=bool)):
            if periodic:
                fractional[..., axis] -= np.round(fractional[..., axis])
        deltas = fractional @ cell

    distances = np.linalg.norm(deltas, axis=-1)
    radii = covalent_radii[numbers]
    bonded = distances < scale * (radii[:, None] + radii[None, :])
    np.fill_diagonal(bonded, False)
    return bonded


def fragment_count(bonds: np.ndarray) -> int:
    """Number of connected components of a boolean ``[N, N]`` bond graph."""
    bonds = np.asarray(bonds, dtype=bool)
    if bonds.ndim != 2 or bonds.shape[0] != bonds.shape[1]:
        raise ValueError("bonds must be a square [N, N] array")
    seen = np.zeros(len(bonds), dtype=bool)
    count = 0
    for start in range(len(bonds)):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in np.nonzero(bonds[node] & ~seen)[0]:
                seen[neighbour] = True
                stack.append(int(neighbour))
    return count


def preserves_fragments(reactant_bonds: np.ndarray, product_bonds: np.ndarray) -> bool:
    """Whether a reactant/product pair keeps the same number of bonded fragments.

    Transition1x contains both local rearrangements and dissociations whose
    product is a separated fragment pair (28.9% of the test split at any
    covalent-radius scale between 1.3 and 1.5).  Only the former are events a
    basin-level proposer is asked for, so the benchmark reports them separately.
    """
    return fragment_count(reactant_bonds) == fragment_count(product_bonds)


def bond_change_labels(reactant_bonds: np.ndarray, product_bonds: np.ndarray) -> np.ndarray:
    """Per-pair ``{0, 1, 2}`` label from the reactant and product bond graphs."""
    reactant_bonds = np.asarray(reactant_bonds, dtype=bool)
    product_bonds = np.asarray(product_bonds, dtype=bool)
    if reactant_bonds.shape != product_bonds.shape:
        raise ValueError("reactant and product bond graphs must have the same shape")
    labels = np.full(reactant_bonds.shape, UNCHANGED, dtype=np.int64)
    labels[product_bonds & ~reactant_bonds] = FORMED
    labels[reactant_bonds & ~product_bonds] = BROKEN
    return labels
