from __future__ import annotations

from typing import Any

import numpy as np
from ase.neighborlist import neighbor_list

from fscgp.data.records import StructureRecord


def build_neighbor_graph(
    structure: StructureRecord,
    cutoff: float,
    max_neighbors: int | None = None,
) -> dict[str, Any]:
    """Build a cutoff graph for molecular or periodic structures.

    Returns NumPy arrays with PyTorch-Geometric-style `edge_index` and explicit
    integer cell offsets. ASE controls periodic image search through the record's
    normalized `pbc` flags.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    atoms = structure.to_ase()
    atoms.set_pbc(structure.pbc)
    src, dst, offsets, distances = neighbor_list(
        "ijSd",
        atoms,
        cutoff=float(cutoff),
        self_interaction=False,
    )
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    distances = np.asarray(distances, dtype=float)

    if max_neighbors is not None and max_neighbors > 0 and src.size:
        keep_parts = []
        for atom_index in np.unique(src):
            edge_ids = np.where(src == atom_index)[0]
            if edge_ids.size > max_neighbors:
                edge_ids = edge_ids[np.argsort(distances[edge_ids])[:max_neighbors]]
            keep_parts.append(edge_ids)
        keep = np.sort(np.concatenate(keep_parts)) if keep_parts else np.array([], dtype=int)
        src, dst, offsets, distances = src[keep], dst[keep], offsets[keep], distances[keep]

    edge_index = np.vstack([src, dst]).astype(np.int64) if src.size else np.zeros((2, 0), dtype=np.int64)
    cell_offset_vectors = offsets @ structure.cell
    edge_vectors = structure.positions[dst] - structure.positions[src] + cell_offset_vectors
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    return {
        "structure_id": structure.structure_id,
        "edge_index": edge_index,
        "edge_vectors": edge_vectors.astype(float),
        "edge_lengths": edge_lengths.astype(float),
        "cell_offsets": offsets.astype(np.int64),
        "atomic_numbers": structure.atomic_numbers,
        "species": list(structure.species),
        "positions": structure.positions.copy(),
        "cell": structure.cell.copy(),
        "pbc": structure.pbc.copy(),
        "n_atoms": structure.n_atoms,
    }
