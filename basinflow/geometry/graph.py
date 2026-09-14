"""Neighbor-graph construction at the structure level (NumPy + ASE).

This module owns everything that turns a *structure* into a graph:

- ``build_neighbor_graph`` builds the cutoff graph for one
  :class:`~basinflow.data.records.StructureRecord`, delegating periodic image
  search to ASE. This is the reference implementation for PBC correctness.
- ``torch_neighbor_graph_from_structure`` and
  ``torch_neighbor_graph_from_batch`` are thin adapters that convert those NumPy
  arrays into torch tensors. They import torch lazily, because torch is an
  optional dependency (see the ``models`` extra in ``pyproject.toml``).

Tensor-level graphs that never touch a ``StructureRecord`` -- the radius graphs
the PaiNN backbone rebuilds on every integration step -- live in
:mod:`basinflow.geometry.radius_graph` instead, so that importing this module
never requires torch.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from ase.neighborlist import neighbor_list

from basinflow.data.records import StructureRecord


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


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "torch graph utilities require optional model dependency: install torch"
        ) from exc
    return torch


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def torch_neighbor_graph_from_structure(
    structure: StructureRecord,
    cutoff: float,
    *,
    device=None,
    dtype=None,
    max_neighbors: int | None = None,
) -> dict[str, Any]:
    """Build a torch graph from a `StructureRecord`.

    The edge vectors come from `build_neighbor_graph`, so periodic images and
    cell offsets are handled by the same ASE-backed path used by core geometry.
    """
    torch = _torch()
    float_dtype = dtype or torch.float32
    graph = build_neighbor_graph(
        structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )
    return {
        "edge_index": torch.as_tensor(
            graph["edge_index"],
            dtype=torch.long,
            device=device,
        ),
        "edge_vectors": torch.as_tensor(
            graph["edge_vectors"],
            dtype=float_dtype,
            device=device,
        ),
        "edge_lengths": torch.as_tensor(
            graph["edge_lengths"],
            dtype=float_dtype,
            device=device,
        ),
        "cell_offsets": torch.as_tensor(
            graph["cell_offsets"],
            dtype=torch.long,
            device=device,
        ),
    }


def _batch_field(batch, name: str):
    if isinstance(batch, dict):
        return batch[name]
    return getattr(batch, name)


def torch_neighbor_graph_from_batch(
    batch,
    cutoff: float,
    *,
    positions_key: str = "pos",
    max_neighbors: int | None = None,
) -> dict[str, Any]:
    """Build a batched torch neighbor graph from a tensor batch dict."""
    torch = _torch()
    if isinstance(batch, dict) and positions_key not in batch and positions_key == "pos":
        positions_key = "reactant_positions"
    positions = _batch_field(batch, positions_key)
    device = positions.device if hasattr(positions, "device") else None
    dtype = positions.dtype if hasattr(positions, "dtype") else torch.float32

    pos_np = _to_numpy(positions).astype(float)
    batch_index = _to_numpy(_batch_field(batch, "batch")).astype(np.int64)
    atomic_numbers = _to_numpy(
        _batch_field(batch, "z" if not isinstance(batch, dict) else ("atomic_numbers" if "atomic_numbers" in batch else "z"))
    ).astype(int)
    cells = _to_numpy(_batch_field(batch, "cell")).astype(float)
    pbc = _to_numpy(_batch_field(batch, "pbc")).astype(bool)

    from ase.data import chemical_symbols

    edge_indices = []
    edge_vectors = []
    edge_lengths = []
    cell_offsets = []

    for sample_id in sorted(np.unique(batch_index).tolist()):
        atom_ids = np.where(batch_index == sample_id)[0]
        if atom_ids.size == 0:
            continue
        local_positions = pos_np[atom_ids]
        species = [chemical_symbols[int(z)] for z in atomic_numbers[atom_ids]]
        structure = StructureRecord(
            structure_id=f"batch:{sample_id}",
            species=species,
            positions=local_positions,
            cell=cells[int(sample_id)],
            pbc=pbc[int(sample_id)],
        )
        graph = build_neighbor_graph(
            structure,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        )
        if graph["edge_index"].size == 0:
            continue
        edge_indices.append(graph["edge_index"] + int(atom_ids[0]))
        edge_vectors.append(graph["edge_vectors"])
        edge_lengths.append(graph["edge_lengths"])
        cell_offsets.append(graph["cell_offsets"])

    if edge_indices:
        edge_index_np = np.concatenate(edge_indices, axis=1)
        edge_vectors_np = np.concatenate(edge_vectors, axis=0)
        edge_lengths_np = np.concatenate(edge_lengths, axis=0)
        cell_offsets_np = np.concatenate(cell_offsets, axis=0)
    else:
        edge_index_np = np.zeros((2, 0), dtype=np.int64)
        edge_vectors_np = np.zeros((0, 3), dtype=float)
        edge_lengths_np = np.zeros((0,), dtype=float)
        cell_offsets_np = np.zeros((0, 3), dtype=np.int64)

    return {
        "edge_index": torch.as_tensor(edge_index_np, dtype=torch.long, device=device),
        "edge_vectors": torch.as_tensor(edge_vectors_np, dtype=dtype, device=device),
        "edge_lengths": torch.as_tensor(edge_lengths_np, dtype=dtype, device=device),
        "cell_offsets": torch.as_tensor(cell_offsets_np, dtype=torch.long, device=device),
    }
