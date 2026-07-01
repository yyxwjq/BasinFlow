from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from fscgp.data.records import StructureRecord


def _concat_field(
    items: list[dict[str, Any]],
    key: str,
    dtype: type,
    ndim: int = 1,
) -> np.ndarray:
    """Concatenate per-item arrays into a single batched array.

    Returns a zero-shaped array when ``items`` is empty.
    """
    if not items:
        shape = (0, 3) if ndim == 2 else (0,)
        return np.zeros(shape, dtype=dtype)
    return np.concatenate([item[key] for item in items], axis=0)


def _concat_structure_field(
    records: Sequence[StructureRecord],
    field: str,
    ndim: int = 2,
) -> np.ndarray:
    """Concatenate a named field from a sequence of `StructureRecord` objects."""
    arrays = [np.asarray(getattr(r, field)) for r in records]
    if not arrays:
        shape = (0, 3) if ndim == 2 else (0,)
        return np.zeros(shape, dtype=arrays[0].dtype if arrays else float)
    return np.concatenate(arrays, axis=0)


def _batch_index(records: Sequence[StructureRecord]) -> np.ndarray:
    if not records:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(
        [np.full(r.n_atoms, i, dtype=np.int64) for i, r in enumerate(records)]
    )


def _ptr(records: Sequence[StructureRecord]) -> np.ndarray:
    counts = [record.n_atoms for record in records]
    return np.asarray([0, *np.cumsum(counts, dtype=np.int64)], dtype=np.int64)


def collate_pairwise(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-size pairwise event items without dropping metadata."""
    reactants = [item["reactant"] for item in items]
    products = [item["product"] for item in items]
    return {
        "event_ids": [item["event_id"] for item in items],
        "basin_ids": [item["basin_id"] for item in items],
        "events": [item["event"] for item in items],
        "reactants": reactants,
        "products": products,
        # --- atom-level concatenated fields ---
        "reactant_positions": _concat_structure_field(reactants, "positions"),
        "product_positions": _concat_structure_field(products, "positions"),
        "displacements": _concat_field(items, "displacement", float, ndim=2),
        "ts_displacements": _concat_field(items, "ts_displacement", float, ndim=2),
        "active_mask": _concat_field(items, "active_mask", bool),
        "movable_mask": _concat_field(items, "movable_mask", bool),
        "event_direction": _concat_field(items, "event_direction", float, ndim=2),
        # --- per-sample fields ---
        "has_transition_state": np.asarray(
            [item["has_transition_state"] for item in items], dtype=bool
        ),
        # --- structure-level fields ---
        "reactant_batch": _batch_index(reactants),
        "product_batch": _batch_index(products),
        "reactant_ptr": _ptr(reactants),
        "product_ptr": _ptr(products),
        "cells": np.stack([record.cell for record in reactants], axis=0)
        if reactants
        else np.zeros((0, 3, 3), dtype=float),
        "pbc": np.stack([record.pbc for record in reactants], axis=0)
        if reactants
        else np.zeros((0, 3), dtype=bool),
        "atomic_numbers": _concat_structure_field(reactants, "atomic_numbers", ndim=1),
        "atom_mappings": [item["atom_mapping"] for item in items],
        "metadata": [item["metadata"] for item in items],
    }


def pairwise_batch_to_pyg_data(batch: dict[str, Any]):
    """Convert a collated pairwise batch to a PyG `Data` object.

    PyTorch and PyTorch Geometric are optional model dependencies.  The core
    data package remains importable without them; this bridge imports both only
    when called.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "pairwise_batch_to_pyg_data requires optional model dependencies: "
            "install torch and torch-geometric"
        ) from exc

    n_atoms = int(batch["reactant_positions"].shape[0])
    edge_index = batch.get("edge_index")
    if edge_index is None:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    cell_offsets = batch.get("cell_offsets")
    if cell_offsets is None:
        cell_offsets = np.zeros((0, 3), dtype=np.int64)

    return Data(
        atomic_numbers=torch.as_tensor(batch["atomic_numbers"], dtype=torch.long),
        pos=torch.as_tensor(batch["reactant_positions"], dtype=torch.float32),
        product_pos=torch.as_tensor(batch["product_positions"], dtype=torch.float32),
        displacement=torch.as_tensor(batch["displacements"], dtype=torch.float32),
        ts_displacement=torch.as_tensor(batch["ts_displacements"], dtype=torch.float32),
        active_mask=torch.as_tensor(batch["active_mask"], dtype=torch.bool),
        movable_mask=torch.as_tensor(batch["movable_mask"], dtype=torch.bool),
        batch=torch.as_tensor(batch["reactant_batch"], dtype=torch.long),
        ptr=torch.as_tensor(batch["reactant_ptr"], dtype=torch.long),
        cell=torch.as_tensor(batch["cells"], dtype=torch.float32),
        pbc=torch.as_tensor(batch["pbc"], dtype=torch.bool),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        cell_offsets=torch.as_tensor(cell_offsets, dtype=torch.long),
        num_nodes=n_atoms,
    )


def collate_basins(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate basin-level items with variable numbers of known events."""
    reactants = [item["reactant"] for item in items]
    return {
        "basin_ids": [item["basin_id"] for item in items],
        "basins": [item["basin"] for item in items],
        "reactants": reactants,
        "events": [item["events"] for item in items],
        "products": [item["products"] for item in items],
        "transition_states": [item["transition_states"] for item in items],
        "known_event_ids": [list(item["known_event_ids"]) for item in items],
        # --- atom-level concatenated fields ---
        "reactant_positions": _concat_structure_field(reactants, "positions"),
        "reactant_batch": _batch_index(reactants),
        # --- structure-level fields ---
        "cells": np.stack([record.cell for record in reactants], axis=0)
        if reactants
        else np.zeros((0, 3, 3), dtype=float),
        "pbc": np.stack([record.pbc for record in reactants], axis=0)
        if reactants
        else np.zeros((0, 3), dtype=bool),
        "atomic_numbers": _concat_structure_field(reactants, "atomic_numbers", ndim=1),
        "metadata": [item["metadata"] for item in items],
    }
