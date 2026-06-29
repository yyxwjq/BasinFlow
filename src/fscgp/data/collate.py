from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from fscgp.data.records import StructureRecord


def _concat_structure_field(
    records: Sequence[StructureRecord], field: str
) -> np.ndarray:
    arrays = [np.asarray(getattr(r, field)) for r in records]
    if not arrays:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate(arrays, axis=0)


def _batch_index(records: Sequence[StructureRecord]) -> np.ndarray:
    if not records:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(
        [
            np.full(r.n_atoms, i, dtype=np.int64)
            for i, r in enumerate(records)
        ]
    )


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
        "reactant_positions": _concat_structure_field(reactants, "positions"),
        "product_positions": _concat_structure_field(products, "positions"),
        "displacements": np.concatenate([item["displacement"] for item in items], axis=0)
        if items
        else np.zeros((0, 3), dtype=float),
        "active_mask": np.concatenate([item["active_mask"] for item in items], axis=0)
        if items
        else np.zeros((0,), dtype=bool),
        "event_direction": np.concatenate([item["event_direction"] for item in items], axis=0)
        if items
        else np.zeros((0, 3), dtype=float),
        "reactant_batch": _batch_index(reactants),
        "product_batch": _batch_index(products),
        "cells": np.stack([record.cell for record in reactants], axis=0)
        if reactants
        else np.zeros((0, 3, 3), dtype=float),
        "pbc": np.stack([record.pbc for record in reactants], axis=0)
        if reactants
        else np.zeros((0, 3), dtype=bool),
        "atomic_numbers": np.concatenate([record.atomic_numbers for record in reactants], axis=0)
        if reactants
        else np.zeros((0,), dtype=np.int64),
        "atom_mappings": [item["atom_mapping"] for item in items],
        "metadata": [item["metadata"] for item in items],
    }


def collate_basins(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate basin-level items with variable numbers of known events."""
    reactants = [item["reactant"] for item in items]
    return {
        "basin_ids": [item["basin_id"] for item in items],
        "basins": [item["basin"] for item in items],
        "reactants": reactants,
        "events": [item["events"] for item in items],
        "products": [item["products"] for item in items],
        "known_event_ids": [list(item["known_event_ids"]) for item in items],
        "reactant_positions": _concat_structure_field(reactants, "positions"),
        "reactant_batch": _batch_index(reactants),
        "cells": np.stack([record.cell for record in reactants], axis=0)
        if reactants
        else np.zeros((0, 3, 3), dtype=float),
        "pbc": np.stack([record.pbc for record in reactants], axis=0)
        if reactants
        else np.zeros((0, 3), dtype=bool),
        "atomic_numbers": np.concatenate([record.atomic_numbers for record in reactants], axis=0)
        if reactants
        else np.zeros((0,), dtype=np.int64),
        "metadata": [item["metadata"] for item in items],
    }
