from __future__ import annotations

from typing import Any

import numpy as np

from fscgp.seeds import EventSeed


def build_product_flow_item(
    pairwise_item: dict[str, Any],
    event_seed: EventSeed,
) -> dict[str, Any]:
    """Build a seed-conditioned product-flow training item."""
    reactant = pairwise_item["reactant"]
    reactant_positions = np.asarray(reactant.positions, dtype=float)
    if event_seed.n_atoms != reactant.n_atoms:
        raise ValueError(
            f"event_seed atom count {event_seed.n_atoms} does not match "
            f"reactant atom count {reactant.n_atoms}"
        )
    target_displacement = np.asarray(pairwise_item["displacement"], dtype=float).copy()
    seed_displacement = np.asarray(event_seed.seed_displacement, dtype=float).copy()
    movable_mask = np.asarray(pairwise_item["movable_mask"], dtype=bool)
    fixed_mask = ~movable_mask
    if not np.array_equal(event_seed.movable_mask, movable_mask):
        raise ValueError("event_seed mask contract does not match pairwise item")
    initial_positions = event_seed.initial_positions(reactant_positions)
    target_positions = reactant_positions + target_displacement
    target_velocity = target_displacement - seed_displacement
    target_velocity[fixed_mask] = 0.0

    metadata = dict(pairwise_item.get("metadata", {}))
    metadata["seed_id"] = event_seed.seed_id
    metadata["seed_type"] = event_seed.seed_type
    return {
        "event_id": pairwise_item["event_id"],
        "basin_id": pairwise_item["basin_id"],
        "reactant_positions": reactant_positions,
        "initial_positions": initial_positions,
        "product_positions": np.asarray(pairwise_item["product"].positions, dtype=float),
        "target_positions": target_positions,
        "seed_displacement": seed_displacement,
        "target_displacement": target_displacement,
        "target_velocity": target_velocity,
        "active_mask": np.asarray(pairwise_item["active_mask"], dtype=bool),
        "movable_mask": movable_mask,
        "event_direction": np.asarray(pairwise_item["event_direction"], dtype=float),
        "cell": np.asarray(reactant.cell, dtype=float),
        "pbc": np.asarray(reactant.pbc, dtype=bool),
        "metadata": metadata,
    }
