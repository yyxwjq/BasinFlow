from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_vector_array(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    return array


def _as_mask_array(name: str, value, n_atoms: int) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != (n_atoms,):
        raise ValueError(f"{name} must have shape ({n_atoms},), got {array.shape}")
    return array


@dataclass
class EventSeed:
    """Physical seed for seed-conditioned event proposal.

    The seed is represented as scalar conditions, vector conditions, and
    an initial geometry displacement.  Fixed atoms must not receive seed
    motion.
    """

    seed_id: str
    seed_type: str
    seed_displacement: np.ndarray
    seed_direction: np.ndarray
    active_prior: np.ndarray
    movable_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.seed_id = str(self.seed_id)
        self.seed_type = str(self.seed_type)
        self.seed_displacement = _as_vector_array(
            "seed_displacement",
            self.seed_displacement,
        )
        n_atoms = self.seed_displacement.shape[0]
        self.seed_direction = _as_vector_array("seed_direction", self.seed_direction)
        if self.seed_direction.shape[0] != n_atoms:
            raise ValueError(
                f"seed_direction must have shape ({n_atoms}, 3), "
                f"got {self.seed_direction.shape}"
            )
        self.active_prior = _as_mask_array("active_prior", self.active_prior, n_atoms)
        self.movable_mask = _as_mask_array("movable_mask", self.movable_mask, n_atoms)
        fixed_mask = ~self.movable_mask
        if np.any(np.abs(self.seed_displacement[fixed_mask]) > 1e-12):
            raise ValueError("fixed atoms must have zero seed_displacement")
        if np.any(np.abs(self.seed_direction[fixed_mask]) > 1e-12):
            raise ValueError("fixed atoms must have zero seed_direction")
        self.metadata = dict(self.metadata or {})

    @property
    def n_atoms(self) -> int:
        return int(self.seed_displacement.shape[0])

    def initial_positions(self, reactant_positions) -> np.ndarray:
        reactant = np.asarray(reactant_positions, dtype=float)
        if reactant.shape != self.seed_displacement.shape:
            raise ValueError(
                "reactant_positions must match seed_displacement shape, "
                f"got {reactant.shape} and {self.seed_displacement.shape}"
            )
        return reactant + self.seed_displacement


def _directions_from_displacement(displacement: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    direction = np.zeros_like(displacement, dtype=float)
    norms = np.linalg.norm(displacement, axis=1)
    valid = norms > eps
    direction[valid] = displacement[valid] / norms[valid, None]
    return direction


def event_seed_from_pairwise_item(
    pairwise_item: dict[str, Any],
    seed_id: str,
    seed_type: str,
    seed_displacement,
    metadata: dict[str, Any] | None = None,
) -> EventSeed:
    displacement = np.asarray(seed_displacement, dtype=float).copy()
    movable_mask = np.asarray(pairwise_item["movable_mask"], dtype=bool)
    fixed_mask = ~movable_mask
    displacement[fixed_mask] = 0.0
    direction = _directions_from_displacement(displacement)
    direction[fixed_mask] = 0.0
    seed_metadata = dict(metadata or {})
    seed_metadata.setdefault("event_id", pairwise_item.get("event_id"))
    seed_metadata.setdefault("basin_id", pairwise_item.get("basin_id"))
    return EventSeed(
        seed_id=seed_id,
        seed_type=seed_type,
        seed_displacement=displacement,
        seed_direction=direction,
        active_prior=np.asarray(pairwise_item["active_mask"], dtype=bool),
        movable_mask=movable_mask,
        metadata=seed_metadata,
    )
