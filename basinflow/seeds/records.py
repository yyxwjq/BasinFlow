from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from basinflow.data.records import StructureRecord


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
    bond_change: np.ndarray | None = None

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
        if self.bond_change is not None:
            labels = np.asarray(self.bond_change)
            if labels.shape != (n_atoms, n_atoms):
                raise ValueError(
                    f"bond_change must have shape ({n_atoms}, {n_atoms}), "
                    f"got {labels.shape}"
                )
            if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() > 2:
                raise ValueError("bond_change labels must be integers in {0, 1, 2}")
            self.bond_change = labels.astype(np.int8, copy=False)
        if np.any(self.bond_change):
            # A bond can only change if at least one of its two atoms moves, so
            # a pair of fixed atoms can never carry a label.
            frozen = ~self.movable_mask
            if np.any(self.bond_change[np.ix_(frozen, frozen)]):
                raise ValueError("a pair of fixed atoms cannot change its bond")

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


@dataclass(frozen=True)
class SeedContext:
    """Reactant-only information an ordinary initializer may consume."""

    event_id: str
    basin_id: str
    reactant: StructureRecord

    @property
    def movable_mask(self) -> np.ndarray:
        return np.asarray(self.reactant.movable_mask, dtype=bool)


def _directions_from_displacement(displacement: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    direction = np.zeros_like(displacement, dtype=float)
    norms = np.linalg.norm(displacement, axis=1)
    valid = norms > eps
    direction[valid] = displacement[valid] / norms[valid, None]
    return direction


def event_seed_from_context(
    context: SeedContext,
    seed_id: str,
    seed_type: str,
    seed_displacement,
    metadata: dict[str, Any] | None = None,
    bond_change=None,
) -> EventSeed:
    displacement = np.asarray(seed_displacement, dtype=float).copy()
    movable_mask = context.movable_mask
    fixed_mask = ~movable_mask
    displacement[fixed_mask] = 0.0
    direction = _directions_from_displacement(displacement)
    direction[fixed_mask] = 0.0
    active_prior = movable_mask & (
        np.linalg.norm(displacement, axis=1) > 1e-12
    )
    seed_metadata = dict(metadata or {})
    seed_metadata.setdefault("event_id", context.event_id)
    seed_metadata.setdefault("basin_id", context.basin_id)
    return EventSeed(
        seed_id=seed_id,
        seed_type=seed_type,
        seed_displacement=displacement,
        seed_direction=direction,
        active_prior=active_prior,
        movable_mask=movable_mask,
        metadata=seed_metadata,
        bond_change=bond_change,
    )

