from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from basinflow.seeds.records import EventSeed, SeedContext, event_seed_from_context


def _metadata(context: SeedContext, init_source: str) -> dict:
    return {
        "event_id": context.event_id,
        "basin_id": context.basin_id,
        "init_source": init_source,
    }


def _rng_for_item(random_seed: int | None, event_id: object, seed_id: str | None) -> np.random.Generator:
    if random_seed is None:
        return np.random.default_rng()
    key = f"{random_seed}:{event_id}:{seed_id or ''}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(derived_seed)


@dataclass(frozen=True)
class ZeroInit:
    seed_type: str = "zero"

    def generate(self, context: SeedContext, seed_id: str | None = None) -> EventSeed:
        reactant = context.reactant
        zeros = np.zeros((reactant.n_atoms, 3), dtype=float)
        return EventSeed(
            seed_id=seed_id or f"{context.event_id}:zero",
            seed_type=self.seed_type,
            seed_displacement=zeros,
            seed_direction=zeros,
            active_prior=np.zeros((reactant.n_atoms,), dtype=bool),
            movable_mask=context.movable_mask,
            metadata=_metadata(context, self.seed_type),
        )


@dataclass(frozen=True)
class GaussianInit:
    scale: float = 0.1
    random_seed: int | None = None
    seed_type: str = "gaussian_movable"

    def generate(self, context: SeedContext, seed_id: str | None = None) -> EventSeed:
        reactant = context.reactant
        movable_mask = context.movable_mask
        fixed_mask = ~movable_mask
        rng = _rng_for_item(self.random_seed, context.event_id, seed_id)
        displacement = np.zeros((reactant.n_atoms, 3), dtype=float)
        displacement[movable_mask] = rng.normal(
            loc=0.0,
            scale=float(self.scale),
            size=(int(movable_mask.sum()), 3),
        )
        direction = np.zeros_like(displacement)
        norms = np.linalg.norm(displacement, axis=1)
        valid = movable_mask & (norms > 1e-12)
        direction[valid] = displacement[valid] / norms[valid, None]
        displacement[fixed_mask] = 0.0
        direction[fixed_mask] = 0.0
        active_prior = movable_mask & (norms > 1e-12)
        return EventSeed(
            seed_id=seed_id or f"{context.event_id}:gaussian:{self.random_seed}",
            seed_type=self.seed_type,
            seed_displacement=displacement,
            seed_direction=direction,
            active_prior=active_prior,
            movable_mask=movable_mask,
            metadata={
                **_metadata(context, self.seed_type),
                "scale": float(self.scale),
                "random_seed": self.random_seed,
            },
        )


@dataclass(frozen=True)
class DirectionalInit:
    direction: tuple[float, float, float]
    scale: float = 0.1
    movable_rank: int = 0
    seed_type: str = "directional"

    def generate(self, context: SeedContext, seed_id: str | None = None) -> EventSeed:
        reactant = context.reactant
        movable_mask = context.movable_mask
        movable_indices = np.where(movable_mask)[0]
        displacement = np.zeros((reactant.n_atoms, 3), dtype=float)
        seed_direction = np.zeros_like(displacement)
        active_prior = np.zeros(reactant.n_atoms, dtype=bool)

        direction = np.asarray(self.direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("DirectionalInit direction must be non-zero")
        unit_direction = direction / norm

        selected_atom = None
        if movable_indices.size:
            selected_atom = int(movable_indices[int(self.movable_rank) % movable_indices.size])
            displacement[selected_atom] = unit_direction * float(self.scale)
            seed_direction[selected_atom] = unit_direction
            active_prior[selected_atom] = True

        direction_list = [float(x) for x in unit_direction.tolist()]
        return EventSeed(
            seed_id=seed_id or f"{context.event_id}:directional:{self.movable_rank}",
            seed_type=self.seed_type,
            seed_displacement=displacement,
            seed_direction=seed_direction,
            active_prior=active_prior,
            movable_mask=movable_mask,
            metadata={
                **_metadata(context, self.seed_type),
                "direction": direction_list,
                "scale": float(self.scale),
                "movable_rank": int(self.movable_rank),
                "selected_atom": selected_atom,
            },
        )


@dataclass(frozen=True)
class ProductInit:
    seed_type: str = "product_displacement"
    requires_target: bool = True

    def generate(
        self,
        context: SeedContext,
        seed_id: str | None = None,
        *,
        target_displacement=None,
    ) -> EventSeed:
        if target_displacement is None:
            raise ValueError("ProductInit requires target_displacement for diagnostics")
        return event_seed_from_context(
            context,
            seed_id=seed_id or f"{context.event_id}:product_displacement",
            seed_type=self.seed_type,
            seed_displacement=target_displacement,
            metadata=_metadata(context, self.seed_type),
        )
