from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from fscgp.seeds.records import EventSeed, event_seed_from_pairwise_item


def _metadata(pairwise_item: dict, seed_source: str) -> dict:
    return {
        "event_id": pairwise_item.get("event_id"),
        "basin_id": pairwise_item.get("basin_id"),
        "seed_source": seed_source,
    }


def _rng_for_item(random_seed: int | None, event_id: object, seed_id: str | None) -> np.random.Generator:
    if random_seed is None:
        return np.random.default_rng()
    key = f"{random_seed}:{event_id}:{seed_id or ''}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(derived_seed)


@dataclass(frozen=True)
class ZeroSeedGenerator:
    seed_type: str = "zero"

    def generate(self, pairwise_item: dict, seed_id: str | None = None) -> EventSeed:
        reactant = pairwise_item["reactant"]
        zeros = np.zeros((reactant.n_atoms, 3), dtype=float)
        return EventSeed(
            seed_id=seed_id or f"{pairwise_item['event_id']}:zero",
            seed_type=self.seed_type,
            seed_displacement=zeros,
            seed_direction=zeros,
            active_prior=np.asarray(pairwise_item["active_mask"], dtype=bool),
            movable_mask=np.asarray(pairwise_item["movable_mask"], dtype=bool),
            metadata=_metadata(pairwise_item, self.seed_type),
        )


@dataclass(frozen=True)
class GaussianMovableSeedGenerator:
    scale: float = 0.1
    random_seed: int | None = None
    seed_type: str = "gaussian_movable"

    def generate(self, pairwise_item: dict, seed_id: str | None = None) -> EventSeed:
        reactant = pairwise_item["reactant"]
        movable_mask = np.asarray(pairwise_item["movable_mask"], dtype=bool)
        fixed_mask = ~movable_mask
        rng = _rng_for_item(self.random_seed, pairwise_item.get("event_id"), seed_id)
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
        return EventSeed(
            seed_id=seed_id or f"{pairwise_item['event_id']}:gaussian:{self.random_seed}",
            seed_type=self.seed_type,
            seed_displacement=displacement,
            seed_direction=direction,
            active_prior=np.asarray(pairwise_item["active_mask"], dtype=bool),
            movable_mask=movable_mask,
            metadata={
                **_metadata(pairwise_item, self.seed_type),
                "scale": float(self.scale),
                "random_seed": self.random_seed,
            },
        )


@dataclass(frozen=True)
class ProductDisplacementSeedGenerator:
    seed_type: str = "product_displacement"

    def generate(self, pairwise_item: dict, seed_id: str | None = None) -> EventSeed:
        return event_seed_from_pairwise_item(
            pairwise_item,
            seed_id=seed_id or f"{pairwise_item['event_id']}:product_displacement",
            seed_type=self.seed_type,
            seed_displacement=pairwise_item["displacement"],
            metadata=_metadata(pairwise_item, self.seed_type),
        )
