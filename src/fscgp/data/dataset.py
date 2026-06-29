from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fscgp.data.records import BasinRecord, EventRecord, StructureRecord
from fscgp.geometry.mic import (
    derive_active_atoms,
    derive_event_direction,
    pairwise_displacements,
)


@dataclass
class EventDataset:
    """Minimal in-memory dataset exposing pairwise and basin-level views."""

    structures: dict[str, StructureRecord]
    events: dict[str, EventRecord]
    basins: dict[str, BasinRecord]

    def __post_init__(self) -> None:
        self.structures = dict(self.structures)
        self.events = dict(self.events)
        self.basins = dict(self.basins)

    @property
    def event_ids(self) -> list[str]:
        return list(self.events.keys())

    @property
    def basin_ids(self) -> list[str]:
        return list(self.basins.keys())

    def iter_pairwise(self, active_threshold: float = 0.1) -> Iterable[dict]:
        for event_id in self.event_ids:
            yield self.pairwise_item(event_id, active_threshold=active_threshold)

    def iter_basins(self) -> Iterable[dict]:
        for basin_id in self.basin_ids:
            yield self.basin_item(basin_id)

    def pairwise_item(self, event_id: str, active_threshold: float = 0.1) -> dict:
        event = self.events[event_id]
        reactant = self.structures[event.reactant_structure_id]
        product = self.structures[event.product_structure_id]
        if reactant.n_atoms != product.n_atoms:
            raise ValueError(
                f"event {event_id} has different reactant/product atom counts: {reactant.n_atoms} and {product.n_atoms}"
            )
        displacement = pairwise_displacements(
            product.positions,
            reactant.positions,
            cell=reactant.cell,
            pbc=reactant.pbc,
        )
        active_mask = derive_active_atoms(displacement, threshold=active_threshold)
        direction = derive_event_direction(displacement, active_mask=active_mask)
        return {
            "event_id": event.event_id,
            "basin_id": event.basin_id,
            "event": event,
            "reactant": reactant,
            "product": product,
            "displacement": displacement,
            "active_mask": active_mask,
            "event_direction": direction,
            "atom_mapping": event.atom_mapping,
            "metadata": dict(event.metadata),
        }

    def basin_item(self, basin_id: str) -> dict:
        basin = self.basins[basin_id]
        reactant = self.structures[basin.reactant_structure_id]
        events = [self.events[event_id] for event_id in basin.known_event_ids]
        products = [self.structures[event.product_structure_id] for event in events]
        return {
            "basin_id": basin.basin_id,
            "basin": basin,
            "reactant": reactant,
            "events": events,
            "products": products,
            "known_event_ids": list(basin.known_event_ids),
            "metadata": dict(basin.metadata),
        }

    @classmethod
    def from_loaded_events(cls, loaded_events: Iterable) -> "EventDataset":
        structures: dict[str, StructureRecord] = {}
        events: dict[str, EventRecord] = {}
        basins: dict[str, BasinRecord] = {}
        for loaded in loaded_events:
            structures.update(loaded.structures)
            events[loaded.event.event_id] = loaded.event
            basin = basins.get(loaded.event.basin_id)
            if basin is None:
                basins[loaded.event.basin_id] = BasinRecord(
                    basin_id=loaded.event.basin_id,
                    reactant_structure_id=loaded.event.reactant_structure_id,
                    known_event_ids=[loaded.event.event_id],
                )
            else:
                basin.known_event_ids.append(loaded.event.event_id)
        return cls(structures=structures, events=events, basins=basins)

    def subset(self, basin_ids: Iterable[str]) -> "EventDataset":
        """Return a new `EventDataset` containing only the given basins.

        Only structures referenced by the retained basins and events
        are kept; orphaned structures are dropped.
        """
        keep_basins = set(basin_ids)
        missing = keep_basins - set(self.basins.keys())
        if missing:
            raise KeyError(f"unknown basin ids: {sorted(missing)}")

        kept_basins = {bid: self.basins[bid] for bid in keep_basins}
        kept_events: dict[str, EventRecord] = {}
        for basin in kept_basins.values():
            for eid in basin.known_event_ids:
                kept_events[eid] = self.events[eid]

        # Collect structure ids referenced by retained basins and events
        kept_sids: set[str] = set()
        for basin in kept_basins.values():
            kept_sids.add(basin.reactant_structure_id)
        for event in kept_events.values():
            kept_sids.add(event.reactant_structure_id)
            kept_sids.add(event.product_structure_id)
            if event.transition_state_structure_id:
                kept_sids.add(event.transition_state_structure_id)
        kept_structures = {sid: self.structures[sid] for sid in kept_sids}
        return EventDataset(
            structures=kept_structures,
            events=kept_events,
            basins=kept_basins,
        )


def split_basins(
    dataset: EventDataset,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[EventDataset, EventDataset, EventDataset]:
    """Randomly split an `EventDataset` into train / validation / test sets.

    Splitting is done at the **basin** level so that all events belonging
    to the same basin stay together.  This prevents information leakage
    between splits.

    Args:
        dataset: the full dataset.
        train: fraction of basins for training (default 0.7).
        val: fraction of basins for validation (default 0.15).
        test: fraction of basins for testing (default 0.15).
        seed: random seed for reproducible shuffling.

    Returns:
        ``(train_ds, val_ds, test_ds)`` — `EventDataset` objects.
        Splits with zero basins are returned as empty datasets.
    """
    total = train + val + test
    if total <= 0:
        raise ValueError("train + val + test must be > 0")
    for name, value in [("train", train), ("val", val), ("test", test)]:
        if value < 0:
            raise ValueError(f"{name} fraction must be non-negative, got {value}")

    basin_ids = sorted(dataset.basin_ids)
    n_basins = len(basin_ids)
    if n_basins == 0:
        empty = EventDataset(structures={}, events={}, basins={})
        return empty, empty, empty

    rng = np.random.default_rng(seed)
    shuffled = list(basin_ids)
    rng.shuffle(shuffled)

    # Compute split counts — use floor division with remainder to avoid
    # banker's-rounding surprises on small basin counts.
    n_train = max(0, min(n_basins, int(n_basins * train / total + 0.5)))
    n_val = max(0, min(n_basins - n_train, int(n_basins * val / total + 0.5)))
    n_test = n_basins - n_train - n_val

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val : n_train + n_val + n_test]

    return (
        dataset.subset(train_ids) if train_ids else _empty_dataset(),
        dataset.subset(val_ids) if val_ids else _empty_dataset(),
        dataset.subset(test_ids) if test_ids else _empty_dataset(),
    )


def _empty_dataset() -> EventDataset:
    return EventDataset(structures={}, events={}, basins={})
