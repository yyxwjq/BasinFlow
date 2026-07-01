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


def _derive_active_atoms_from_constraints(
    structure: StructureRecord,
) -> list[int] | None:
    """Return movable atom indices from a per-atom constraint mask.

    Returns ``None`` when no constraint information is available,
    allowing downstream code to fall back to MIC-derived labels.
    """
    if structure.constraints is None:
        return None
    movable = ~structure.constraints
    if not movable.any():
        return []
    return [int(i) for i in movable.nonzero()[0]]


def _fixed_mask(structure: StructureRecord) -> np.ndarray:
    if structure.constraints is None:
        return np.zeros((structure.n_atoms,), dtype=bool)
    return np.asarray(structure.constraints, dtype=bool)


def _active_mask_from_event(
    event: EventRecord,
    n_atoms: int,
) -> np.ndarray | None:
    if event.active_atoms is None:
        return None
    active_mask = np.zeros((n_atoms,), dtype=bool)
    active_mask[np.asarray(event.active_atoms, dtype=int)] = True
    return active_mask


def _ts_item(
    dataset: EventDataset,
    event: EventRecord,
    reactant: StructureRecord,
) -> tuple[StructureRecord | None, np.ndarray]:
    """Return (transition_state, ts_displacement) for an event.

    When no TS exists, returns ``(None, zeros)`` so downstream tensors
    keep a consistent shape.
    """
    ts = (
        dataset.structures[event.transition_state_structure_id]
        if event.transition_state_structure_id
        else None
    )
    if ts is None:
        return None, np.zeros((reactant.n_atoms, 3), dtype=float)
    if reactant.n_atoms != ts.n_atoms:
        raise ValueError(
            f"event {event.event_id} has different reactant/TS atom "
            f"counts: {reactant.n_atoms} and {ts.n_atoms}"
        )
    ts_disp = pairwise_displacements(
        ts.positions,
        reactant.positions,
        cell=reactant.cell,
        pbc=reactant.pbc,
    )
    return ts, ts_disp


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
        """Event ids in insertion order (CSV row order for dir-loaded datasets)."""
        return list(self.events.keys())

    @property
    def basin_ids(self) -> list[str]:
        """Basin ids in insertion order."""
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
        active_mask = _active_mask_from_event(event, reactant.n_atoms)
        if active_mask is None:
            active_mask = derive_active_atoms(
                displacement,
                threshold=active_threshold,
            )
        direction = derive_event_direction(displacement, active_mask=active_mask)
        fixed_mask = _fixed_mask(reactant)
        movable_mask = ~fixed_mask
        transition_state, ts_displacement = _ts_item(self, event, reactant)

        return {
            "event_id": event.event_id,
            "basin_id": event.basin_id,
            "event": event,
            "reactant": reactant,
            "product": product,
            "transition_state": transition_state,
            "has_transition_state": transition_state is not None,
            "displacement": displacement,
            "ts_displacement": ts_displacement,
            "active_mask": active_mask,
            "fixed_mask": fixed_mask,
            "movable_mask": movable_mask,
            "event_direction": direction,
            "atom_mapping": event.atom_mapping,
            "metadata": dict(event.metadata),
        }

    def basin_item(self, basin_id: str) -> dict:
        basin = self.basins[basin_id]
        reactant = self.structures[basin.reactant_structure_id]
        events = [self.events[event_id] for event_id in basin.known_event_ids]
        products = [self.structures[event.product_structure_id] for event in events]
        transition_states = [
            self.structures[event.transition_state_structure_id]
            if event.transition_state_structure_id
            else None
            for event in events
        ]
        return {
            "basin_id": basin.basin_id,
            "basin": basin,
            "reactant": reactant,
            "events": events,
            "products": products,
            "transition_states": transition_states,
            "known_event_ids": list(basin.known_event_ids),
            "metadata": dict(basin.metadata),
        }

    @classmethod
    def from_loaded_events(cls, loaded_events: Iterable) -> "EventDataset":
        structures: dict[str, StructureRecord] = {}
        events: dict[str, EventRecord] = {}
        basins: dict[str, BasinRecord] = {}
        for loaded in loaded_events:
            for sid, structure in loaded.structures.items():
                if sid in structures and not np.array_equal(
                    structures[sid].positions, structure.positions
                ):
                    raise ValueError(
                        f"duplicate structure_id with different positions: {sid}"
                    )
                structures[sid] = structure

            event = loaded.event
            if event.active_atoms is None:
                active = _derive_active_atoms_from_constraints(loaded.reactant)
                if active is not None:
                    event.active_atoms = active
            events[event.event_id] = event

            basin = basins.get(event.basin_id)
            if basin is None:
                basins[event.basin_id] = BasinRecord(
                    basin_id=event.basin_id,
                    reactant_structure_id=event.reactant_structure_id,
                    known_event_ids=[event.event_id],
                )
            else:
                basin.known_event_ids.append(event.event_id)
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
