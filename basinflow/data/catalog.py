"""Domain catalog and basin-level split primitives for event datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from basinflow.data.records import BasinRecord, EventRecord, StructureRecord
from basinflow.geometry.mic import (
    derive_active_atoms,
    derive_event_direction,
    pairwise_displacements,
)


@dataclass(frozen=True)
class EventTarget:
    """Supervised R-to-P quantities derived from one catalog event."""

    event: EventRecord
    reactant: StructureRecord
    product: StructureRecord
    transition_state: StructureRecord | None
    displacement: np.ndarray
    target_positions: np.ndarray
    active_mask: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class BasinSplit:
    """Ordered basin ids for reproducible train/validation/test selection."""

    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        assigned: set[str] = set()
        for partition in ("train_ids", "val_ids", "test_ids"):
            identifiers = getattr(self, partition)
            if not isinstance(identifiers, (list, tuple)):
                raise ValueError(f"{partition} must be a list or tuple of basin ids")
            if any(not isinstance(value, str) or not value.strip() for value in identifiers):
                raise ValueError(f"{partition} must contain only nonempty string basin ids")
            unique = set(identifiers)
            if len(unique) != len(identifiers):
                raise ValueError(f"{partition} contains duplicate basin ids")
            overlap = assigned & unique
            if overlap:
                raise ValueError(f"split partitions overlap at basin ids: {sorted(overlap)}")
            assigned.update(unique)
            object.__setattr__(self, partition, tuple(identifiers))

    def validate(self, catalog: "EventCatalog", *, require_complete: bool = True) -> None:
        """Check source membership, optionally requiring every catalog basin."""
        assigned = set(self.train_ids) | set(self.val_ids) | set(self.test_ids)
        available = set(catalog.basin_ids)
        unknown = assigned - available
        if unknown:
            raise ValueError(f"split contains unknown basin ids: {sorted(unknown)}")
        missing = available - assigned
        if require_complete and missing:
            raise ValueError(f"split is missing catalog basin ids: {sorted(missing)}")

    @classmethod
    def create(
        cls,
        catalog: "EventCatalog",
        *,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 42,
    ) -> "BasinSplit":
        total = float(train + val + test)
        if total <= 0:
            raise ValueError("train + val + test must be > 0")
        if any(value < 0 for value in (train, val, test)):
            raise ValueError("split fractions must be non-negative")
        basin_ids = sorted(catalog.basin_ids)
        rng = np.random.default_rng(seed)
        rng.shuffle(basin_ids)
        count = len(basin_ids)
        train_count = max(0, min(count, int(count * train / total + 0.5))) # round to nearest int
        val_count = max(
            0,
            min(count - train_count, int(count * val / total + 0.5)), # round to nearest int
        )
        return cls(
            train_ids=tuple(basin_ids[:train_count]),
            val_ids=tuple(basin_ids[train_count : train_count + val_count]),
            test_ids=tuple(basin_ids[train_count + val_count :]),
            seed=int(seed),
        )

    def select(
        self,
        catalog: "EventCatalog",
        split: Literal["train", "val", "test"],
    ) -> "EventCatalog":
        selected = {
            "train": self.train_ids,
            "val": self.val_ids,
            "test": self.test_ids,
        }[split]
        return catalog.subset(selected)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "train": list(self.train_ids),
                    "val": list(self.val_ids),
                    "test": list(self.test_ids),
                    "seed": self.seed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BasinSplit":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("split manifest must be a JSON object")
        required = {"train", "val", "test", "seed"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"split manifest is missing keys: {sorted(missing)}")
        return cls(
            train_ids=payload["train"],
            val_ids=payload["val"],
            test_ids=payload["test"],
            seed=int(payload["seed"]),
        )


@dataclass
class EventCatalog:
    """Validated domain records grouped by reactant basin.

    The catalog deliberately has no tensor, seed, or graph responsibilities.
    """

    structures: dict[str, StructureRecord]
    events: dict[str, EventRecord]
    basins: dict[str, BasinRecord]

    def __post_init__(self) -> None:
        self.structures = dict(self.structures)
        self.events = dict(self.events)
        self.basins = dict(self.basins)
        self._validate()

    @property
    def event_ids(self) -> list[str]:
        return list(self.events)

    @property
    def basin_ids(self) -> list[str]:
        return list(self.basins)

    @classmethod
    def from_eon_directory(cls, events_dir: str | Path, **kwargs) -> "EventCatalog":
        from basinflow.data.raw_events import load_eon_catalog

        return load_eon_catalog(events_dir, **kwargs)

    def _validate(self) -> None:
        for event_id, event in self.events.items():
            if event.event_id != event_id:
                raise ValueError(f"event key {event_id!r} does not match event_id")
            for structure_id in (
                event.reactant_structure_id,
                event.product_structure_id,
                event.transition_state_structure_id,
            ):
                if structure_id is not None and structure_id not in self.structures:
                    raise ValueError(
                        f"event {event_id} references unknown structure {structure_id!r}"
                    )
            reactant = self.structures[event.reactant_structure_id]
            for role, structure_id in (
                ("product", event.product_structure_id),
                ("transition state", event.transition_state_structure_id),
            ):
                if structure_id is None:
                    continue
                structure = self.structures[structure_id]
                if structure.n_atoms != reactant.n_atoms:
                    raise ValueError(
                        f"event {event_id} reactant/{role} atom counts differ"
                    )
                if structure.species != reactant.species:
                    raise ValueError(
                        f"event {event_id} reactant/{role} species order differs"
                    )
                if not np.allclose(structure.cell, reactant.cell):
                    raise ValueError(f"event {event_id} reactant/{role} cells differ")
                if not np.array_equal(structure.pbc, reactant.pbc):
                    raise ValueError(f"event {event_id} reactant/{role} PBC differs")
            if event.basin_id not in self.basins:
                raise ValueError(f"event {event_id} references unknown basin {event.basin_id!r}")

        for basin_id, basin in self.basins.items():
            if basin.basin_id != basin_id:
                raise ValueError(f"basin key {basin_id!r} does not match basin_id")
            if basin.reactant_structure_id not in self.structures:
                raise ValueError(
                    f"basin {basin_id} references unknown reactant structure"
                )
            for event_id in basin.known_event_ids:
                if event_id not in self.events:
                    raise ValueError(f"basin {basin_id} references unknown event {event_id!r}")
                event = self.events[event_id]
                if event.basin_id != basin_id:
                    raise ValueError(f"event {event_id} belongs to another basin")

    def event_target(self, event_id: str, *, active_threshold: float = 0.1) -> EventTarget:
        event = self.events[event_id]
        reactant = self.structures[event.reactant_structure_id]
        product = self.structures[event.product_structure_id]
        displacement = pairwise_displacements(
            product.positions,
            reactant.positions,
            cell=reactant.cell,
            pbc=reactant.pbc,
        )
        if event.active_atoms is None:
            active_mask = derive_active_atoms(displacement, threshold=active_threshold)
        else:
            active_mask = np.zeros((reactant.n_atoms,), dtype=bool)
            active_mask[np.asarray(event.active_atoms, dtype=int)] = True
        direction = derive_event_direction(displacement, active_mask=active_mask)
        transition_state = (
            self.structures[event.transition_state_structure_id]
            if event.transition_state_structure_id is not None
            else None
        )
        return EventTarget(
            event=event,
            reactant=reactant,
            product=product,
            transition_state=transition_state,
            displacement=displacement,
            target_positions=reactant.positions + displacement,
            active_mask=active_mask,
            direction=direction,
        )

    def single_event_subset(self) -> "EventCatalog":
        """Keep only events whose product preserves the reactant's fragments.

        Transition1x mixes local rearrangements with dissociations whose
        product is a separated fragment pair; only the former are events a
        basin-level proposer is asked for (``docs/22`` section 二之二).
        """
        from basinflow.geometry.bonds import bond_adjacency, preserves_fragments

        keep = []
        for basin_id in self.basin_ids:
            basin = self.basins[basin_id]
            target = self.event_target(basin.known_event_ids[0])
            reactant = target.reactant
            if preserves_fragments(
                bond_adjacency(reactant.positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
                bond_adjacency(target.target_positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
            ):
                keep.append(basin_id)
        if not keep:
            raise ValueError("no single-event records in this catalog")
        return self.subset(keep)

    def subset(self, basin_ids: Iterable[str]) -> "EventCatalog":
        ordered_ids = [str(basin_id) for basin_id in basin_ids]
        missing = set(ordered_ids) - set(self.basins)
        if missing:
            raise KeyError(f"unknown basin ids: {sorted(missing)}")
        basins = {basin_id: self.basins[basin_id] for basin_id in ordered_ids}
        event_ids = [
            event_id
            for basin in basins.values()
            for event_id in basin.known_event_ids
        ]
        events = {event_id: self.events[event_id] for event_id in event_ids}
        structure_ids = {basin.reactant_structure_id for basin in basins.values()}
        for event in events.values():
            structure_ids.update(
                (event.reactant_structure_id, event.product_structure_id)
            )
            if event.transition_state_structure_id is not None:
                structure_ids.add(event.transition_state_structure_id)
        return EventCatalog(
            structures={sid: self.structures[sid] for sid in structure_ids},
            events=events,
            basins=basins,
        )
