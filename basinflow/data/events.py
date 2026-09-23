"""Events: three inlined frames plus the one supervised R-to-P quantity.

An event is reactant -> product, optionally with a transition state. The frames are
inlined rather than referenced through a structure registry, so there is no id
bookkeeping and no dangling reference can survive a subset.

The event is the unit of training, evaluation and splitting: one event is one
recorded elementary step, and a split assigns whole events to partitions. The only
supervised quantity here is :class:`EventTarget`: the minimum-image displacement
``P - R`` and the lifted endpoint ``R + displacement``. Active-atom and direction
labels are absent because both are derivable from the displacement, and a derived
label shipped next to its source is just a second thing to keep in sync.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import ase.io
import numpy as np

from basinflow.data.structures import Structure
from basinflow.geometry.mic import pairwise_displacements

FRAME_ROLES = ("reactant", "product", "transition_state")
# EON names its third frame "saddle"; the data layer calls that role
# "transition_state", so the two spellings must be accepted on read.
ROLE_ALIASES = {"saddle": "transition_state"}
EVENT_FILE_RE = re.compile(r"^(?P<prefix>.*?)event_(?P<index>\d+)$")


def event_id_from_path(path: Path) -> str:
    """The integer event id encoded in an event file name.

    ``event_00007.traj`` is event ``7`` and ``state3_event_00007.traj`` is event
    ``7`` of that state. The id is what ``split.csv`` stores, so it must be a
    plain integer and unique inside the directory.

    Raises:
        ValueError: If the file name does not carry a numeric event index.
    """
    match = EVENT_FILE_RE.match(path.stem)
    if match is None:
        raise ValueError(
            f"event file {path.name} must be named '<prefix>event_<digits>.<ext>'; "
            "the digits are the integer event id used by split.csv"
        )
    return str(int(match.group("index")))


@dataclass
class Event:
    """One known event: the reactant it starts from, its product, and an optional TS."""

    event_id: str
    reactant: Structure
    product: Structure
    transition_state: Structure | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_id = str(self.event_id)
        self.metadata = dict(self.metadata or {})
        # Invariant: R/P/TS must agree on atom count, species order, cell and pbc.
        reactant = self.reactant
        for role, structure in (
            ("product", self.product),
            ("transition_state", self.transition_state),
        ):
            if structure is None:
                continue
            if structure.n_atoms != reactant.n_atoms:
                raise ValueError(f"event {self.event_id} reactant/{role} atom counts differ")
            if structure.species != reactant.species:
                raise ValueError(f"event {self.event_id} reactant/{role} species order differs")
            if not np.allclose(structure.cell, reactant.cell):
                raise ValueError(f"event {self.event_id} reactant/{role} cells differ")
            if not np.array_equal(structure.pbc, reactant.pbc):
                raise ValueError(f"event {self.event_id} reactant/{role} PBC differs")


@dataclass
class EventCatalog:
    """Events by id, one entry per recorded elementary step.

    The catalog is a flat mapping: an event carries everything it needs, so there
    is no second index to keep consistent with it.
    """

    events: dict[str, Event]

    def __post_init__(self) -> None:
        self.events = dict(self.events)
        for event_id, event in self.events.items():
            if event.event_id != event_id:
                raise ValueError(f"event key {event_id!r} does not match its event_id")

    @property
    def event_ids(self) -> list[str]:
        return list(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __contains__(self, event_id: str) -> bool:
        return event_id in self.events

    def select(self, event_ids: Iterable[str], *, name: str | None = None) -> "EventCatalog":
        """Restrict the catalog to the named events, in the requested order.

        An empty selection raises instead of quietly yielding an empty dataset: a
        split manifest with an empty partition is legal data, but training or
        evaluating on it is a configuration error, not a result.
        """
        ordered = [str(event_id) for event_id in event_ids]
        unknown = [event_id for event_id in ordered if event_id not in self.events]
        if unknown:
            raise KeyError(f"unknown event ids: {unknown}")
        if not ordered:
            raise ValueError(f"{f'{name} ' if name else ''}event selection is empty")
        return EventCatalog(events={event_id: self.events[event_id] for event_id in ordered})


def ordered_event_ids(catalog: EventCatalog) -> list[str]:
    """Event ids in ascending integer order: ``"2"`` before ``"10"``.

    The order is the contract that keeps an on-disk ``split.csv`` valid, and it is
    numeric rather than lexicographic so that ``event_10`` does not sort before
    ``event_2``.
    """
    return sorted(catalog.events, key=int)


@dataclass(frozen=True)
class EventTarget:
    """The supervised R-to-P quantities, derived, never stored."""

    event: Event
    reactant: Structure
    product: Structure
    transition_state: Structure | None
    displacement: np.ndarray
    target_positions: np.ndarray


def comparison_groups(
    catalog: EventCatalog, groups: dict[str, str] | None
) -> dict[str, str] | None:
    """Fill a group table out to every event of ``catalog``.

    Each event maps to its basin key, and an event the table does not mention
    forms a group of its own. That way a partial table degrades to event-level
    comparison for the missing events instead of failing at scoring time, while
    the events the table does describe are still compared basin-wide.

    Returns ``None`` when there is no table, which means "compare each event with
    itself".
    """
    if groups is None:
        return None
    filled = dict(groups)
    for event_id in catalog.events:
        filled.setdefault(event_id, f"__event__:{event_id}")
    return filled


def event_target(event: Event) -> EventTarget:
    """MIC displacement ``P - R`` and the lifted endpoint ``R + displacement``.

    The lifted endpoint is the reaction coordinate's endpoint, not the recorded
    product geometry: for a periodic system the two differ by a lattice vector, and
    only the lifted one crosses the cell boundary without a discontinuity.
    """
    displacement = pairwise_displacements(
        event.product.positions,
        event.reactant.positions,
        cell=event.reactant.cell,
        pbc=event.reactant.pbc,
    )
    return EventTarget(
        event=event,
        reactant=event.reactant,
        product=event.product,
        transition_state=event.transition_state,
        displacement=displacement,
        target_positions=event.reactant.positions + displacement,
    )


def select_events(
    catalog: EventCatalog, event_ids: Iterable[str], *, name: str | None = None
) -> EventCatalog:
    """Restrict a catalog to the named events, in the requested order."""
    return catalog.select(event_ids, name=name)


def single_event_scope(catalog: EventCatalog) -> EventCatalog:
    """Keep only events whose product preserves the reactant's fragments.

    Transition1x mixes local rearrangements with dissociations whose product is a
    separated fragment pair; only the former are events a proposer is asked for
    (``docs/22`` section 二之二). A multi-event catalog is kept only if all of its
    events are local, so the scope never mixes classes.
    """
    from basinflow.geometry.bonds import bond_adjacency, preserves_fragments

    keep = []
    for event_id, event in catalog.events.items():
        reactant = event.reactant
        reference = bond_adjacency(
            reactant.positions, reactant.atomic_numbers, reactant.cell, reactant.pbc
        )
        target = event_target(event)
        if preserves_fragments(
            reference,
            bond_adjacency(
                target.target_positions,
                reactant.atomic_numbers,
                reactant.cell,
                reactant.pbc,
            ),
        ):
            keep.append(event_id)
    if not keep:
        raise ValueError("no single-event records in this catalog")
    return select_events(catalog, keep, name="single_event")


def load_events(
    events_dir: str | Path,
    *,
    glob_patterns: Iterable[str] = ("event_*.traj",),
    pbc_override: bool | Iterable[bool] | None = None,
) -> EventCatalog:
    """Load one event directory: every matching file is one event.

    Only ``.traj`` is read. ASE's extended XYZ writer does not serialize
    constraints unless a ``move_mask`` column is requested explicitly, so an
    ``.extxyz`` file silently loses the fixed-atom constraint that this layer
    depends on; refusing the format is cheaper than a dataset whose mask is
    quietly all-True. An ``.extxyz`` file in the directory is reported by name.

    This is the only data source mode. There is no side table: an event file
    carries its own identity (the integer in its name) and its own metadata
    (frame ``info``), and the train/val/test assignment lives in a separate
    ``split.csv`` (:mod:`basinflow.data.splits`).

    Frame order is part of the data contract: 0 reactant, 1 product, 2 transition
    state. Two frames are allowed (a known event without a TS); anything else is
    rejected rather than guessed at. ``source_file`` defaults to the trajectory but
    frame ``info`` wins, so a converted dataset keeps naming the pickle it came from.
    """
    events_dir = Path(events_dir)
    if not events_dir.is_dir():
        raise FileNotFoundError(f"events directory not found: {events_dir}")

    files: list[Path] = []
    for pattern in glob_patterns:
        files.extend(events_dir.glob(pattern))
    if not files:
        rejected = sorted(events_dir.glob("event_*.extxyz"))
        if rejected:
            raise ValueError(
                f"{events_dir} holds .extxyz event files ({rejected[0].name}, ...); "
                "only .traj is supported because extxyz does not carry ASE constraints"
            )
        raise FileNotFoundError(f"no event files matching {tuple(glob_patterns)} in {events_dir}")

    events: dict[str, Event] = {}
    for path in sorted(files, key=lambda item: int(event_id_from_path(item))):
        frames = ase.io.read(path, index=":")
        if not isinstance(frames, list):
            frames = [frames]
        if len(frames) not in {2, 3}:
            raise ValueError(f"event file {path} must contain exactly two or three frames")
        if pbc_override is not None:
            for frame in frames:
                frame.set_pbc(pbc_override)
        for index, frame in enumerate(frames):
            declared = frame.info.get("role")
            if declared is not None:
                declared = ROLE_ALIASES.get(str(declared), str(declared))
            if declared is not None and declared != FRAME_ROLES[index]:
                raise ValueError(
                    f"event file {path} frame {index} declares role {declared!r}, "
                    f"expected {FRAME_ROLES[index]!r}"
                )
        metadata = {"source_file": str(path)}
        metadata.update(frames[0].info)
        event_id = event_id_from_path(path)
        if event_id in events:
            raise ValueError(f"duplicate event id {event_id!r} from {path}")
        events[event_id] = Event(
            event_id=event_id,
            reactant=Structure.from_ase(frames[0]),
            product=Structure.from_ase(frames[1]),
            transition_state=Structure.from_ase(frames[2]) if len(frames) == 3 else None,
            metadata=metadata,
        )
    return EventCatalog(events=events)
