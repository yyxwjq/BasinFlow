"""Events: three inlined frames plus the supervised R-to-P quantity.

An event is reactant -> product, optionally with a transition state. Events are the
unit of training, evaluation and splitting. The supervised quantity is
:class:`EventTarget`: the minimum-image displacement ``P - R`` and the lifted
endpoint ``R + displacement``.
"""
from __future__ import annotations

from basinflow.geometry.mic import get_displacements
from basinflow.data.structures import Structure
from dataclasses import dataclass, field
from typing import Any, Iterable
from pathlib import Path
import numpy as np
import ase.io


# --- Types ---


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
    """Events by id, one entry per recorded elementary step."""

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

        Raises:
            KeyError: If an id is not in the catalog.
            ValueError: If the selection is empty.
        """
        ordered = [str(event_id) for event_id in event_ids]
        unknown = [event_id for event_id in ordered if event_id not in self.events]
        if unknown:
            raise KeyError(f"unknown event ids: {unknown}")
        if not ordered:
            raise ValueError(f"{f'{name} ' if name else ''}event selection is empty")
        return EventCatalog(events={event_id: self.events[event_id] for event_id in ordered})


# --- Loading ---


def event_id_from_path(path: Path) -> str:
    """The integer event id encoded in an event file name.

    ``event_00007.traj`` is event ``7``: the digits after ``event_`` are the
    integer event id used by ``split.csv``, with zero padding ignored.

    Raises:
        ValueError: If the file name is not ``event_<digits>.<ext>``.
    """
    index = path.stem.removeprefix("event_")
    if not index.isdecimal():
        raise ValueError(
            f"event file {path.name} must be named 'event_<digits>.traj'; "
            "the digits are the integer event id used by split.csv"
        )
    return str(int(index))


def load_events(events_dir: str | Path) -> EventCatalog:
    """Load one event directory: every ``event_<digits>.traj`` file is one event.

    Returns:
        An :class:`EventCatalog` keyed by the integer event id encoded in each file
        name. ``metadata`` starts from the first frame's ``info``, with
        ``source_file`` defaulting to the file path. Frame order is 0 reactant,
        1 product, 2 transition state.

    Raises:
        FileNotFoundError: If the directory holds no ``event_*.traj`` file.
        ValueError: If a name carries no numeric event id, a file does not hold
            exactly three frames, or an event id repeats.
    """
    events_dir = Path(events_dir)
    if not events_dir.is_dir():
        raise FileNotFoundError(f"events directory not found: {events_dir}")

    files = sorted(
        events_dir.glob("event_*.traj"), key=lambda item: int(event_id_from_path(item))
    )
    if not files:
        raise FileNotFoundError(f"no event_*.traj files in {events_dir}")

    events: dict[str, Event] = {}
    for path in files:
        frames = ase.io.read(path, index=":")
        if not isinstance(frames, list):
            frames = [frames]
        if len(frames) != 3:
            raise ValueError(f"event file {path} must contain exactly three frames")
        metadata = {"source_file": str(path)}
        metadata.update(frames[0].info)
        event_id = event_id_from_path(path)
        if event_id in events:
            raise ValueError(f"duplicate event id {event_id!r} from {path}")
        events[event_id] = Event(
            event_id=event_id,
            reactant=Structure.from_ase(frames[0]),
            product=Structure.from_ase(frames[1]),
            transition_state=Structure.from_ase(frames[2]),
            metadata=metadata,
        )
    return EventCatalog(events=events)


# --- Supervised quantity ---


@dataclass(frozen=True)
class EventTarget:
    """The supervised R-to-P quantities, derived, never stored."""

    event: Event
    reactant: Structure
    product: Structure
    transition_state: Structure | None
    displacement: np.ndarray
    target_positions: np.ndarray


def event_target(event: Event) -> EventTarget:
    """MIC displacement ``P - R`` and the endpoint ``R + displacement``.

    ``target_positions`` is ``R`` plus the displacement, so for a periodic frame it
    can differ from the recorded product geometry by a lattice vector.
    """
    displacement = get_displacements(
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


# --- Selection ---


def ordered_event_ids(catalog: EventCatalog) -> list[str]:
    """Event ids in ascending integer order: ``"2"`` before ``"10"``."""
    return sorted(catalog.events, key=int)


def select_events(
    catalog: EventCatalog, event_ids: Iterable[str], *, name: str | None = None
) -> EventCatalog:
    """Restrict a catalog to the named events, in the requested order."""
    return catalog.select(event_ids, name=name)
