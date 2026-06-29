from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase.io

from fscgp.data.dataset import EventDataset
from fscgp.data.records import EventRecord, StructureRecord


@dataclass(frozen=True)
class LoadedEvent:
    event: EventRecord
    reactant: StructureRecord
    product: StructureRecord
    transition_state: StructureRecord | None = None

    @property
    def structures(self) -> dict[str, StructureRecord]:
        records = {
            self.reactant.structure_id: self.reactant,
            self.product.structure_id: self.product,
        }
        if self.transition_state is not None:
            records[self.transition_state.structure_id] = self.transition_state
        return records


def read_event_file(
    path: str | Path,
    basin_id: str | None = None,
    event_id: str | None = None,
    validation_status: str = "known_valid",
    metadata: dict[str, Any] | None = None,
    ase_index: str = ":3",
    **ase_read_kwargs: Any,
) -> LoadedEvent:
    """Read a multi-frame ASE event file.

    Frame 0 is the reactant, frame 1 is the product, and frame 2 is an
    optional transition state.  By default only the first three frames
    are read (``ase_index=":3"``).
    """
    event_path = Path(path)
    frames = ase.io.read(event_path, index=ase_index, **ase_read_kwargs)
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) < 2:
        raise ValueError(f"event file {event_path} must contain at least two frames")

    inferred_event_id = event_id or event_path.stem
    inferred_basin_id = basin_id or inferred_event_id
    base_metadata = dict(metadata or {})
    base_metadata.setdefault("source_file", str(event_path))

    reactant = StructureRecord.from_ase(
        frames[0],
        structure_id=f"{inferred_event_id}:reactant",
        metadata={**base_metadata, "frame_index": 0, "role": "reactant"},
    )
    product = StructureRecord.from_ase(
        frames[1],
        structure_id=f"{inferred_event_id}:product",
        metadata={**base_metadata, "frame_index": 1, "role": "product"},
    )

    transition_state = None
    if len(frames) >= 3:
        transition_state = StructureRecord.from_ase(
            frames[2],
            structure_id=f"{inferred_event_id}:transition_state",
            metadata={**base_metadata, "frame_index": 2, "role": "transition_state"},
        )

    event = EventRecord(
        event_id=inferred_event_id,
        reactant_structure_id=reactant.structure_id,
        product_structure_id=product.structure_id,
        basin_id=inferred_basin_id,
        transition_state_structure_id=transition_state.structure_id if transition_state else None,
        validation_status=validation_status,
        metadata=base_metadata,
    )
    return LoadedEvent(
        event=event,
        reactant=reactant,
        product=product,
        transition_state=transition_state,
    )


def read_event_files(
    paths: list[str | Path],
    basin_ids: list[str] | None = None,
    event_ids: list[str] | None = None,
    **kwargs: Any,
) -> list[LoadedEvent]:
    """Read multiple event files into `LoadedEvent` objects."""
    loaded = []
    for index, path in enumerate(paths):
        loaded.append(
            read_event_file(
                path,
                basin_id=basin_ids[index] if basin_ids is not None else None,
                event_id=event_ids[index] if event_ids is not None else None,
                **kwargs,
            )
        )
    return loaded


def _read_basin_table(csv_path: Path) -> dict[str, str]:
    """Read a ``basin_table.csv`` → ``{filename: basin_id}`` mapping."""
    mapping: dict[str, str] = {}
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            fname = (row.get("file") or "").strip()
            basin = (row.get("basin") or "").strip()
            if fname and basin:
                mapping[fname] = basin
    return mapping


def read_events_directory(
    events_dir: str | Path,
    basin_table: str | Path | None = None,
    glob_pattern: str = "event_*.extxyz",
    **kwargs: Any,
) -> EventDataset:
    """Read an events directory into an `EventDataset`.

    Scans for ``event_*.extxyz`` files and reads the optional
    ``basin_table.csv`` to map each event file to its correct basin.
    Each extxyz file is loaded via `read_event_file`.

    Args:
        events_dir: directory containing extxyz event files.
        basin_table: path to a CSV mapping event files to basins
            (columns ``file``, ``basin``). Defaults to
            ``<events_dir>/basin_table.csv`` if present.
        glob_pattern: glob pattern for event files relative
            to ``events_dir``.
        **kwargs: forwarded to `read_event_file`.

    Returns:
        `EventDataset` with all loaded events grouped by basin.
    """
    events_dir = Path(events_dir)
    if not events_dir.is_dir():
        raise FileNotFoundError(f"events directory not found: {events_dir}")

    # Resolve basin mapping
    basin_map: dict[str, str] = {}
    if basin_table is None:
        default_table = events_dir / "basin_table.csv"
        if default_table.is_file():
            basin_table = default_table
    if basin_table is not None:
        basin_map = _read_basin_table(Path(basin_table))

    # Find event files
    event_files = sorted(events_dir.glob(glob_pattern))
    if not event_files:
        raise FileNotFoundError(
            f"no event files matching {glob_pattern!r} in {events_dir}"
        )

    # Load each event
    loaded_events: list[LoadedEvent] = []
    for path in event_files:
        basin_id = basin_map.get(path.name)
        if basin_id is None and basin_map:
            print(f"  ⚠ {path.name}: not listed in basin table")
        loaded = read_event_file(
            path,
            basin_id=basin_id,
            event_id=path.stem,
            **kwargs,
        )
        loaded_events.append(loaded)

    # Check for entries in basin_table with no matching file
    if basin_map:
        on_disk = {p.name for p in event_files}
        for fname in basin_map:
            if fname not in on_disk:
                print(f"  ⚠ basin table references missing file: {fname}")

    return EventDataset.from_loaded_events(loaded_events)
