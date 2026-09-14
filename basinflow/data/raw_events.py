from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase.io
import numpy as np

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord


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
    ase_index: str = ":",
    pbc_override: bool | list[bool] | tuple[bool, bool, bool] | None = None,
    **ase_read_kwargs: Any,
) -> LoadedEvent:
    """Read a multi-frame ASE event file.

    Frame 0 is the reactant, frame 1 is the product, and frame 2 is an
    optional transition state. Event files must contain exactly two or three
    frames; path images belong to later validation stages.
    """
    event_path = Path(path)
    frames = ase.io.read(event_path, index=ase_index, **ase_read_kwargs)
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) not in {2, 3}:
        raise ValueError(
            f"event file {event_path} must contain exactly two or three frames"
        )
    if pbc_override is not None:
        for frame in frames:
            frame.set_pbc(pbc_override)

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


def _read_basin_table(csv_path: Path) -> list[dict[str, str]]:
    """Read ``basin_table.csv`` rows in file order."""
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"file", "basin"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"basin table {csv_path} is missing columns: {sorted(missing)}"
            )
        for row in reader:
            fname = (row.get("file") or "").strip()
            basin = (row.get("basin") or "").strip()
            if not fname or not basin:
                raise ValueError(
                    f"basin table {csv_path} has row without file/basin: {row}"
                )
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def load_eon_catalog(
    events_dir: str | Path,
    basin_table: str | Path | None = None,
    glob_pattern: str = "event_*.extxyz",
    **kwargs: Any,
) -> EventCatalog:
    """Load a strict EON-style event directory into an ``EventCatalog``.

    This reader preserves ``move_mask`` only as the canonical movement
    constraint on ``StructureRecord``. Active supervision is derived later by
    ``EventCatalog.event_target`` from explicit event labels or MIC movement.
    """
    events_dir = Path(events_dir)
    if not events_dir.is_dir():
        raise FileNotFoundError(f"events directory not found: {events_dir}")

    basin_rows: list[dict[str, str]] = []
    if basin_table is None:
        default_table = events_dir / "basin_table.csv"
        if default_table.is_file():
            basin_table = default_table
    if basin_table is not None:
        basin_rows = _read_basin_table(Path(basin_table))

    event_files = sorted(events_dir.glob(glob_pattern))
    if not event_files and glob_pattern == "event_*.extxyz":
        event_files = sorted(events_dir.glob("event_*.traj"))
    if not event_files:
        raise FileNotFoundError(f"no event files matching {glob_pattern!r} in {events_dir}")

    if basin_rows:
        path_by_name = {path.name: path for path in event_files}
        missing = [row["file"] for row in basin_rows if row["file"] not in path_by_name]
        if missing:
            raise FileNotFoundError(f"basin table references missing event files: {missing}")
        load_specs = [
            (path_by_name[row["file"]], row["basin"], row)
            for row in basin_rows
        ]
    else:
        load_specs = [(path, None, {}) for path in event_files]

    structures: dict[str, StructureRecord] = {}
    events: dict[str, EventRecord] = {}
    basins: dict[str, BasinRecord] = {}
    for path, basin_id, row_metadata in load_specs:
        loaded = read_event_file(
            path,
            basin_id=basin_id,
            event_id=path.stem,
            metadata=row_metadata,
            **kwargs,
        )
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
            reference = structures[basin.reactant_structure_id]
            candidate = loaded.reactant
            if (
                reference.species != candidate.species
                or not np.allclose(reference.cell, candidate.cell)
                or not np.array_equal(reference.pbc, candidate.pbc)
                or not np.array_equal(reference.movable_mask, candidate.movable_mask)
            ):
                raise ValueError(
                    f"basin {loaded.event.basin_id!r} contains reactants with inconsistent topology"
                )
            if np.allclose(reference.positions, candidate.positions, atol=1e-3, rtol=0.0):
                loaded.event.reactant_structure_id = basin.reactant_structure_id
            basin.known_event_ids.append(loaded.event.event_id)
    return EventCatalog(structures=structures, events=events, basins=basins)
