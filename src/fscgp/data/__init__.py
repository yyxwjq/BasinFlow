"""Data records, raw event readers, datasets, and collation utilities."""

from fscgp.data.dataset import EventDataset, split_basins
from fscgp.data.raw_events import (
    LoadedEvent,
    read_event_file,
    read_event_files,
    read_events_directory,
)
from fscgp.data.records import BasinRecord, CandidateRecord, EventRecord, StructureRecord

__all__ = [
    "StructureRecord",
    "EventRecord",
    "BasinRecord",
    "CandidateRecord",
    "LoadedEvent",
    "EventDataset",
    "read_event_file",
    "read_event_files",
    "read_events_directory",
    "split_basins",
]
