"""Domain records, source readers, and optional PyG training views."""

from basinflow.data.catalog import BasinSplit, EventCatalog, EventTarget
from basinflow.data.pyg import BasinDataset, EventData, EventFlowDataset

from basinflow.data.raw_events import load_eon_catalog
from basinflow.data.records import BasinRecord, CandidateRecord, EventRecord, StructureRecord

__all__ = [
    "StructureRecord",
    "EventRecord",
    "BasinRecord",
    "CandidateRecord",
    "BasinSplit",
    "EventCatalog",
    "EventTarget",
    "EventData",
    "EventFlowDataset",
    "BasinDataset",
    "load_eon_catalog",
]
