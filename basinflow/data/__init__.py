"""Event data layer: structures, events, splits and the bridge sample view."""
from basinflow.data.events import (
    Event,
    EventCatalog,
    EventTarget,
    event_id_from_path,
    event_target,
    load_events,
    ordered_event_ids,
)
from basinflow.data.samples import BridgeFlowDataset, EventData, require_pyg
from basinflow.data.splits import (
    Split,
    check_split,
    load_split,
    partition,
    resolve_split,
    save_split,
    split_events,
)
from basinflow.data.structures import Structure

__all__ = [
    "BridgeFlowDataset",
    "Event",
    "EventCatalog",
    "EventData",
    "EventTarget",
    "Split",
    "Structure",
    "check_split",
    "event_id_from_path",
    "event_target",
    "load_events",
    "load_split",
    "ordered_event_ids",
    "partition",
    "require_pyg",
    "resolve_split",
    "save_split",
    "split_events",
]
