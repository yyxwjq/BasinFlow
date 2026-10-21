"""Event data layer: structures, events, splits and the sample views.

The exports below are the public surface of the R + P -> TS task: the two
baselines (PaiNN/EGNN) and the ReactOT bridge.
"""
from basinflow.data.events import (
    Event,
    EventCatalog,
    EventTarget,
    event_id_from_path,
    event_target,
    load_events,
    ordered_event_ids,
    select_events,
)
from basinflow.data.samples import (
    BridgeFlowDataset,
    EventData,
    TransitionStateDataset,
    require_pyg,
)
from basinflow.data.splits import (
    EventSplit,
    all_in_train,
    load_split,
    resolve_split,
    save_split,
    split_events,
    split_path,
)
from basinflow.data.structures import Structure

__all__ = [
    "BridgeFlowDataset",
    "Event",
    "EventCatalog",
    "EventData",
    "EventSplit",
    "EventTarget",
    "Structure",
    "TransitionStateDataset",
    "all_in_train",
    "event_id_from_path",
    "event_target",
    "load_events",
    "load_split",
    "ordered_event_ids",
    "require_pyg",
    "resolve_split",
    "save_split",
    "select_events",
    "split_events",
    "split_path",
]
