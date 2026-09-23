"""Minimal event data layer: structures, events, splits, and the sample views.

The public surface is exactly what the three paths need -- mode A (R -> P), mode B
(R + P -> TS) and target-free event inference -- and nothing else. The export list
below is kept in step with the implementation; there is no hand-maintained
"compatibility" entry.
"""
from basinflow.data.events import (
    Event,
    EventCatalog,
    EventTarget,
    comparison_groups,
    event_id_from_path,
    event_target,
    load_events,
    ordered_event_ids,
    select_events,
    single_event_scope,
)
from basinflow.data.samples import (
    BridgeFlowDataset,
    EventData,
    EventFlowDataset,
    EventProposalDataset,
    TransitionStateDataset,
    require_pyg,
)
from basinflow.data.splits import (
    EventSplit,
    all_in_train,
    load_groups,
    load_split,
    resolve_split,
    save_split,
    split_events,
    split_events_by_group,
    split_path,
)
from basinflow.data.structures import Structure

__all__ = [
    "BridgeFlowDataset",
    "Event",
    "EventCatalog",
    "EventData",
    "EventFlowDataset",
    "EventProposalDataset",
    "EventSplit",
    "EventTarget",
    "Structure",
    "TransitionStateDataset",
    "all_in_train",
    "comparison_groups",
    "event_id_from_path",
    "event_target",
    "load_events",
    "load_groups",
    "load_split",
    "ordered_event_ids",
    "require_pyg",
    "resolve_split",
    "save_split",
    "select_events",
    "single_event_scope",
    "split_events",
    "split_events_by_group",
    "split_path",
]
