"""Resolve a workflow's data source: one event directory plus its ``split.csv``.

There is exactly one mode. ``[data] events_dir`` names the directory of
``event_<id>.traj``/``event_<id>.extxyz`` files, and each file is one event.
``[data] auto_split`` decides where the train/val/test assignment comes from:

* ``true`` -- split the events by the ``[split]`` fractions and ``seed``, writing
  ``split.csv`` into the events directory. An existing file is reused, so a rerun
  never silently changes which events are held out; set
  ``[data] regenerate_split = true`` to resplit and overwrite it.
* ``false`` -- read the existing ``split.csv``; when there is none, every event is
  used for training instead of inventing a split.
"""
from __future__ import annotations

from pathlib import Path

from basinflow.data import load_events, resolve_split
from basinflow.workflows.config import parse_bool, parse_pbc_override


def load_configured_dataset(config, *, seed: int, groups: dict[str, str] | None = None):
    """Return ``(catalog, split, provenance)`` for a parsed workflow config.

    ``groups`` maps event id to a grouping key and is passed through to the
    automatic split, so that all events of one group land in one partition.
    """
    data = config["data"]
    split_config = config["split"] if config.has_section("split") else {}

    def fraction(name: str, default: float) -> float:
        return float(split_config[name]) if name in split_config else default

    split_seed = int(split_config.get("seed", seed))
    events_dir = Path(data["events_dir"].strip())
    catalog = load_events(events_dir, pbc_override=parse_pbc_override(data.get("pbc_override")))
    split, provenance = resolve_split(
        events_dir,
        catalog,
        auto=parse_bool(data.get("auto_split", "false")),
        train=fraction("train", 0.7),
        val=fraction("val", 0.15),
        test=fraction("test", 0.15),
        seed=split_seed,
        groups=groups,
        regenerate=parse_bool(data.get("regenerate_split", "false")),
    )
    provenance.setdefault("split_seed", split_seed)
    return catalog, split, provenance
