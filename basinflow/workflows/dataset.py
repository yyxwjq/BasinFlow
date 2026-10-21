"""Resolve a workflow's data source: one event directory plus its ``split.csv``.

``[data] events_dir`` names the directory of event files, and each file is one
event. ``[data] auto_split`` decides where the train/val/test assignment comes
from:

* ``true`` -- split the events by the ``[split]`` fractions and ``seed``, writing
  ``split.csv`` into the events directory. An existing file is reused unless
  ``[data] regenerate_split = true``.
* ``false`` -- read the existing ``split.csv``; when there is none, every event is
  used for training.
"""
from __future__ import annotations

from pathlib import Path

from basinflow.data import load_events, resolve_split
from basinflow.workflows.config import parse_bool


def load_configured_dataset(config, *, seed: int):
    """Return ``(catalog, split, provenance)`` for a parsed workflow config."""
    data = config["data"]
    split_config = config["split"] if config.has_section("split") else {}

    def fraction(name: str, default: float) -> float:
        return float(split_config[name]) if name in split_config else default

    split_seed = int(split_config.get("seed", seed))
    events_dir = Path(data["events_dir"].strip())
    catalog = load_events(events_dir)
    split, provenance = resolve_split(
        events_dir,
        catalog,
        auto=parse_bool(data.get("auto_split", "false")),
        train=fraction("train", 0.7),
        val=fraction("val", 0.15),
        test=fraction("test", 0.15),
        seed=split_seed,
        regenerate=parse_bool(data.get("regenerate_split", "false")),
    )
    provenance.setdefault("split_seed", split_seed)
    return catalog, split, provenance
