"""Event-level train/val/test assignment, stored as ``split.csv`` next to events.

A split is a plain ``{partition: tuple[event_id, ...]}`` mapping. Every event is
assigned as a whole to exactly one partition; empty partitions are allowed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from basinflow.data.events import EventCatalog, ordered_event_ids

SPLIT_NAME = "split.csv"
PARTITIONS = ("train", "val", "test")
Split = dict[str, tuple[str, ...]]

def check_split(split: Mapping[str, Iterable[str]]) -> Split:
    """Validate, freeze and return a split as tuples of event ids."""
    if set(split) != set(PARTITIONS):
        raise ValueError(f"split must have exactly the partitions {PARTITIONS}")
    checked: Split = {}
    assigned: set[str] = set()
    for name in PARTITIONS:
        identifiers = tuple(str(value) for value in split[name])
        if any(not identifier.strip() for identifier in identifiers):
            raise ValueError(f"{name} contains an empty event id")
        seen: set[str] = set()
        repeated: list[str] = []
        for identifier in identifiers:
            if identifier in seen or identifier in assigned:
                repeated.append(identifier)
            seen.add(identifier)
        if repeated:
            raise ValueError(f"split event ids repeat across partitions: {sorted(set(repeated))}")
        checked[name] = identifiers
        assigned.update(identifiers)
    return checked


def partition(split: Split, name: str) -> tuple[str, ...]:
    """The ids of one partition."""
    if name not in PARTITIONS:
        raise ValueError(f"unknown partition {name!r}; expected one of {PARTITIONS}")
    return split[name]


def split_events(
    catalog: EventCatalog,
    *,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Split:
    """Split events by fraction, deterministically for a given ``seed``."""
    total = float(train + val + test)
    if total <= 0.0 or min(train, val, test) < 0.0:
        raise ValueError("split fractions must be non-negative and sum to > 0")
    count = len(catalog)
    train_count = max(0, min(count, int(count * train / total + 0.5)))
    val_count = max(0, min(count - train_count, int(count * val / total + 0.5)))
    shuffled = ordered_event_ids(catalog)
    np.random.default_rng(seed).shuffle(shuffled)
    return check_split(
        {
            "train": shuffled[:train_count],
            "val": shuffled[train_count : train_count + val_count],
            "test": shuffled[train_count + val_count :],
        }
    )


def split_path(events_dir: str | Path) -> Path:
    """Location of the split file for an events directory."""
    return Path(events_dir) / SPLIT_NAME


def save_split(split: Mapping[str, Iterable[str]], path: str | Path) -> Path:
    """Write ``split`` as the three-column CSV, padding short columns with blanks."""
    checked = check_split(split)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PARTITIONS)
        for row in range(max(len(ids) for ids in checked.values())):
            writer.writerow(
                [checked[name][row] if row < len(checked[name]) else "" for name in PARTITIONS]
            )
    return output


def load_split(path: str | Path, catalog: EventCatalog) -> Split:
    """Read a split CSV and check it against ``catalog``."""
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"split file {source} is empty")
    if [cell.strip() for cell in rows[0]] != list(PARTITIONS):
        raise ValueError(
            f"split file {source} must start with the header {','.join(PARTITIONS)}, "
            f"got {rows[0]}"
        )
    columns: dict[str, list[str]] = {name: [] for name in PARTITIONS}
    for row in rows[1:]:
        for name, cell in zip(PARTITIONS, (row + [""] * len(PARTITIONS))[: len(PARTITIONS)]):
            if cell.strip():
                columns[name].append(cell.strip())
    known = set(catalog.events)
    for name, identifiers in columns.items():
        invalid = [identifier for identifier in identifiers if not identifier.isdigit()]
        if invalid:
            raise ValueError(f"split file {source} must name events by integer id: {invalid[:5]}")
        unknown = [identifier for identifier in identifiers if identifier not in known]
        if unknown:
            raise ValueError(f"split file {source} names {len(unknown)} unknown {name} events: {unknown[:5]}")
    return check_split(columns)


def resolve_split(
    events_dir: str | Path,
    catalog: EventCatalog,
    *,
    auto: bool,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    regenerate: bool = False,
    log: Any = print,
) -> tuple[Split, dict[str, Any]]:
    """Load or create the split for an events directory.

    ``auto=True`` writes ``split.csv`` when needed; an existing file is reused
    unless ``regenerate`` is set. ``auto=False`` reads the existing file, and
    falls back to "everything in train" when there is none.
    """
    directory = Path(events_dir)
    target = split_path(directory)
    provenance: dict[str, Any] = {"events_dir": str(directory)}

    if auto and target.is_file() and not regenerate:
        split = load_split(target, catalog)
        provenance.update({"split_file": str(target), "split_mode": "reused"})
        log(f"[data] auto_split=true and {target} exists; reusing it (delete it to resplit)")
        return split, provenance
    if auto:
        split = split_events(catalog, train=train, val=val, test=test, seed=seed)
        save_split(split, target)
        provenance.update(
            {
                "split_file": str(target),
                "split_mode": "regenerated" if regenerate else "created",
                "split_seed": int(seed),
                "split_fractions": {"train": train, "val": val, "test": test},
            }
        )
        log(
            f"[data] auto_split=true: wrote {target} "
            f"({_counts(split)})"
        )
        return split, provenance
    if target.is_file():
        split = load_split(target, catalog)
        provenance.update({"split_file": str(target), "split_mode": "loaded"})
        log(f"[data] auto_split=false: using {target} ({_counts(split)})")
        return split, provenance

    split = check_split(
        {"train": tuple(ordered_event_ids(catalog)), "val": (), "test": ()}
    )
    provenance.update({"split_file": None, "split_mode": "all_train"})
    log(
        f"[data] auto_split=false and no {target}: "
        f"all {len(split['train'])} events are used for training"
    )
    return split, provenance


def _counts(split: Split) -> str:
    return " ".join(f"{name}={len(split[name])}" for name in PARTITIONS)
