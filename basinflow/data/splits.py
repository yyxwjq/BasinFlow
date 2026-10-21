"""Event-level train/val/test selection, stored as ``split.csv`` next to the events.

The split is a three-column CSV of integer event ids -- ``train,val,test`` -- in
:func:`basinflow.data.events.ordered_event_ids` order, and each event is assigned
to a partition as a whole.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from basinflow.data.events import EventCatalog, ordered_event_ids, select_events

SPLIT_NAME = "split.csv"
PARTITIONS = ("train", "val", "test")


@dataclass(frozen=True)
class EventSplit:
    """Ordered event ids per partition, with empty partitions allowed."""

    train_ids: tuple[str, ...] = field(default_factory=tuple)
    val_ids: tuple[str, ...] = field(default_factory=tuple)
    test_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        assigned: set[str] = set()
        for partition in PARTITIONS:
            identifiers = getattr(self, partition + "_ids")
            if isinstance(identifiers, str) or not isinstance(identifiers, (list, tuple)):
                raise ValueError(f"{partition}_ids must be a list or tuple of event ids")
            if any(not isinstance(value, str) or not value.strip() for value in identifiers):
                raise ValueError(f"{partition}_ids must contain only nonempty string event ids")
            unique = set(identifiers)
            if len(unique) != len(identifiers):
                raise ValueError(f"{partition}_ids contains duplicate event ids")
            overlap = assigned & unique
            if overlap:
                raise ValueError(f"split partitions overlap at event ids: {sorted(overlap)}")
            assigned.update(unique)
            object.__setattr__(self, partition + "_ids", tuple(identifiers))

    def partition(self, name: str) -> tuple[str, ...]:
        if name not in PARTITIONS:
            raise ValueError(f"unknown partition {name!r}; expected one of {PARTITIONS}")
        return getattr(self, name + "_ids")

    def ids(self, name: str) -> tuple[str, ...]:
        """Partition contents; ``"all"`` returns train + val + test."""
        if name == "all":
            return self.train_ids + self.val_ids + self.test_ids
        return self.partition(name)

    def select(self, catalog: EventCatalog, name: str) -> EventCatalog:
        return select_events(catalog, self.partition(name), name=name)


def split_events(
    catalog: EventCatalog,
    *,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> EventSplit:
    """Split events by fraction, deterministically for a given ``seed``."""
    counts = _partition_counts(len(catalog), train=train, val=val, test=test)
    shuffled = ordered_event_ids(catalog)
    np.random.default_rng(seed).shuffle(shuffled)
    train_count, val_count = counts
    return EventSplit(
        train_ids=tuple(shuffled[:train_count]),
        val_ids=tuple(shuffled[train_count : train_count + val_count]),
        test_ids=tuple(shuffled[train_count + val_count :]),
    )


def all_in_train(catalog: EventCatalog) -> EventSplit:
    """Every event in train, with empty val and test partitions."""
    return EventSplit(train_ids=tuple(ordered_event_ids(catalog)))


def _partition_counts(count: int, *, train: float, val: float, test: float) -> tuple[int, int]:
    """Rounded ``(train, val)`` counts; test takes the remainder."""
    total = float(train + val + test)
    if total <= 0.0:
        raise ValueError("train + val + test must be > 0")
    if any(value < 0.0 for value in (train, val, test)):
        raise ValueError("split fractions must be non-negative")
    train_count = max(0, min(count, int(count * train / total + 0.5)))
    val_count = max(0, min(count - train_count, int(count * val / total + 0.5)))
    return train_count, val_count


def split_path(events_dir: str | Path) -> Path:
    """Location of the split file for an events directory."""
    return Path(events_dir) / SPLIT_NAME


def save_split(split: EventSplit, path: str | Path) -> Path:
    """Write ``path`` as the three-column CSV, padding short columns with blanks."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: list(split.partition(name)) for name in PARTITIONS}
    width = max((len(values) for values in columns.values()), default=0)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(PARTITIONS))
        for row in range(width):
            writer.writerow(
                [columns[name][row] if row < len(columns[name]) else "" for name in PARTITIONS]
            )
    return output


def load_split(path: str | Path, catalog: EventCatalog) -> EventSplit:
    """Read the CSV and check it against ``catalog``.

    Events missing from the file are allowed: they are simply in no partition.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the header is not ``train,val,test``, if a partition repeats
            an id, or if the file names an id the catalog does not have.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"split file not found: {source}")
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration:
            raise ValueError(f"split file {source} is empty") from None
        if header != list(PARTITIONS):
            raise ValueError(
                f"split file {source} must start with the header {','.join(PARTITIONS)}, got {header}"
            )
        columns: dict[str, list[str]] = {name: [] for name in PARTITIONS}
        for row in reader:
            padded = list(row) + [""] * (len(PARTITIONS) - len(row))
            for name, cell in zip(PARTITIONS, padded):
                value = cell.strip()
                if value:
                    columns[name].append(value)
    known = set(catalog.events)
    for name in PARTITIONS:
        non_integer = [event_id for event_id in columns[name] if not event_id.isdigit()]
        if non_integer:
            raise ValueError(
                f"split file {source} must name events by integer id; "
                f"{name} contains {non_integer[:5]}"
            )
        unknown = [event_id for event_id in columns[name] if event_id not in known]
        if unknown:
            raise ValueError(
                f"split file {source} assigns {len(unknown)} unknown event ids to {name}: "
                f"{unknown[:5]}"
            )
    return EventSplit(
        train_ids=tuple(columns["train"]),
        val_ids=tuple(columns["val"]),
        test_ids=tuple(columns["test"]),
    )


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
) -> tuple[EventSplit, dict[str, Any]]:
    """Resolve the split for an events directory, creating ``split.csv`` if asked.

    ``auto=True`` splits the catalog and writes ``split.csv`` next to the events. An existing file is reused unless
    ``regenerate`` is set. ``auto=False`` reads the existing file and, when there is
    none, puts every event in train.
    """
    directory = Path(events_dir)
    target = split_path(directory)
    provenance: dict[str, Any] = {"events_dir": str(directory)}
    if auto:
        if target.is_file() and not regenerate:
            split = load_split(target, catalog)
            provenance.update({"split_file": str(target), "split_mode": "reused"})
            log(f"[data] auto_split=true and {target} exists; reusing it (delete it to resplit)")
            return split, provenance
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
            f"(train={len(split.train_ids)} val={len(split.val_ids)} test={len(split.test_ids)})"
        )
        return split, provenance

    if target.is_file():
        split = load_split(target, catalog)
        provenance.update({"split_file": str(target), "split_mode": "loaded"})
        log(
            f"[data] auto_split=false: using {target} "
            f"(train={len(split.train_ids)} val={len(split.val_ids)} test={len(split.test_ids)})"
        )
        return split, provenance

    split = all_in_train(catalog)
    provenance.update({"split_file": None, "split_mode": "all_train"})
    log(
        f"[data] auto_split=false and no {target}: "
        f"all {len(split.train_ids)} events are used for training"
    )
    return split, provenance
