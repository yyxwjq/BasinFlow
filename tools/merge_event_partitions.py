"""Merge a legacy three-directory event dataset into the canonical single-directory layout.

Earlier Transition1x preparation wrote ``<root>/{train,val,test}/event_*.traj``,
each with its own ``basin_table.csv``. The data layer supports one source mode
only -- a single ``events_dir`` of ``event_<int>`` files plus a ``split.csv`` of
integer event ids -- so those datasets have to be materialised into that shape
once.

Event files are hard-linked by default, so the merge costs no extra disk space and
stays on the same filesystem. Event ids are renumbered ``0..N-1`` in
train-then-val-then-test order; ``split.csv`` records which id belongs where, and
the original source index and partition stay in every frame's ``info``.

    python tools/merge_event_partitions.py <dataset_root> [--output DIR] [--copy]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

PARTITIONS = ("train", "val", "test")


def merge_event_partitions(
    root: str | Path,
    output: str | Path | None = None,
    *,
    partitions: tuple[str, ...] = PARTITIONS,
    copy: bool = False,
) -> dict:
    """Build ``<output>/events`` with integer-named files and a ``split.csv``."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    output = Path(output) if output is not None else root.with_name(root.name + "_merged")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    files_by_partition: dict[str, list[str]] = {}
    for partition in partitions:
        table = root / partition / "basin_table.csv"
        if not table.is_file():
            raise FileNotFoundError(f"partition table not found: {table}")
        with table.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        names = [(row.get("file") or "").strip() for row in rows]
        if any(not name for name in names):
            raise ValueError(f"{table} has a row without a file name")
        files_by_partition[partition] = names

    seen: set[str] = set()
    for partition in partitions:
        for name in files_by_partition[partition]:
            if name in seen:
                raise ValueError(f"event file {name} appears in more than one partition")
            seen.add(name)

    events_dir = output / "events"
    events_dir.mkdir(parents=True)
    split_columns: dict[str, list[str]] = {partition: [] for partition in PARTITIONS}
    next_id = 0
    for partition in partitions:
        for name in files_by_partition[partition]:
            event_id = str(next_id)
            next_id += 1
            target = events_dir / f"event_{event_id}{Path(name).suffix}"
            source = root / partition / name
            if copy:
                shutil.copy2(source, target)
            else:
                os.link(source, target)
            split_columns[partition].append(event_id)

    from basinflow.data.splits import EventSplit, save_split

    split = EventSplit(
        train_ids=tuple(split_columns["train"]),
        val_ids=tuple(split_columns["val"]),
        test_ids=tuple(split_columns["test"]),
    )
    save_split(split, events_dir / "split.csv")

    from basinflow.data import load_events, load_split, resolve_split

    catalog = load_events(events_dir)
    loaded = load_split(events_dir / "split.csv", catalog)
    if loaded != split:
        raise ValueError("split.csv does not round-trip through the loader")
    if len(catalog) != next_id:
        raise ValueError(f"merged directory has {len(catalog)} events, expected {next_id}")
    _, provenance = resolve_split(events_dir, catalog, auto=False)
    counts = {partition: len(split.partition(partition)) for partition in PARTITIONS}
    return {
        "output": str(output),
        "events_dir": str(events_dir),
        "split_file": str(events_dir / "split.csv"),
        "num_events": len(catalog),
        "split": counts,
        "source_counts": {name: len(files) for name, files in files_by_partition.items()},
        "split_mode": provenance["split_mode"],
        "linked": not copy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="dataset root containing train/ val/ test/")
    parser.add_argument("--output", help="new output directory (default: <root>_merged)")
    parser.add_argument("--copy", action="store_true", help="copy instead of hard-linking")
    args = parser.parse_args()
    report = merge_event_partitions(args.root, args.output, copy=args.copy)
    print(
        f"{report['output']}: {report['split']} events "
        f"(from {report['source_counts']}); wrote {report['split_file']}"
    )


if __name__ == "__main__":
    main()
