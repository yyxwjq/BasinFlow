from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ase.io import read, write

SEP = "=" * 50
ROLES = ["reactant", "product", "saddle"]
EVENT_RE = re.compile(r"(reactant|product|saddle)_(\d+)\.con")


def process_basin(
    basin_id: int,
    event_counter: int,
    events_output_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Process events under a single basin (``<basin_id>/procdata``)."""
    procdata = Path(str(basin_id)) / "procdata"
    if not procdata.is_dir():
        print(f"Skipping basin {basin_id}: missing procdata")
        return [], event_counter

    # Group .con files by event number: {num: {role: path}}
    by_event: dict[int, dict[str, Path]] = {}
    for f in procdata.glob("*.con"):
        m = EVENT_RE.match(f.name)
        if m:
            by_event.setdefault(int(m.group(2)), {})[m.group(1)] = f

    if not by_event:
        print(f"Basin {basin_id}: no matching events")
        return [], event_counter

    basin_events: list[dict[str, Any]] = []
    for local_num in sorted(by_event):
        files = by_event[local_num]
        missing = [r for r in ROLES if r not in files]
        if missing:
            print(f"Basin {basin_id}, Event {local_num}: missing {missing}")
            continue
        try:
            frames = [read(files[r]) for r in ROLES]
            out_name = f"event_{event_counter}.extxyz"
            write(events_output_dir / out_name, frames)
            basin_events.append(
                {"global_event": event_counter, "local_event": local_num,
                 "basin": basin_id, "file": out_name}
            )
            print(f"√ Basin {basin_id}, Event {local_num} → Global {event_counter}")
            event_counter += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"× Basin {basin_id}, Event {local_num}: {e}")

    return basin_events, event_counter


def save_basin_table(all_events: list[dict[str, Any]], output_dir: Path) -> None:
    """Write basin mapping CSV."""
    path = output_dir / "basin_table.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["global_event", "local_event", "basin", "file"])
        w.writeheader()
        w.writerows(all_events)
    print(f"\n📋 Basin table saved: {path}")


def generate_summary(all_events: list[dict[str, Any]]) -> None:
    """Print processing summary."""
    if not all_events:
        print("\n No events processed")
        return
    counts = Counter(e["basin"] for e in all_events)
    print(f"\n{SEP}\n Summary — {len(all_events)} events, {len(counts)} basins\n{'-' * 30}")
    for basin in sorted(counts):
        print(f"Basin {basin}: {counts[basin]} events")
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert basin .con files to extxyz events")
    parser.add_argument(
        "work_dir", nargs="?", default=".",
        help="root directory containing integer-named basin folders",
    )
    args = parser.parse_args()

    root = Path(args.work_dir).resolve()
    print(f"{root}")

    out_dir = root / "events"
    out_dir.mkdir(exist_ok=True)

    basins = sorted(int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit())
    if not basins:
        print("No integer-named basin folders found")
        return
    print(f"{len(basins)} basins: {basins}")

    all_events: list[dict[str, Any]] = []
    counter = 0
    for bid in basins:
        print(f"\n{SEP}\n Basin {bid}\n{SEP}")
        records, counter = process_basin(bid, counter, out_dir)
        all_events.extend(records)

    if all_events:
        save_basin_table(all_events, out_dir)
        generate_summary(all_events)
    else:
        print("\n No events processed")


if __name__ == "__main__":
    main()
