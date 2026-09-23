from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ase.io import read, write

SEP = "=" * 50
REQUIRED_ROLES = ["reactant", "product"]
OPTIONAL_ROLES = ["saddle"]
ROLES = [*REQUIRED_ROLES, *OPTIONAL_ROLES]
EVENT_RE = re.compile(r"(reactant|product|saddle)_(\d+)\.con")


def process_basin(
    state_id: int,
    event_id: int,
    events_output_dir: Path,
    output_format: str = "traj",
) -> tuple[list[dict[str, Any]], int]:
    """Process one EON state's events (``<state_id>/procdata``)."""
    if output_format != "traj":
        raise ValueError("only 'traj' output is supported: extxyz does not carry ASE constraints")
    procdata = Path(str(state_id)) / "procdata"
    if not procdata.is_dir():
        print(f"Skipping state {state_id}: missing procdata")
        return [], event_id

    # Group .con files by event number: {num: {role: path}}
    by_event: dict[int, dict[str, Path]] = {}
    for f in procdata.glob("*.con"):
        m = EVENT_RE.match(f.name)
        if m:
            by_event.setdefault(int(m.group(2)), {})[m.group(1)] = f

    if not by_event:
        print(f"State {state_id}: no matching events")
        return [], event_id

    records: list[dict[str, Any]] = []
    for local_num in sorted(by_event):
        files = by_event[local_num]
        missing = [r for r in REQUIRED_ROLES if r not in files]
        if missing:
            print(f"State {state_id}, process {local_num}: missing {missing}")
            continue
        try:
            roles = [*REQUIRED_ROLES, *[r for r in OPTIONAL_ROLES if r in files]]
            frames = [read(files[r]) for r in roles]
            # Provenance travels with the frames, so no side table is needed.
            for frame, role in zip(frames, roles):
                frame.info.update(
                    {
                        "dataset_kind": "eon_akmc_event",
                        "source_state": int(state_id),
                        "source_process": int(local_num),
                        "source_file": str(files[role]),
                        "role": frame.info.get("role", role) if role == "reactant" else role,
                    }
                )
            out_name = f"event_{event_id}.{output_format}"
            write(events_output_dir / out_name, frames)
            records.append(
                {"event_id": event_id, "state": state_id, "process": local_num, "file": out_name}
            )
            print(f"√ state {state_id}, process {local_num} → event {event_id}")
            event_id += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"× state {state_id}, process {local_num}: {e}")

    return records, event_id


def write_group_table(all_events: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write ``basin_table.csv`` mapping each event to its source state.

    The table is an evaluation side file: it says which recorded steps belong to
    the same EON state, so a candidate can be scored against all known products of
    that state. Training never reads it.
    """
    path = Path(output_dir) / "basin_table.csv"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing group table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "basin"])
        for record in all_events:
            writer.writerow([record["event_id"], record["state"]])
    print(f"\n🗂  Group table saved: {path}")
    return path


def generate_summary(all_events: list[dict[str, Any]]) -> None:
    """Print processing summary."""
    if not all_events:
        print("\n No events processed")
        return
    counts = Counter(e["state"] for e in all_events)
    print(f"\n{SEP}\n Summary — {len(all_events)} events, {len(counts)} states\n{'-' * 30}")
    for state in sorted(counts):
        print(f"State {state}: {counts[state]} events")
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert basin .con files to BasinFlow events")
    parser.add_argument(
        "work_dir", nargs="?", default=".",
        help="root directory containing integer-named basin folders",
    )
    parser.add_argument(
        "--basin-table",
        action="store_true",
        help=(
            "also write basin_table.csv (event_id,basin), mapping every event to its "
            "source EON state; used by evaluation to score against a whole basin"
        ),
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
        records, counter = process_basin(
            bid,
            counter,
            out_dir,
            output_format="traj",
        )
        all_events.extend(records)

    if all_events:
        if args.basin_table:
            write_group_table(all_events, out_dir)
        generate_summary(all_events)
    else:
        print("\n No events processed")


if __name__ == "__main__":
    main()
