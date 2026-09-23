"""One-off migration: legacy ``split_manifest.json`` to the new ``split.csv``.

Earlier datasets stored their partitions as a JSON list of *basin* ids next to a
``basin_table.csv`` that mapped every event file to one basin. The data layer no
longer reads either file: a dataset now carries a three-column ``split.csv`` of
integer event ids inside its events directory (:mod:`basinflow.data.splits`).

This tool performs that conversion once per dataset directory, so an existing
export can be used without re-running its converter. It never modifies the legacy
files and refuses to overwrite an existing ``split.csv`` unless ``--force``.

Usage::

    python tools/migrate_split_manifest.py /path/to/dataset            # dry run
    python tools/migrate_split_manifest.py /path/to/dataset --write
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.data.events import EVENT_FILE_RE  # noqa: E402
from basinflow.data.splits import EventSplit, save_split  # noqa: E402

PARTITIONS = ("train", "val", "test")


def _event_id(file_name: str) -> str:
    match = EVENT_FILE_RE.match(Path(file_name).stem)
    if match is None:
        raise ValueError(f"cannot read an integer event id from {file_name!r}")
    return str(int(match.group("index")))


def _basin_to_event(events_dir: Path, table_path: Path) -> dict[str, str]:
    """Map every legacy basin id to the event id of its file."""
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = {"file", "basin"} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{table_path} is missing columns: {sorted(missing)}")
    return {row["basin"]: _event_id(row["file"]) for row in rows}


def _basin_to_event_from_names(events_dir: Path, basin_ids: list[str]) -> dict[str, str]:
    """Fallback when there is no basin table: the basin id is the event index."""
    found = {}
    for basin_id in basin_ids:
        if not re.fullmatch(r"\d+", str(basin_id)):
            raise ValueError(
                f"no basin table and basin id {basin_id!r} is not an integer event index"
            )
        path = next(events_dir.glob(f"event_{int(basin_id)}.*"), None) or next(
            events_dir.glob(f"event_{int(basin_id):05d}.*"), None
        )
        if path is None:
            raise ValueError(f"basin {basin_id} has no event file in {events_dir}")
        found[str(basin_id)] = _event_id(path.name)
    return found


def migrate(
    dataset_dir: Path, *, write: bool, force: bool, manifest_path: Path | None = None
) -> dict[str, object]:
    events_dir = dataset_dir / "events"
    if not events_dir.is_dir():
        events_dir = dataset_dir if any(dataset_dir.glob("event_*")) else events_dir
    manifest_path = manifest_path or (dataset_dir / "split_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no split_manifest.json in {dataset_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_path = events_dir / "basin_table.csv"
    listed = [str(b) for name in PARTITIONS for b in manifest.get(name, [])]
    if table_path.is_file():
        mapping = _basin_to_event(events_dir, table_path)
    else:
        mapping = _basin_to_event_from_names(events_dir, listed)

    unknown = [basin for basin in listed if basin not in mapping]
    if unknown:
        raise ValueError(f"{len(unknown)} manifest basins have no event file: {unknown[:5]}")
    split = EventSplit(
        train_ids=tuple(mapping[b] for b in manifest.get("train", [])),
        val_ids=tuple(mapping[b] for b in manifest.get("val", [])),
        test_ids=tuple(mapping[b] for b in manifest.get("test", [])),
    )
    assigned = set(split.ids("all"))
    all_events = {_event_id(path.name) for path in events_dir.glob("event_*")}
    unassigned = sorted(all_events - assigned, key=int)
    target = events_dir / "split.csv"
    report = {
        "dataset": str(dataset_dir),
        "events_dir": str(events_dir),
        "num_events": len(all_events),
        "train": len(split.train_ids),
        "val": len(split.val_ids),
        "test": len(split.test_ids),
        "unassigned": len(unassigned),
        "split_csv": str(target),
    }
    if write:
        if target.exists() and not force:
            raise FileExistsError(f"{target} already exists; pass --force to overwrite")
        save_split(split, target)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="directory holding events/ and split_manifest.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest to read when it does not sit inside dataset_dir (Au/Pt keep theirs under a run directory)",
    )
    parser.add_argument("--write", action="store_true", help="write split.csv (default is a dry run)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split.csv")
    args = parser.parse_args()

    report = migrate(
        args.dataset_dir, write=args.write, force=args.force, manifest_path=args.manifest
    )
    action = "wrote" if args.write else "would write"
    print(
        f"{report['dataset']}: {report['num_events']} events, "
        f"train={report['train']} val={report['val']} test={report['test']}, "
        f"unassigned={report['unassigned']} -- {action} {report['split_csv']}"
    )


if __name__ == "__main__":
    main()
