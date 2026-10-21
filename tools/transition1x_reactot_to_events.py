"""Convert the official ReactOT bridge pickles into a BasinFlow events directory.

Input: the Zenodo release 13131875 files ``train_rpsb_all.pkl`` and
``valid_rpsb_all.pkl`` (``README.md`` in the reference repo). Both files hold the
*same* 10073 Transition1x rows; they differ only in ``use_ind``, the row list the
reference trains on (9000) and evaluates on (1073). That was verified on the
downloaded files: the two ``use_ind`` sets are disjoint and their union is every
row, and the coordinates agree row by row.

Output: one flat directory of ``event_<ordinal>.traj`` files plus ``split.csv``,
which is the layout the data layer reads
(``basinflow/data/events.load_events`` + ``splits.load_split``).

Conversion rules, all forced by the reference implementation
------------------------------------------------------------
* **Frame order is R, P, TS.** The event contract is
  ``("reactant", "product", "transition_state")``
  (``basinflow/data/events.FRAME_ROLES``), while ReactOT's internal fragment
  order for this task is ``["R", "TS", "P"]``. The mapping is by *name*; the
  on-disk order must not follow the reference's.
* **Per-frame centroid centering, no alignment.** ReactOT's dataset centers each
  frame at its own centroid (``base_dataset.process_molecules``, ``center=True``)
  and never Kabsch-aligns product to reactant, so the bridge prior ``(R + P) / 2``
  is the reference's. There is no ``--align-product-to-reactant`` option.
* **``movable_mask`` is all True.** ReactOT's ``fixed_idx = [0, 2]`` clamps the R
  and P *fragments* inside the model; it is not a data-side constraint. Writing it
  into ``movable_mask`` would make the loss and the sampler treat every atom as
  fixed.
* **No cell, no PBC.** Transition1x is molecular, so ``pbc=False`` and a zero
  cell. The config must not set ``pbc_override``.
* **``split.csv`` holds the exported ordinals, not source row indices.** Each
  event file is named by its position in the export, and that integer is what
  ``split.csv`` stores.

Usage::

    python tools/transition1x_reactot_to_events.py \\
        train_rpsb_all.pkl valid_rpsb_all.pkl /path/to/dataset/events
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write

#: BasinFlow frame order. ReactOT's ``fragment_names`` is ["R", "TS", "P"].
FRAME_ROLES = ("reactant", "product", "transition_state")
REQUIRED_FIELDS = ("num_atoms", "charges", "positions")
REQUIRED_TOP_LEVEL = ("reactant", "product", "transition_state", "single_fragment")
SPLIT_HEADER = ("train", "val", "test")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ReactOT pickle not found: {path}")
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a dictionary")
    for key in REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise ValueError(f"{path} is missing top-level key {key!r}")
    total = len(raw["single_fragment"])
    for role in FRAME_ROLES:
        frame = raw[role]
        if not isinstance(frame, dict):
            raise ValueError(f"{path} frame {role!r} must be a dictionary")
        for field in REQUIRED_FIELDS:
            if field not in frame:
                raise ValueError(f"{path} frame {role!r} is missing {field!r}")
            if len(frame[field]) != total:
                raise ValueError(
                    f"{path} frame {role!r} field {field!r} has {len(frame[field])} rows, "
                    f"expected {total}"
                )
    return raw


def _row_indices(raw: dict[str, Any], path: Path) -> list[int]:
    indices = [int(value) for value in raw["use_ind"]]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{path} 'use_ind' contains duplicate rows")
    total = len(raw["single_fragment"])
    invalid = [index for index in indices if index < 0 or index >= total]
    if invalid:
        raise ValueError(f"{path} 'use_ind' has out-of-range rows: {invalid[:5]}")
    return indices


def _symbols(charges: Iterable[int], *, row: int, role: str) -> list[str]:
    symbols = []
    for charge in charges:
        atomic_number = int(charge)
        if atomic_number <= 0 or atomic_number >= len(chemical_symbols):
            raise ValueError(f"row {row} frame {role}: invalid atomic number {atomic_number}")
        symbols.append(chemical_symbols[atomic_number])
    return symbols


def _frame_atoms(
    raw: dict[str, Any],
    *,
    role: str,
    row: int,
    metadata: dict[str, Any],
    center: bool,
) -> Atoms:
    frame = raw[role]
    n_atoms = int(frame["num_atoms"][row])
    charges = list(frame["charges"][row])[:n_atoms]
    positions = np.asarray(frame["positions"][row], dtype=float)[:n_atoms]
    if positions.shape != (n_atoms, 3):
        raise ValueError(
            f"row {row} frame {role}: positions must be ({n_atoms}, 3), got {positions.shape}"
        )
    if len(charges) != n_atoms:
        raise ValueError(f"row {row} frame {role}: charges length {len(charges)} != {n_atoms}")
    if center:
        positions = positions - positions.mean(axis=0, keepdims=True)
    atoms = Atoms(
        symbols=_symbols(charges, row=row, role=role), positions=positions, pbc=False
    )
    atoms.info.update({**metadata, "role": role})
    return atoms


def _metadata(raw: dict[str, Any], row: int, *, dataset: str, source: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_dataset": dataset,
        "source_file": source,
        "source_index": int(row),
        "dataset_kind": "reactot_transition1x_bridge",
        "coordinate_convention": "per_frame_centroid_centered",
        "single_fragment": bool(raw["single_fragment"][row]),
        "movable_mask_source": "all_atoms_movable",
    }
    for key in ("rxn", "formula"):
        values = raw["reactant"].get(key)
        if values is not None:
            value = values[row]
            if value is not None:
                metadata[key] = str(value)
    return metadata


def _write_split(path: Path, partitions: dict[str, list[str]]) -> None:
    width = max((len(values) for values in partitions.values()), default=0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(SPLIT_HEADER))
        for index in range(width):
            writer.writerow(
                [partitions.get(name, [])[index] if index < len(partitions.get(name, [])) else ""
                 for name in SPLIT_HEADER]
            )


def convert(
    train_pickle: str | Path,
    valid_pickle: str | Path,
    output_events_dir: str | Path,
    *,
    center: bool = True,
    single_fragment_only: bool = False,
) -> dict[str, Any]:
    """Write the ReactOT bridge dataset and return a provenance report."""
    train_path, valid_path = Path(train_pickle), Path(valid_pickle)
    train_raw, valid_raw = _load(train_path), _load(valid_path)
    total = len(train_raw["single_fragment"])
    if len(valid_raw["single_fragment"]) != total:
        raise ValueError("the two pickles hold a different number of rows")

    train_rows, valid_rows = _row_indices(train_raw, train_path), _row_indices(valid_raw, valid_path)
    overlap = sorted(set(train_rows) & set(valid_rows))
    if overlap:
        raise ValueError(f"the two use_ind sets overlap at rows: {overlap[:5]}")
    union = sorted(set(train_rows) | set(valid_rows))
    if union != list(range(total)):
        missing = sorted(set(range(total)) - set(union))
        raise ValueError(
            f"use_ind does not cover every row (total={total}, covered={len(union)}); "
            f"first missing rows: {missing[:5]}"
        )

    if single_fragment_only:
        train_rows = [row for row in train_rows if train_raw["single_fragment"][row] == 1]
        valid_rows = [row for row in valid_rows if valid_raw["single_fragment"][row] == 1]

    events_dir = Path(output_events_dir)
    if events_dir.exists() and any(events_dir.glob("event_*")):
        raise FileExistsError(f"{events_dir} already holds event files; refusing to overwrite")
    events_dir.mkdir(parents=True, exist_ok=True)

    # Export order: train rows first, then the set-aside test rows, so the ordinals
    # and the split columns are produced by one pass and cannot drift.
    partitions = {"train": train_rows, "test": valid_rows}
    ordinal_of: dict[tuple[str, int], str] = {}
    rows_report: list[dict[str, str]] = []
    ordinal = 0
    for partition in ("train", "test"):
        for row in partitions[partition]:
            metadata = _metadata(train_raw, row, dataset="ReactOT", source=str(train_path))
            frames = [
                _frame_atoms(train_raw, role=role, row=row, metadata=metadata, center=center)
                for role in FRAME_ROLES
            ]
            reference_species = frames[0].get_chemical_symbols()
            for frame in frames[1:]:
                if frame.get_chemical_symbols() != reference_species:
                    raise ValueError(f"row {row}: the three frames do not share a species order")
            destination = events_dir / f"event_{ordinal}.traj"
            write(destination, frames)
            ordinal_of[(partition, row)] = str(ordinal)
            rows_report.append(
                {
                    "event_id": str(ordinal),
                    "partition": partition,
                    "source_index": str(row),
                    "single_fragment": str(int(train_raw["single_fragment"][row])),
                }
            )
            ordinal += 1

    split = {
        "train": [ordinal_of[("train", row)] for row in train_rows],
        "val": [],
        "test": [ordinal_of[("test", row)] for row in valid_rows],
    }
    _write_split(events_dir / "split.csv", split)

    return {
        "events_dir": str(events_dir),
        "num_events": len(rows_report),
        "split": {name: len(values) for name, values in split.items()},
        "center": bool(center),
        "single_fragment_only": bool(single_fragment_only),
        "coordinate_convention": "per_frame_centroid_centered" if center else "source_coordinates",
        "source_files": {"train": str(train_path), "valid": str(valid_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_pickle", type=Path, help="official train_rpsb_all.pkl")
    parser.add_argument("valid_pickle", type=Path, help="official valid_rpsb_all.pkl (the 1073 set-aside rows)")
    parser.add_argument("output_events_dir", type=Path, help="new directory for event_*.traj and split.csv")
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="keep the source coordinates; the reference always centers per frame, so this is a diagnostic arm",
    )
    parser.add_argument("--single-fragment-only", action="store_true")
    args = parser.parse_args()
    report = convert(
        args.train_pickle,
        args.valid_pickle,
        args.output_events_dir,
        center=not args.no_center,
        single_fragment_only=args.single_fragment_only,
    )
    print(
        f"wrote {report['num_events']} events to {report['events_dir']} "
        f"(train={report['split']['train']} val={report['split']['val']} test={report['split']['test']}, "
        f"{report['coordinate_convention']})"
    )


if __name__ == "__main__":
    main()
