"""Convert a Transition1x pickle into BasinFlow standard event files."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols
from ase.io import write


ROLES = ("reactant", "product", "transition_state")
REQUIRED_FIELDS = ("num_atoms", "charges", "positions")
ENERGY_KEY = "wB97x_6-31G(d).energy"
FORCES_KEY = "wB97x_6-31G(d).forces"


def _frame_value(frame: dict[str, Any], key: str, index: int, role: str) -> Any:
    if key not in frame:
        raise ValueError(f"Transition1x {role} frame is missing {key!r}")
    values = frame[key]
    if index >= len(values):
        raise ValueError(f"Transition1x {role} field {key!r} has no row {index}")
    return values[index]


def _validate_indices(indices: Any, total: int) -> list[int]:
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise ValueError("Transition1x source indices must be a sequence of integers")
    if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) for index in indices):
        raise ValueError("Transition1x source indices must contain only integers, not booleans or floats")
    indices = [int(index) for index in indices]
    if len(set(indices)) != len(indices):
        raise ValueError("Transition1x selection contains duplicate source indices")
    invalid = [index for index in indices if index < 0 or index >= total]
    if invalid:
        raise ValueError(f"Transition1x selection contains invalid source indices: {invalid[:5]}")
    return indices


def _selected_indices(
    raw: dict[str, Any], selection: str, *, single_fragment_only: bool = False,
    source_indices: list[int] | None = None,
) -> list[int]:
    total = len(raw["reactant"]["num_atoms"])
    if source_indices is not None:
        if selection != "use_ind":
            raise ValueError("explicit source indices cannot be combined with a nondefault selection")
        if not isinstance(source_indices, list):
            raise ValueError("Transition1x explicit source indices must be a list of integers")
        indices = _validate_indices(source_indices, total)
    elif selection in {"use_ind", "complement"}:
        if "use_ind" not in raw:
            raise ValueError("Transition1x pickle is missing 'use_ind'")
        indices = _validate_indices(raw["use_ind"], total)
        if selection == "complement":
            excluded = set(indices)
            indices = [index for index in range(total) if index not in excluded]
    elif selection == "all":
        indices = list(range(total))
    else:
        raise ValueError("selection must be 'use_ind', 'complement' or 'all'")
    if single_fragment_only:
        indices = [index for index in indices if raw["single_fragment"][index] == 1]
    return indices


def _load_source(path: Path) -> dict[str, Any]:
    with path.open("rb") as source_file:
        raw = pickle.load(source_file)
    if not isinstance(raw, dict):
        raise ValueError("Transition1x pickle must contain a dictionary")
    for role in ROLES:
        frame = raw.get(role)
        if not isinstance(frame, dict):
            raise ValueError(f"Transition1x pickle is missing frame dictionary {role!r}")
        for field in REQUIRED_FIELDS:
            if field not in frame:
                raise ValueError(f"Transition1x {role} frame is missing {field!r}")
    if "single_fragment" not in raw:
        raise ValueError("Transition1x pickle is missing 'single_fragment'")
    if len(raw["single_fragment"]) != len(raw["reactant"]["num_atoms"]):
        raise ValueError("Transition1x single_fragment must contain one value per source row")
    return raw


def _symbols(charges: Any, *, source_index: int, role: str) -> list[str]:
    symbols: list[str] = []
    for charge in charges:
        atomic_number = int(charge)
        if atomic_number <= 0 or atomic_number >= len(chemical_symbols):
            raise ValueError(
                f"Transition1x {role} row {source_index} has invalid atomic number {atomic_number}"
            )
        symbols.append(chemical_symbols[atomic_number])
    return symbols


def _metadata(raw: dict[str, Any], source_path: Path, source_index: int, selection: str, center: bool) -> dict[str, Any]:
    reactant = raw["reactant"]
    metadata: dict[str, Any] = {
        "source_dataset": "Transition1x",
        "source_file": str(source_path),
        "source_index": int(source_index),
        "selection": selection,
        "dataset_kind": "transition1x_single_event_molecule",
        "coordinate_convention": "per_frame_centroid_centered" if center else "source_coordinates",
        "single_fragment": bool(raw["single_fragment"][source_index]),
    }
    for key in ("rxn", "formula"):
        if key in reactant:
            value = _frame_value(reactant, key, source_index, "reactant")
            if value is not None:
                metadata[key] = str(value)
    return metadata


def _atoms(raw: dict[str, Any], *, role: str, source_index: int, metadata: dict[str, Any], center: bool) -> Atoms:
    frame = raw[role]
    n_atoms = int(_frame_value(frame, "num_atoms", source_index, role))
    charges = list(_frame_value(frame, "charges", source_index, role))[:n_atoms]
    positions = np.asarray(_frame_value(frame, "positions", source_index, role), dtype=float)[:n_atoms]
    if positions.shape != (n_atoms, 3):
        raise ValueError(
            f"Transition1x {role} row {source_index} positions must have shape ({n_atoms}, 3), got {positions.shape}"
        )
    if len(charges) != n_atoms:
        raise ValueError(f"Transition1x {role} row {source_index} charges have length {len(charges)}, expected {n_atoms}")
    if center:
        positions = positions - positions.mean(axis=0, keepdims=True)
    atoms = Atoms(symbols=_symbols(charges, source_index=source_index, role=role), positions=positions, pbc=False)
    atoms.info.update({**metadata, "role": role})
    calculator_data: dict[str, Any] = {}
    if ENERGY_KEY in frame:
        calculator_data["energy"] = float(_frame_value(frame, ENERGY_KEY, source_index, role))
    if FORCES_KEY in frame:
        forces = np.asarray(_frame_value(frame, FORCES_KEY, source_index, role), dtype=float)[:n_atoms]
        if forces.shape != (n_atoms, 3):
            raise ValueError(f"Transition1x {role} row {source_index} forces must have shape ({n_atoms}, 3), got {forces.shape}")
        calculator_data["forces"] = forces
    if calculator_data:
        atoms.calc = SinglePointCalculator(atoms, **calculator_data)
    return atoms


def _align_frames_to_reactant(reactant: Atoms, *others: Atoms) -> None:
    """Put every other frame into the reactant frame, each by its own Kabsch fit.

    The source frames do **not** share an orientation: OAReactDiff's
    preprocessing rotated each frame independently, which is why the relative
    R -> P rotation was measured as unrelated noise in two different copies of
    the same reaction (0% agreement, median 136 degrees; docs/18 section 8).
    There is therefore no path geometry to preserve, and each frame has to be
    aligned on its own.

    Aligning the transition state is what makes the R+P -> TS task well posed:
    left in its source frame the target carries an arbitrary rigid motion, so
    the raw per-atom RMS from the midpoint bridge is 3.7 A while the aligned
    distance is 0.49 A, and the model spends its capacity on the rotation.
    """
    if reactant.pbc.any() or any(frame.pbc.any() for frame in others):
        raise ValueError("frame alignment requires nonperiodic structures")
    reactant_center = reactant.positions.mean(axis=0)
    centered_reactant = reactant.positions - reactant_center
    for frame in others:
        frame_center = frame.positions.mean(axis=0)
        centered = frame.positions - frame_center
        left_vectors, _, right_vectors = np.linalg.svd(centered.T @ centered_reactant)
        correction = np.diag([1.0, 1.0, np.linalg.det(left_vectors @ right_vectors)])
        rotation = left_vectors @ correction @ right_vectors
        calculator_data = dict(frame.calc.results) if frame.calc is not None else {}
        if "forces" in calculator_data:
            calculator_data["forces"] = calculator_data["forces"] @ rotation
        frame.set_positions(centered @ rotation + reactant_center)
        if calculator_data:
            frame.calc = SinglePointCalculator(frame, **calculator_data)
        frame.info["coordinate_convention"] = "proper_kabsch_to_reactant"
        frame.info["frame_alignment_rotation"] = rotation.tolist()


def convert_transition1x(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    selection: str = "use_ind",
    source_indices: list[int] | None = None,
    single_fragment_only: bool = False,
    align_product_to_reactant: bool = False,
    center: bool = False,
    output_format: str = "traj",
) -> list[dict[str, str]]:
    """Write selected Transition1x rows as BasinFlow standard event files."""
    if output_format not in {"traj", "extxyz"}:
        raise ValueError("output_format must be 'traj' or 'extxyz'")
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Transition1x pickle not found: {source}")
    output = Path(output_dir)
    raw = _load_source(source)
    indices = _selected_indices(raw, selection, single_fragment_only=single_fragment_only, source_indices=source_indices)
    # Each target frame gets its own rigid transform onto the reactant, because
    # the source frames do not share an orientation. It is well defined for
    # multi-fragment records too, so the only requirement is a nonperiodic
    # structure (checked in _align_frames_to_reactant).
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for ordinal, source_index in enumerate(indices):
        file_name = f"event_{source_index}.{output_format}"
        destination = output / file_name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing event file: {destination}")
        metadata = _metadata(raw, source, source_index, "explicit_indices" if source_indices is not None else selection, center)
        metadata["single_fragment_only"] = bool(single_fragment_only)
        metadata["align_product_to_reactant"] = bool(align_product_to_reactant)
        if align_product_to_reactant:
            metadata["transition_state_usage"] = "proper_kabsch_to_reactant"
        frames = [
            _atoms(raw, role=role, source_index=source_index, metadata=metadata, center=center)
            for role in ROLES
        ]
        if any(frame.get_chemical_symbols() != frames[0].get_chemical_symbols() for frame in frames[1:]):
            raise ValueError(f"Transition1x row {source_index} frame species order differs")
        if align_product_to_reactant:
            _align_frames_to_reactant(*frames)
        write(destination, frames)
        rows.append(
            {
                "global_event": str(ordinal),
                "local_event": str(source_index),
                "basin": f"transition1x:{source_index}",
                "file": file_name,
                "dataset_kind": "transition1x_single_event_molecule",
                "source_index": str(source_index),
            }
        )
    table_path = output / "basin_table.csv"
    if table_path.exists():
        raise FileExistsError(f"refusing to overwrite existing basin table: {table_path}")
    with table_path.open("w", newline="", encoding="utf-8") as table_file:
        writer = csv.DictWriter(
            table_file,
            fieldnames=[
                "global_event",
                "local_event",
                "basin",
                "file",
                "dataset_kind",
                "source_index",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Transition1x pickle rows to BasinFlow event files.")
    parser.add_argument("pickle_path", help="Transition1x pickle input")
    parser.add_argument("output_dir", help="empty or new output events directory")
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--selection", choices=["use_ind", "complement", "all"], default="use_ind")
    selection_group.add_argument("--indices-json", type=Path, help="JSON list of explicit source row indices, in export order")
    parser.add_argument("--single-fragment-only", action="store_true", help="intersect selection with source single_fragment flags")
    parser.add_argument("--align-product-to-reactant", action="store_true", help="proper Kabsch alignment of single-fragment product targets to reactants")
    parser.add_argument("--center", action="store_true", help="center each frame at its centroid before writing")
    parser.add_argument("--format", dest="output_format", choices=["traj", "extxyz"], default="traj")
    args = parser.parse_args()
    source_indices = None
    if args.indices_json is not None:
        with args.indices_json.open(encoding="utf-8") as indices_file:
            source_indices = json.load(indices_file)
        if not isinstance(source_indices, list):
            parser.error("--indices-json must contain a JSON list of integer source indices")
    rows = convert_transition1x(
        args.pickle_path,
        args.output_dir,
        selection=args.selection,
        source_indices=source_indices,
        single_fragment_only=args.single_fragment_only,
        align_product_to_reactant=args.align_product_to_reactant,
        center=args.center,
        output_format=args.output_format,
    )
    print(f"Converted {len(rows)} Transition1x rows to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
