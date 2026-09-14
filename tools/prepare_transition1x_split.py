"""Prepare explicit Transition1x source partitions without resplitting reactions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from transition1x_to_events import ROLES, _load_source, _selected_indices, convert_transition1x


def _identity_hash(raw: dict) -> str:
    records = []
    for source_index in range(len(raw["reactant"]["num_atoms"])):
        for role in ROLES:
            frame = raw[role]
            if "rxn" not in frame:
                raise ValueError(f"source identity requires {role}.rxn")
            records.append([
                str(frame["rxn"][source_index]),
                int(frame["num_atoms"][source_index]),
                np.asarray(frame["charges"][source_index]).tolist(),
            ])
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def prepare_transition1x_split(
    train_source: str | Path,
    valid_source: str | Path,
    output_dir: str | Path,
    *,
    single_fragment_only: bool = True,
    align_product_to_reactant: bool = False,
) -> dict:
    """Preserve source coordinates and source train/validation index membership."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    sources = {"train": Path(train_source).resolve(), "val": Path(valid_source).resolve()}
    raw_sources = {partition: _load_source(path) for partition, path in sources.items()}
    identities = {partition: _identity_hash(raw) for partition, raw in raw_sources.items()}
    if identities["train"] != identities["val"]:
        raise ValueError("source record identity/order differs between train and validation")
    source_indices = {partition: _selected_indices(raw, "use_ind") for partition, raw in raw_sources.items()}
    if set(source_indices["train"]) & set(source_indices["val"]):
        raise ValueError("source train/validation use_ind partitions overlap")
    indices = {
        partition: _selected_indices(raw, "use_ind", single_fragment_only=single_fragment_only)
        for partition, raw in raw_sources.items()
    }
    if any(not selected for selected in indices.values()):
        raise ValueError("source train and validation selections must both be nonempty")
    if align_product_to_reactant and any(
        raw_sources[partition]["single_fragment"][index] != 1
        for partition, selected in indices.items() for index in selected
    ):
        raise ValueError("product alignment requires single-fragment records")
    provenance = {
        "partition_method": "source_use_ind",
        "single_fragment_only": single_fragment_only,
        "align_product_to_reactant": align_product_to_reactant,
        "coordinate_convention": "centroid_centered_product_kabsch_to_reactant" if align_product_to_reactant else "per_frame_centroid_centered",
        "coordinate_source": "each_partition_uses_its_own_source_file",
        "test_available": False,
        "source_identity_order_sha256": identities["train"],
        "sources": {
            partition: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_row_count": len(raw_sources[partition]["reactant"]["num_atoms"]),
                "use_ind_count": len(source_indices[partition]),
                "selected_count": len(indices[partition]),
                "selected_source_indices": indices[partition],
            }
            for partition, path in sources.items()
        },
    }
    if align_product_to_reactant:
        provenance["alignment"] = {
            "method": "proper_kabsch",
            "atom_mapping": "source_order_preserved",
            "reflections_allowed": False,
            "product_forces": "rotated_with_product_positions",
            "transition_state_usage": "source_frame_not_an_aligned_path_guess",
            "inference_inputs": "reactant_only_unchanged",
        }
    split = {
        "train": [f"transition1x:{index}" for index in indices["train"]],
        "val": [f"transition1x:{index}" for index in indices["val"]],
        "test": [],
        "seed": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="transition1x-", dir=output.parent) as temporary:
        staging = Path(temporary)
        prepared = staging / "prepared"
        events_dir = prepared / "events"
        events_dir.mkdir(parents=True)
        rows = []
        for partition, source in sources.items():
            partition_dir = staging / partition
            partition_rows = convert_transition1x(
                source, partition_dir, selection="use_ind", center=True,
                single_fragment_only=single_fragment_only,
                align_product_to_reactant=align_product_to_reactant,
            )
            for row in partition_rows:
                (partition_dir / row["file"]).rename(events_dir / row["file"])
                rows.append({**row, "global_event": str(len(rows)), "source_partition": partition})
        with (events_dir / "basin_table.csv").open("w", newline="", encoding="utf-8") as table_file:
            writer = csv.DictWriter(table_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for filename, payload in (("split_manifest.json", split), ("source_partitions.json", provenance)):
            (prepared / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prepared.rename(output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_source", help="OAReactDiff train_addprop.pkl")
    parser.add_argument("valid_source", help="OAReactDiff valid_addprop.pkl")
    parser.add_argument("output_dir", help="new output directory")
    parser.add_argument("--include-multifragment", action="store_true", help="keep all use_ind rows instead of OA's single-fragment subset")
    parser.add_argument("--align-product-to-reactant", action="store_true", help="proper Kabsch alignment of single-fragment product targets to reactants")
    args = parser.parse_args()
    provenance = prepare_transition1x_split(
        args.train_source, args.valid_source, args.output_dir,
        single_fragment_only=not args.include_multifragment,
        align_product_to_reactant=args.align_product_to_reactant,
    )
    counts = {partition: source["selected_count"] for partition, source in provenance["sources"].items()}
    print(f"Prepared Transition1x partitions {counts}; test is empty; output={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
