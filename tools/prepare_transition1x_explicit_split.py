"""Freeze custom Transition1x train/validation/test directories with pinned validation rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from prepare_transition1x_split import _identity_hash
from transition1x_to_events import _load_source, _selected_indices, _validate_indices, convert_transition1x


def prior_validation_indices(prior_run: str | Path) -> list[int]:
    path = Path(prior_run) / "sampling" / "trial_metrics.csv"
    indices = set()
    with path.open(newline="", encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)
        if "pseudo_basin_id" not in (reader.fieldnames or []):
            raise ValueError("prior trial metrics require pseudo_basin_id")
        for row in reader:
            match = re.fullmatch(r"transition1x:(0|[1-9][0-9]*)", row["pseudo_basin_id"] or "")
            if match is None:
                raise ValueError("prior pseudo_basin_id must identify a Transition1x source index")
            indices.add(int(match.group(1)))
    if not indices:
        raise ValueError("prior trial metrics contain no evaluated source indices")
    return sorted(indices)


def prepare_transition1x_explicit_split(
    train_source: str | Path,
    valid_source: str | Path,
    output_dir: str | Path,
    *,
    validation_count: int = 536,
    seed: int = 20260910,
    required_validation_indices: list[int] | None = None,
    single_fragment_only: bool = False,
    align_product_to_reactant: bool = False,
) -> dict:
    """Keep train use_ind; freeze a custom split of its complement without filtering."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    sources = {"train": Path(train_source).resolve(), "val": Path(valid_source).resolve()}
    raw_sources = {partition: _load_source(path) for partition, path in sources.items()}
    identities = {partition: _identity_hash(raw) for partition, raw in raw_sources.items()}
    if identities["train"] != identities["val"]:
        raise ValueError("source record identity/order differs between train and validation")
    train_indices = _selected_indices(raw_sources["train"], "use_ind")
    valid_indices = _selected_indices(raw_sources["val"], "use_ind")
    complement = _selected_indices(raw_sources["train"], "complement")
    if set(train_indices) & set(valid_indices):
        raise ValueError("source train/validation use_ind partitions overlap")
    if set(valid_indices) != set(complement):
        raise ValueError("validation use_ind must exactly equal the training use_ind complement")
    if not train_indices:
        raise ValueError("training source selection must be nonempty")
    if isinstance(validation_count, (bool, np.bool_)) or not isinstance(validation_count, Integral):
        raise ValueError("validation_count must be an integer")
    if not 0 < validation_count < len(complement):
        raise ValueError("validation_count must leave nonempty validation and test partitions")
    if single_fragment_only:
        flags = raw_sources["train"]["single_fragment"]
        train_indices = [index for index in train_indices if flags[index]]
        valid_indices = [index for index in valid_indices if flags[index]]
        complement = [index for index in complement if flags[index]]
        if not train_indices:
            raise ValueError("single_fragment_only removed every training selection")
        if not 0 < validation_count < len(complement):
            raise ValueError(
                "validation_count must leave nonempty validation and test partitions after "
                "single_fragment_only filtering")
    if required_validation_indices is None:
        required_validation_indices = []
    if not isinstance(required_validation_indices, list):
        raise ValueError("required_validation_indices must be a list")
    required = _validate_indices(required_validation_indices, len(raw_sources["train"]["reactant"]["num_atoms"]))
    if not set(required).issubset(complement):
        excluded = sorted(set(required) - set(complement))
        if single_fragment_only:
            raise ValueError(
                f"single_fragment_only excluded pinned validation indices {excluded}; a filtered, "
                "aligned export is a separate protocol and must not pin rows")
        raise ValueError("required validation indices must belong to the source complement")
    if len(required) > validation_count:
        raise ValueError("required validation indices exceed validation_count")
    available = [index for index in complement if index not in set(required)]
    chosen = np.random.default_rng(seed).choice(available, size=validation_count - len(required), replace=False)
    selected_validation = sorted(required + chosen.tolist())
    indices = {
        "train": sorted(train_indices),
        "val": selected_validation,
        "test": sorted(set(complement) - set(selected_validation)),
    }
    sources["test"] = sources["val"]
    split = {partition: [f"transition1x:{index}" for index in selected] for partition, selected in indices.items()}
    split["seed"] = seed
    provenance = {
        "partition_method": "custom_train_use_ind_complement_pinned_validation",
        "reference_split": False,
        "single_fragment_only": bool(single_fragment_only),
        "align_product_to_reactant": bool(align_product_to_reactant),
        "coordinate_convention": ("centroid_centered_product_kabsch_to_reactant"
                                  if align_product_to_reactant else "per_frame_centroid_centered"),
        "coordinate_source": "train_from_train_source_validation_and_test_from_valid_source",
        "source_identity_order_sha256": identities["train"],
        "seed": seed,
        "validation_count": validation_count,
        "required_validation_indices": sorted(required),
        "required_validation_basin_ids": [f"transition1x:{index}" for index in sorted(required)],
        "sources": {
            partition: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selected_count": len(indices[partition]),
                "selected_source_indices": indices[partition],
            }
            for partition, path in sources.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="explicit-transition1x-", dir=output.parent) as temporary:
        prepared = Path(temporary) / "prepared"
        prepared.mkdir()
        for partition, path in sources.items():
            convert_transition1x(path, prepared / partition, source_indices=indices[partition],
                                 single_fragment_only=single_fragment_only,
                                 align_product_to_reactant=align_product_to_reactant, center=True)
            (prepared / f"{partition}_indices.json").write_text(json.dumps(indices[partition]) + "\n", encoding="utf-8")
        for filename, payload in (("split_manifest.json", split), ("source_partitions.json", provenance)):
            (prepared / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        prepared.rename(output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_source")
    parser.add_argument("valid_source")
    parser.add_argument("output_dir")
    parser.add_argument("--validation-count", type=int, default=536)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--prior-run", type=Path, help="pin every source index from sampling/trial_metrics.csv in validation")
    parser.add_argument("--prior-plan", type=Path, action="append", default=[], help="also pin basin ids from a diagnostic sampling_plan.json")
    parser.add_argument("--single-fragment-only", action="store_true",
                        help="keep only records marked single_fragment; required before aligning products")
    parser.add_argument("--align-product-to-reactant", action="store_true",
                        help="proper Kabsch alignment of product targets onto reactants, removing the "
                             "arbitrary per-file source orientation; requires --single-fragment-only")
    args = parser.parse_args()
    required = prior_validation_indices(args.prior_run) if args.prior_run is not None else []
    for path in args.prior_plan:
        payload = json.loads(path.read_text())
        for row in payload['plan']:
            match = re.fullmatch(r"transition1x:(0|[1-9][0-9]*)", row['basin_id'])
            if match is None:
                raise ValueError('diagnostic basin_id must identify a Transition1x source index')
            required.append(int(match.group(1)))
    required = sorted(set(required))
    provenance = prepare_transition1x_explicit_split(
        args.train_source, args.valid_source, args.output_dir,
        validation_count=args.validation_count, seed=args.seed, required_validation_indices=required,
        single_fragment_only=args.single_fragment_only,
        align_product_to_reactant=args.align_product_to_reactant,
    )
    counts = {partition: source["selected_count"] for partition, source in provenance["sources"].items()}
    print(f"Prepared custom Transition1x partitions {counts}; pinned validation rows={len(required)}; "
          f"coordinate_convention={provenance['coordinate_convention']}")


if __name__ == "__main__":
    main()
