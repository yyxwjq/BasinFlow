"""Audit the two official ReactOT pickles before converting them.

Answers the questions that decide the training set: how many rows each file holds,
which rows ``use_ind`` selects, whether the two files agree row by row, and whether
the frames are internally consistent (atom counts, species order, elements,
per-frame centroid).

    python tools/audit_reactot_pickles.py train_rpsb_all.pkl valid_rpsb_all.pkl
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

ROLES = ("reactant", "product", "transition_state")
EXPECTED_ELEMENTS = {1, 6, 7, 8, 9}


def _row_digest(value) -> str:
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def _audit(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    total = len(raw["single_fragment"])
    use_ind = [int(index) for index in raw["use_ind"]]
    print(f"== {path}")
    print(f"  rows (all arrays): {total}")
    print(f"  use_ind: {len(use_ind)} rows, unique={len(set(use_ind))}, "
          f"sorted={use_ind == sorted(use_ind)}, range=({min(use_ind)}, {max(use_ind)})")
    all_flags = Counter(int(v) for v in raw["single_fragment"])
    selected_flags = Counter(int(raw["single_fragment"][index]) for index in use_ind)
    print(f"  single_fragment over all rows: {dict(all_flags)}")
    print(f"  single_fragment in use_ind:    {dict(selected_flags)}")
    for role in ROLES:
        fields = sorted(raw[role])
        print(f"  {role}: {fields}")

    # Frame-level consistency over every row, not just use_ind.
    max_atoms = 0
    bad_shape = bad_species = 0
    max_centroid = 0.0
    elements: Counter[int] = Counter()
    for role in ROLES:
        frame = raw[role]
        for index, count in enumerate(frame["num_atoms"]):
            count = int(count)
            max_atoms = max(max_atoms, count)
            positions = np.asarray(frame["positions"][index], dtype=float)[:count]
            charges = [int(value) for value in frame["charges"][index][:count]]
            if positions.shape != (count, 3) or len(charges) != count:
                bad_shape += 1
            max_centroid = max(max_centroid, float(np.abs(positions.mean(axis=0)).max()))
            elements.update(charges)
    for index in range(total):
        count = int(raw["reactant"]["num_atoms"][index])
        species = [tuple(int(v) for v in raw[role]["charges"][index][:count]) for role in ROLES]
        if not species[0] == species[1] == species[2]:
            bad_species += 1
    print(f"  max atoms: {max_atoms}  bad shapes: {bad_shape}  species mismatches: {bad_species}")
    print(f"  elements: {dict(sorted(elements.items()))}  max |frame centroid|: {max_centroid:.3e}")
    if set(elements) - EXPECTED_ELEMENTS:
        print(f"  !! unexpected elements: {sorted(set(elements) - EXPECTED_ELEMENTS)}")
    if max_centroid > 1e-3:
        print("  note: frames are not centred in the pickle; the converter and")
        print("        ProcessedTS1x(center=True) subtract the per-frame centroid")
    return {"raw": raw, "total": total, "use_ind": use_ind}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_pickle", type=Path)
    parser.add_argument("valid_pickle", type=Path)
    args = parser.parse_args()

    train = _audit(args.train_pickle)
    valid = _audit(args.valid_pickle)

    train_idx = set(train["use_ind"])
    valid_idx = set(valid["use_ind"])
    everything = set(range(train["total"]))
    print("== partitions")
    print(f"  train={len(train_idx)} valid={len(valid_idx)} overlap={len(train_idx & valid_idx)}")
    print(f"  union={len(train_idx | valid_idx)} total={train['total']} "
          f"unassigned={len(everything - (train_idx | valid_idx))}")

    # Row-by-row identity of the three frames between the two files.
    mismatches = 0
    for role in ROLES:
        for field in train["raw"][role]:
            if field not in valid["raw"][role]:
                print(f"  !! {role}.{field} missing from the valid pickle")
                mismatches += 1
                continue
            left, right = train["raw"][role][field], valid["raw"][role][field]
            if len(left) != train["total"] or len(right) != valid["total"]:
                continue
            for index in range(train["total"]):
                if _row_digest(left[index]) != _row_digest(right[index]):
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"  !! {role}.{field} differs at row {index}")
    print(f"  row-level mismatches between the files: {mismatches}")

    rxns = [str(value) for value in train["raw"]["reactant"]["rxn"]]
    selected = [rxns[index] for index in train["use_ind"]]
    print(f"  train rxn ids: {len(set(selected))} unique of {len(selected)} selected "
          f"({len(set(rxns))} unique over all rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
