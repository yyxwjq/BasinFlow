from __future__ import annotations

import copy
import json
import pickle
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read


def _sources(tmp_path):
    raw = {"single_fragment": [True, False, True, False, True, False], "use_ind": [1, 0]}
    for role in ("reactant", "product", "transition_state"):
        raw[role] = {
            "num_atoms": [2] * 6, "charges": [[1, 8]] * 6,
            "positions": [np.array([[0., 0., 0.], [1. + index / 10, 0., 0.]]) for index in range(6)],
            "rxn": [f"reaction-{index}" for index in range(6)],
        }
    train = tmp_path / "train.pkl"
    valid = tmp_path / "valid.pkl"
    train.write_bytes(pickle.dumps(raw))
    valid_raw = copy.deepcopy(raw)
    valid_raw["use_ind"] = [5, 3, 4, 2]
    valid_raw["product"]["positions"][2][1, 0] = 2.4
    valid.write_bytes(pickle.dumps(valid_raw))
    return train, valid


def _prepare(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("tools").resolve()))
    return runpy.run_path("tools/prepare_transition1x_explicit_split.py")["prepare_transition1x_explicit_split"]


def test_explicit_transition1x_freezes_all_fragments_and_pins_validation(tmp_path, monkeypatch):
    train, valid = _sources(tmp_path)
    prepare = _prepare(monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        prepare(train, valid, output, validation_count=2, seed=7, required_validation_indices=[2])
    split = json.loads((first / "split_manifest.json").read_text())
    assert split == json.loads((second / "split_manifest.json").read_text())
    assert split["train"] == ["transition1x:0", "transition1x:1"]
    assert len(split["val"]) == len(split["test"]) == 2
    assert "transition1x:2" in split["val"]
    assert set(split["val"]).isdisjoint(split["test"])
    assert set(split["train"] + split["val"] + split["test"]) == {f"transition1x:{index}" for index in range(6)}
    for partition in ("train", "val", "test"):
        indices = json.loads((first / f"{partition}_indices.json").read_text())
        assert indices == sorted(indices)
        assert split[partition] == [f"transition1x:{index}" for index in indices]
        assert (first / partition / "basin_table.csv").exists()
    frames = read(first / "val" / "event_2.traj", index=":")
    assert np.allclose(frames[1].positions, [[-1.2, 0, 0], [1.2, 0, 0]])
    assert frames[0].info["source_file"] == str(valid.resolve())
    assert frames[0].info["align_product_to_reactant"] is False
    assert read(first / "train" / "event_1.traj").info["single_fragment"] is False
    provenance = json.loads((first / "source_partitions.json").read_text())
    assert provenance["reference_split"] is False
    assert provenance["required_validation_indices"] == [2]
    assert provenance["single_fragment_only"] is False
    assert provenance["seed"] == 7
    assert len(provenance["sources"]["train"]["sha256"]) == 64


@pytest.mark.parametrize("failure", ["overlap", "complement", "identity", "pinned_train", "pinned_bool", "pinned_duplicate", "count"])
def test_explicit_transition1x_rejects_bad_partitions_without_output(tmp_path, monkeypatch, failure):
    train, valid = _sources(tmp_path)
    prepare = _prepare(monkeypatch)
    raw = pickle.loads(valid.read_bytes())
    if failure == "overlap":
        raw["use_ind"] = [0, 2, 3, 4]
    elif failure == "complement":
        raw["use_ind"] = [2, 3, 4]
    elif failure == "identity":
        raw["reactant"]["rxn"][0] = "other"
    valid.write_bytes(pickle.dumps(raw))
    pinned = {"pinned_train": [0], "pinned_bool": [True], "pinned_duplicate": [2, 2]}.get(failure, [2])
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        prepare(train, valid, output, validation_count=5 if failure == "count" else 2,
                required_validation_indices=pinned)
    assert not output.exists()


def test_explicit_transition1x_refuses_existing_directory(tmp_path, monkeypatch):
    train, valid = _sources(tmp_path)
    prepare = _prepare(monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        prepare(train, valid, output, validation_count=2)
    assert marker.read_text() == "keep"


def test_explicit_transition1x_cli_pins_unique_prior_run_ids(tmp_path):
    train, valid = _sources(tmp_path)
    prior_run = tmp_path / "prior"
    (prior_run / "sampling").mkdir(parents=True)
    (prior_run / "sampling" / "trial_metrics.csv").write_text(
        "pseudo_basin_id,trial\ntransition1x:2,0\ntransition1x:2,1\ntransition1x:3,0\n"
    )
    output = tmp_path / "output"
    result = subprocess.run([
        sys.executable, "tools/prepare_transition1x_explicit_split.py", str(train), str(valid), str(output),
        "--validation-count", "2", "--prior-run", str(prior_run),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "val_indices.json").read_text()) == [2, 3]


def test_explicit_transition1x_cli_also_pins_diagnostic_plan(tmp_path):
    train, valid = _sources(tmp_path)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'evaluation_split': 'val', 'plan': [
        {'basin_id': 'transition1x:4', 'trial_index': 0},
        {'basin_id': 'transition1x:4', 'trial_index': 1},
        {'basin_id': 'transition1x:5', 'trial_index': 0},
    ]}))
    output = tmp_path / 'output'
    result = subprocess.run([
        sys.executable, 'tools/prepare_transition1x_explicit_split.py', str(train), str(valid), str(output),
        '--validation-count', '2', '--prior-plan', str(plan),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / 'val_indices.json').read_text()) == [4, 5]


def _rotated_sources(tmp_path):
    """Three single-fragment records whose product is a 90-degree rotation of the reactant."""
    base = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    raw = {"single_fragment": [True, True, True], "use_ind": [0]}
    for role in ("reactant", "product", "transition_state"):
        raw[role] = {"num_atoms": [3, 3, 3], "charges": [[6, 6, 6]] * 3,
                     "positions": [base.copy() for _ in range(3)],
                     "rxn": [f"rxn{index}" for index in range(3)]}
    raw["product"]["positions"] = [base @ turn for _ in range(3)]
    train = tmp_path / "train.pkl"
    valid = tmp_path / "valid.pkl"
    train.write_bytes(pickle.dumps(raw))
    other = copy.deepcopy(raw)
    other["use_ind"] = [1, 2]
    valid.write_bytes(pickle.dumps(other))
    return train, valid, base


def _any_event_frame(root, index):
    for partition in ("val", "test", "train"):
        path = root / partition / f"event_{index}.traj"
        if path.exists():
            return read(path, index=":")
    raise AssertionError(f"event_{index} not written to {root}")


def test_explicit_transition1x_alignment_removes_the_arbitrary_source_orientation(tmp_path, monkeypatch):
    train, valid, base = _rotated_sources(tmp_path)
    prepare = _prepare(monkeypatch)
    centered = base - base.mean(axis=0)
    plain = tmp_path / "plain"
    prepare(train, valid, plain, validation_count=1, seed=3)
    aligned = tmp_path / "aligned"
    prepare(train, valid, aligned, validation_count=1, seed=3,
            single_fragment_only=True, align_product_to_reactant=True)

    plain_frames = _any_event_frame(plain, 1)
    assert np.allclose(plain_frames[0].positions, centered)
    assert not np.allclose(plain_frames[1].positions, centered)
    assert plain_frames[1].info["align_product_to_reactant"] is False

    aligned_frames = _any_event_frame(aligned, 1)
    assert np.allclose(aligned_frames[0].positions, centered)
    assert np.allclose(aligned_frames[1].positions, centered, atol=1e-9)
    assert aligned_frames[1].info["coordinate_convention"] == "proper_kabsch_to_reactant"
    assert aligned_frames[1].info["align_product_to_reactant"] is True

    provenance = json.loads((aligned / "source_partitions.json").read_text())
    assert provenance["align_product_to_reactant"] is True
    assert provenance["single_fragment_only"] is True
    assert provenance["coordinate_convention"] == "centroid_centered_product_kabsch_to_reactant"


def test_explicit_transition1x_alignment_does_not_require_the_fragment_filter(tmp_path, monkeypatch):
    """Alignment is a rigid transform, so it must work with multi-fragment records.

    The converter used to refuse that combination, which forced the MolGEN-style
    9000-record split down to the 6733 single-fragment subset.
    """
    train, valid, base = _rotated_sources(tmp_path)
    prepare = _prepare(monkeypatch)
    output = tmp_path / "output"
    provenance = prepare(train, valid, output, validation_count=1,
                         align_product_to_reactant=True)
    assert provenance["align_product_to_reactant"] is True
    assert provenance["single_fragment_only"] is False
    assert provenance["coordinate_convention"] == "centroid_centered_product_kabsch_to_reactant"


def test_explicit_transition1x_single_fragment_filter_drops_mixed_records(tmp_path, monkeypatch):
    train, valid = _sources(tmp_path)
    prepare = _prepare(monkeypatch)
    output = tmp_path / "output"
    prepare(train, valid, output, validation_count=1, seed=5, single_fragment_only=True)
    split = json.loads((output / "split_manifest.json").read_text())
    assert split["train"] == ["transition1x:0"]
    assert set(split["val"]).isdisjoint(split["test"])
    provenance = json.loads((output / "source_partitions.json").read_text())
    assert provenance["single_fragment_only"] is True

    pinned = tmp_path / "pinned"
    with pytest.raises(ValueError, match="excluded pinned validation indices"):
        prepare(train, valid, pinned, validation_count=1, seed=5,
                required_validation_indices=[3], single_fragment_only=True)
    assert not pinned.exists()
