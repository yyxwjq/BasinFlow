from __future__ import annotations

import pickle
import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from basinflow.data import event_target, load_events


def _write_transition1x_fixture(path):
    def frame(positions):
        return {
            "num_atoms": [2, 3, 2],
            "charges": [[1, 8], [6, 1, 1], [7, 1]],
            "positions": positions,
            "rxn": ["r0", "r1", "r2"],
            "formula": ["HO", "CH2", "NH"],
            "fragments": [[[0, 1]], [[0, 1, 2]], [[0, 1]]],
            "wB97x_6-31G(d).energy": [-1.0, -2.0, -3.0],
            "wB97x_6-31G(d).forces": [
                np.zeros((2, 3)),
                np.zeros((3, 3)),
                np.zeros((2, 3)),
            ],
        }

    raw = {
        "reactant": frame(
            [
                np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ]
        ),
        "product": frame(
            [
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
            ]
        ),
        "transition_state": frame(
            [
                np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [1.9, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
            ]
        ),
        "single_fragment": [True, False, True],
        "use_ind": [2, 0],
    }
    with path.open("wb") as file:
        pickle.dump(raw, file)


def test_transition1x_converter_writes_standard_event_directory(tmp_path):
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)

    result = subprocess.run(
        [
            sys.executable,
            "tools/transition1x_to_events.py",
            str(source),
            str(output_dir),
            "--selection",
            "use_ind",
            "--center",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    # Event ids are the export positions 0..N-1, so split.csv can hold integers.
    assert [path.name for path in sorted(output_dir.glob("event_*.traj"))] == [
        "event_0.traj",
        "event_1.traj",
    ]
    assert not (output_dir / "basin_table.csv").exists()

    frames = read(output_dir / "event_0.traj", index=":")
    assert len(frames) == 3
    assert frames[0].get_chemical_symbols() == ["N", "H"]
    assert frames[0].get_pbc().tolist() == [False, False, False]
    assert np.allclose(frames[0].get_positions().mean(axis=0), 0.0)
    # The source row stays identifiable through each frame's info.
    assert frames[0].info["source_index"] == 2
    assert frames[0].info["dataset_kind"] == "transition1x_single_event_molecule"

    catalog = load_events(output_dir)
    assert catalog.event_ids == ["0", "1"]
    target = event_target(catalog.events["0"])
    assert target.reactant.species == ["N", "H"]
    assert np.allclose(target.transition_state.positions.mean(axis=0), 0.0)
    assert int(catalog.events["0"].metadata["source_index"]) == 2
    assert catalog.events["0"].metadata["dataset_kind"] == "transition1x_single_event_molecule"


def test_transition1x_converter_rejects_unknown_selection(tmp_path):
    source = tmp_path / "train.pkl"
    _write_transition1x_fixture(source)

    result = subprocess.run(
        [sys.executable, "tools/transition1x_to_events.py", str(source), str(tmp_path / "events"), "--selection", "unknown"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "selection" in result.stderr


@pytest.mark.parametrize("selection, expected_indices", [("use_ind", [2]), ("all", [0, 2]), ("complement", [0])])
def test_transition1x_single_fragment_filter_intersects_selection(tmp_path, selection, expected_indices):
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)
    with source.open("rb") as source_file:
        raw = pickle.load(source_file)
    raw["use_ind"] = [2, 1]
    with source.open("wb") as source_file:
        pickle.dump(raw, source_file)

    result = subprocess.run(
        [sys.executable, "tools/transition1x_to_events.py", str(source), str(output_dir),
         "--selection", selection, "--single-fragment-only"],
        check=False, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    written = sorted(output_dir.glob("event_*.traj"), key=lambda item: int(item.stem.split("_")[1]))
    assert [int(read(path, index=0).info["source_index"]) for path in written] == expected_indices
    for path in written:
        frames = read(path, index=":")
        assert all(frame.info["single_fragment_only"] is True for frame in frames)
        assert all(frame.info["single_fragment"] is True for frame in frames)


def test_transition1x_converter_rejects_incomplete_single_fragment_metadata(tmp_path):
    source = tmp_path / "train.pkl"
    _write_transition1x_fixture(source)
    with source.open("rb") as source_file:
        raw = pickle.load(source_file)
    raw["single_fragment"] = [True]
    with source.open("wb") as source_file:
        pickle.dump(raw, source_file)

    result = subprocess.run(
        [sys.executable, "tools/transition1x_to_events.py", str(source), str(tmp_path / "events")],
        check=False, capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert "single_fragment must contain one value per source row" in result.stderr
    assert not list((tmp_path / "events").glob("event_*"))


def test_transition1x_complement_preserves_source_partition(tmp_path):
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]

    rows = converter(source, output_dir, selection="complement")

    assert [row["source_index"] for row in rows] == ["1"]
    frames = read(output_dir / "event_0.traj", index=":")
    assert all(frame.info["selection"] == "complement" for frame in frames)


@pytest.mark.parametrize("single_fragment_only, expected", [(False, [2, 1, 0]), (True, [2, 0])])
def test_transition1x_explicit_indices_preserve_order_and_filter(tmp_path, single_fragment_only, expected):
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]

    rows = converter(source, output_dir, source_indices=[2, 1, 0], single_fragment_only=single_fragment_only)

    assert [int(row["source_index"]) for row in rows] == expected
    for row in rows:
        frames = read(output_dir / row["file"], index=":")
        assert all(frame.info["selection"] == "explicit_indices" for frame in frames)


@pytest.mark.parametrize("indices", [[True], [1.0], ["1"], [0, 0], [-1], [3]])
@pytest.mark.parametrize("selection", ["use_ind", "complement", "explicit_indices"])
def test_transition1x_rejects_invalid_indices_before_output(tmp_path, indices, selection):
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]
    if selection == "explicit_indices":
        options = {"source_indices": indices}
    else:
        with source.open("rb") as source_file:
            raw = pickle.load(source_file)
        raw["use_ind"] = indices
        with source.open("wb") as source_file:
            pickle.dump(raw, source_file)
        options = {"selection": selection}

    with pytest.raises(ValueError, match="indices"):
        converter(source, output_dir, **options)
    assert not output_dir.exists()


@pytest.mark.parametrize("selection", ["all", "complement"])
def test_transition1x_rejects_conflicting_explicit_selection(tmp_path, selection):
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]

    with pytest.raises(ValueError, match="selection"):
        converter(source, output_dir, source_indices=[0], selection=selection)
    assert not output_dir.exists()


@pytest.mark.parametrize("indices, succeeds", [([2, 0], True), ({"indices": [2]}, False), ([False], False)])
def test_transition1x_indices_json_cli(tmp_path, indices, succeeds):
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    indices_path = tmp_path / "indices.json"
    _write_transition1x_fixture(source)
    indices_path.write_text(json.dumps(indices))

    result = subprocess.run(
        [sys.executable, "tools/transition1x_to_events.py", str(source), str(output_dir),
         "--indices-json", str(indices_path)],
        check=False, capture_output=True, text=True,
    )

    if not succeeds:
        assert result.returncode != 0
        assert "indices" in result.stderr
        assert not output_dir.exists()
        return
    assert result.returncode == 0, result.stderr + result.stdout
    written = sorted(output_dir.glob("event_*.traj"), key=lambda item: int(item.stem.split("_")[1]))
    assert [int(read(path, index=0).info["source_index"]) for path in written] == indices


@pytest.mark.parametrize("align_product", [False, True])
def test_transition1x_converter_writes_partition_directories(tmp_path, align_product):
    """``--split-file`` expresses train/val/test as directories, not as a manifest."""
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "prepared"
    _write_transition1x_fixture(source)
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train": [2], "val": [1], "test": [0]}))

    result = subprocess.run(
        [sys.executable, "tools/transition1x_to_events.py", str(source), str(output_dir),
         "--selection", "all", "--split-file", str(split_file)]
        + (["--align-product-to-reactant"] if align_product else []),
        check=False, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    counts = {
        name: len(list((output_dir / name).glob("event_*.traj")))
        for name in ("train", "val", "test")
    }
    assert counts == {"train": 1, "val": 1, "test": 1}
    for name in counts:
        assert not (output_dir / name / "basin_table.csv").exists()
    # Event ids are global to the export, so each partition keeps a disjoint slice
    # of 0..N-1 and nothing has to be renumbered when the partitions are merged.
    written = {
        name: sorted(
            (output_dir / name).glob("event_*.traj"),
            key=lambda item: int(item.stem.split("_")[1]),
        )
        for name in ("train", "val", "test")
    }
    assert [path.name for path in written["train"]] == ["event_0.traj"]
    assert [path.name for path in written["val"]] == ["event_1.traj"]
    assert [path.name for path in written["test"]] == ["event_2.traj"]
    frames = read(written["val"][0], index=":")
    assert int(frames[0].info["source_index"]) == 1
    if align_product:
        assert frames[1].info["coordinate_convention"] == "proper_kabsch_to_reactant"


def test_transition1x_split_file_rejects_unknown_and_overlapping_rows(tmp_path):
    source = tmp_path / "train.pkl"
    _write_transition1x_fixture(source)
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]

    with pytest.raises(ValueError, match="outside the selection"):
        converter(source, tmp_path / "a", selection="all", partitions={"train": [5]})
    assert not (tmp_path / "a").exists()

    with pytest.raises(ValueError, match="overlap"):
        converter(
            source, tmp_path / "b", selection="all", partitions={"train": [0], "val": [0]}
        )
    assert not (tmp_path / "b").exists()


def _write_alignment_fixture(path, reactant, product, forces, state=None):
    raw = {"single_fragment": [True], "use_ind": [0]}
    if state is None:
        state = reactant + 0.37
    for role, positions in [("reactant", reactant), ("product", product), ("transition_state", state)]:
        raw[role] = {
            "num_atoms": [len(reactant)], "charges": [[6, 1, 7, 8]],
            "positions": [positions], "rxn": ["alignment-fixture"],
            "wB97x_6-31G(d).energy": [-1.5], "wB97x_6-31G(d).forces": [forces],
        }
    with path.open("wb") as source_file:
        pickle.dump(raw, source_file)


@pytest.mark.parametrize("planar", [False, True])
def test_product_alignment_removes_independent_rotation_and_rotates_forces(tmp_path, planar):
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]
    reactant = np.array([[0, 0, 0], [1.3, 0.1, 0], [0.2, 1.1, 0], [0.1, 0.2, 1.4]], dtype=float)
    product = reactant + np.array([[0.1, 0, 0], [0.2, 0.1, 0], [0, -0.1, 0], [0.1, 0, 0.1]])
    if planar:
        reactant[:, 2] = 0
        product[:, 2] = 0
    forces = np.arange(12).reshape(4, 3) / 10
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    source_state = reactant + 0.37
    exported = []
    exported_states = []
    for name, source_product, source_forces in [
        ("original", product, forces),
        ("rotated", product @ rotation + [4, -3, 2], forces @ rotation),
    ]:
        source = tmp_path / f"{name}.pkl"
        # The whole path is rotated together, exactly as a raw source file
        # would be; only then is the export invariance well defined.
        _write_alignment_fixture(
            source, reactant, source_product, source_forces,
            state=source_state @ rotation + [4, -3, 2] if name == "rotated" else source_state,
        )
        converter(source, tmp_path / name, center=True, align_product_to_reactant=True)
        frames = read(tmp_path / name / "event_0.traj", index=":")
        exported.append(frames[1])
        exported_states.append(frames[2])
        assert np.allclose(frames[0].positions, reactant - reactant.mean(0))
        applied = np.asarray(frames[1].info["frame_alignment_rotation"])
        assert np.linalg.det(applied) == pytest.approx(1.0)
        assert np.allclose(frames[1].get_forces(), source_forces @ applied)
        # The transition state is fitted to the reactant independently, so its
        # rotation is its own but its geometry is preserved.
        assert frames[2].info["coordinate_convention"] == "proper_kabsch_to_reactant"
        assert np.linalg.det(np.asarray(frames[2].info["frame_alignment_rotation"])) == pytest.approx(1.0)
        state_rotation = np.asarray(frames[2].info["frame_alignment_rotation"])
        assert np.allclose(frames[2].get_forces(), source_forces @ state_rotation)
        assert frames[2].info["transition_state_usage"] == "proper_kabsch_to_reactant"
        # Each frame is fitted on its own, so the transition state's internal
        # geometry is exactly the source geometry even though its orientation
        # is not the product's.
        assert np.allclose(frames[2].get_all_distances(),
                           np.linalg.norm(reactant[:, None] - reactant[None, :], axis=-1))
        assert frames[1].get_potential_energy() == pytest.approx(-1.5)
        assert np.isfinite(frames[1].positions).all()
        assert np.allclose(frames[1].get_all_distances(), np.linalg.norm(product[:, None] - product[None, :], axis=-1))
    assert np.allclose(exported[0].positions, exported[1].positions, atol=1e-12)
    assert np.allclose(exported[0].get_forces(), exported[1].get_forces(), atol=1e-12)
    assert np.allclose(exported_states[0].positions, exported_states[1].positions, atol=1e-12)


def test_product_alignment_preserves_chirality_and_target_free_inference(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from basinflow.data import EventProposalDataset
    from basinflow.seeds import ZeroInit

    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]
    reactant = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    product = reactant * [-1, 1, 1]
    source = tmp_path / "reflected.pkl"
    _write_alignment_fixture(source, reactant, product, np.zeros((4, 3)))
    converter(source, tmp_path / "events", align_product_to_reactant=True)
    frames = read(tmp_path / "events/event_0.traj", index=":")
    assert not np.allclose(frames[1].positions, reactant)
    assert np.linalg.det(frames[1].positions[1:] - frames[1].positions[0]) == pytest.approx(
        np.linalg.det(product[1:] - product[0])
    )
    catalog = load_events(tmp_path / "events")
    before = EventProposalDataset(catalog, [ZeroInit()])[0]
    catalog.events["0"].product.positions[:] += 30
    catalog.events["0"].transition_state.positions[:] -= 20
    after = EventProposalDataset(catalog, [ZeroInit()])[0]
    assert not any(key.startswith("target_") for key in after.keys())
    assert set(before.keys()) == set(after.keys())
    for key in before.keys():
        if torch.is_tensor(before[key]):
            assert torch.equal(before[key], after[key])


def test_frame_alignment_accepts_multifragment_selection(tmp_path):
    """A rigid product-to-reactant alignment is valid for multi-fragment records."""
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]
    source = tmp_path / "source.pkl"
    _write_transition1x_fixture(source)
    rows = converter(source, tmp_path / "events", selection="all",
                     align_product_to_reactant=True)
    assert rows
    written = sorted((tmp_path / "events").glob("event_*.traj"))
    assert written
    frames = read(written[0], index=":")
    assert frames[1].info["coordinate_convention"] == "proper_kabsch_to_reactant"
    # The transition state is aligned to the reactant too, or the R+P -> TS
    # target carries a rigid motion the model cannot learn.
    assert frames[2].info["coordinate_convention"] == "proper_kabsch_to_reactant"
    assert np.linalg.det(np.asarray(frames[2].info["frame_alignment_rotation"])) == pytest.approx(1.0)


def test_each_target_frame_is_aligned_to_the_reactant_on_its_own(tmp_path):
    """Source frames are independently oriented, so each is fitted separately."""
    from ase import Atoms

    align = runpy.run_path("tools/transition1x_to_events.py")["_align_frames_to_reactant"]
    reactant = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    # Product and state are the same shape but rotated independently.
    spin = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    product = Atoms("H2", positions=np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]) @ spin + [3, 4, 5])
    state = Atoms("H2", positions=np.array([[0.0, 0.4, 0.3], [0.0, 1.7, -0.2]]) @ spin @ spin + [-2, 1, 0])

    align(reactant, product, state)

    share = np.linalg.norm([0.0, 1.3, -0.5])
    assert np.allclose(product.get_all_distances(), np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert np.allclose(state.get_all_distances(), np.array([[0.0, share], [share, 0.0]]))
    assert product.info["coordinate_convention"] == "proper_kabsch_to_reactant"
    assert state.info["coordinate_convention"] == "proper_kabsch_to_reactant"
    # Both are centred on the reactant, so their centroids coincide.
    assert np.allclose(product.positions.mean(0), reactant.positions.mean(0))
    assert np.allclose(state.positions.mean(0), reactant.positions.mean(0))


def test_frame_alignment_rejects_periodic_structures():
    from ase import Atoms

    align = runpy.run_path("tools/transition1x_to_events.py")["_align_frames_to_reactant"]
    reactant = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    with pytest.raises(ValueError, match="nonperiodic"):
        align(reactant, reactant.copy())


def test_converter_writes_an_optional_group_table_keyed_by_reaction(tmp_path):
    """``--basin-table`` maps every event to its source reaction, for evaluation."""
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]

    rows = converter(source, output_dir, selection="all", write_groups=True)

    table = output_dir / "basin_table.csv"
    assert table.is_file()
    written = list(csv.DictReader(table.open(newline="", encoding="utf-8")))
    assert [row["event_id"] for row in written] == [row["event_id"] for row in rows]
    assert all(row["basin"] for row in written)

    # The table loads against the exported catalog and groups same-reaction rows.
    from basinflow.data import load_groups, load_events

    catalog = load_events(output_dir)
    groups = load_groups(table, catalog)
    assert set(groups) == set(catalog.event_ids)


def test_converter_refuses_to_overwrite_an_existing_group_table(tmp_path):
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)
    converter = runpy.run_path("tools/transition1x_to_events.py")["convert_transition1x"]
    output_dir.mkdir()
    (output_dir / "basin_table.csv").write_text("event_id,basin\n")

    with pytest.raises(FileExistsError, match="group table"):
        converter(source, output_dir, selection="all", write_groups=True)
