import csv
import os

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write
import pytest

from fscgp.data.raw_events import read_event_file, read_events_directory
from fscgp.data.dataset import split_basins


def test_two_frame_event_file_loads_reactant_and_product(tmp_path):
    path = tmp_path / "h2_event.extxyz"
    frames = [
        Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=np.eye(3) * 12, pbc=False),
        Atoms("H2", positions=[[0, 0, 0], [0.90, 0, 0]], cell=np.eye(3) * 12, pbc=False),
    ]
    write(path, frames)

    loaded = read_event_file(path, basin_id="basin-h2", event_id="stretch")

    assert loaded.reactant.structure_id == "stretch:reactant"
    assert loaded.product.structure_id == "stretch:product"
    assert loaded.transition_state is None
    assert loaded.event.reactant_structure_id == loaded.reactant.structure_id
    assert loaded.event.product_structure_id == loaded.product.structure_id
    assert np.allclose(loaded.product.positions[1], [0.90, 0, 0])


def test_three_frame_event_file_preserves_transition_state(tmp_path):
    path = tmp_path / "h2_ts.xyz"
    frames = [
        Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=np.eye(3) * 12, pbc=False),
        Atoms("H2", positions=[[0, 0, 0], [0.90, 0, 0]], cell=np.eye(3) * 12, pbc=False),
        Atoms("H2", positions=[[0, 0, 0], [0.82, 0, 0]], cell=np.eye(3) * 12, pbc=False),
    ]
    write(path, frames)

    loaded = read_event_file(path, basin_id="basin-h2", event_id="stretch")

    assert loaded.transition_state is not None
    assert loaded.transition_state.structure_id == "stretch:transition_state"
    assert loaded.event.transition_state_structure_id == loaded.transition_state.structure_id


def test_molecule_and_periodic_files_share_record_interface(tmp_path):
    molecule_path = tmp_path / "mol.extxyz"
    periodic_path = tmp_path / "crystal.extxyz"
    write(
        molecule_path,
        [
            Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("He", positions=[[0.1, 0, 0]], cell=np.eye(3) * 10, pbc=False),
        ],
    )
    write(
        periodic_path,
        [
            Atoms("NaCl", positions=[[0, 0, 0], [2.8, 0, 0]], cell=np.eye(3) * 5.6, pbc=True),
            Atoms("NaCl", positions=[[0.1, 0, 0], [2.8, 0, 0]], cell=np.eye(3) * 5.6, pbc=True),
        ],
    )

    molecule = read_event_file(molecule_path, basin_id="b-mol", event_id="mol").reactant
    periodic = read_event_file(periodic_path, basin_id="b-pbc", event_id="pbc").reactant

    assert molecule.cell.shape == periodic.cell.shape == (3, 3)
    assert molecule.pbc.tolist() == [False, False, False]
    assert periodic.pbc.tolist() == [True, True, True]


def test_events_directory_uses_basin_table_file_to_basin_mapping(tmp_path):
    event_frames = {
        "event_0.extxyz": [
            Atoms("H2", positions=[[0, 0, 0], [0.70, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("H2", positions=[[0, 0, 0], [0.90, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("H2", positions=[[0, 0, 0], [0.80, 0, 0]], cell=np.eye(3) * 10, pbc=False),
        ],
        "event_1.extxyz": [
            Atoms("H2", positions=[[0, 0, 0], [0.70, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("H2", positions=[[0, 0, 0], [0.50, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("H2", positions=[[0, 0, 0], [0.60, 0, 0]], cell=np.eye(3) * 10, pbc=False),
        ],
        "event_2.extxyz": [
            Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("He", positions=[[0.2, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("He", positions=[[0.1, 0, 0]], cell=np.eye(3) * 10, pbc=False),
        ],
    }
    for filename, frames in event_frames.items():
        write(tmp_path / filename, frames)

    with (tmp_path / "basin_table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["global_event", "local_event", "basin", "file"])
        writer.writeheader()
        writer.writerow({"global_event": 10, "local_event": 0, "basin": "basin_A", "file": "event_0.extxyz"})
        writer.writerow({"global_event": 11, "local_event": 1, "basin": "basin_A", "file": "event_1.extxyz"})
        writer.writerow({"global_event": 12, "local_event": 0, "basin": "basin_B", "file": "event_2.extxyz"})

    dataset = read_events_directory(tmp_path)

    assert dataset.event_ids == ["event_0", "event_1", "event_2"]
    assert dataset.basin_ids == ["basin_A", "basin_B"]
    assert dataset.basins["basin_A"].known_event_ids == ["event_0", "event_1"]
    assert dataset.basins["basin_B"].known_event_ids == ["event_2"]
    assert dataset.events["event_0"].transition_state_structure_id == "event_0:transition_state"
    assert dataset.events["event_1"].basin_id == "basin_A"


def test_events_directory_loads_in_basin_table_order_and_preserves_metadata(tmp_path):
    for filename, shift in [
        ("event_10.extxyz", 0.10),
        ("event_2.extxyz", 0.20),
        ("event_1.extxyz", 0.30),
    ]:
        write(
            tmp_path / filename,
            [
                Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=False),
                Atoms("He", positions=[[shift, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            ],
        )

    with (tmp_path / "basin_table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["global_event", "local_event", "basin", "file"],
        )
        writer.writeheader()
        writer.writerow({"global_event": 2, "local_event": 0, "basin": "b2", "file": "event_2.extxyz"})
        writer.writerow({"global_event": 10, "local_event": 1, "basin": "b10", "file": "event_10.extxyz"})
        writer.writerow({"global_event": 1, "local_event": 2, "basin": "b1", "file": "event_1.extxyz"})

    dataset = read_events_directory(tmp_path)

    assert dataset.event_ids == ["event_2", "event_10", "event_1"]
    assert dataset.events["event_2"].metadata["global_event"] == "2"
    assert dataset.events["event_2"].metadata["local_event"] == "0"
    assert dataset.events["event_2"].metadata["basin"] == "b2"
    assert dataset.events["event_2"].metadata["file"] == "event_2.extxyz"


def test_read_event_file_derives_active_atoms_from_fixatoms_constraints(tmp_path):
    path = tmp_path / "event_0.extxyz"
    frames = [
        Atoms("Au3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]], cell=np.eye(3) * 10, pbc=False),
        Atoms("Au3", positions=[[0, 0, 0], [1, 0.2, 0], [2, 0, 0]], cell=np.eye(3) * 10, pbc=False),
        Atoms("Au3", positions=[[0, 0, 0], [1, 0.1, 0], [2, 0, 0]], cell=np.eye(3) * 10, pbc=False),
    ]
    for atoms in frames:
        atoms.set_constraint(FixAtoms(indices=[0, 2]))
    write(path, frames)

    dataset = read_events_directory(tmp_path)
    item = dataset.pairwise_item("event_0", active_threshold=1.0)

    assert dataset.events["event_0"].active_atoms == [1]
    assert item["active_mask"].tolist() == [False, True, False]
    assert item["fixed_mask"].tolist() == [True, False, True]
    assert item["movable_mask"].tolist() == [False, True, False]
    assert item["has_transition_state"] is True
    assert item["transition_state"].structure_id == "event_0:transition_state"
    assert np.allclose(item["ts_displacement"][1], [0, 0.1, 0])


def test_pairwise_item_falls_back_to_displacement_active_atoms_without_constraints(tmp_path):
    path = tmp_path / "event_0.extxyz"
    write(
        path,
        [
            Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("H2", positions=[[0, 0, 0], [0.9, 0, 0]], cell=np.eye(3) * 10, pbc=False),
        ],
    )

    dataset = read_events_directory(tmp_path)
    item = dataset.pairwise_item("event_0", active_threshold=0.1)

    assert dataset.events["event_0"].active_atoms is None
    assert item["active_mask"].tolist() == [False, True]
    assert item["fixed_mask"].tolist() == [False, False]
    assert item["movable_mask"].tolist() == [True, True]


def test_split_basins_keeps_events_from_same_basin_together(tmp_path):
    for index in range(4):
        write(
            tmp_path / f"event_{index}.extxyz",
            [
                Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=False),
                Atoms("He", positions=[[0.1 * (index + 1), 0, 0]], cell=np.eye(3) * 10, pbc=False),
            ],
        )
    with (tmp_path / "basin_table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["global_event", "local_event", "basin", "file"])
        writer.writeheader()
        writer.writerow({"global_event": 0, "local_event": 0, "basin": "b0", "file": "event_0.extxyz"})
        writer.writerow({"global_event": 1, "local_event": 1, "basin": "b0", "file": "event_1.extxyz"})
        writer.writerow({"global_event": 2, "local_event": 0, "basin": "b1", "file": "event_2.extxyz"})
        writer.writerow({"global_event": 3, "local_event": 0, "basin": "b2", "file": "event_3.extxyz"})

    dataset = read_events_directory(tmp_path)
    train, val, test = split_basins(dataset, train=0.34, val=0.33, test=0.33, seed=1)

    split_event_sets = [
        set(train.event_ids),
        set(val.event_ids),
        set(test.event_ids),
    ]
    containing_b0 = [events for events in split_event_sets if {"event_0", "event_1"} & events]

    assert len(containing_b0) == 1
    assert {"event_0", "event_1"} <= containing_b0[0]


def test_stage2_real_events_dataset_when_env_is_set():
    events_dir = os.environ.get("BASINFLOW_EVENTS_DIR")
    if not events_dir:
        pytest.skip("BASINFLOW_EVENTS_DIR is not set")

    dataset = read_events_directory(events_dir)

    assert len(dataset.event_ids) == 24
    assert len(dataset.basin_ids) == 13
    assert dataset.event_ids[:3] == ["event_0", "event_1", "event_2"]

    for event_id in dataset.event_ids:
        item = dataset.pairwise_item(event_id)
        assert item["reactant"].n_atoms == 101
        assert item["product"].n_atoms == 101
        assert item["has_transition_state"] is True
        assert item["movable_mask"].sum() == 1
        assert item["fixed_mask"].sum() == 100
        assert item["active_mask"].sum() == 1

    train, val, test = split_basins(dataset, train=0.7, val=0.15, test=0.15, seed=42)
    split_basin_sets = [set(train.basin_ids), set(val.basin_ids), set(test.basin_ids)]
    assert split_basin_sets[0].isdisjoint(split_basin_sets[1])
    assert split_basin_sets[0].isdisjoint(split_basin_sets[2])
    assert split_basin_sets[1].isdisjoint(split_basin_sets[2])
