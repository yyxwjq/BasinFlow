from ase.io import read, write
from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np

from tools.eon_to_events import process_basin


def _write_con(path, shift):
    atoms = Atoms(
        "H",
        positions=[[shift, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=False,
    )
    write(path, atoms)


def _write_con_with_constraint(path, shift):
    atoms = Atoms(
        "H2",
        positions=[[shift, 0.0, 0.0], [1.0 + shift, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=False,
    )
    atoms.set_constraint(FixAtoms(indices=[0]))
    write(path, atoms)


def test_process_basin_writes_two_frame_event_when_saddle_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    procdata = tmp_path / "0" / "procdata"
    procdata.mkdir(parents=True)
    output_dir = tmp_path / "events"
    output_dir.mkdir()
    _write_con(procdata / "reactant_0.con", 0.0)
    _write_con(procdata / "product_0.con", 0.2)

    records, next_counter = process_basin(0, 0, output_dir, output_format="extxyz")

    frames = read(output_dir / "event_0.extxyz", index=":")
    assert next_counter == 1
    assert records == [{"global_event": 0, "local_event": 0, "basin": 0, "file": "event_0.extxyz"}]
    assert len(frames) == 2


def test_process_basin_writes_three_frame_event_when_saddle_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    procdata = tmp_path / "0" / "procdata"
    procdata.mkdir(parents=True)
    output_dir = tmp_path / "events"
    output_dir.mkdir()
    _write_con(procdata / "reactant_0.con", 0.0)
    _write_con(procdata / "product_0.con", 0.2)
    _write_con(procdata / "saddle_0.con", 0.1)

    records, next_counter = process_basin(0, 0, output_dir, output_format="extxyz")

    frames = read(output_dir / "event_0.extxyz", index=":")
    assert next_counter == 1
    assert records[0]["file"] == "event_0.extxyz"
    assert len(frames) == 3


def test_process_basin_can_write_traj_with_fixatoms_constraints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    procdata = tmp_path / "0" / "procdata"
    procdata.mkdir(parents=True)
    output_dir = tmp_path / "events"
    output_dir.mkdir()
    _write_con_with_constraint(procdata / "reactant_0.con", 0.0)
    _write_con_with_constraint(procdata / "product_0.con", 0.2)

    records, next_counter = process_basin(0, 0, output_dir)

    frames = read(output_dir / "event_0.traj", index=":")
    assert next_counter == 1
    assert records == [{"global_event": 0, "local_event": 0, "basin": 0, "file": "event_0.traj"}]
    assert len(frames) == 2
    assert frames[0].constraints
    assert frames[0].constraints[0].get_indices().tolist() == [0]
