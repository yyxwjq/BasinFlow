import numpy as np
from ase import Atoms
from ase.io import write

from fscgp.data.raw_events import read_event_file


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
