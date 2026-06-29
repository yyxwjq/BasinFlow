import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from fscgp.data.records import BasinRecord, EventRecord, StructureRecord


def test_structure_record_represents_molecule_with_vacuum_cell_and_no_pbc():
    structure = StructureRecord(
        structure_id="h2",
        species=["H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        cell=np.eye(3) * 20.0,
        pbc=[False, False, False],
        metadata={"system_type": "molecule"},
    )

    assert structure.cell.shape == (3, 3)
    assert structure.pbc.tolist() == [False, False, False]
    assert structure.n_atoms == 2


def test_structure_record_represents_periodic_crystal():
    structure = StructureRecord(
        structure_id="nacl",
        species=["Na", "Cl"],
        positions=[[0.0, 0.0, 0.0], [2.8, 2.8, 2.8]],
        cell=np.eye(3) * 5.6,
        pbc=[True, True, True],
        metadata={"system_type": "periodic_bulk"},
    )

    assert structure.cell.shape == (3, 3)
    assert structure.pbc.dtype == np.bool_
    assert structure.pbc.tolist() == [True, True, True]


def test_structure_record_from_ase_reads_singlepoint_energy_and_forces():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], pbc=False)
    forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    atoms.info["energy"] = 99.0
    atoms.arrays["forces"] = np.zeros((2, 3))
    atoms.calc = SinglePointCalculator(atoms, energy=-1.25, forces=forces)

    structure = StructureRecord.from_ase(atoms, structure_id="h2")

    assert structure.energy == -1.25
    assert np.allclose(structure.forces, forces)


def test_structure_record_to_ase_stores_energy_and_forces_in_singlepoint_calculator():
    forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    structure = StructureRecord(
        structure_id="h2",
        species=["H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        energy=-1.25,
        forces=forces,
    )

    atoms = structure.to_ase()

    assert "energy" not in atoms.info
    assert "forces" not in atoms.arrays
    assert atoms.get_potential_energy() == -1.25
    assert np.allclose(atoms.get_forces(), forces)


def test_event_and_basin_records_support_multiple_events_from_same_basin():
    event_a = EventRecord(
        event_id="event-a",
        reactant_structure_id="r0",
        product_structure_id="p0",
        basin_id="basin-0",
    )
    event_b = EventRecord(
        event_id="event-b",
        reactant_structure_id="r0",
        product_structure_id="p1",
        basin_id="basin-0",
    )
    basin = BasinRecord(
        basin_id="basin-0",
        reactant_structure_id="r0",
        known_event_ids=[event_a.event_id, event_b.event_id],
    )

    assert basin.known_event_ids == ["event-a", "event-b"]
    assert event_a.basin_id == event_b.basin_id == basin.basin_id
