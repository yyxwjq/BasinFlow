import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from basinflow.data import Event, EventCatalog, Structure


def test_structure_represents_molecule_with_vacuum_cell_and_no_pbc():
    structure = Structure(
        species=["H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        cell=np.eye(3) * 20.0,
        pbc=[False, False, False],
    )

    assert structure.cell.shape == (3, 3)
    assert structure.pbc.tolist() == [False, False, False]
    assert structure.n_atoms == 2
    # Molecular data still carries every field; the zero cell and the all-False
    # flags are data, not omissions.
    assert structure.movable_mask.all()


def test_structure_represents_periodic_crystal():
    structure = Structure(
        species=["Na", "Cl"],
        positions=[[0.0, 0.0, 0.0], [2.8, 2.8, 2.8]],
        cell=np.eye(3) * 5.6,
        pbc=[True, True, True],
    )

    assert structure.cell.shape == (3, 3)
    assert structure.pbc.dtype == np.bool_
    assert structure.pbc.tolist() == [True, True, True]


def test_missing_cell_normalizes_to_zero_and_pbc_to_false():
    structure = Structure(species=["H"], positions=[[0.0, 0.0, 0.0]])

    assert np.array_equal(structure.cell, np.zeros((3, 3)))
    assert structure.pbc.tolist() == [False, False, False]


def test_from_ase_reads_fixatoms_as_the_movement_constraint():
    atoms = Atoms("H3", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], pbc=False)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))

    structure = Structure.from_ase(atoms)

    assert structure.movable_mask.tolist() == [False, True, False]


def test_to_ase_derives_fixatoms_from_the_movement_constraint():
    structure = Structure(
        species=["H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        movable_mask=[False, True],
    )

    atoms = structure.to_ase()

    assert atoms.constraints[0].get_indices().tolist() == [0]


def test_event_rejects_frames_that_disagree_on_topology():
    reactant = Structure(["H", "O"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    wrong_species = Structure(["H", "C"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    wrong_cell = Structure(["H", "O"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=np.eye(3) * 5.0)

    for product, expected in ((wrong_species, "species"), (wrong_cell, "cells")):
        try:
            Event("e", reactant, product)
        except ValueError as error:
            assert expected in str(error)
        else:  # pragma: no cover
            raise AssertionError(f"reactant/product {expected} mismatch must be an error")


def test_catalog_keeps_two_events_with_different_reactants_side_by_side():
    """Two events are two units; no basin index can call them one."""
    first = Structure(["H", "O"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    second = Structure(["H", "O"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], movable_mask=[False, True])
    product = Structure(["H", "O"], [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])

    catalog = EventCatalog(
        events={
            "event-a": Event("event-a", first, product),
            "event-b": Event("event-b", second, product),
        },
    )

    assert catalog.event_ids == ["event-a", "event-b"]
    assert catalog.events["event-b"].reactant.movable_mask.tolist() == [False, True]


def test_a_singlepoint_calculator_does_not_become_structure_state():
    """Energy and forces are not part of the data-layer structure contract."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], pbc=False)
    atoms.calc = SinglePointCalculator(atoms, energy=-1.25, forces=np.zeros((2, 3)))

    structure = Structure.from_ase(atoms)
    exported = structure.to_ase()

    assert not hasattr(structure, "energy")
    assert not hasattr(structure, "forces")
    assert exported.calc is None
