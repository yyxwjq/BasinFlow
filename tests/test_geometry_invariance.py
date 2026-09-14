import numpy as np

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord


def _rotation_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _dataset_from_arrays(
    reactant_positions,
    product_positions,
    ts_positions=None,
    constraints=None,
    cell=None,
    pbc=None,
) -> EventCatalog:
    species = ["H"] * len(reactant_positions)
    cell = np.eye(3) * 10.0 if cell is None else cell
    pbc = [False, False, False] if pbc is None else pbc
    structures = {
        "r": StructureRecord(
            "r",
            species,
            reactant_positions,
            cell=cell,
            pbc=pbc,
            constraints=constraints,
        ),
        "p": StructureRecord("p", species, product_positions, cell=cell, pbc=pbc),
    }
    ts_id = None
    if ts_positions is not None:
        structures["ts"] = StructureRecord("ts", species, ts_positions, cell=cell, pbc=pbc)
        ts_id = "ts"
    events = {
        "e": EventRecord(
            "e",
            "r",
            "p",
            "b",
            transition_state_structure_id=ts_id,
        )
    }
    basins = {"b": BasinRecord("b", "r", ["e"])}
    return EventCatalog(structures=structures, events=events, basins=basins)


def test_pairwise_item_is_invariant_to_global_translation():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    product = np.array([[0.0, 0.0, 0.0], [1.0, 0.4, 0.0], [2.0, 0.0, 0.0]])
    ts = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.0, 0.0]])
    constraints = np.array([True, False, True])
    shift = np.array([3.2, -1.7, 5.5])

    original = _dataset_from_arrays(reactant, product, ts, constraints).event_target("e")
    translated = _dataset_from_arrays(
        reactant + shift,
        product + shift,
        ts + shift,
        constraints,
    ).event_target("e")

    assert np.allclose(translated.displacement, original.displacement)
    assert np.array_equal(translated.active_mask, original.active_mask)
    assert np.allclose(translated.direction, original.direction)


def test_pairwise_item_vectors_rotate_equivariantly_for_molecules():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    product = np.array([[0.0, 0.0, 0.0], [1.4, 0.3, 0.0], [2.0, 0.0, 0.0]])
    ts = np.array([[0.0, 0.0, 0.0], [1.2, 0.15, 0.0], [2.0, 0.0, 0.0]])
    rotation = _rotation_z(np.pi / 2.0)

    original = _dataset_from_arrays(reactant, product, ts).event_target("e")
    rotated = _dataset_from_arrays(
        reactant @ rotation.T,
        product @ rotation.T,
        ts @ rotation.T,
    ).event_target("e")

    assert np.allclose(rotated.displacement, original.displacement @ rotation.T)
    assert np.array_equal(rotated.active_mask, original.active_mask)
    assert np.allclose(rotated.direction, original.direction @ rotation.T)


def test_pairwise_item_fields_permute_consistently():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    product = np.array([[0.0, 0.2, 0.0], [1.0, 0.0, 0.0], [2.0, -0.3, 0.0]])
    ts = np.array([[0.0, 0.1, 0.0], [1.0, 0.0, 0.0], [2.0, -0.15, 0.0]])
    constraints = np.array([False, True, False])
    permutation = np.array([2, 0, 1])

    original = _dataset_from_arrays(reactant, product, ts, constraints).event_target("e")
    permuted = _dataset_from_arrays(
        reactant[permutation],
        product[permutation],
        ts[permutation],
        constraints[permutation],
    ).event_target("e")

    assert np.allclose(permuted.displacement, original.displacement[permutation])
    assert np.array_equal(permuted.active_mask, original.active_mask[permutation])
    assert np.allclose(permuted.direction, original.direction[permutation])


def test_periodic_pairwise_item_is_invariant_to_mic_safe_translation():
    cell = np.eye(3) * 5.0
    pbc = [True, True, True]
    reactant = np.array([[4.8, 1.0, 1.0], [2.0, 2.0, 2.0]])
    product = np.array([[0.2, 1.0, 1.0], [2.0, 2.3, 2.0]])
    shift = np.array([0.1, 0.2, 0.0])

    original = _dataset_from_arrays(
        reactant,
        product,
        cell=cell,
        pbc=pbc,
    ).event_target("e")
    translated = _dataset_from_arrays(
        reactant + shift,
        product + shift,
        cell=cell,
        pbc=pbc,
    ).event_target("e")

    assert np.allclose(translated.displacement, original.displacement)
    assert np.array_equal(translated.active_mask, original.active_mask)
    assert np.allclose(translated.direction, original.direction)
