import numpy as np

from basinflow.data import Event, EventCatalog, Structure, event_target


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


def _catalog_from_arrays(
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
    movable_mask = None if constraints is None else ~np.asarray(constraints, dtype=bool)
    reactant = Structure(
        species, reactant_positions, cell=cell, pbc=pbc, movable_mask=movable_mask
    )
    product = Structure(species, product_positions, cell=cell, pbc=pbc)
    state = None if ts_positions is None else Structure(species, ts_positions, cell=cell, pbc=pbc)
    return EventCatalog(
        events={"e": Event("e", reactant, product, state)})


def _target(reactant_positions, product_positions, ts_positions=None, constraints=None,
            cell=None, pbc=None):
    catalog = _catalog_from_arrays(
        reactant_positions, product_positions, ts_positions, constraints, cell, pbc
    )
    return event_target(catalog.events["e"])


def test_pairwise_item_is_invariant_to_global_translation():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    product = np.array([[0.0, 0.0, 0.0], [1.0, 0.4, 0.0], [2.0, 0.0, 0.0]])
    ts = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.0, 0.0]])
    constraints = np.array([True, False, True])
    shift = np.array([3.2, -1.7, 5.5])

    original = _target(reactant, product, ts, constraints)
    translated = _target(reactant + shift, product + shift, ts + shift, constraints)

    assert np.allclose(translated.displacement, original.displacement)
    assert np.allclose(translated.target_positions - shift, original.target_positions)


def test_pairwise_item_vectors_rotate_equivariantly_for_molecules():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    product = np.array([[0.0, 0.0, 0.0], [1.4, 0.3, 0.0], [2.0, 0.0, 0.0]])
    ts = np.array([[0.0, 0.0, 0.0], [1.2, 0.15, 0.0], [2.0, 0.0, 0.0]])
    rotation = _rotation_z(np.pi / 2.0)

    original = _target(reactant, product, ts)
    rotated = _target(reactant @ rotation.T, product @ rotation.T, ts @ rotation.T)

    assert np.allclose(rotated.displacement, original.displacement @ rotation.T)
    assert np.allclose(rotated.target_positions, original.target_positions @ rotation.T)


def test_pairwise_item_fields_permute_consistently():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    product = np.array([[0.0, 0.2, 0.0], [1.0, 0.0, 0.0], [2.0, -0.3, 0.0]])
    ts = np.array([[0.0, 0.1, 0.0], [1.0, 0.0, 0.0], [2.0, -0.15, 0.0]])
    constraints = np.array([False, True, False])
    permutation = np.array([2, 0, 1])

    original = _target(reactant, product, ts, constraints)
    permuted = _target(
        reactant[permutation], product[permutation], ts[permutation], constraints[permutation]
    )

    assert np.allclose(permuted.displacement, original.displacement[permutation])
    # The movement constraint is per atom and must follow the permutation.
    assert np.array_equal(
        permuted.reactant.movable_mask, original.reactant.movable_mask[permutation]
    )


def test_periodic_pairwise_item_is_invariant_to_mic_safe_translation():
    cell = np.eye(3) * 5.0
    pbc = [True, True, True]
    reactant = np.array([[4.8, 1.0, 1.0], [2.0, 2.0, 2.0]])
    product = np.array([[0.2, 1.0, 1.0], [2.0, 2.3, 2.0]])
    shift = np.array([0.1, 0.2, 0.0])

    original = _target(reactant, product, cell=cell, pbc=pbc)
    translated = _target(reactant + shift, product + shift, cell=cell, pbc=pbc)

    assert np.allclose(translated.displacement, original.displacement)


def test_minimum_image_displacement_is_the_short_way_round_the_cell():
    cell = np.eye(3) * 5.0
    pbc = [True, True, True]
    # 4.8 -> 0.2 is +0.4 across the boundary, not -4.6 back through the cell.
    target = _target(
        np.array([[4.8, 1.0, 1.0]]), np.array([[0.2, 1.0, 1.0]]), cell=cell, pbc=pbc
    )

    assert np.allclose(target.displacement, [[0.4, 0.0, 0.0]])
    # The lifted endpoint follows the interpolant instead of jumping a cell.
    assert np.allclose(target.target_positions, [[5.2, 1.0, 1.0]])
