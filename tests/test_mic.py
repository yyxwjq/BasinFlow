import numpy as np

from basinflow.geometry.mic import (
    derive_active_atoms,
    derive_event_direction,
    displacement_norms,
    minimum_image_displacement,
    pairwise_displacements,
)


def test_nonperiodic_molecule_displacement_is_direct_even_with_cell():
    disp = minimum_image_displacement(
        [9.8, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        cell=np.eye(3) * 10.0,
        pbc=[False, False, False],
    )

    assert np.allclose(disp, [9.6, 0.0, 0.0])


def test_periodic_boundary_crossing_uses_minimum_image():
    disp = minimum_image_displacement(
        [9.8, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        cell=np.eye(3) * 10.0,
        pbc=[True, True, True],
    )

    assert np.allclose(disp, [-0.4, 0.0, 0.0])


def test_mixed_pbc_flags_only_wrap_periodic_directions():
    disp = minimum_image_displacement(
        [9.8, 9.8, 0.0],
        [0.2, 0.2, 0.0],
        cell=np.eye(3) * 10.0,
        pbc=[True, False, False],
    )

    assert np.allclose(disp, [-0.4, 9.6, 0.0])


def test_pairwise_displacements_norms_active_masks_and_event_directions():
    reactant = np.array([[0.0, 0.0, 0.0], [9.8, 0.0, 0.0], [1.0, 1.0, 1.0]])
    product = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.02, 1.0, 1.0]])
    disp = pairwise_displacements(product, reactant, cell=np.eye(3) * 10.0, pbc=[True, True, True])
    norms = displacement_norms(disp)
    active = derive_active_atoms(disp, threshold=0.1)
    direction = derive_event_direction(disp, active_mask=active)

    assert np.allclose(disp[1], [0.4, 0.0, 0.0])
    assert np.allclose(norms, [0.0, 0.4, 0.02])
    assert active.tolist() == [False, True, False]
    assert np.allclose(direction[1], [1.0, 0.0, 0.0])
    assert np.allclose(direction[[0, 2]], 0.0)
