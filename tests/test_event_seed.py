import numpy as np
import pytest

from fscgp.data.dataset import EventDataset
from fscgp.data.records import BasinRecord, EventRecord, StructureRecord
from fscgp.seeds import EventSeed, event_seed_from_pairwise_item


def _pairwise_item():
    structures = {
        "r": StructureRecord(
            "r",
            ["H", "H", "H"],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            cell=np.eye(3) * 10.0,
            pbc=[False, False, False],
            constraints=[True, False, True],
        ),
        "p": StructureRecord(
            "p",
            ["H", "H", "H"],
            [[0.0, 0.0, 0.0], [1.0, 0.4, 0.0], [2.0, 0.0, 0.0]],
            cell=np.eye(3) * 10.0,
            pbc=[False, False, False],
        ),
    }
    events = {"e": EventRecord("e", "r", "p", "b")}
    basins = {"b": BasinRecord("b", "r", ["e"])}
    return EventDataset(structures, events, basins).pairwise_item("e")


def test_event_seed_validates_array_shapes():
    with pytest.raises(ValueError, match="seed_displacement"):
        EventSeed(
            seed_id="bad",
            seed_type="test",
            seed_displacement=np.zeros((3,)),
            seed_direction=np.zeros((3, 3)),
            active_prior=np.zeros(3),
            movable_mask=np.ones(3, dtype=bool),
        )


def test_event_seed_rejects_nonzero_fixed_atom_vectors():
    movable_mask = np.array([False, True, False])
    displacement = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="fixed atoms"):
        EventSeed(
            seed_id="bad-fixed",
            seed_type="test",
            seed_displacement=displacement,
            seed_direction=np.zeros((3, 3)),
            active_prior=np.zeros(3),
            movable_mask=movable_mask,
        )


def test_event_seed_initial_positions_are_reactant_plus_seed_displacement():
    pairwise = _pairwise_item()
    seed = EventSeed(
        seed_id="manual",
        seed_type="test",
        seed_displacement=np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]),
        seed_direction=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        active_prior=pairwise["active_mask"],
        movable_mask=pairwise["movable_mask"],
    )

    assert np.allclose(
        seed.initial_positions(pairwise["reactant"].positions),
        pairwise["reactant"].positions + seed.seed_displacement,
    )


def test_event_seed_can_be_built_from_pairwise_item():
    pairwise = _pairwise_item()

    seed = event_seed_from_pairwise_item(
        pairwise,
        seed_id="from-pair",
        seed_type="product_displacement",
        seed_displacement=pairwise["displacement"],
    )

    assert seed.seed_id == "from-pair"
    assert seed.seed_type == "product_displacement"
    assert np.array_equal(seed.movable_mask, pairwise["movable_mask"])
    assert np.array_equal(seed.active_prior, pairwise["active_mask"])
    assert np.allclose(seed.seed_displacement[0], 0.0)
    assert np.allclose(seed.seed_displacement[2], 0.0)
    assert np.allclose(seed.seed_direction[1], [0.0, 1.0, 0.0])
