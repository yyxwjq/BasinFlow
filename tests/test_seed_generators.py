import numpy as np

from fscgp.data.dataset import EventDataset
from fscgp.data.records import BasinRecord, EventRecord, StructureRecord
from fscgp.seeds import (
    GaussianMovableSeedGenerator,
    ProductDisplacementSeedGenerator,
    ZeroSeedGenerator,
)


def _pairwise_item():
    structures = {
        "r": StructureRecord(
            "r",
            ["Au", "Au", "Au"],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            cell=np.eye(3) * 10.0,
            pbc=[False, False, False],
            constraints=[True, False, True],
        ),
        "p": StructureRecord(
            "p",
            ["Au", "Au", "Au"],
            [[0.0, 0.0, 0.0], [1.0, 0.4, 0.0], [2.0, 0.0, 0.0]],
            cell=np.eye(3) * 10.0,
            pbc=[False, False, False],
        ),
    }
    events = {"e": EventRecord("e", "r", "p", "b")}
    basins = {"b": BasinRecord("b", "r", ["e"])}
    return EventDataset(structures, events, basins).pairwise_item("e")


def _pairwise_item_with_event_id(event_id: str):
    item = _pairwise_item()
    item["event_id"] = event_id
    return item


def test_zero_seed_generator_keeps_initial_geometry_equal_to_reactant():
    item = _pairwise_item()
    seed = ZeroSeedGenerator().generate(item)

    assert seed.seed_type == "zero"
    assert np.allclose(seed.seed_displacement, 0.0)
    assert np.allclose(seed.seed_direction, 0.0)
    assert np.allclose(seed.initial_positions(item["reactant"].positions), item["reactant"].positions)


def test_gaussian_movable_seed_generator_is_reproducible():
    item = _pairwise_item()

    first = GaussianMovableSeedGenerator(scale=0.2, random_seed=7).generate(item)
    second = GaussianMovableSeedGenerator(scale=0.2, random_seed=7).generate(item)

    assert np.allclose(first.seed_displacement, second.seed_displacement)
    assert np.allclose(first.seed_direction, second.seed_direction)


def test_gaussian_movable_seed_generator_changes_with_random_seed():
    item = _pairwise_item()

    first = GaussianMovableSeedGenerator(scale=0.2, random_seed=7).generate(item)
    second = GaussianMovableSeedGenerator(scale=0.2, random_seed=8).generate(item)

    assert not np.allclose(first.seed_displacement, second.seed_displacement)


def test_gaussian_movable_seed_generator_varies_by_event_id_when_seeded():
    first = GaussianMovableSeedGenerator(scale=0.2, random_seed=7).generate(
        _pairwise_item_with_event_id("e1")
    )
    second = GaussianMovableSeedGenerator(scale=0.2, random_seed=7).generate(
        _pairwise_item_with_event_id("e2")
    )

    assert not np.allclose(first.seed_displacement, second.seed_displacement)


def test_gaussian_movable_seed_generator_keeps_fixed_atoms_zero():
    item = _pairwise_item()

    seed = GaussianMovableSeedGenerator(scale=0.2, random_seed=7).generate(item)

    fixed = ~item["movable_mask"]
    assert np.allclose(seed.seed_displacement[fixed], 0.0)
    assert np.allclose(seed.seed_direction[fixed], 0.0)
    assert np.linalg.norm(seed.seed_displacement[item["movable_mask"]]) > 0.0


def test_product_displacement_seed_matches_movable_pairwise_displacement():
    item = _pairwise_item()

    seed = ProductDisplacementSeedGenerator().generate(item)

    assert seed.seed_type == "product_displacement"
    assert np.allclose(
        seed.seed_displacement[item["movable_mask"]],
        item["displacement"][item["movable_mask"]],
    )
    assert np.allclose(seed.seed_displacement[~item["movable_mask"]], 0.0)
