import numpy as np

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord
from basinflow.seeds import (
    DirectionalInit,
    GaussianInit,
    ProductInit,
    SeedContext,
    ZeroInit,
)


def _seed_context():
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
    catalog = EventCatalog(structures, events, basins)
    target = catalog.event_target("e")
    return SeedContext(target.event.event_id, target.event.basin_id, target.reactant), target


def _seed_context_with_event_id(event_id: str):
    context, target = _seed_context()
    return SeedContext(event_id, context.basin_id, context.reactant), target


def test_zero_seed_generator_keeps_initial_geometry_equal_to_reactant():
    context, _ = _seed_context()
    seed = ZeroInit().generate(context)

    assert seed.seed_type == "zero"
    assert np.allclose(seed.seed_displacement, 0.0)
    assert np.allclose(seed.seed_direction, 0.0)
    assert np.allclose(seed.initial_positions(context.reactant.positions), context.reactant.positions)


def test_gaussian_movable_seed_generator_is_reproducible():
    context, _ = _seed_context()

    first = GaussianInit(scale=0.2, random_seed=7).generate(context)
    second = GaussianInit(scale=0.2, random_seed=7).generate(context)

    assert np.allclose(first.seed_displacement, second.seed_displacement)
    assert np.allclose(first.seed_direction, second.seed_direction)


def test_gaussian_movable_seed_generator_changes_with_random_seed():
    context, _ = _seed_context()

    first = GaussianInit(scale=0.2, random_seed=7).generate(context)
    second = GaussianInit(scale=0.2, random_seed=8).generate(context)

    assert not np.allclose(first.seed_displacement, second.seed_displacement)


def test_gaussian_movable_seed_generator_varies_by_event_id_when_seeded():
    first = GaussianInit(scale=0.2, random_seed=7).generate(
        _seed_context_with_event_id("e1")[0]
    )
    second = GaussianInit(scale=0.2, random_seed=7).generate(
        _seed_context_with_event_id("e2")[0]
    )

    assert not np.allclose(first.seed_displacement, second.seed_displacement)


def test_gaussian_movable_seed_generator_keeps_fixed_atoms_zero():
    context, _ = _seed_context()

    seed = GaussianInit(scale=0.2, random_seed=7).generate(context)

    fixed = ~context.movable_mask
    assert np.allclose(seed.seed_displacement[fixed], 0.0)
    assert np.allclose(seed.seed_direction[fixed], 0.0)
    assert np.linalg.norm(seed.seed_displacement[context.movable_mask]) > 0.0


def test_directional_init_uses_single_movable_atom_and_direction():
    context, _ = _seed_context()

    seed = DirectionalInit(
        direction=(0.0, 2.0, 0.0),
        scale=0.3,
        movable_rank=0,
    ).generate(context)

    assert seed.seed_type == "directional"
    assert np.array_equal(seed.active_prior, [False, True, False])
    assert np.allclose(seed.seed_displacement[0], 0.0)
    assert np.allclose(seed.seed_displacement[1], [0.0, 0.3, 0.0])
    assert np.allclose(seed.seed_displacement[2], 0.0)
    assert np.allclose(seed.seed_direction[1], [0.0, 1.0, 0.0])
    assert seed.metadata["selected_atom"] == 1


def test_product_displacement_seed_matches_movable_pairwise_displacement():
    context, target = _seed_context()

    seed = ProductInit().generate(context, target_displacement=target.displacement)

    assert seed.seed_type == "product_displacement"
    assert np.allclose(
        seed.seed_displacement[context.movable_mask],
        target.displacement[context.movable_mask],
    )
    assert np.allclose(seed.seed_displacement[~context.movable_mask], 0.0)
