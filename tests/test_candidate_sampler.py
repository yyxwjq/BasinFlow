import numpy as np
import pytest

from basinflow.data import Event, EventCatalog, Structure
from basinflow.seeds import DirectionalInit, GaussianInit, ZeroInit


def _two_event_basin():
    reactant = Structure(
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
        movable_mask=[False, True, False],
    )
    first = Structure(
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
    )
    second = Structure(
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [0.55, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
    )
    return EventCatalog(
        events={
            "e1": Event("e1", reactant, first),
            "e2": Event("e2", reactant, second),
        },
    )


def test_candidate_sampler_generates_multiple_finite_candidates():
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.sampling import CandidateSampler

    torch.manual_seed(0)
    catalog = _two_event_basin()
    model = EGNNFlow(hidden_dim=24, num_layers=2, cutoff=2.5)
    sampler = CandidateSampler(
        model,
        [ZeroInit(), GaussianInit(scale=0.05, random_seed=7)],
        cutoff=2.5,
        num_steps=2,
    )

    result = sampler.sample(catalog, "e1")

    assert len(result.candidates) == 2
    assert len(result.generated_structures) == 2
    for candidate in result.candidates:
        generated = result.generated_structures[candidate.generated_structure_id]
        assert generated.n_atoms == 3
        assert generated.species == ["H", "O", "H"]
        assert np.isfinite(generated.positions).all()
        assert np.array_equal(generated.pbc, catalog.events["e1"].reactant.pbc)


def test_candidate_sampler_rejects_nonfinite_velocity():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class NonfiniteModel:
        builds_own_graph = True

        def eval(self):
            return self

        def __call__(self, batch, t):
            return {"velocity": torch.full_like(batch.pos, float("nan"))}

    with pytest.raises(FloatingPointError, match="non-finite velocity"):
        CandidateSampler(NonfiniteModel(), [ZeroInit()], num_steps=2).sample(_two_event_basin(), "e1")


def test_candidate_sampler_preserves_original_precision_on_fixed_atoms():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class ZeroModel:
        builds_own_graph = True

        def eval(self):
            return self

        def __call__(self, batch, t):
            return {"velocity": torch.zeros_like(batch.pos)}

    catalog = _two_event_basin()
    reactant = catalog.events["e1"].reactant
    reactant.positions[0] = [0.123456789, 0.987654321, 0.111111111]
    result = CandidateSampler(ZeroModel(), [ZeroInit()], num_steps=2).sample(catalog, "e1")
    generated = next(iter(result.generated_structures.values()))
    fixed = ~reactant.movable_mask
    np.testing.assert_array_equal(generated.positions[fixed], reactant.positions[fixed])


def test_egnn_graph_is_rebuilt_on_every_field_evaluation():
    """An adaptive solver's steps are non-uniform, so the graph cannot be cached
    on a step counter: EGNN (which does not build its own graph) must be rebuilt
    at each field evaluation, exactly as PaiNN and MolGEN's backbone do."""
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class CountingSampler(CandidateSampler):
        def __post_init__(self):
            super().__post_init__()
            self.graph_builds = 0

        def _build_graph(self, batch):
            self.graph_builds += 1
            return super()._build_graph(batch)

    class ZeroVelocityModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            return {"velocity": torch.zeros_like(batch.pos)}

    sampler = CountingSampler(ZeroVelocityModel(), [ZeroInit()], cutoff=2.5, num_steps=5)
    sampler.sample(_two_event_basin(), "e1")

    # dopri5 evaluates the field several times per saved interval, so a run over
    # five saved points makes far more than five graph builds.
    assert sampler.graph_builds > 5


def test_candidate_sampler_uses_active_prior_without_target_active_labels():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class RequiresPriorOnlyModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            assert "active_prior" in batch
            assert not any(key.startswith("target") for key in batch.keys())
            return {"velocity": torch.zeros_like(batch["pos"])}

    sampler = CandidateSampler(
        RequiresPriorOnlyModel(),
        [ZeroInit()],
        cutoff=2.5,
        num_steps=2,
    )

    result = sampler.sample(_two_event_basin(), "e1")

    assert len(result.candidates) == 1


def test_candidate_sampler_records_directional_init_metadata():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class ZeroVelocityModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            return {"velocity": torch.zeros_like(batch["pos"])}

    sampler = CandidateSampler(
        ZeroVelocityModel(),
        [DirectionalInit(direction=(1.0, 0.0, 0.0), scale=0.2, movable_rank=0)],
        cutoff=2.5,
        num_steps=2,
    )

    result = sampler.sample(_two_event_basin(), "e1")
    candidate = result.candidates[0]

    assert candidate.seed_type == "directional"
    assert candidate.seed_id == "e1:seed:0:directional"
    assert candidate.candidate_id == "e1:candidate:0"


def test_sampling_queries_the_field_at_current_positions_with_consistent_edges():
    """Every field evaluation must see edges built from the *current* positions.

    The EGNN path does not carry a prebuilt graph, so the sampler rebuilds it
    inside the field. An edge vector left over from an earlier state would make
    the message passing read a stale geometry.
    """
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class InspectModel:
        def __init__(self):
            self.times = []

        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            self.times.append(float(t))
            source, target = batch.edge_index
            expected = batch.pos[target] - batch.pos[source]
            assert torch.allclose(batch.edge_vectors, expected)
            assert torch.allclose(batch.edge_lengths, torch.linalg.norm(expected, dim=1))
            return {"velocity": torch.ones_like(batch.pos)}

    model = InspectModel()
    CandidateSampler(model, [ZeroInit()], num_steps=4).sample(_two_event_basin(), "e1")

    # A non-uniform set of query times spanning the interval, not an Euler grid.
    assert len(model.times) > 4
    assert min(model.times) >= 0.0
    assert max(model.times) >= 0.9


def test_sampling_records_the_actual_seed_without_regenerating():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class StatefulInit:
        seed_type = "gaussian_movable"

        def __init__(self):
            self.calls = 0

        def generate(self, context, seed_id=None):
            self.calls += 1
            seed = GaussianInit(scale=0.1, random_seed=self.calls).generate(context, seed_id)
            seed.metadata["generation_call"] = self.calls
            return seed

    class ZeroModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            return {"velocity": torch.zeros_like(batch.pos)}

    initializer = StatefulInit()
    result = CandidateSampler(ZeroModel(), [initializer], num_steps=2).sample(_two_event_basin(), "e1")

    assert initializer.calls == 1
    # The candidate records which seed actually produced it instead of
    # regenerating one, so the geometry and the reported provenance cannot drift.
    assert result.candidates[0].seed_id.endswith(":seed:0:gaussian_movable")
    assert result.candidates[0].seed_type == "gaussian_movable"


def test_sampler_leaves_graph_construction_to_self_contained_models():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class OwnGraphModel:
        builds_own_graph = True

        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            assert "edge_vectors" not in batch
            return {"velocity": torch.zeros_like(batch.pos)}

    class NoExternalGraphSampler(CandidateSampler):
        def _build_graph(self, batch):
            raise AssertionError("external graph should not be constructed")

    result = NoExternalGraphSampler(OwnGraphModel(), [ZeroInit()], num_steps=2).sample(_two_event_basin(), "e1")

    assert result.candidates[0].candidate_id == "e1:candidate:0"
