import numpy as np
import pytest

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord
from basinflow.seeds import DirectionalInit, GaussianInit, ZeroInit


def _two_event_basin():
    reactant = StructureRecord(
        "r",
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
        constraints=[True, False, True],
    )
    p1 = StructureRecord(
        "p1",
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
    )
    p2 = StructureRecord(
        "p2",
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [0.55, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
    )
    return EventCatalog(
        structures={"r": reactant, "p1": p1, "p2": p2},
        events={
            "e1": EventRecord("e1", "r", "p1", "b"),
            "e2": EventRecord("e2", "r", "p2", "b"),
        },
        basins={"b": BasinRecord("b", "r", ["e1", "e2"])},
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

    result = sampler.sample(catalog, "b")

    assert len(result.candidates) == 2
    assert len(result.generated_structures) == 2
    for candidate in result.candidates:
        generated = result.generated_structures[candidate.generated_structure_id]
        assert generated.n_atoms == 3
        assert generated.species == ["H", "O", "H"]
        assert np.isfinite(generated.positions).all()
        assert np.array_equal(generated.pbc, catalog.structures["r"].pbc)


def test_candidate_sampler_rejects_nonfinite_velocity():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class NonfiniteModel:
        builds_own_graph = True

        def eval(self):
            return self

        def __call__(self, batch, t):
            return {"velocity": torch.full_like(batch.pos, float("nan"))}

    with pytest.raises(FloatingPointError, match="basin b.*step 0"):
        CandidateSampler(NonfiniteModel(), [ZeroInit()], num_steps=1).sample(_two_event_basin(), "b")


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
    catalog.structures["r"].positions[0] = [0.123456789, 0.987654321, 0.111111111]
    result = CandidateSampler(ZeroModel(), [ZeroInit()], num_steps=1).sample(catalog, "b")
    generated = next(iter(result.generated_structures.values()))
    fixed = ~catalog.structures["r"].movable_mask
    np.testing.assert_array_equal(generated.positions[fixed], catalog.structures["r"].positions[fixed])


def test_candidate_sampler_honors_graph_update_interval():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class ZeroVelocityModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            return {
                "velocity": torch.zeros_like(batch["pos"]),
                "active_logits": torch.zeros_like(batch["movable_mask"], dtype=batch["pos"].dtype),
                "direction": torch.zeros_like(batch["pos"]),
            }

    class CountingSampler(CandidateSampler):
        def __post_init__(self):
            super().__post_init__()
            self.graph_builds = 0

        def _build_graph(self, batch):
            self.graph_builds += 1
            return super()._build_graph(batch)

    sampler = CountingSampler(
        ZeroVelocityModel(),
        [ZeroInit()],
        cutoff=2.5,
        num_steps=5,
        graph_update_interval=2,
    )

    sampler.sample(_two_event_basin(), "b")

    assert sampler.graph_builds == 3


def test_candidate_sampler_uses_active_prior_without_target_active_labels():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class RequiresPriorOnlyModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            assert "active_prior" in batch
            assert "target_active_mask" not in batch
            assert "active_mask" not in batch
            return {
                "velocity": torch.zeros_like(batch["pos"]),
                "active_logits": torch.zeros_like(
                    batch["active_prior"],
                    dtype=batch["pos"].dtype,
                ),
                "direction": torch.zeros_like(batch["pos"]),
            }

    sampler = CandidateSampler(
        RequiresPriorOnlyModel(),
        [ZeroInit()],
        cutoff=2.5,
        num_steps=1,
    )

    result = sampler.sample(_two_event_basin(), "b")

    assert len(result.candidates) == 1


def test_candidate_sampler_records_directional_init_metadata():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class ZeroVelocityModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            return {
                "velocity": torch.zeros_like(batch["pos"]),
                "active_logits": torch.zeros_like(
                    batch["active_prior"],
                    dtype=batch["pos"].dtype,
                ),
                "direction": torch.zeros_like(batch["pos"]),
            }

    sampler = CandidateSampler(
        ZeroVelocityModel(),
        [DirectionalInit(direction=(1.0, 0.0, 0.0), scale=0.2, movable_rank=0)],
        cutoff=2.5,
        num_steps=1,
    )

    result = sampler.sample(_two_event_basin(), "b")
    candidate = result.candidates[0]

    assert candidate.metadata["seed_type"] == "directional"
    assert candidate.metadata["seed_metadata"]["selected_atom"] == 1
    assert candidate.metadata["seed_metadata"]["direction"] == [1.0, 0.0, 0.0]


def test_velocity_only_sampling_derives_mic_event_semantics():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class VelocityModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            return {"velocity": torch.tensor([[10.0, 0.0, 0.0], [9.75, 0.0, 0.0], [10.0, 0.0, 0.0]])}

    catalog = _two_event_basin()
    for structure in catalog.structures.values():
        structure.pbc[:] = True
    result = CandidateSampler(VelocityModel(), [ZeroInit()], num_steps=1).sample(catalog, "b")
    candidate = result.candidates[0]

    assert candidate.predicted_active_atoms == [1]
    np.testing.assert_allclose(candidate.event_direction, [[0, 0, 0], [-1, 0, 0], [0, 0, 0]])
    assert candidate.metadata["active_atoms_source"] == "final_mic_displacement"
    assert candidate.metadata["event_direction_source"] == "final_mic_displacement"
    assert candidate.sampling_config["active_threshold"] == 0.1


def test_sampling_uses_left_endpoint_euler_and_fresh_edge_vectors():
    torch = pytest.importorskip("torch")
    from basinflow.sampling import CandidateSampler

    class InspectModel:
        def __init__(self):
            self.times = []

        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            self.times.append(t)
            source, target = batch.edge_index
            expected = batch.pos[target] - batch.pos[source]
            assert torch.allclose(batch.edge_vectors, expected)
            assert torch.allclose(batch.edge_lengths, torch.linalg.norm(expected, dim=1))
            return {"velocity": torch.ones_like(batch.pos), "active_logits": torch.zeros(batch.num_nodes), "direction": torch.zeros_like(batch.pos)}

    model = InspectModel()
    CandidateSampler(model, [ZeroInit()], num_steps=3, graph_update_interval=3).sample(_two_event_basin(), "b")

    assert model.times == [0.0, 1 / 3, 2 / 3]


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
            return {"velocity": torch.zeros_like(batch.pos), "active_logits": torch.zeros(batch.num_nodes), "direction": torch.zeros_like(batch.pos)}

    initializer = StatefulInit()
    result = CandidateSampler(ZeroModel(), [initializer], num_steps=1).sample(_two_event_basin(), "b")

    assert initializer.calls == 1
    assert result.candidates[0].metadata["seed_metadata"]["generation_call"] == 1


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

    result = NoExternalGraphSampler(OwnGraphModel(), [ZeroInit()], num_steps=2).sample(_two_event_basin(), "b")

    assert result.candidates[0].predicted_active_atoms == []
