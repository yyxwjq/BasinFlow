"""Reference training-contract tests: batching, clipping and the loss scale."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from basinflow.models import flow_loss  # noqa: E402
from basinflow.training.batch_sampler import NodeSquareBatchSampler  # noqa: E402
from basinflow.training.grad_clip import AdaptiveGradClipper  # noqa: E402


def test_node_square_sampler_packs_by_squared_atom_count():
    sampler = NodeSquareBatchSampler([10, 10, 10, 10], max_num=200, drop_last=False)
    batches = list(sampler)

    assert [len(batch) for batch in batches] == [2, 2]
    for batch in batches:
        assert sum(sampler.costs[index] for index in batch) <= 200
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]


def test_node_square_sampler_yields_a_single_batch_for_a_small_dataset():
    sampler = NodeSquareBatchSampler([4, 4], max_num=2800)  # drop_last defaults True
    assert [batch for batch in sampler] == [[0, 1]]


def test_node_square_sampler_keeps_an_oversized_item_alone():
    sampler = NodeSquareBatchSampler([50], max_num=200, drop_last=False)
    assert [batch for batch in sampler] == [[0]]


def test_node_square_sampler_reshuffles_between_epochs():
    sampler = NodeSquareBatchSampler(list(range(1, 21)), max_num=10**6, drop_last=False)

    first = [index for batch in sampler for index in batch]
    second = [index for batch in sampler for index in batch]

    assert sorted(first) == sorted(second) == list(range(20))
    assert first != second


def test_adaptive_clipper_starts_at_the_reference_bound_and_records_the_clipped_norm():
    clipper = AdaptiveGradClipper()
    assert clipper.bound() == pytest.approx(4500.0)

    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])  # norm 5
    norm, bound = clipper.step([parameter])

    assert norm == pytest.approx(5.0)
    assert bound == pytest.approx(4500.0)
    assert clipper.queue.items[0] == pytest.approx(5.0)


def test_adaptive_clipper_clips_to_the_rolling_bound():
    clipper = AdaptiveGradClipper()
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3000.0, 4000.0])  # norm 5000 > 4500

    norm, bound = clipper.step([parameter])

    assert norm == pytest.approx(5000.0)
    assert bound == pytest.approx(4500.0)
    assert float(parameter.grad.norm()) == pytest.approx(bound, rel=1e-6)
    assert clipper.queue.items[0] == pytest.approx(bound)


def test_bridge_loss_is_plain_mse_like_en_sb():
    prediction = torch.zeros(2, 3)
    target = torch.full((2, 3), 2.0)
    batch = {"target_scaled": target, "movable_mask": torch.ones(2, dtype=torch.bool)}

    total, parts = flow_loss({"scaled": prediction}, batch)

    # SBModule returns EnSB.forward's F.mse_loss, with no fragment scale.
    assert float(total) == pytest.approx(4.0)
    assert float(parts["scaled_loss"]) == pytest.approx(4.0)


def test_reference_checkpoint_key_mapping_roundtrips(tmp_path):
    """The released ckpt stores LEFTNet as ddpm.dynamics.{model,encoders,decoders}."""
    from basinflow.models.leftnet.adapter import LeftNetBridge
    from basinflow.models.reference import load_reactot_checkpoint

    original = LeftNetBridge(hidden_channels=32, num_layers=2, num_radial=8, cutoff=5.0)
    state = {}
    for key, value in original.state_dict().items():
        if key.startswith("leftnet."):
            state["ddpm.dynamics.model." + key[len("leftnet."):]] = value
        else:
            state["ddpm.dynamics." + key] = value
    checkpoint = {
        "state_dict": state,
        "hyper_parameters": {
            "model_config": {
                "hidden_channels": 32,
                "num_layers": 2,
                "num_radial": 8,
                "cutoff": 5.0,
                "reflect_equiv": True,
                "legacy": True,
                "update": True,
                "single_layer_output": True,
                "object_aware": True,
            },
            "node_nfs": [9, 9, 9],
        },
    }
    path = tmp_path / "fake-reactot.ckpt"
    torch.save(checkpoint, path)

    loaded, hyper = load_reactot_checkpoint(path)

    assert hyper["node_nfs"] == [9, 9, 9]
    torch.testing.assert_close(
        loaded.state_dict()["leftnet.embedding.weight"],
        original.state_dict()["leftnet.embedding.weight"],
    )


def test_bridge_validation_metric_is_zero_for_a_perfect_prediction():
    import numpy as np

    from basinflow.data import Event, EventCatalog, Structure
    from basinflow.data.samples import BridgeFlowDataset
    from basinflow.training.validation import evaluate_bridge_scaled_error

    def structure(offset: float) -> Structure:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) + offset
        return Structure(["H", "H"], positions)

    catalog = EventCatalog(
        events={
            "0": Event("0", structure(0.0), structure(0.2), transition_state=structure(0.1)),
            "1": Event("1", structure(0.3), structure(0.4), transition_state=structure(0.35)),
        }
    )
    dataset = BridgeFlowDataset(catalog, timesteps=16, lowest_step=1)

    class PerfectModel(torch.nn.Module):
        def forward(self, batch):
            return {"scaled": batch.target_scaled}

    assert evaluate_bridge_scaled_error(PerfectModel(), dataset, max_batches=2) == 0.0


def test_load_initial_weights_reads_a_basinflow_checkpoint(tmp_path):
    from basinflow.models.leftnet.adapter import LeftNetBridge
    from basinflow.training.init import load_initial_weights

    source = LeftNetBridge(hidden_channels=32, num_layers=2, num_radial=8, cutoff=5.0)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": source.state_dict()}, path)
    target = LeftNetBridge(hidden_channels=32, num_layers=2, num_radial=8, cutoff=5.0)

    assert load_initial_weights(target, path) == "basinflow"
    torch.testing.assert_close(
        target.state_dict()["leftnet.embedding.weight"],
        source.state_dict()["leftnet.embedding.weight"],
    )
