import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.loader import DataLoader

from basinflow.data.catalog import EventCatalog
from basinflow.data.pyg import BasinDataset, EventFlowDataset
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord
from basinflow.seeds import GaussianInit, ProductInit, ZeroInit
from basinflow.sampling import CandidateSampler
from basinflow.models import EGNNFlow, flow_loss


def _catalog():
    structures = {
        "r": StructureRecord("r", ["H"], [[4.8, 0.0, 0.0]], cell=np.eye(3) * 5, pbc=True),
        "p": StructureRecord("p", ["H"], [[0.2, 0.0, 0.0]], cell=np.eye(3) * 5, pbc=True),
    }
    events = {"e": EventRecord("e", "r", "p", "b")}
    basins = {"b": BasinRecord("b", "r", ["e"])}
    return EventCatalog(structures, events, basins)


def test_product_flow_dataset_emits_one_supervised_pyg_sample():
    dataset = EventFlowDataset(_catalog(), [ZeroInit()], flow_time=0.25)

    sample = dataset[0]

    assert sample.pos.shape == (1, 3)
    assert torch.allclose(sample.target_pos, torch.tensor([[5.2, 0.0, 0.0]]))
    assert sample.flow_time.tolist() == [0.25]
    assert hasattr(sample, "target_active_mask")
    assert not hasattr(sample, "transition_state")


def test_basin_proposal_dataset_never_exposes_event_targets():
    dataset = BasinDataset(_catalog(), [ZeroInit()])

    sample = dataset[0]

    assert sample.basin_id == "b"
    assert not hasattr(sample, "target_pos")
    assert not hasattr(sample, "target_velocity")
    assert not hasattr(sample, "target_active_mask")


def test_product_init_requires_explicit_oracle_diagnostic_mode():
    with pytest.raises(ValueError, match="diagnostic_oracle"):
        EventFlowDataset(_catalog(), [ProductInit()])


def test_event_data_batches_graph_fields_by_graph_not_atom():
    dataset = EventFlowDataset(_catalog(), [ZeroInit()], flow_time=0.5)
    batch = next(iter(DataLoader([dataset[0], dataset[0]], batch_size=2)))

    assert batch.cell.shape == (2, 3, 3)
    assert batch.pbc.shape == (2, 3)
    assert batch.flow_time.shape == (2, 1)
    assert batch.seed_type_id.shape == (2, 1)


def test_event_data_batches_initializers_without_internal_metadata_dicts():
    dataset = EventFlowDataset(
        _catalog(),
        [ZeroInit(), GaussianInit(scale=0.05, random_seed=7)],
        flow_time=0.5,
    )

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch.num_graphs == 2
    assert not hasattr(batch, "seed_metadata")


def test_candidate_sampler_accepts_catalog_and_never_needs_event_targets():
    class ZeroModel:
        def eval(self):
            return self

        def __call__(self, batch, t=0.0):
            assert not hasattr(batch, "target_pos")
            return {
                "velocity": torch.zeros_like(batch.pos),
                "active_logits": torch.zeros(batch.num_nodes),
                "direction": torch.zeros_like(batch.pos),
            }

    result = CandidateSampler(ZeroModel(), [ZeroInit()], num_steps=1).sample(_catalog(), "b")

    assert len(result.candidates) == 1
    assert result.candidates[0].sampling_config["evaluation_mode"] == "basin_level_no_oracle_no_relaxation"


def test_egnn_consumes_pyg_batch_and_rebuilds_graph_from_pos():
    batch = next(iter(DataLoader(EventFlowDataset(_catalog(), [ZeroInit()], flow_time=0.5), batch_size=1)))

    output = EGNNFlow(hidden_dim=8, num_layers=1, cutoff=2.5)(batch)
    loss, _ = flow_loss(output, batch)

    assert output["velocity"].shape == (1, 3)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("random_seed", [7, None])
def test_gaussian_training_noise_is_epoch_fresh_and_reproducible(random_seed):
    dataset = EventFlowDataset(_catalog(), [GaussianInit(random_seed=random_seed)], seed=19)
    first = dataset[0]
    dataset.set_epoch(1)
    second = dataset[0]
    replica = EventFlowDataset(_catalog(), [GaussianInit(random_seed=random_seed)], seed=19)
    replica.set_epoch(1)

    assert not torch.equal(first.seed_displacement, second.seed_displacement)
    assert torch.equal(second.seed_displacement, replica[0].seed_displacement)
    assert torch.allclose(second.target_velocity, second.target_pos - second.source_pos, atol=5e-7)
    assert torch.allclose(second.pos, second.source_pos + second.flow_time * second.target_velocity)


def test_velocity_only_loss_never_reads_disabled_auxiliary_labels():
    from basinflow.models import FlowLossWeights

    velocity = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    batch = {"movable_mask": torch.tensor([True]), "target_velocity": torch.zeros_like(velocity)}
    loss, parts = flow_loss({"velocity": velocity}, batch)
    loss.backward()

    assert FlowLossWeights().active == 0.0
    assert FlowLossWeights().direction == 0.0
    assert loss.item() == pytest.approx(14 / 3)
    assert torch.allclose(velocity.grad, 2 * velocity.detach() / 3)
    assert parts["active_loss"].item() == 0
    assert parts["direction_loss"].item() == 0
