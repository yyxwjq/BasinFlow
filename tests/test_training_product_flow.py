import numpy as np
import pytest

from basinflow.data.catalog import EventCatalog
from basinflow.data.pyg import EventFlowDataset
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord
from basinflow.seeds import ProductInit, ZeroInit


def _toy_dataset():
    structures = {
        "r": StructureRecord(
            "r",
            ["H", "O", "H"],
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
            cell=np.eye(3) * 10,
            pbc=[False, False, False],
            constraints=[True, False, True],
        ),
        "p": StructureRecord(
            "p",
            ["H", "O", "H"],
            [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [1.6, 0.0, 0.0]],
            cell=np.eye(3) * 10,
            pbc=[False, False, False],
        ),
    }
    return EventCatalog(
        structures=structures,
        events={"e": EventRecord("e", "r", "p", "b")},
        basins={"b": BasinRecord("b", "r", ["e"])},
    )


def test_epoch_resume_matches_continuous_training_and_restores_optimizer(tmp_path):
    import copy
    import torch
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow
    from basinflow.seeds import GaussianInit

    torch.manual_seed(17)
    initial = EGNNFlow(hidden_dim=8, num_layers=1, cutoff=2.5)
    continuous = copy.deepcopy(initial)
    interrupted = copy.deepcopy(initial)
    initializers = [ZeroInit(), GaussianInit(scale=.1, random_seed=31)]
    torch.manual_seed(99)
    expected = train_product_flow(continuous, _toy_dataset(), initializers, epochs=2, lr=.001)
    torch.manual_seed(99)
    train_product_flow(interrupted, _toy_dataset(), initializers, epochs=1, lr=.001,
                       epoch_checkpoint_dir=tmp_path / 'first')
    restored = copy.deepcopy(initial)
    torch.manual_seed(700)
    actual = train_product_flow(restored, _toy_dataset(), initializers, epochs=2, lr=.001,
                               resume_from=tmp_path / 'first/epoch_0001.pt')
    assert actual == expected
    for name, value in continuous.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_resume_rejects_partial_epoch_or_changed_training_data(tmp_path):
    import torch
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow
    from basinflow.seeds import GaussianInit

    model = EGNNFlow(hidden_dim=8, num_layers=1, cutoff=2.5)
    initializers = [ZeroInit(), GaussianInit(scale=.1, random_seed=31)]
    train_product_flow(model, _toy_dataset(), initializers, epochs=1, max_steps=1,
                       epoch_checkpoint_dir=tmp_path / 'partial')
    with pytest.raises(ValueError, match='complete epoch'):
        train_product_flow(model, _toy_dataset(), initializers, epochs=2,
                           resume_from=tmp_path / 'partial/epoch_0001.pt')
    train_product_flow(model, _toy_dataset(), initializers, epochs=1,
                       epoch_checkpoint_dir=tmp_path / 'complete')
    changed = _toy_dataset()
    changed.structures['p'].positions[1, 0] += .1
    with pytest.raises(ValueError, match='dataset'):
        train_product_flow(model, changed, initializers, epochs=2,
                           resume_from=tmp_path / 'complete/epoch_0001.pt')


def test_train_product_flow_epoch_reduces_loss_on_toy_dataset():
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow_epoch

    torch.manual_seed(0)
    dataset = _toy_dataset()
    model = EGNNFlow(hidden_dim=24, num_layers=2, cutoff=2.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    first = train_product_flow_epoch(
        model,
        dataset,
        [ZeroInit()],
        optimizer,
    )
    last = first
    for _ in range(30):
        last = train_product_flow_epoch(
            model,
            dataset,
            [ZeroInit()],
            optimizer,
        )

    assert last["loss"] < first["loss"]


def test_train_product_flow_saves_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    torch.manual_seed(0)
    checkpoint = tmp_path / "model.pt"
    model = EGNNFlow(hidden_dim=16, num_layers=1, cutoff=2.5)

    history = train_product_flow(
        model,
        _toy_dataset(),
        [ZeroInit()],
        epochs=1,
        lr=0.01,
        checkpoint_path=checkpoint,
    )

    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "model_state_dict" in saved
    assert saved["config"]["epochs"] == 1
    assert len(history) == 1


def test_init_rmsd_baseline_reports_oracle_product_init():
    from basinflow.evaluation import evaluate_init_rmsd_baseline

    report = evaluate_init_rmsd_baseline(
        _toy_dataset(),
        {"oracle_product_displacement": ProductInit()},
    )

    overall = report["oracle_product_displacement"]["overall"]
    assert overall["num_items"] == 1
    assert overall["mean_rmsd"] == 0.0


def test_train_product_flow_writes_step_log(tmp_path):
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    torch.manual_seed(0)
    log_path = tmp_path / "training.log"
    model = EGNNFlow(hidden_dim=16, num_layers=1, cutoff=2.5)

    history = train_product_flow(
        model,
        EventFlowDataset(_toy_dataset(), [ZeroInit(), ProductInit()], diagnostic_oracle=True),
        epochs=1,
        lr=0.01,
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "epoch step total_loss velocity_loss active_loss direction_loss"
    assert len(lines) == 3
    assert [(int(parts[0]), int(parts[1])) for parts in (line.split() for line in lines[1:])] == [
        (1, 1),
        (1, 2),
    ]
    assert len(lines[1].split()) == 6
    assert history[0]["num_items"] == 2


def test_train_product_flow_full_batch_logs_one_step_for_all_items(tmp_path):
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    torch.manual_seed(0)
    log_path = tmp_path / "training.log"
    model = EGNNFlow(hidden_dim=16, num_layers=1, cutoff=2.5)

    history = train_product_flow(
        model,
        EventFlowDataset(_toy_dataset(), [ZeroInit(), ProductInit()], diagnostic_oracle=True),
        epochs=1,
        lr=0.01,
        log_path=log_path,
        batch_size="full",
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "epoch step total_loss velocity_loss active_loss direction_loss"
    assert len(lines) == 2
    assert [int(value) for value in lines[1].split()[:2]] == [1, 1]
    assert history[0]["num_items"] == 2
    assert history[0]["num_batches"] == 1
    assert history[0]["last_step"] == 1


def test_active_threshold_controls_catalog_supervision_labels():
    catalog = _toy_dataset()

    assert catalog.event_target("e", active_threshold=0.3).active_mask.sum() == 0


def test_active_pos_weight_uses_movable_train_atoms_only():
    from basinflow.training import active_pos_weight

    assert active_pos_weight(_toy_dataset(), active_threshold=0.2) == 0.0


def test_active_pos_weight_returns_one_for_all_negative_train_labels():
    from basinflow.training import active_pos_weight

    assert active_pos_weight(_toy_dataset(), active_threshold=0.3) == 1.0


def test_train_epoch_uses_one_optimizer_step_per_pyg_batch():
    torch = pytest.importorskip("torch")
    from basinflow.data import EventCatalog, EventFlowDataset
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow_epoch

    class TrackingOptimizer:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.steps = 0

        def zero_grad(self):
            self.optimizer.zero_grad()

        def step(self):
            self.optimizer.step()
            self.steps += 1

    torch.manual_seed(0)
    model = EGNNFlow(hidden_dim=16, num_layers=1, cutoff=2.5)
    optimizer = TrackingOptimizer(torch.optim.Adam(model.parameters(), lr=0.01))
    catalog = _toy_dataset()
    dataset = EventFlowDataset(catalog, [ZeroInit(), ZeroInit()], flow_time=0.5)

    metrics = train_product_flow_epoch(
        model,
        dataset,
        optimizer,
        batch_size=1,
    )

    assert metrics["num_batches"] == 2
    assert optimizer.steps == 2


def test_training_catalog_does_not_implicitly_enable_oracle():
    pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    with pytest.raises(ValueError, match="diagnostic_oracle"):
        train_product_flow(EGNNFlow(hidden_dim=8, num_layers=1), _toy_dataset(), [ProductInit()])


def test_training_respects_global_optimizer_update_budget(tmp_path):
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    checkpoint = tmp_path / "checkpoint.pt"
    history = train_product_flow(
        EGNNFlow(hidden_dim=8, num_layers=1),
        _toy_dataset(),
        [ZeroInit(), ZeroInit(), ZeroInit()],
        epochs=5,
        max_steps=4,
        checkpoint_path=checkpoint,
    )

    assert [epoch["num_batches"] for epoch in history] == [3, 1]
    assert [epoch["num_items"] for epoch in history] == [3, 1]
    assert history[-1]["last_step"] == 4
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["config"]["max_steps"] == 4
    assert saved["config"]["max_grad_norm"] == 10.0


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_training_rejects_nonfinite_updates(failure):
    torch = pytest.importorskip("torch")
    from basinflow.training import train_product_flow_epoch

    class NonfiniteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            if failure == "gradient":
                self.weight.register_hook(lambda gradient: gradient * float("nan"))

        def forward(self, batch):
            velocity = self.weight * torch.ones_like(batch.pos)
            if failure == "loss":
                velocity = velocity * float("nan")
            return {"velocity": velocity}

    model = NonfiniteModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises((ValueError, RuntimeError, FloatingPointError), match="[Nn]on.?finite|finite"):
        train_product_flow_epoch(model, _toy_dataset(), [ZeroInit()], optimizer)
    assert model.weight.item() == 1.0


def test_training_clips_finite_gradients_before_optimizer_update():
    torch = pytest.importorskip("torch")
    from basinflow.training import train_product_flow_epoch

    class ConstantVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(100.0))

        def forward(self, batch):
            return {"velocity": self.weight * torch.ones_like(batch.pos)}

    model = ConstantVelocity()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    train_product_flow_epoch(
        model, _toy_dataset(), [ZeroInit()], optimizer, max_grad_norm=0.5
    )

    assert model.weight.item() == pytest.approx(99.5)


def test_velocity_validation_is_fixed_order_deterministic_and_restores_state():
    torch = pytest.importorskip("torch")
    from basinflow.seeds import GaussianInit
    from basinflow.training import evaluate_product_flow_velocity

    class TrackingVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.inputs = []

        def forward(self, batch):
            assert not self.training
            assert not torch.is_grad_enabled()
            self.inputs.append(batch.pos.clone())
            return {"velocity": self.weight * batch.pos}

    dataset = EventFlowDataset(
        _toy_dataset(), [GaussianInit(scale=0.05)] * 3, seed=71
    )
    model = TrackingVelocity()
    expected_positions = [dataset[index].pos for index in range(len(dataset))]
    squared_error = sum(
        float(((0.5 * sample.pos - sample.target_velocity)[sample.movable_mask] ** 2).sum())
        for sample in (dataset[index] for index in range(len(dataset)))
    )
    dataset.set_epoch(9)
    rng_state = torch.get_rng_state().clone()

    first = evaluate_product_flow_velocity(model, dataset, batch_size=2)
    assert model.training
    assert dataset.epoch == 9
    assert model.weight.grad is None
    assert torch.equal(torch.get_rng_state(), rng_state)
    torch.testing.assert_close(torch.cat(model.inputs), torch.cat(expected_positions))
    assert first["velocity_loss"] == pytest.approx(squared_error / 9)
    assert first["num_items"] == 3
    assert first["num_batches"] == 2
    assert first["num_components"] == 9

    model.eval()
    second = evaluate_product_flow_velocity(model, dataset, batch_size=1)
    assert not model.training
    assert second["velocity_loss"] == pytest.approx(first["velocity_loss"])


def test_velocity_validation_restores_model_on_nonfinite_prediction():
    torch = pytest.importorskip("torch")
    from basinflow.training import evaluate_product_flow_velocity

    class InvalidVelocity(torch.nn.Module):
        def forward(self, batch):
            return {"velocity": torch.full_like(batch.pos, float("nan"))}

    dataset = EventFlowDataset(_toy_dataset(), [ZeroInit()])
    dataset.set_epoch(7)
    model = InvalidVelocity()
    with pytest.raises(FloatingPointError, match="non-finite"):
        evaluate_product_flow_velocity(model, dataset)
    assert model.training
    assert dataset.epoch == 7


def test_training_validates_each_epoch_and_saves_budget_truncated_state(tmp_path):
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    train_dataset = EventFlowDataset(_toy_dataset(), [ZeroInit()] * 3)
    validation_dataset = EventFlowDataset(_toy_dataset(), [ZeroInit()] * 2)
    checkpoint = tmp_path / "checkpoint.pt"
    epoch_dir = tmp_path / "epochs"
    model = EGNNFlow(hidden_dim=8, num_layers=1)
    history = train_product_flow(
        model, train_dataset, epochs=5, max_steps=4,
        validation_dataset=validation_dataset, validation_batch_size=2,
        checkpoint_path=checkpoint, epoch_checkpoint_dir=epoch_dir,
        optimizer_name="adamw", amsgrad=True, weight_decay=0.01,
    )

    assert len(history) == 2
    assert all(np.isfinite(item["validation_velocity_loss"]) for item in history)
    assert all(item["validation_num_items"] == 2 for item in history)
    assert all(item["validation_num_batches"] == 1 for item in history)
    first = torch.load(epoch_dir / "epoch_0001.pt", weights_only=False)
    last = torch.load(epoch_dir / "epoch_0002.pt", weights_only=False)
    assert first["epoch_complete"] is True
    assert first["completed_epochs"] == 1
    assert first["completed_steps"] == 3
    assert first["budget_truncated"] is False
    assert len(first["history"]) == 1
    assert last["epoch"] == 2
    assert last["epoch_complete"] is False
    assert last["completed_epochs"] == 1
    assert last["completed_steps"] == 4
    assert last["budget_truncated"] is True
    assert last["history"] == history
    assert last["config"]["optimizer_name"] == "adamw"
    assert last["optimizer_state_dict"]["param_groups"][0]["amsgrad"] is True
    assert last["optimizer_state_dict"]["param_groups"][0]["weight_decay"] == 0.01
    assert last["optimizer_state_dict"]["state"]
    assert torch.equal(last["torch_rng_state"], torch.get_rng_state())
    final = torch.load(checkpoint, weights_only=False)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(last["model_state_dict"][name], parameter)
        torch.testing.assert_close(final["model_state_dict"][name], parameter)


@pytest.mark.parametrize("configuration", [
    {"optimizer_name": "sgd"},
    {"lr": float("nan")},
    {"lr": -0.1},
    {"weight_decay": -0.1},
    {"weight_decay": float("inf")},
    {"amsgrad": "false"},
])
def test_training_rejects_invalid_optimizer_configuration(configuration):
    pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    with pytest.raises(ValueError):
        train_product_flow(
            EGNNFlow(hidden_dim=8, num_layers=1),
            _toy_dataset(), [ZeroInit()], **configuration,
        )


@pytest.mark.parametrize("max_steps, expected_epochs, truncated", [(3, 1, True), (6, 2, False)])
def test_epoch_boundary_budget_records_completed_epochs(tmp_path, max_steps, expected_epochs, truncated):
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    checkpoint = tmp_path / "checkpoint.pt"
    train_product_flow(
        EGNNFlow(hidden_dim=8, num_layers=1),
        _toy_dataset(), [ZeroInit()] * 3,
        epochs=2, max_steps=max_steps, checkpoint_path=checkpoint,
    )

    saved = torch.load(checkpoint, weights_only=False)
    assert saved["completed_epochs"] == expected_epochs
    assert saved["epoch_complete"] is True
    assert saved["budget_truncated"] is truncated


def test_training_rejects_empty_validation_before_updating_model():
    torch = pytest.importorskip("torch")
    from basinflow.models import EGNNFlow
    from basinflow.training import train_product_flow

    model = EGNNFlow(hidden_dim=8, num_layers=1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    empty_catalog = EventCatalog(structures={}, events={}, basins={})
    with pytest.raises(ValueError, match="validation dataset"):
        train_product_flow(
            model, _toy_dataset(), [ZeroInit()],
            validation_dataset=EventFlowDataset(empty_catalog, [ZeroInit()]),
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_cosine_schedule_warms_up_then_decays_and_can_be_disabled():
    torch = pytest.importorskip("torch")
    from basinflow.training.product_flow import build_scheduler

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)

    assert build_scheduler(optimizer, name="constant") is None

    scheduler = build_scheduler(optimizer, name="cosine", total_steps=100,
                               warmup_fraction=0.1, min_lr_fraction=0.05)
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert rates[0] < rates[9]                     # warmup rises
    assert rates[9] == pytest.approx(1.0, abs=1e-6)  # peak at the end of warmup
    # The decay reaches the floor one step past the last recorded rate.
    assert rates[-1] == pytest.approx(0.05, abs=5e-3)
    assert rates[-1] < rates[50]
    assert all(a >= b - 1e-9 for a, b in zip(rates[9:], rates[10:]))  # monotone decay

    with pytest.raises(ValueError, match="scheduler must be"):
        build_scheduler(optimizer, name="linear")
    with pytest.raises(ValueError, match="warmup_fraction"):
        build_scheduler(optimizer, name="cosine", total_steps=10, warmup_fraction=1.0)


def test_scheduler_step_budget_covers_the_requested_epochs():
    pytest.importorskip("torch")
    from basinflow.training.product_flow import total_scheduler_steps

    class _Dataset:
        def __len__(self):
            return 130

    assert total_scheduler_steps(10, _Dataset(), 64, None) == 30   # ceil(130/64) * 10
    assert total_scheduler_steps(10, _Dataset(), 64, 12) == 12
    assert total_scheduler_steps(0, _Dataset(), 64, None) == 1     # never zero


def test_scheduler_continues_across_a_resumed_budget():
    torch = pytest.importorskip("torch")
    from basinflow.training.product_flow import build_scheduler

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, name="cosine", total_steps=1000,
                                warmup_fraction=0.05, min_lr_fraction=0.05)

    for _ in range(500):
        optimizer.step()
        scheduler.step()
    halfway = optimizer.param_groups[0]["lr"]

    resumed = build_scheduler(optimizer, name="cosine", total_steps=1000,
                              warmup_fraction=0.05, min_lr_fraction=0.05)
    resumed.last_epoch = 500
    resumed.step()  # advances to step 501, one past the recorded rate
    assert optimizer.param_groups[0]["lr"] == pytest.approx(halfway, rel=1e-2)
    assert optimizer.param_groups[0]["lr"] > 0.5  # not restarted at warmup
