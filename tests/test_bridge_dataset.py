"""The bridge dataset view must reproduce ReactOT's training pair exactly.

``EnSB.sample_batch`` / ``q_sample`` / ``compute_label`` define the pair the
reference fits: the target is the recorded transition state, the prior is the
midpoint ``(R + P) / 2``, the sampled state is
``x_t = mu_x0 * TS + mu_x1 * midpoint`` and the supervised residual is
``(x_t - TS) / sigma_t`` with a per-sample bridge time. These tests pin that
contract on the ``EventData`` object the training loop will consume.
"""
from __future__ import annotations

import numpy as np
import pytest

from basinflow.data import Event, EventCatalog, Structure
from basinflow.data.samples import BridgeFlowDataset
from basinflow.models.sb import SBSchedule

SYMBOLS = ["C", "H", "O", "H"]
STEPS = 64
SCHEDULE_KWARGS = dict(timesteps=STEPS, beta_max=0.3, power=0.5, inv_power=1.0)


def _centered(rng: np.random.Generator) -> np.ndarray:
    positions = rng.normal(size=(len(SYMBOLS), 3))
    return positions - positions.mean(axis=0)


def _catalog(events: int = 2) -> EventCatalog:
    rng = np.random.default_rng(0)
    built = {}
    for index in range(events):
        reactant = _centered(rng)
        product = reactant + 0.4
        product -= product.mean(axis=0)
        state = 0.5 * (reactant + product) + 0.05
        state -= state.mean(axis=0)
        built[str(index)] = Event(
            event_id=str(index),
            reactant=Structure(SYMBOLS, reactant),
            product=Structure(SYMBOLS, product),
            transition_state=Structure(SYMBOLS, state),
            metadata={"source_file": f"event_{index}.traj"},
        )
    return EventCatalog(events=built)


def _dataset(**overrides) -> BridgeFlowDataset:
    parameters = dict(SCHEDULE_KWARGS, seed=7)
    parameters.update(overrides)
    return BridgeFlowDataset(_catalog(), **parameters)


def test_bridge_dataset_exposes_the_training_contract() -> None:
    dataset = _dataset()
    assert len(dataset) == 2
    assert dataset.catalog is not None
    assert dataset.epoch == 0
    assert dataset.seed == 7
    assert dataset.diagnostic_oracle is False
    assert dataset.fixed_flow_time is None


def test_bridge_sample_matches_the_schedule_algebra() -> None:
    dataset = _dataset()
    schedule = SBSchedule(**SCHEDULE_KWARGS)
    item = dataset[0]

    event = dataset.catalog.events["0"]
    target = event.transition_state.positions
    midpoint = 0.5 * (event.reactant.positions + event.product.positions)

    step = int(round(float(item.flow_time.reshape(-1)[0]) * STEPS))
    mu_x0, mu_x1 = float(schedule.mu_x0[step]), float(schedule.mu_x1[step])
    x_t = mu_x0 * target + mu_x1 * midpoint
    assert np.allclose(item.pos.numpy(), x_t, atol=1e-5)

    std_fwd = float(schedule.std_fwd[step])
    assert np.allclose(item.target_scaled.numpy(), (x_t - target) / std_fwd, atol=1e-5)


def test_bridge_sample_carries_both_endpoints_and_the_target() -> None:
    dataset = _dataset()
    event = dataset.catalog.events["1"]
    item = dataset[1]
    assert np.allclose(item.reactant_pos.numpy(), event.reactant.positions, atol=1e-6)
    assert np.allclose(item.endpoint_pos.numpy(), event.product.positions, atol=1e-6)
    assert np.allclose(item.target_pos.numpy(), event.transition_state.positions, atol=1e-6)
    assert np.allclose(item.source_pos.numpy(), 0.5 * (event.reactant.positions + event.product.positions), atol=1e-6)
    assert bool(item.movable_mask.all())
    assert item.z.tolist() == [6, 1, 8, 1]


def test_bridge_time_is_drawn_from_the_uniform_schedule_range() -> None:
    dataset = _dataset()
    times = {float(dataset[index].flow_time.reshape(-1)[0]) for index in range(len(dataset))}
    assert all(0.0 <= value < 1.0 for value in times)


def test_bridge_sample_is_reproducible_for_a_seed_and_epoch() -> None:
    first = _dataset()
    second = _dataset()
    assert np.array_equal(first[0].pos.numpy(), second[0].pos.numpy())
    assert float(first[0].flow_time) == float(second[0].flow_time)

    # A single index can legitimately redraw the same bridge step, so compare the
    # whole epoch's time draws rather than one item.
    before = [float(first[index].flow_time) for index in range(len(first))]
    first.set_epoch(3)
    after = [float(first[index].flow_time) for index in range(len(first))]
    assert before != after


def test_bridge_dataset_requires_a_transition_state() -> None:
    catalog = _catalog()
    broken = EventCatalog(
        events={
            "0": Event(
                event_id="0",
                reactant=catalog.events["0"].reactant,
                product=catalog.events["0"].product,
                transition_state=None,
            )
        }
    )
    with pytest.raises(ValueError, match="transition state"):
        BridgeFlowDataset(broken, **dict(SCHEDULE_KWARGS, seed=7))


def test_swap_reactant_product_only_swaps_the_two_endpoints():
    import numpy as np
    import torch

    from basinflow.data import Event, EventCatalog, Structure
    from basinflow.data.samples import BridgeFlowDataset

    reactant = Structure(["H", "H"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    product = Structure(["H", "H"], np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]))
    state = Structure(["H", "H"], np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]))
    catalog = EventCatalog(events={"0": Event("0", reactant, product, transition_state=state)})

    plain = BridgeFlowDataset(catalog, timesteps=16, seed=1)[0]
    swapped = BridgeFlowDataset(catalog, timesteps=16, seed=1, swap_reactant_product=True)[0]

    torch.testing.assert_close(swapped.reactant_pos, plain.endpoint_pos)
    torch.testing.assert_close(swapped.endpoint_pos, plain.reactant_pos)
    torch.testing.assert_close(swapped.source_pos, plain.source_pos)
    torch.testing.assert_close(swapped.target_pos, plain.target_pos)
