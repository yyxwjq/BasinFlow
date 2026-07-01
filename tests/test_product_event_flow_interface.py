import os

import numpy as np
import pytest

from fscgp.data.dataset import EventDataset
from fscgp.data.raw_events import read_events_directory
from fscgp.data.records import BasinRecord, EventRecord, StructureRecord
from fscgp.flow import build_product_flow_item
from fscgp.models import DummyProductEventFlow, masked_velocity_mse
from fscgp.seeds import ProductDisplacementSeedGenerator, ZeroSeedGenerator


def _flow_item():
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
    item = EventDataset(structures, events, basins).pairwise_item("e")
    return build_product_flow_item(item, ZeroSeedGenerator().generate(item))


def test_dummy_product_event_flow_forward_returns_vector_field_shape():
    flow_item = _flow_item()
    model = DummyProductEventFlow()

    output = model.forward(flow_item, t=0.5)

    assert output.shape == (3, 3)


def test_dummy_product_event_flow_zeros_fixed_atom_outputs():
    flow_item = _flow_item()
    model = DummyProductEventFlow(mode="target_velocity")

    output = model.forward(flow_item, t=0.5)

    assert np.allclose(output[~flow_item["movable_mask"]], 0.0)
    assert np.allclose(output[flow_item["movable_mask"]], flow_item["target_velocity"][flow_item["movable_mask"]])


def test_masked_velocity_mse_uses_movable_atoms_only():
    flow_item = _flow_item()
    prediction = np.zeros_like(flow_item["target_velocity"])
    prediction[~flow_item["movable_mask"]] = 100.0

    loss = masked_velocity_mse(prediction, flow_item)
    expected = np.mean(flow_item["target_velocity"][flow_item["movable_mask"]] ** 2)

    assert np.isclose(loss, expected)


def test_dummy_product_event_flow_can_return_zero_velocity_for_product_seed():
    item = EventDataset(
        {
            "r": StructureRecord("r", ["H"], [[0.0, 0.0, 0.0]], np.eye(3) * 10, [False] * 3),
            "p": StructureRecord("p", ["H"], [[0.2, 0.0, 0.0]], np.eye(3) * 10, [False] * 3),
        },
        {"e": EventRecord("e", "r", "p", "b")},
        {"b": BasinRecord("b", "r", ["e"])},
    ).pairwise_item("e")
    seed = ProductDisplacementSeedGenerator().generate(item)
    flow_item = build_product_flow_item(item, seed)

    output = DummyProductEventFlow(mode="target_velocity").forward(flow_item, t=0.5)

    assert np.allclose(output, 0.0)


def test_dummy_product_event_flow_works_on_real_au_event_when_env_is_set():
    events_dir = os.environ.get("BASINFLOW_EVENTS_DIR")
    if not events_dir:
        pytest.skip("BASINFLOW_EVENTS_DIR is not set")

    dataset = read_events_directory(events_dir)
    item = dataset.pairwise_item(dataset.event_ids[0])
    seed = ZeroSeedGenerator().generate(item)
    flow_item = build_product_flow_item(item, seed)

    output = DummyProductEventFlow(mode="target_velocity").forward(flow_item, t=0.5)

    assert output.shape == (101, 3)
    assert np.allclose(output[~flow_item["movable_mask"]], 0.0)
    assert masked_velocity_mse(output, flow_item) == 0.0
