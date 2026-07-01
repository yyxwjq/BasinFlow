import os

import numpy as np
import pytest

from fscgp.data.dataset import EventDataset
from fscgp.data.raw_events import read_events_directory
from fscgp.data.records import BasinRecord, EventRecord, StructureRecord
from fscgp.flow import build_product_flow_item
from fscgp.seeds import ZeroSeedGenerator, event_seed_from_pairwise_item


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


def test_product_flow_item_initial_positions_are_reactant_plus_seed():
    item = _pairwise_item()
    seed = ZeroSeedGenerator().generate(item)

    flow_item = build_product_flow_item(item, seed)

    assert np.allclose(
        flow_item["initial_positions"],
        flow_item["reactant_positions"] + flow_item["seed_displacement"],
    )


def test_product_flow_item_target_velocity_matches_product_minus_initial_for_non_pbc():
    item = _pairwise_item()
    seed = event_seed_from_pairwise_item(
        item,
        seed_id="half",
        seed_type="manual",
        seed_displacement=item["displacement"] * 0.5,
    )

    flow_item = build_product_flow_item(item, seed)

    assert np.allclose(
        flow_item["target_velocity"],
        item["product"].positions - flow_item["initial_positions"],
    )


def test_product_flow_item_rejects_seed_from_different_mask_contract():
    item = _pairwise_item()
    bad_seed = ZeroSeedGenerator().generate(item)
    bad_seed.movable_mask = np.array([True, True, True])

    with pytest.raises(ValueError, match="mask contract"):
        build_product_flow_item(item, bad_seed)


def test_product_flow_item_exposes_mic_target_positions_for_periodic_events():
    cell = np.eye(3) * 5.0
    structures = {
        "r": StructureRecord(
            "r",
            ["H"],
            [[4.8, 0.0, 0.0]],
            cell=cell,
            pbc=[True, True, True],
        ),
        "p": StructureRecord(
            "p",
            ["H"],
            [[0.2, 0.0, 0.0]],
            cell=cell,
            pbc=[True, True, True],
        ),
    }
    events = {"e": EventRecord("e", "r", "p", "b")}
    basins = {"b": BasinRecord("b", "r", ["e"])}
    item = EventDataset(structures, events, basins).pairwise_item("e")

    flow_item = build_product_flow_item(item, ZeroSeedGenerator().generate(item))

    assert np.allclose(flow_item["target_displacement"], [[0.4, 0.0, 0.0]])
    assert np.allclose(flow_item["target_positions"], [[5.2, 0.0, 0.0]])
    assert np.allclose(flow_item["product_positions"], [[0.2, 0.0, 0.0]])


def test_product_flow_item_zeros_fixed_atom_target_velocity():
    item = _pairwise_item()
    seed = ZeroSeedGenerator().generate(item)

    flow_item = build_product_flow_item(item, seed)

    fixed = ~item["movable_mask"]
    assert np.allclose(flow_item["seed_displacement"][fixed], 0.0)
    assert np.allclose(flow_item["target_velocity"][fixed], 0.0)


def test_product_flow_item_passes_masks_and_direction_through():
    item = _pairwise_item()
    seed = ZeroSeedGenerator().generate(item)

    flow_item = build_product_flow_item(item, seed)

    assert np.array_equal(flow_item["active_mask"], item["active_mask"])
    assert np.array_equal(flow_item["movable_mask"], item["movable_mask"])
    assert np.allclose(flow_item["event_direction"], item["event_direction"])


def test_product_flow_item_works_on_real_events_when_env_is_set():
    events_dir = os.environ.get("BASINFLOW_EVENTS_DIR")
    if not events_dir:
        pytest.skip("BASINFLOW_EVENTS_DIR is not set")

    dataset = read_events_directory(events_dir)
    item = dataset.pairwise_item(dataset.event_ids[0])
    seed = ZeroSeedGenerator().generate(item)

    flow_item = build_product_flow_item(item, seed)

    assert flow_item["reactant_positions"].shape == (101, 3)
    assert flow_item["initial_positions"].shape == (101, 3)
    assert flow_item["target_velocity"].shape == (101, 3)
    assert np.allclose(flow_item["target_velocity"][~item["movable_mask"]], 0.0)
