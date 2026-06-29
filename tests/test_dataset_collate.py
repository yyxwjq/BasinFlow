import numpy as np

from fscgp.data.collate import collate_basins, collate_pairwise
from fscgp.data.dataset import EventDataset
from fscgp.data.records import BasinRecord, EventRecord, StructureRecord


def _dataset():
    structures = {
        "r0": StructureRecord("r0", ["H", "H"], [[0, 0, 0], [0.7, 0, 0]], np.eye(3) * 10, [False] * 3),
        "p0": StructureRecord("p0", ["H", "H"], [[0, 0, 0], [0.9, 0, 0]], np.eye(3) * 10, [False] * 3),
        "p1": StructureRecord("p1", ["H", "H"], [[0, 0, 0], [0.5, 0, 0]], np.eye(3) * 10, [False] * 3),
        "r1": StructureRecord("r1", ["Na"], [[0, 0, 0]], np.eye(3) * 5, [True] * 3),
        "p2": StructureRecord("p2", ["Na"], [[4.8, 0, 0]], np.eye(3) * 5, [True] * 3),
    }
    events = {
        "e0": EventRecord("e0", "r0", "p0", "b0"),
        "e1": EventRecord("e1", "r0", "p1", "b0"),
        "e2": EventRecord("e2", "r1", "p2", "b1"),
    }
    basins = {
        "b0": BasinRecord("b0", "r0", ["e0", "e1"]),
        "b1": BasinRecord("b1", "r1", ["e2"]),
    }
    return EventDataset(structures=structures, events=events, basins=basins)


def test_pairwise_view_derives_displacement_active_mask_and_event_direction():
    item = _dataset().pairwise_item("e0", active_threshold=0.1)

    assert item["event_id"] == "e0"
    assert item["reactant"].structure_id == "r0"
    assert item["product"].structure_id == "p0"
    assert item["active_mask"].tolist() == [False, True]
    assert np.allclose(item["event_direction"][1], [1, 0, 0])


def test_basin_view_preserves_multiple_known_events():
    item = _dataset().basin_item("b0")

    assert item["basin"].basin_id == "b0"
    assert [event.event_id for event in item["events"]] == ["e0", "e1"]
    assert [product.structure_id for product in item["products"]] == ["p0", "p1"]


def test_pairwise_collate_supports_variable_atom_counts_and_keeps_metadata():
    dataset = _dataset()
    batch = collate_pairwise([dataset.pairwise_item("e0"), dataset.pairwise_item("e2")])

    assert batch["reactant_positions"].shape == (3, 3)
    assert batch["product_positions"].shape == (3, 3)
    assert batch["reactant_batch"].tolist() == [0, 0, 1]
    assert batch["event_ids"] == ["e0", "e2"]
    assert batch["basin_ids"] == ["b0", "b1"]
    assert batch["cells"].shape == (2, 3, 3)
    assert batch["pbc"].tolist() == [[False, False, False], [True, True, True]]


def test_basin_collate_supports_variable_known_event_counts():
    dataset = _dataset()
    batch = collate_basins([dataset.basin_item("b0"), dataset.basin_item("b1")])

    assert batch["basin_ids"] == ["b0", "b1"]
    assert batch["known_event_ids"] == [["e0", "e1"], ["e2"]]
    assert batch["reactant_positions"].shape == (3, 3)
    assert batch["reactant_batch"].tolist() == [0, 0, 1]
    assert batch["cells"].shape == (2, 3, 3)
    assert batch["pbc"].tolist() == [[False, False, False], [True, True, True]]
