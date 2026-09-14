import json

import numpy as np
import pytest
from ase import Atoms
import ase.io

from basinflow.data.catalog import BasinSplit, EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord


def _catalog(*, active_atoms=None):
    structures = {
        "reactant": StructureRecord(
            "reactant",
            ["H", "H"],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            movable_mask=[True, True],
        ),
        "product": StructureRecord(
            "product",
            ["H", "H"],
            [[0.3, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ),
    }
    events = {
        "event": EventRecord(
            "event",
            "reactant",
            "product",
            "basin",
            active_atoms=active_atoms,
        )
    }
    basins = {"basin": BasinRecord("basin", "reactant", ["event"])}
    return EventCatalog(structures=structures, events=events, basins=basins)


def test_catalog_preserves_explicit_active_labels_over_displacement():
    catalog = _catalog(active_atoms=[1])

    target = catalog.event_target("event", active_threshold=0.1)

    assert target.active_mask.tolist() == [False, True]
    assert np.allclose(target.displacement, [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_catalog_derives_active_labels_from_mic_displacement_without_annotation():
    catalog = _catalog()

    target = catalog.event_target("event", active_threshold=0.1)

    assert target.active_mask.tolist() == [True, False]


def test_catalog_allows_event_specific_reactant_within_basin():
    catalog = _catalog()
    catalog.events["event"].reactant_structure_id = "product"

    validated = EventCatalog(catalog.structures, catalog.events, catalog.basins)

    assert validated.events["event"].reactant_structure_id == "product"


def test_basin_split_round_trips_and_selects_whole_basins(tmp_path):
    catalog = _catalog()
    split = BasinSplit(train_ids=("basin",), val_ids=(), test_ids=(), seed=7)
    path = tmp_path / "split.json"

    split.save(path)
    loaded = BasinSplit.load(path)

    assert loaded == split
    assert loaded.select(catalog, "train").basin_ids == ["basin"]


@pytest.mark.parametrize(
    "train_ids,val_ids,test_ids,error",
    [
        (("basin", "basin"), (), (), "duplicate"),
        (("basin",), ("basin",), (), "overlap"),
        ((), ("basin",), ("basin",), "overlap"),
        (("",), (), (), "nonempty string"),
        (("   ",), (), (), "nonempty string"),
        ((1,), (), (), "nonempty string"),
        ((None,), (), (), "nonempty string"),
        ("basin", (), (), "list or tuple"),
    ],
)
def test_basin_split_rejects_ambiguous_membership(train_ids, val_ids, test_ids, error):
    with pytest.raises(ValueError, match=error):
        BasinSplit(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids, seed=42)


def test_basin_split_validates_catalog_membership_and_optional_completeness():
    catalog = _catalog()
    exact = BasinSplit(train_ids=("basin",), val_ids=(), test_ids=(), seed=42)
    exact.validate(catalog)
    empty = BasinSplit(train_ids=(), val_ids=(), test_ids=(), seed=42)
    empty.validate(catalog, require_complete=False)
    with pytest.raises(ValueError, match="missing.*basin"):
        empty.validate(catalog)
    unknown = BasinSplit(train_ids=("unknown",), val_ids=(), test_ids=(), seed=42)
    with pytest.raises(ValueError, match="unknown.*unknown"):
        unknown.validate(catalog, require_complete=False)


@pytest.mark.parametrize("train_ids", [[1], "basin", ["basin", "basin"]])
def test_basin_split_load_rejects_malformed_ids_without_coercion(tmp_path, train_ids):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": train_ids, "val": [], "test": [], "seed": 42}))
    with pytest.raises(ValueError):
        BasinSplit.load(path)


def test_basin_split_freezes_list_membership_without_reordering():
    identifiers = ["second", "first"]
    split = BasinSplit(train_ids=identifiers, val_ids=[], test_ids=[], seed=42)
    identifiers.append("third")
    assert split.train_ids == ("second", "first")


def test_eon_catalog_keeps_movable_constraint_separate_from_active_label(tmp_path):
    (tmp_path / "event_0.extxyz").write_text(
        """2
Lattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:move_mask:L:1 pbc=\"F F F\"
H 0 0 0 T
H 1 0 0 F
2
Lattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:move_mask:L:1 pbc=\"F F F\"
H 0 0 0 T
H 1.3 0 0 F
""",
        encoding="utf-8",
    )

    catalog = EventCatalog.from_eon_directory(tmp_path)
    target = catalog.event_target("event_0", active_threshold=0.1)

    assert target.reactant.movable_mask.tolist() == [True, False]
    assert target.active_mask.tolist() == [False, True]
    assert catalog.events["event_0"].active_atoms is None


def test_eon_catalog_rejects_event_files_with_more_than_three_frames(tmp_path):
    frames = [Atoms("H", positions=[[value, 0.0, 0.0]]) for value in range(4)]
    ase.io.write(tmp_path / "event_0.extxyz", frames)

    with pytest.raises(ValueError, match="two or three frames"):
        EventCatalog.from_eon_directory(tmp_path)
