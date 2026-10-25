import numpy as np
import pytest
from ase import Atoms
import ase.io

from basinflow.data import (
    Event,
    EventCatalog,
    Structure,
    check_split,
    event_target,
    load_events,
    load_split,
    ordered_event_ids,
    partition,
    save_split,
)


def _catalog():
    reactant = Structure(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        movable_mask=[True, True],
    )
    product = Structure(["H", "H"], [[0.3, 0.0, 0.0], [1.0, 0.0, 0.0]])
    return EventCatalog(events={"0": Event("0", reactant, product)})


def test_event_target_is_the_mic_displacement_and_the_lifted_endpoint():
    target = event_target(_catalog().events["0"])

    assert np.allclose(target.displacement, [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(target.target_positions, target.reactant.positions + target.displacement)
    # A target carries no stored active-atom or direction labels.
    assert not hasattr(target, "active_mask")
    assert not hasattr(target, "direction")


def test_split_round_trips_and_selects_whole_events(tmp_path):
    catalog = _catalog()
    split = {"train": ("0",), "val": (), "test": ()}
    path = tmp_path / "split.csv"

    save_split(split, path)
    loaded = load_split(path, catalog)

    assert loaded == check_split(split)
    assert catalog.select(partition(loaded, "train")).event_ids == ["0"]


@pytest.mark.parametrize(
    "split",
    [
        {"train": ("0",), "val": ("0",), "test": ()},
        {"train": ("0",), "val": (), "test": ("0",)},
        {"train": ("0", "0"), "val": (), "test": ()},
    ],
)
def test_check_split_rejects_ambiguous_membership(split):
    with pytest.raises(ValueError, match="repeat"):
        check_split(split)


def test_check_split_requires_every_partition():
    with pytest.raises(ValueError, match="exactly the partitions"):
        check_split({"train": ("0",), "val": ()})


def test_selecting_an_unknown_event_is_an_error_not_an_empty_catalog():
    catalog = _catalog()
    split = {"train": ("7",), "val": (), "test": ()}

    with pytest.raises(KeyError, match="unknown event ids"):
        catalog.select(partition(split, "train"))


def test_check_split_freezes_list_membership_without_reordering():
    identifiers = ["2", "1"]
    split = check_split({"train": identifiers, "val": [], "test": []})
    identifiers.append("3")

    assert split["train"] == ("2", "1")


def test_event_catalog_has_no_second_index_to_keep_in_sync():
    """One event is one unit: there is no basin index beside the event mapping."""
    catalog = _catalog()

    assert not hasattr(catalog, "basins")
    assert not hasattr(catalog.events["0"], "basin_id")
    assert len(catalog) == 1
    assert "0" in catalog


def test_ordered_event_ids_are_numeric_not_lexicographic(tmp_path):
    for index in (2, 10, 1):
        ase.io.write(
            tmp_path / f"event_{index}.traj",
            [
                Atoms("H", positions=[[0.0, 0.0, 0.0]]),
                Atoms("H", positions=[[0.4, 0.0, 0.0]]),
                Atoms("H", positions=[[0.2, 0.0, 0.0]]),
            ],
        )
    catalog = load_events(tmp_path)

    # Numeric order is the contract that keeps split.csv valid: "10" comes last,
    # even though the file name sorts before "2" as a string.
    assert catalog.event_ids == ["1", "2", "10"]
    assert ordered_event_ids(catalog) == ["1", "2", "10"]


def test_load_events_keeps_the_movement_constraint_separate_from_supervision(tmp_path):
    from ase.constraints import FixAtoms

    frames = []
    for shift in (0.0, 0.3, 0.15):
        frame = Atoms("H2", positions=[[0, 0, 0], [1.0 + shift, 0, 0]], cell=np.eye(3) * 10, pbc=False)
        # Fixed atoms live in the ASE constraint, which .traj carries and extxyz
        # does not; the data layer reads it and nothing else.
        frame.set_constraint(FixAtoms(indices=[1]))
        frames.append(frame)
    ase.io.write(tmp_path / "event_0.traj", frames)

    catalog = load_events(tmp_path)
    target = event_target(catalog.events["0"])

    # movable_mask is the given constraint; it is never derived from the event.
    assert target.reactant.movable_mask.tolist() == [True, False]
    assert target.product.movable_mask.tolist() == [True, False]
    assert np.allclose(target.displacement[1], [0.3, 0.0, 0.0])


def test_load_events_ignores_files_that_are_not_event_files(tmp_path):
    frames = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[0.4, 0.0, 0.0]]),
        Atoms("H", positions=[[0.2, 0.0, 0.0]]),
    ]
    ase.io.write(tmp_path / "reactant_0.traj", frames)
    ase.io.write(tmp_path / "event_0.traj", frames)

    catalog = load_events(tmp_path)

    assert catalog.event_ids == ["0"]


def test_load_events_rejects_event_files_with_other_than_three_frames(tmp_path):
    frames = [Atoms("H", positions=[[value, 0.0, 0.0]]) for value in range(4)]
    ase.io.write(tmp_path / "event_0.traj", frames)

    with pytest.raises(ValueError, match="exactly three frames"):
        load_events(tmp_path)
