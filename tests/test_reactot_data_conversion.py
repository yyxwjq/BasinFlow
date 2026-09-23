"""The ReactOT pickle converter must produce a layout the data layer accepts.

These tests use synthetic pickles with the official schema (nested
``reactant``/``product``/``transition_state`` frame dictionaries, ``num_atoms``,
``charges``, ``positions``, top-level ``single_fragment`` and ``use_ind``) and
then read the result back through ``basinflow.data``, which is the contract that
matters: a converted dataset must load, split and centrifuge exactly like any
other events directory.
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

from basinflow.data import load_events, load_split, resolve_split

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "transition1x_reactot_to_events", REPO_ROOT / "tools" / "transition1x_reactot_to_events.py"
)
assert _SPEC is not None and _SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CONVERTER)

ATOMS = 4
SYMBOLS = [6, 1, 8, 7]  # C, H, O, N


def _frame(rows: int, atoms: int, generator: np.random.Generator, *, pad: int = 2) -> dict:
    return {
        "num_atoms": [atoms] * rows,
        # Rows are padded in the real pickles; the converter must truncate.
        "charges": [[*SYMBOLS[:atoms], 99, 99] for _ in range(rows)],
        "positions": [generator.normal(size=(atoms + pad, 3)) for _ in range(rows)],
        "rxn": [f"rxn{index}" for index in range(rows)],
    }


def _raw(rows: int = 4, atoms: int = ATOMS, seed: int = 0) -> dict:
    generator = np.random.default_rng(seed)
    raw = {
        role: _frame(rows, atoms, generator)
        for role in ("reactant", "product", "transition_state")
    }
    raw["single_fragment"] = [1, 0, 1, 1][:rows]
    raw["use_ind"] = []
    return raw


def _write(path: Path, raw: dict, use_ind: list[int]) -> Path:
    raw = {**raw, "use_ind": list(use_ind)}
    with path.open("wb") as handle:
        pickle.dump(raw, handle)
    return path


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = _raw()
    train = _write(tmp_path / "train_rpsb_all.pkl", raw, [0, 1])
    valid = _write(tmp_path / "valid_rpsb_all.pkl", raw, [2, 3])
    events = tmp_path / "dataset" / "events"
    return train, valid, events


def test_conversion_writes_the_canonical_layout(tmp_path: Path) -> None:
    train, valid, events = _dataset(tmp_path)
    report = CONVERTER.convert(train, valid, events)

    assert report["num_events"] == 4
    assert report["split"] == {"train": 2, "val": 0, "test": 2}
    assert sorted(path.name for path in events.glob("event_*")) == [
        f"event_{index}.traj" for index in range(4)
    ]
    catalog = load_events(events)
    split = load_split(events / "split.csv", catalog)
    assert split.train_ids == ("0", "1")
    assert split.val_ids == ()
    assert split.test_ids == ("2", "3")


def test_conversion_satisfies_the_event_invariants(tmp_path: Path) -> None:
    train, valid, events = _dataset(tmp_path)
    CONVERTER.convert(train, valid, events)
    catalog = load_events(events)
    for event in catalog.events.values():
        # Event.__post_init__ already enforces equal atom counts, species order,
        # cell and pbc; asserting the counts makes the failure mode explicit.
        assert event.reactant.n_atoms == ATOMS  # padded rows were truncated
        assert event.product.species == event.reactant.species
        assert event.transition_state is not None
        assert not event.reactant.pbc.any()
        assert bool(np.all(event.reactant.movable_mask))  # never fixed_idx
        assert event.metadata["dataset_kind"] == "reactot_transition1x_bridge"


def test_centered_frames_have_zero_centroid(tmp_path: Path) -> None:
    train, valid, events = _dataset(tmp_path)
    CONVERTER.convert(train, valid, events)
    catalog = load_events(events)
    for event in catalog.events.values():
        for structure in (event.reactant, event.product, event.transition_state):
            assert np.allclose(structure.positions.mean(axis=0), np.zeros(3), atol=1e-9)


def test_no_center_keeps_the_source_coordinates(tmp_path: Path) -> None:
    train, valid, events = _dataset(tmp_path)
    report = CONVERTER.convert(train, valid, events, center=False)
    assert report["coordinate_convention"] == "source_coordinates"
    catalog = load_events(events)
    first = catalog.events["0"].reactant.positions
    assert not np.allclose(first.mean(axis=0), np.zeros(3), atol=1e-9)


def test_overlapping_use_ind_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    train = _write(tmp_path / "train_rpsb_all.pkl", raw, [0, 1])
    valid = _write(tmp_path / "valid_rpsb_all.pkl", raw, [1, 2, 3])
    with pytest.raises(ValueError, match="overlap"):
        CONVERTER.convert(train, valid, tmp_path / "events")


def test_incomplete_use_ind_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    train = _write(tmp_path / "train_rpsb_all.pkl", raw, [0, 1])
    valid = _write(tmp_path / "valid_rpsb_all.pkl", raw, [2])
    with pytest.raises(ValueError, match="cover every row"):
        CONVERTER.convert(train, valid, tmp_path / "events")


def test_species_mismatch_between_frames_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["product"]["charges"][0] = [1, 1, 1, 1, 99, 99]
    train = _write(tmp_path / "train_rpsb_all.pkl", raw, [0, 1])
    valid = _write(tmp_path / "valid_rpsb_all.pkl", raw, [2, 3])
    with pytest.raises(ValueError, match="species order"):
        CONVERTER.convert(train, valid, tmp_path / "events")


def test_existing_event_files_are_not_overwritten(tmp_path: Path) -> None:
    train, valid, events = _dataset(tmp_path)
    CONVERTER.convert(train, valid, events)
    with pytest.raises(FileExistsError):
        CONVERTER.convert(train, valid, events)


def test_resolve_split_reads_the_written_file(tmp_path: Path) -> None:
    """The workflow path must accept the dataset without regenerating the split."""
    train, valid, events = _dataset(tmp_path)
    CONVERTER.convert(train, valid, events)
    catalog = load_events(events)
    split, provenance = resolve_split(events, catalog, auto=False)
    assert provenance["split_mode"] == "loaded"
    assert (len(split.train_ids), len(split.val_ids), len(split.test_ids)) == (2, 0, 2)


def test_single_fragment_filter_shrinks_both_partitions(tmp_path: Path) -> None:
    train, valid, events = _dataset(tmp_path)
    report = CONVERTER.convert(
        train, valid, events, single_fragment_only=True
    )
    # single_fragment is [1, 0, 1, 1]: only row 1 (a train row) is multi-fragment,
    # so train keeps row 0 while the test partition keeps both of its rows.
    assert report["split"]["train"] == 1
    assert report["split"]["test"] == 2
