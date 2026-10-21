"""Shared synthetic ReactOT dataset for the end-to-end workflow tests.

Builds official-schema ``*_rpsb_all.pkl`` pickles (nested frames, ``single_fragment``,
``use_ind``) and runs the real converter, so the workflow tests exercise the same
layout the reference data has: 4 rows, 2 train + 2 test, no validation split.
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "transition1x_reactot_to_events",
    REPO_ROOT / "tools" / "transition1x_reactot_to_events.py",
)
assert _SPEC is not None and _SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CONVERTER)

ATOMS = 4
SYMBOLS = [6, 1, 8, 7]  # C, H, O, N


def _frame(rows: int, generator: np.random.Generator, *, pad: int = 2) -> dict:
    return {
        "num_atoms": [ATOMS] * rows,
        "charges": [[*SYMBOLS, 99, 99] for _ in range(rows)],
        "positions": [generator.normal(size=(ATOMS + pad, 3)) for _ in range(rows)],
        "rxn": [f"rxn{index}" for index in range(rows)],
    }


def _raw(rows: int = 4, seed: int = 0) -> dict:
    generator = np.random.default_rng(seed)
    raw = {
        role: _frame(rows, generator)
        for role in ("reactant", "product", "transition_state")
    }
    raw["single_fragment"] = [1] * rows
    raw["use_ind"] = []
    return raw


def _write(path: Path, raw: dict, use_ind: list[int]) -> Path:
    with path.open("wb") as handle:
        pickle.dump({**raw, "use_ind": list(use_ind)}, handle)
    return path


def write_reactot_events(tmp_path: Path) -> Path:
    """Write the two pickles and convert them into ``tmp_path/events``."""
    raw = _raw()
    train = _write(tmp_path / "train_rpsb_all.pkl", raw, [0, 1])
    valid = _write(tmp_path / "valid_rpsb_all.pkl", raw, [2, 3])
    events = tmp_path / "events"
    CONVERTER.convert(train, valid, events)
    return events
