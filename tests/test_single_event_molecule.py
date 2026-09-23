from __future__ import annotations

import csv

import numpy as np
import pytest

from basinflow.data import Event, EventCatalog, Structure


def _dataset() -> EventCatalog:
    events = {}
    for index in range(3):
        event_id = str(index)
        reactant = Structure(
            species=["H", "O"],
            positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            pbc=False,
        )
        product = Structure(
            species=["H", "O"],
            positions=[[0.0, 0.0, 0.0], [0.8 + 0.1 * (index + 1), 0.0, 0.0]],
            pbc=False,
        )
        events[event_id] = Event(
            event_id,
            reactant,
            product,
            metadata={
                "dataset_kind": "transition1x_single_event_molecule",
                "source_index": str(index),
            },
        )
    return EventCatalog(events=events)


class _ZeroFlow:
    def eval(self):
        return self

    def __call__(self, batch, t):
        torch = pytest.importorskip("torch")
        n_atoms = batch["pos"].shape[0]
        return {
            "velocity": torch.zeros_like(batch["pos"]),
            "direction": torch.zeros_like(batch["pos"]),
        }


def test_run_single_event_trials_writes_exact_selected_trial_count(tmp_path):
    pytest.importorskip("torch")
    from basinflow.evaluation.single_event_molecule import run_single_event_molecule_trials

    summary = run_single_event_molecule_trials(
        model=_ZeroFlow(),
        test_catalog=_dataset(),
        output_dir=tmp_path,
        cutoff=3.0,
        num_steps=2,
        gaussian_scale=0.05,
        num_trials=2,
        sampling_seed=17,
        selection_seed=23,
    )

    metrics_path = tmp_path / "sampling" / "trial_metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert summary["num_trials"] == 2
    assert len(rows) == 2
    assert len({row["event_id"] for row in rows}) == 2
    assert all(row["evaluation_protocol"] == "single_event_pairwise_no_relaxation" for row in rows)
    assert (tmp_path / "sampling" / "generated_candidates.extxyz").is_file()
    assert (tmp_path / "sampling" / "reference_products.extxyz").is_file()
    assert (tmp_path / "sampling" / "summary.json").is_file()


def test_run_single_event_trials_samples_each_selected_event_repeatedly_and_plots(tmp_path):
    pytest.importorskip("torch")
    from basinflow.evaluation.single_event_molecule import run_single_event_molecule_trials

    summary = run_single_event_molecule_trials(
        model=_ZeroFlow(),
        test_catalog=_dataset(),
        output_dir=tmp_path,
        cutoff=3.0,
        num_steps=2,
        gaussian_scale=0.05,
        num_trials=None,
        trials_per_event=3,
        num_test_events=2,
        sampling_seed=17,
        selection_seed=23,
    )

    metrics_path = tmp_path / "sampling" / "trial_metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert summary["num_test_events"] == 2
    assert summary["trials_per_test_event"] == 3
    assert summary["num_trials"] == 6
    assert len(rows) == 6
    counts = {}
    for row in rows:
        counts[row["event_id"]] = counts.get(row["event_id"], 0) + 1
    assert sorted(counts.values()) == [3, 3]
    assert summary["raw_rmsd_angstrom"]["count"] == 6
    assert "below_mean_fraction" in summary["raw_rmsd_mean_split"]
    assert "above_mean_fraction" in summary["raw_rmsd_mean_split"]
    plots = tmp_path / "sampling" / "plots"
    assert len(list(plots.glob("*_rmsd_100_trials.png"))) == 2
    assert (plots / "rmsd_by_event.png").is_file()
    assert (plots / "rmsd_distribution.png").is_file()


