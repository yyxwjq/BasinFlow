from __future__ import annotations

import csv

import numpy as np
import pytest

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord


def _dataset() -> EventCatalog:
    structures = {}
    events = {}
    basins = {}
    for index in range(3):
        event_id = f"transition1x:{index}"
        reactant = StructureRecord(
            structure_id=f"{event_id}:reactant",
            species=["H", "O"],
            positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            pbc=False,
            metadata={"source_index": index},
        )
        product = StructureRecord(
            structure_id=f"{event_id}:product",
            species=["H", "O"],
            positions=[[0.0, 0.0, 0.0], [0.8 + 0.1 * (index + 1), 0.0, 0.0]],
            pbc=False,
        )
        structures[reactant.structure_id] = reactant
        structures[product.structure_id] = product
        events[event_id] = EventRecord(
            event_id,
            reactant.structure_id,
            product.structure_id,
            event_id,
            metadata={
                "dataset_kind": "transition1x_single_event_molecule",
                "source_index": str(index),
            },
        )
        basins[event_id] = BasinRecord(event_id, reactant.structure_id, [event_id])
    return EventCatalog(structures=structures, events=events, basins=basins)


class _ZeroFlow:
    def eval(self):
        return self

    def __call__(self, batch, t):
        torch = pytest.importorskip("torch")
        n_atoms = batch["pos"].shape[0]
        return {
            "velocity": torch.zeros_like(batch["pos"]),
            "active_logits": torch.zeros(n_atoms, device=batch["pos"].device),
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
        graph_update_interval=1,
        gaussian_scale=0.05,
        num_trials=2,
        sampling_seed=17,
        selection_seed=23,
    )

    metrics_path = tmp_path / "sampling" / "trial_metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert summary["num_trials"] == 2
    assert len(rows) == 2
    assert len({row["pseudo_basin_id"] for row in rows}) == 2
    assert all(row["evaluation_protocol"] == "single_event_pairwise_no_relaxation" for row in rows)
    assert (tmp_path / "sampling" / "generated_candidates.extxyz").is_file()
    assert (tmp_path / "sampling" / "reference_products.extxyz").is_file()
    assert (tmp_path / "sampling" / "summary.json").is_file()


def test_run_single_event_trials_samples_each_selected_basin_repeatedly_and_plots(tmp_path):
    pytest.importorskip("torch")
    from basinflow.evaluation.single_event_molecule import run_single_event_molecule_trials

    summary = run_single_event_molecule_trials(
        model=_ZeroFlow(),
        test_catalog=_dataset(),
        output_dir=tmp_path,
        cutoff=3.0,
        num_steps=2,
        graph_update_interval=1,
        gaussian_scale=0.05,
        num_trials=None,
        trials_per_basin=3,
        num_test_basins=2,
        sampling_seed=17,
        selection_seed=23,
    )

    metrics_path = tmp_path / "sampling" / "trial_metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert summary["num_test_basins"] == 2
    assert summary["trials_per_test_basin"] == 3
    assert summary["num_trials"] == 6
    assert len(rows) == 6
    counts = {}
    for row in rows:
        counts[row["pseudo_basin_id"]] = counts.get(row["pseudo_basin_id"], 0) + 1
    assert sorted(counts.values()) == [3, 3]
    assert summary["raw_rmsd_angstrom"]["count"] == 6
    assert "below_mean_fraction" in summary["raw_rmsd_mean_split"]
    assert "above_mean_fraction" in summary["raw_rmsd_mean_split"]
    plots = tmp_path / "sampling" / "plots"
    assert len(list(plots.glob("*_rmsd_100_trials.png"))) == 2
    assert (plots / "rmsd_by_basin.png").is_file()
    assert (plots / "rmsd_distribution.png").is_file()


@pytest.mark.parametrize("runner_kind", ["single_event", "semantic"])
def test_trial_runner_propagates_activity_threshold_without_changing_products(tmp_path, monkeypatch, runner_kind):
    torch = pytest.importorskip("torch")
    from basinflow.evaluation import semantic_flow, single_event_molecule
    from basinflow.sampling import CandidateSampler

    class ConstantFlow:
        builds_own_graph = True

        def eval(self):
            return self

        def __call__(self, batch, t):
            return {"velocity": torch.full_like(batch.pos, 0.15 / np.sqrt(3))}

    recorded = []

    class RecordingSampler(CandidateSampler):
        def sample(self, catalog, basin_id):
            result = super().sample(catalog, basin_id)
            recorded.append(result)
            return result

    module = single_event_molecule if runner_kind == "single_event" else semantic_flow
    monkeypatch.setattr(module, "CandidateSampler", RecordingSampler)
    for active_threshold in [0.1, 0.2]:
        kwargs = dict(
            model=ConstantFlow(), test_catalog=_dataset().subset(["transition1x:0"]),
            output_dir=tmp_path / str(active_threshold), cutoff=3.0, num_steps=1,
            graph_update_interval=1, gaussian_scale=0.0, sampling_seed=17,
            active_threshold=active_threshold,
        )
        if runner_kind == "single_event":
            module.run_single_event_molecule_trials(**kwargs, selection_seed=23, num_trials=1)
        else:
            module.run_gaussian_trials(**kwargs, system="fixture", trials_per_basin=1)

    low, high = recorded
    assert low.candidates[0].predicted_active_atoms == [0, 1]
    assert high.candidates[0].predicted_active_atoms == []
    assert high.candidates[0].sampling_config["active_threshold"] == 0.2
    np.testing.assert_array_equal(
        next(iter(low.generated_structures.values())).positions,
        next(iter(high.generated_structures.values())).positions,
    )
