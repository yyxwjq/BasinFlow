import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord, StructureRecord


def _diagnostic_module():
    path = Path("tools/diagnose_flow_sampling.py")
    spec = importlib.util.spec_from_file_location("diagnose_flow_sampling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _catalog():
    reactant = StructureRecord("reactant", ["H", "C", "O"], [[-1, 0, 0], [1, 0, 0], [0, 2, 0]])
    product = StructureRecord("product", reactant.species, reactant.positions * np.exp(-1))
    event = EventRecord("event", "reactant", "product", "transition1x:0", metadata={"dataset_kind": "transition1x_single_event_molecule"})
    return EventCatalog({"reactant": reactant, "product": product}, {"event": event}, {"transition1x:0": BasinRecord("transition1x:0", "reactant", ["event"])})


def test_diagnostic_pairs_exact_initial_states_and_measures_euler_discretization():
    module = _diagnostic_module()
    catalog = _catalog()
    plan = module.make_sampling_plan(catalog.basin_ids, basin_count=8, trials_per_basin=2, seed=20260910)

    class LinearField:
        builds_own_graph = True

        def eval(self):
            return self

        def __call__(self, batch, t=0):
            assert not any(key.startswith("target_") for key in batch.keys())
            return {"velocity": -batch.pos}

    report = module.diagnose_sampling(
        LinearField(), catalog, plan, steps=(2, 4, 8), gaussian_scale=0.05,
        cutoff=5.0, checkpoint_id="toy", device="cpu",
    )

    assert len(report["rows"]) == 6
    for trial_index in range(2):
        rows = [row for row in report["rows"] if row["trial_index"] == trial_index]
        assert len({row["initial_positions_sha256"] for row in rows}) == 1
        assert len({row["init_random_seed"] for row in rows}) == 1
        assert all(row["finite"] for row in rows)
        assert rows[-1]["raw_rmsd_angstrom"] < rows[0]["raw_rmsd_angstrom"]
        assert all(row["pair_distance_mae_angstrom"] >= 0 for row in rows)
        assert all(row["kabsch_rmsd_angstrom"] <= row["raw_rmsd_angstrom"] + 1e-6 for row in rows)
    assert report["graph_update_interval"] == 1
    assert report["integrator"] == "euler_left_endpoint"
    assert module.make_sampling_plan(list(reversed(catalog.basin_ids)), basin_count=8, trials_per_basin=2, seed=20260910) == plan


def test_diagnostic_records_nonfinite_failures_without_dropping_pairs():
    module = _diagnostic_module()
    catalog = _catalog()

    class InvalidField:
        builds_own_graph = True

        def eval(self):
            return self

        def __call__(self, batch, t=0):
            return {"velocity": torch.full_like(batch.pos, float("nan"))}

    plan = module.make_sampling_plan(catalog.basin_ids, basin_count=1, trials_per_basin=1, seed=20260910)
    report = module.diagnose_sampling(InvalidField(), catalog, plan, steps=(2, 4), gaussian_scale=0.05, cutoff=5.0, checkpoint_id="toy", device="cpu")
    assert len(report["rows"]) == 2
    assert all(not row["finite"] and row["raw_rmsd_angstrom"] is None for row in report["rows"])
    assert all("non-finite velocity" in row["error"] for row in report["rows"])


def test_diagnostic_cli_restores_toy_checkpoint_and_persists_preselected_plan(tmp_path):
    from ase.io import write
    from basinflow.models.factory import build_model

    run_dir = tmp_path / "run"
    events_dir = tmp_path / "events"
    run_dir.mkdir()
    events_dir.mkdir()
    catalog = _catalog()
    frames = [catalog.structures[name].to_ase() for name in ("reactant", "product")]
    for frame in frames:
        frame.info["dataset_kind"] = "transition1x_single_event_molecule"
    write(events_dir / "event_0.traj", frames)
    (events_dir / "basin_table.csv").write_text("basin,file,dataset_kind\ntransition1x:0,event_0.traj,transition1x_single_event_molecule\n")
    split = {"train": [], "val": ["transition1x:0"], "test": [], "seed": 42}
    (run_dir / "split_manifest.json").write_text(json.dumps(split))
    (run_dir / "config.resolved.ini").write_text(f"[data]\nevents_dir = {events_dir}\n[init]\ngaussian_scale = 0.05\n[evaluation]\nsplit = val\n")
    model_config = {"hidden_dim": 4, "num_layers": 1, "cutoff": 5.0}
    model = build_model("egnn", model_config)
    torch.save({"model_backend": "egnn", "model_config": model_config,
                "model_state_dict": model.state_dict(), "split_manifest": split}, run_dir / "checkpoint.pt")
    output = tmp_path / "diagnostic"

    result = subprocess.run([sys.executable, "tools/diagnose_flow_sampling.py", "--run-dir", str(run_dir),
                             "--output", str(output), "--steps", "1", "2", "--num-basins", "1",
                             "--trials-per-basin", "1", "--cpu-threads", "1"], capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads((output / "sampling_plan.json").read_text())
    report = json.loads((output / "diagnostic.json").read_text())
    assert report["plan"] == plan["plan"]
    assert report["evaluation_split"] == "val"
    assert len(report["rows"]) == 2
    assert len(report["source_sha256"]["checkpoint.pt"]) == 64
