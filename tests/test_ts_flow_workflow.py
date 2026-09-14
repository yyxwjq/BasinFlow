"""End-to-end contract for the R+P -> TS *flow* workflow.

The TS task used to be trained with ``flow_time = 0.0`` and ``source_scale =
0.0``, which pins the supervision to a single point: the network only ever fits
``TS - (R+P)/2`` and one sampler step is the whole model. These tests walk the
real pipeline (converter -> workflow -> checkpoint -> scorer) and assert the
properties that make it a flow instead (see ``docs/25``).
"""
from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from test_transition1x import _write_transition1x_fixture  # noqa: E402

EVENTS_DIR_NAME = "events"


@pytest.fixture()
def events_dir(tmp_path: Path) -> Path:
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / EVENTS_DIR_NAME
    _write_transition1x_fixture(source)
    result = subprocess.run(
        [sys.executable, "tools/transition1x_to_events.py", str(source), str(output_dir),
         "--selection", "use_ind", "--center"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return output_dir


def _config(tmp_path: Path, events_dir: Path, *, source_scale: float, time_distribution: str,
            flow_time: float | None, output_name: str) -> Path:
    parser = configparser.ConfigParser()
    parser["data"] = {"events_dir": str(events_dir), "active_threshold": "0.1"}
    parser["output"] = {"output_dir": str(tmp_path / output_name), "log_to_stdout": "false"}
    parser["split"] = {"train": "0.5", "val": "0.5", "test": "0.0", "seed": "7"}
    parser["model"] = {
        "backend": "painn", "num_features": "8", "num_layers": "1", "num_radial_basis": "6",
        "r_max": "5", "stability_mode": "none", "endpoint_condition": "true",
        "endpoint_equivariant": "true",
    }
    init = {"source_scale": str(source_scale), "time_distribution": time_distribution}
    if flow_time is not None:
        init["flow_time"] = str(flow_time)
    parser["init"] = init
    parser["training"] = {"epochs": "2", "lr": "0.001", "batch_size": "2",
                          "save_each_epoch": "true", "seed": "7"}
    parser["evaluation"] = {"eval_num_steps": "3", "trials_per_basin": "2", "sampling_seed": "5",
                            "split": "val"}
    parser["loss"] = {"velocity_weight": "1.0", "active_weight": "0.0", "direction_weight": "0.0",
                      "active_pos_weight": "1.0", "norm": "l1"}
    parser["runtime"] = {"device": "cpu", "cpu_threads": "1"}
    path = tmp_path / f"{output_name}.ini"
    with path.open("w", encoding="utf-8") as file:
        parser.write(file)
    return path


def test_ts_workflow_trains_a_field_and_scores_it_with_an_ode(tmp_path, events_dir):
    from basinflow.workflows.transition_state import run_transition1x_ts

    config = _config(tmp_path, events_dir, source_scale=1.0, time_distribution="beta",
                     flow_time=None, output_name="flow_run")
    report = run_transition1x_ts(config)

    evaluation = report["evaluation"]
    assert evaluation["evaluation_protocol"] == "transition_state_flow_ode"
    assert evaluation["num_steps"] == 3
    assert evaluation["time_distribution"] == "beta"
    # Two trials means two *different* source draws, which is what makes
    # best-of-N meaningful rather than a repeat of one deterministic output.
    assert evaluation["num_trials"] == 2 * evaluation["num_events"]
    assert evaluation["best_of_trials"]["selector"] == "oracle_min_rmsd"
    # The pinned-time parameterisation is gone from the resolved config.
    resolved = (tmp_path / "flow_run" / "config.resolved.ini").read_text(encoding="utf-8")
    assert "flow_time" not in resolved
    assert (tmp_path / "flow_run" / "checkpoint.pt").is_file()


def test_ts_workflow_still_accepts_the_deterministic_bridge(tmp_path, events_dir):
    """The old parameterisation stays legal, it just is not called a flow."""
    from basinflow.workflows.transition_state import run_transition1x_ts

    config = _config(tmp_path, events_dir, source_scale=0.0, time_distribution="fixed",
                     flow_time=0.0, output_name="bridge_run")
    report = run_transition1x_ts(config)

    assert report["evaluation"]["evaluation_protocol"] == "transition_state_flow_ode"
    assert report["flow_time"] == 0.0
    assert report["source_scale"] == 0.0


def test_scorer_records_the_step_and_trial_regime(tmp_path, events_dir):
    from tools.evaluate_transition_state import _score
    from basinflow.workflows.transition_state import run_transition1x_ts

    config = _config(tmp_path, events_dir, source_scale=1.0, time_distribution="beta",
                     flow_time=None, output_name="scored_run")
    run_transition1x_ts(config)
    checkpoint = sorted((tmp_path / "scored_run" / "epoch_checkpoints").glob("epoch_*.pt"))[-1]

    record = _score(tmp_path / "scored_run", checkpoint, tmp_path / "score_1step", "val", None,
                    steps=1, trials=3)

    assert record["ode_steps"] == 1
    assert record["trials"] == 3
    # Averaging over several source draws cannot be worse than the best single
    # draw, so the best-of-N median is bounded by the per-trial median.
    assert record["best_of_trials_median"] <= record["aligned_median"] + 1e-9
    assert (tmp_path / "score_1step" / "sampling" / "summary.json").is_file()
