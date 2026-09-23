"""End-to-end contract for the ``leftnet_sb`` (ReactOT bridge) workflow.

This walks the real pipeline -- converter -> workflow -> checkpoint -> bridge
scorer -- and asserts that a bridge run switches its data view, its objective,
its sampler and its reported weights together. A run that trained on the bridge
pairs but scored a velocity field, or reported raw weights when EMA was
requested, would silently measure a different model.
"""
from __future__ import annotations

import configparser
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from test_transition1x import _write_transition1x_fixture  # noqa: E402


@pytest.fixture()
def events_dir(tmp_path: Path) -> Path:
    source = tmp_path / "train.pkl"
    output_dir = tmp_path / "events"
    _write_transition1x_fixture(source)
    result = subprocess.run(
        [sys.executable, "tools/transition1x_to_events.py", str(source), str(output_dir),
         "--selection", "use_ind", "--center"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return output_dir


def _config(tmp_path: Path, events_dir: Path, *, output_name: str = "bridge_run",
            validate_each_epoch: bool = False, ema_decay: str | None = "0.9") -> Path:
    parser = configparser.ConfigParser()
    parser["data"] = {"events_dir": str(events_dir), "auto_split": "true"}
    parser["output"] = {"output_dir": str(tmp_path / output_name), "log_to_stdout": "false"}
    parser["split"] = {"train": "0.5", "val": "0.5", "test": "0.0", "seed": "7"}
    parser["model"] = {
        "backend": "leftnet_sb", "num_features": "16", "num_layers": "1", "num_radial": "6",
        "cutoff": "5.0", "num_elements": "5", "legacy": "true", "update": "true",
        "single_layer_output": "true", "object_aware": "true", "reflect_equiv": "true",
    }
    parser["bridge"] = {"timesteps": "16", "beta_max": "0.3", "power": "0.5", "inv_power": "1"}
    training = {"epochs": "1", "max_steps": "2", "lr": "0.001", "batch_size": "2",
                "optimizer": "adamw", "amsgrad": "true", "save_each_epoch": "false", "seed": "7",
                "validate_each_epoch": "true" if validate_each_epoch else "false"}
    if ema_decay is not None:
        training["ema_decay"] = ema_decay
    parser["training"] = training
    parser["loss"] = {"velocity_weight": "1.0", "norm": "mse"}
    parser["runtime"] = {"device": "cpu", "cpu_threads": "1"}
    parser["evaluation"] = {"eval_num_steps": "4", "trials_per_event": "1", "sampling_seed": "5",
                            "split": "val", "integrator": "euler", "solver_atol": "0.01",
                            "solver_rtol": "0.01"}
    path = tmp_path / f"{output_name}.ini"
    with path.open("w", encoding="utf-8") as file:
        parser.write(file)
    return path


def test_bridge_workflow_trains_and_scores_with_the_bridge_sampler(tmp_path, events_dir):
    from basinflow.workflows.transition_state import run_transition1x_ts

    report = run_transition1x_ts(_config(tmp_path, events_dir))
    assert report["backend"] == "leftnet_sb"
    assert report["training_contract"] == "schrodinger_bridge_rp_to_transition_state"
    assert report["evaluated_weights"] == "ema"
    assert report["ema_decay"] == pytest.approx(0.9)
    assert report["bridge"]["timesteps"] == 16
    assert report["bridge"]["beta"] > 0
    assert report["evaluation"]["evaluation_protocol"] == "reactot_bridge_ode"
    assert report["evaluation"]["integrator"] == "euler"
    assert report["evaluation"]["reactot_reference_protocol"]["rmsd_angstrom"]["count"] > 0


def test_bridge_workflow_writes_both_checkpoints_and_the_dual_protocol(tmp_path, events_dir):
    from basinflow.workflows.transition_state import run_transition1x_ts

    run_dir = tmp_path / "bridge_run"
    run_transition1x_ts(_config(tmp_path, events_dir))
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "training_state.pt").is_file()
    assert (run_dir / "config.resolved.ini").is_file()
    assert (run_dir / "training.log").is_file()
    summary = json.loads((run_dir / "sampling" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["reactot_reference_protocol"]) >= {
        "label", "rmsd_angstrom", "uncapped_rmsd_angstrom", "chirality_gain_angstrom",
    }
    # the strict column is still reported and unchanged in meaning
    assert summary["aligned_rmsd_angstrom"]["count"] == summary["reactot_reference_protocol"]["rmsd_angstrom"]["count"]

    saved = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["model_backend"] == "leftnet_sb"
    assert saved["evaluated_weights"] == "ema"


def test_bridge_workflow_rejects_velocity_style_epoch_validation(tmp_path, events_dir):
    from basinflow.workflows.transition_state import run_transition1x_ts

    with pytest.raises(ValueError, match="validate_each_epoch"):
        run_transition1x_ts(_config(tmp_path, events_dir, validate_each_epoch=True))


def test_bridge_workflow_can_run_without_ema(tmp_path, events_dir):
    from basinflow.workflows.transition_state import run_transition1x_ts

    report = run_transition1x_ts(_config(tmp_path, events_dir, output_name="no_ema", ema_decay=None))
    assert report["evaluated_weights"] == "raw"
    assert report["ema_decay"] is None
