"""End-to-end contract for the ``leftnet_sb`` (ReactOT bridge) workflow.

This walks the real pipeline -- converter -> workflow -> checkpoint -> bridge
scorer -- and asserts that every backend uses the same data view, objective and
sampler, and reports the weights it actually scored. A run that trained one
objective but scored another, or reported raw weights when EMA was requested,
would silently measure a different model.
"""
from __future__ import annotations

import configparser
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from _fixtures import write_reactot_events  # noqa: E402


@pytest.fixture()
def events_dir(tmp_path: Path) -> Path:
    return write_reactot_events(tmp_path)


def _config(tmp_path: Path, events_dir: Path, *, output_name: str = "bridge_run",
            backend: str = "leftnet_sb", validate_each_epoch: bool = False,
            ema_decay: str | None = "0.9") -> Path:
    parser = configparser.ConfigParser()
    parser["data"] = {"events_dir": str(events_dir), "auto_split": "true"}
    parser["output"] = {"output_dir": str(tmp_path / output_name), "log_to_stdout": "false"}
    parser["split"] = {"train": "0.5", "val": "0.5", "test": "0.0", "seed": "7"}
    model = {"backend": backend, "num_layers": "1", "cutoff": "5.0"}
    if backend == "leftnet_sb":
        model.update({
            "num_features": "16", "num_radial": "6", "num_elements": "5",
            "legacy": "true", "update": "true", "single_layer_output": "true",
            "object_aware": "true", "reflect_equiv": "true",
        })
    elif backend == "painn":
        model.update({
            "num_features": "16", "num_radial_basis": "6", "num_elements": "119",
            "r_max": "5.0", "time_dim": "8", "stability_mode": "scaled",
        })
    else:
        model.update({"hidden_dim": "16", "max_atomic_number": "10"})
    parser["model"] = model
    parser["bridge"] = {"timesteps": "16", "beta_max": "0.3", "power": "0.5", "inv_power": "1"}
    training = {"epochs": "1", "max_steps": "2", "lr": "0.001", "batch_size": "2",
                "optimizer": "adamw", "amsgrad": "true", "save_each_epoch": "false", "seed": "7",
                "validate_each_epoch": "true" if validate_each_epoch else "false"}
    if ema_decay is not None:
        training["ema_decay"] = ema_decay
    parser["training"] = training
    parser["loss"] = {"norm": "mse"}
    parser["runtime"] = {"device": "cpu", "cpu_threads": "1"}
    parser["evaluation"] = {"eval_num_steps": "4", "trials_per_event": "1", "sampling_seed": "5",
                            "split": "test", "integrator": "euler", "solver_atol": "0.01",
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

    # The bridge arm defaults to the reference training contract, without needing
    # either key in the config.
    state = torch.load(run_dir / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["config"]["batch_sampler"] == "node2"
    assert state["config"]["max_num"] == 2800
    assert state["config"]["adaptive_clip"] is True
    # The sampler must actually produce a batch for this two-event fixture; a
    # drop-everything epoch would still write checkpoints and pass weaker checks.
    assert state["config"]["completed_steps"] >= 1


def test_bridge_workflow_validates_each_epoch_and_keeps_the_best(tmp_path, events_dir):
    """The reference selects on val_ep_scaled_err; the best weights are scored."""
    from basinflow.workflows.transition_state import run_transition1x_ts

    report = run_transition1x_ts(
        _config(tmp_path, events_dir, output_name="validated", validate_each_epoch=True)
    )
    run_dir = tmp_path / "validated"
    assert (run_dir / "best.pt").is_file()
    assert report["validation"]["best_epoch"] == 1
    assert math.isfinite(report["validation"]["val_scaled_err"])
    state = torch.load(run_dir / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["config"]["bridge_validation"] is True
    assert state["config"]["best_epoch"] == 1


def test_bridge_workflow_can_run_without_ema(tmp_path, events_dir):
    from basinflow.workflows.transition_state import run_transition1x_ts

    report = run_transition1x_ts(_config(tmp_path, events_dir, output_name="no_ema", ema_decay=None))
    assert report["evaluated_weights"] == "raw"
    assert report["ema_decay"] is None


def test_bridge_workflow_resumes_from_the_latest_complete_checkpoint(tmp_path, events_dir):
    """An HPC job killed by wall time must continue, not restart."""
    from basinflow.workflows.transition_state import run_transition1x_ts

    config = _config(tmp_path, events_dir, output_name="resume_run")
    text = config.read_text(encoding="utf-8")
    text = text.replace("epochs = 1", "epochs = 2").replace("max_steps = 2", "max_steps = 1")
    text = text.replace("[training]", "[training]\nresume_from = auto")
    config.write_text(text, encoding="utf-8")

    run_transition1x_ts(config)  # first job: one optimizer step

    text = config.read_text(encoding="utf-8").replace("max_steps = 1", "max_steps = 2")
    config.write_text(text, encoding="utf-8")
    run_transition1x_ts(config)  # resubmission: continues from step 1

    state = torch.load(tmp_path / "resume_run" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["config"]["completed_steps"] == 2


@pytest.mark.parametrize("backend", ["painn", "egnn"])
def test_every_backend_uses_the_same_bridge_workflow(tmp_path, events_dir, backend):
    """PaiNN and EGNN are interchangeable with LEFTNet, not separate pipelines."""
    from basinflow.workflows.transition_state import run_transition1x_ts

    config = _config(tmp_path, events_dir, output_name=backend, backend=backend,
                     ema_decay=None)
    text = config.read_text(encoding="utf-8").replace("max_steps = 2", "max_steps = 1")
    config.write_text(text, encoding="utf-8")

    report = run_transition1x_ts(config)

    assert report["backend"] == backend
    assert report["training_contract"] == "schrodinger_bridge_rp_to_transition_state"
    assert report["evaluation"]["evaluation_protocol"] == "reactot_bridge_ode"
    saved = torch.load(tmp_path / backend / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["model_backend"] == backend
    assert saved["evaluated_weights"] == "raw"
