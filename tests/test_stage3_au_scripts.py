import csv
import configparser
import json
import subprocess
import sys

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write
import numpy as np
import pytest


def _write_tiny_events(path):
    for index, shift in enumerate([0.25, -0.25]):
        frames = [
            Atoms(
                "HOH",
                positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
                cell=np.eye(3) * 10.0,
                pbc=False,
            ),
            Atoms(
                "HOH",
                positions=[[0.0, 0.0, 0.0], [0.8 + shift, 0.0, 0.0], [1.6, 0.0, 0.0]],
                cell=np.eye(3) * 10.0,
                pbc=False,
            ),
        ]
        frames[0].set_constraint(FixAtoms(mask=[True, False, True]))
        frames[1].set_constraint(FixAtoms(mask=[True, False, True]))
        write(path / f"event_{index}.extxyz", frames, columns=["symbols", "positions", "move_mask"])

    with (path / "basin_table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["global_event", "local_event", "basin", "file"])
        writer.writeheader()
        writer.writerow({"global_event": 0, "local_event": 0, "basin": "b0", "file": "event_0.extxyz"})
        writer.writerow({"global_event": 1, "local_event": 0, "basin": "b1", "file": "event_1.extxyz"})


def test_stage3_train_and_sample_scripts_produce_artifacts(tmp_path):
    torch = pytest.importorskip("torch")
    events_dir = tmp_path / "events"
    run_dir = tmp_path / "run"
    sample_dir = tmp_path / "sample"
    events_dir.mkdir()
    _write_tiny_events(events_dir)

    train = subprocess.run(
        [
            sys.executable,
            "examples/train_product_flow.py",
            "--events-dir",
            str(events_dir),
            "--output-dir",
            str(run_dir),
            "--epochs",
            "1",
            "--active-threshold",
            "0.2",
            "--hidden-dim",
            "8",
            "--num-layers",
            "1",
            "--batch-size",
            "full",
            "--train",
            "0.5",
            "--val",
            "0.0",
            "--test",
            "0.5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert train.returncode == 0, train.stderr + train.stdout
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "split_manifest.json").is_file()
    assert (run_dir / "training.log").is_file()
    assert not (run_dir / "train_history.json").exists()
    assert (run_dir / "eval_metrics.json").is_file()
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["training_config"]["batch_size"] == "full"
    train_metrics = json.loads((run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert "model_rollout" in train_metrics
    assert "init_baseline" in train_metrics
    assert "train" in train_metrics["model_rollout"]["splits"]
    assert "test" in train_metrics["model_rollout"]["splits"]
    assert train_metrics["init_baseline"]["oracle_product_displacement"]["overall"]["mean_rmsd"] == 0.0
    assert "Test rollout RMSD" in train.stdout
    assert "Oracle product init RMSD" in train.stdout
    assert "epoch step total_loss velocity_loss active_loss direction_loss" in train.stdout
    log_lines = (run_dir / "training.log").read_text(encoding="utf-8").strip().splitlines()
    assert log_lines[0] == "epoch step total_loss velocity_loss active_loss direction_loss"
    assert len(log_lines) > 1
    assert len(log_lines[1].split()) == 6

    sample = subprocess.run(
        [
            sys.executable,
            "examples/sample_product_flow.py",
            "--events-dir",
            str(events_dir),
            "--checkpoint",
            str(run_dir / "checkpoint.pt"),
            "--split-manifest",
            str(run_dir / "split_manifest.json"),
            "--output-dir",
            str(sample_dir),
            "--num-steps",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert sample.returncode == 0, sample.stderr + sample.stdout
    assert (sample_dir / "metrics.json").is_file()
    assert (sample_dir / "candidates.json").is_file()
    assert (sample_dir / "generated_candidates.extxyz").is_file()
    candidates = json.loads((sample_dir / "candidates.json").read_text(encoding="utf-8"))
    assert all(candidate["sampling_config"]["active_threshold"] == 0.2 for candidate in candidates)
    metrics = json.loads((sample_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "average_test_best_rmsd" in metrics
    assert metrics["average_test_best_rmsd"] is not None
    assert metrics["test_events"]
    assert metrics["test_events"][0]["file"].startswith("event_")
    assert "Average test best RMSD" in sample.stdout
    assert "Test events:" in sample.stdout
    assert ".extxyz" in sample.stdout


def test_train_script_accepts_config_ini_and_writes_resolved_config(tmp_path):
    pytest.importorskip("torch")
    events_dir = tmp_path / "events"
    run_dir = tmp_path / "run_from_config"
    config_path = tmp_path / "config.ini"
    events_dir.mkdir()
    _write_tiny_events(events_dir)

    config_path.write_text(
        "\n".join(
            [
                "[data]",
                f"events_dir = {events_dir}",
                "pbc_override = auto",
                "active_source = auto",
                "active_threshold = 0.1",
                "",
                "[output]",
                f"output_dir = {run_dir}",
                "log_to_stdout = false",
                "",
                "[split]",
                "train = 0.5",
                "val = 0.0",
                "test = 0.5",
                "seed = 7",
                "",
                "[model]",
                "hidden_dim = 8",
                "num_layers = 1",
                "cutoff = 3.0",
                "",
                "[init]",
                "types = zero, gaussian",
                "gaussian_scale = 0.05",
                "",
                "[training]",
                "epochs = 1",
                "lr = 0.01",
                "batch_size = full",
                "",
                "[loss]",
                "velocity_weight = 1.0",
                "active_weight = 0.2",
                "direction_weight = 0.3",
                "active_pos_weight = auto",
                "",
                "[runtime]",
                "device = cpu",
                "cpu_threads = 1",
                "gpu_ids = ",
                "gpu_count = 0",
                "",
                "[evaluation]",
                "eval_num_steps = 2",
                "eval_graph_update_interval = 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    train = subprocess.run(
        [
            sys.executable,
            "examples/train_product_flow.py",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert train.returncode == 0, train.stderr + train.stdout
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "split_manifest.json").is_file()
    assert (run_dir / "training.log").is_file()
    assert (run_dir / "config.resolved.ini").is_file()
    assert "epoch step total_loss velocity_loss active_loss direction_loss" not in train.stdout

    log_lines = (run_dir / "training.log").read_text(encoding="utf-8").strip().splitlines()
    assert log_lines[0] == "epoch step total_loss velocity_loss active_loss direction_loss"
    resolved = configparser.ConfigParser()
    resolved.read(run_dir / "config.resolved.ini")
    assert resolved["data"]["events_dir"] == str(events_dir)
    assert resolved["output"]["output_dir"] == str(run_dir)
    assert resolved["training"]["batch_size"] == "full"
    assert resolved["runtime"]["device"] == "cpu"
    assert resolved["runtime"]["cpu_threads"] == "1"
    assert resolved["loss"]["active_pos_weight"] == "auto"

    checkpoint = __import__("torch").load(
        run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["training_config"]["config"] == str(config_path)
    assert checkpoint["training_config"]["batch_size"] == "full"
    assert checkpoint["runtime_config"]["requested_device"] == "cpu"
    assert checkpoint["runtime_config"]["device"] == "cpu"
    assert checkpoint["runtime_config"]["cpu_threads"] == 1
    assert checkpoint["training_config"]["loss_weights"]["active"] == 0.2
    assert checkpoint["training_config"]["loss_weights"]["direction"] == 0.3
    assert checkpoint["training_config"]["loss_weights"]["active_pos_weight"] == 0.0
