from __future__ import annotations

import configparser
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from basinflow.evaluation.semantic_flow import _write_representatives
from basinflow.data.records import StructureRecord


def _write_events(events_dir):
    for index, shift in enumerate([0.2, -0.2]):
        reactant = Atoms(
            "HOH",
            positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
            cell=np.eye(3) * 10.0,
            pbc=False,
        )
        product = reactant.copy()
        product.positions[1, 0] += shift
        reactant.arrays["move_mask"] = np.array([False, True, False])
        product.arrays["move_mask"] = np.array([False, True, False])
        write(events_dir / f"event_{index}.extxyz", [reactant, product])

    with (events_dir / "basin_table.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["global_event", "local_event", "basin", "file"])
        writer.writeheader()
        for index in range(2):
            writer.writerow({
                "global_event": index,
                "local_event": 0,
                "basin": f"basin-{index}",
                "file": f"event_{index}.extxyz",
            })


def test_semantic_benchmark_continues_after_eam_failure_and_writes_artifacts(tmp_path):
    pytest.importorskip("torch")
    events_dir = tmp_path / "events"
    output_dir = tmp_path / "run"
    events_dir.mkdir()
    _write_events(events_dir)
    config_path = tmp_path / "benchmark.ini"
    config_path.write_text(
        "\n".join(
            [
                "[data]",
                f"events_dir = {events_dir}",
                "pbc_override = auto",
                "active_source = displacement",
                "active_threshold = 0.1",
                "",
                "[output]",
                f"output_dir = {output_dir}",
                "log_to_stdout = false",
                "",
                "[split]",
                "train = 0.5",
                "val = 0.0",
                "test = 0.5",
                "seed = 3",
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
                "active_weight = 0.1",
                "direction_weight = 0.1",
                "active_pos_weight = auto",
                "",
                "[runtime]",
                "device = cpu",
                "cpu_threads = 1",
                "gpu_ids =",
                "gpu_count = 0",
                "",
                "[evaluation]",
                "eval_num_steps = 2",
                "eval_graph_update_interval = 1",
                "trials_per_test_basin = 2",
                "sampling_seed = 11",
                "",
                "[relaxation]",
                f"eam_potential = {tmp_path / 'missing.eam'}",
                "eam_form = eam",
                "relax_fmax = 0.05",
                "relax_steps = 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "examples/benchmark_semantic_flow.py", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    metrics_path = output_dir / "sampling" / "trial_metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert {row["trial_index"] for row in rows} == {"0", "1"}
    assert {row["relax_status"] for row in rows} == {"failed"}
    assert (output_dir / "sampling" / "generated_before_eam.traj").is_file()
    assert (output_dir / "sampling" / "summary.json").is_file()
    assert (output_dir / "loss_by_epoch.csv").is_file()
    assert (output_dir / "loss_curves.png").is_file()
    assert (output_dir / "sampling" / "plots" / "basin-0_rmsd_100_trials.png").is_file()
    assert (output_dir / "sampling" / "plots" / "rmsd_by_basin.png").is_file()
    assert (output_dir / "sampling" / "plots" / "eam_rmsd_before_after.png").is_file()
    representative_frames = read(
        output_dir / "sampling" / "representatives" / "basin-0.traj",
        ":",
    )
    assert representative_frames
    assert all("mic_error" in frame.info for frame in representative_frames)

    resolved = configparser.ConfigParser()
    resolved.read(output_dir / "config.resolved.ini")
    assert resolved["evaluation"]["trials_per_test_basin"] == "2"


def test_representatives_write_event_mapping_manifest(tmp_path):
    def structure(structure_id, x):
        return StructureRecord(
            structure_id=structure_id,
            species=["Au"],
            positions=[[x, 0.0, 0.0]],
            cell=np.eye(3) * 10.0,
            pbc=False,
        )

    products = {
        "event_a": structure("product_a", 0.0),
        "event_b": structure("product_b", 0.2),
    }
    records = defaultdict(list)
    for trial_index, raw_x, raw_event, relaxed_x, relaxed_event in [
        (0, 0.03, "event_a", 0.04, "event_a"),
        (1, 0.10, "event_a", 0.12, "event_b"),
        (2, 0.30, "event_b", 0.25, "event_b"),
    ]:
        raw_match = {
            "event_id": raw_event,
            "source_file": f"/data/{raw_event}.traj",
            "product": products[raw_event],
            "rmsd_angstrom": raw_x,
        }
        relaxed_match = {
            "event_id": relaxed_event,
            "source_file": f"/data/{relaxed_event}.traj",
            "product": products[relaxed_event],
            "rmsd_angstrom": relaxed_x,
        }
        raw = structure(f"raw_{trial_index}", raw_x)
        relaxed = structure(f"relaxed_{trial_index}", relaxed_x)
        records["basin-x"].append(
            {
                "row": {
                    "system": "au",
                    "basin_id": "basin-x",
                    "trial_index": trial_index,
                    "candidate_id": f"basin-x:trial:{trial_index}",
                    "init_random_seed": 100 + trial_index,
                    "nearest_event_before": raw_event,
                    "source_file_before": f"{raw_event}.traj",
                    "raw_rmsd_angstrom": raw_x,
                    "relax_status": "converged",
                    "relax_converged": True,
                    "relax_steps": 1,
                    "relax_error": None,
                    "nearest_event_after": relaxed_event,
                    "source_file_after": f"{relaxed_event}.traj",
                    "relaxed_rmsd_angstrom": relaxed_x,
                    "relaxed_rmsd_to_before_event": relaxed_x,
                    "event_changed_after_relax": raw_event != relaxed_event,
                },
                "raw": raw,
                "raw_match": raw_match,
                "relaxed": relaxed,
                "relaxed_match": relaxed_match,
            }
        )

    output_dir = tmp_path / "representatives"
    _write_representatives(records, output_dir)

    manifest = json.loads((output_dir / "basin-x.json").read_text(encoding="utf-8"))
    assert manifest["basin_id"] == "basin-x"
    assert [item["rank"] for item in manifest["representatives"]] == [
        "best",
        "median",
        "worst",
    ]
    median = manifest["representatives"][1]
    assert median["candidate_id"] == "basin-x:trial:1"
    assert median["nearest_event_before"] == "event_a"
    assert median["source_file_before"] == "event_a.traj"
    assert median["nearest_event_after"] == "event_b"
    assert median["source_file_after"] == "event_b.traj"
    assert median["event_changed_after_relax"] is True
    assert manifest["event_transition_counts"]["event_a -> event_b"] == 1

    frames = read(output_dir / "basin-x.traj", ":")
    assert [frame.info["role"] for frame in frames[:4]] == [
        "raw_candidate",
        "raw_reference_product",
        "eam_relaxed_candidate",
        "eam_reference_product",
    ]
    assert frames[0].info["nearest_event_id"] == "event_a"
    assert frames[0].info["source_file"] == "event_a.traj"
    assert frames[2].info["nearest_event_id"] == "event_a"
    assert frames[2].info["source_file"] == "event_a.traj"
    assert frames[2].info["event_changed_after_relax"] is False
    assert all("candidate_id" in frame.info for frame in frames)
    assert all("mic_error" in frame.info for frame in frames)
