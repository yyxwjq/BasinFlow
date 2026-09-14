import os
import subprocess
import sys

from ase import Atoms
from ase.io import write
import numpy as np
import pytest


def test_demo_pipeline_runs_on_real_events_when_env_is_set():
    events_dir = os.environ.get("BASINFLOW_EVENTS_DIR")
    if not events_dir:
        pytest.skip("BASINFLOW_EVENTS_DIR is not set")

    result = subprocess.run(
        [sys.executable, "examples/demo_pipeline.py", events_dir],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Stage" in result.stdout
    assert "pipeline complete" in result.stdout


def test_demo_pipeline_runs_without_event_zero_or_numeric_basin(tmp_path):
    event_path = tmp_path / "event_5.extxyz"
    write(
        event_path,
        [
            Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=False),
            Atoms("He", positions=[[0.2, 0, 0]], cell=np.eye(3) * 10, pbc=False),
        ],
    )
    (tmp_path / "basin_table.csv").write_text(
        "global_event,local_event,basin,file\n"
        "5,0,basin_A,event_5.extxyz\n"
    )

    result = subprocess.run(
        [sys.executable, "examples/demo_pipeline.py", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "basin_A" in result.stdout
    assert "event_5" in result.stdout
