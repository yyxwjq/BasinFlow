import hashlib
import json
import subprocess
import sys


def test_provenance_captures_current_source_and_actual_logged_steps(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training.log").write_text(
        "epoch step total_loss velocity_loss active_loss direction_loss\n"
        "1 1 0.4 0.4 0.0 0.0\n1 2 0.3 0.3 0.0 0.0\n"
    )
    (run_dir / "config.resolved.ini").write_text("[training]\nmax_steps = 1000\n")
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint fixture, never deserialize")
    (run_dir / "split_manifest.json").write_text('{"train": ["basin"], "val": [], "test": []}')
    (run_dir / "generated.traj").write_bytes(b"not archived")
    output = tmp_path / "provenance"
    command = [sys.executable, "tools/capture_provenance.py", "--run-dir", str(run_dir), "--output", str(output)]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["capture_timing"] == "post_run_current_workspace_not_launch_snapshot"
    assert manifest["runs"][0]["training_progress"]["completed_steps_from_log"] == 2
    assert manifest["runs"][0]["training_progress"]["logged_optimizer_updates"] == 2
    assert manifest["runs"][0]["training_progress"]["strictly_increasing_steps"] is True
    artifacts = manifest["runs"][0]["artifacts"]
    assert artifacts["checkpoint.pt"]["sha256"] == hashlib.sha256((run_dir / "checkpoint.pt").read_bytes()).hexdigest()
    assert "generated.traj" not in artifacts
    source = output / "source/basinflow/data/catalog.py"
    assert source.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_files"]["basinflow/data/catalog.py"]["sha256"]
    assert manifest["environment"]["python_executable"] == sys.executable
    assert "torch" in manifest["environment"]["packages"]
    assert len(manifest["git"]["diff_head_sha256"]) == 64
    assert (output / "working_tree.patch").is_file()
    repeat = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeat.returncode != 0
    assert "refusing to overwrite" in repeat.stderr


def test_provenance_log_records_reset_and_missing_progress(tmp_path):
    import runpy

    progress = runpy.run_path("tools/capture_provenance.py")["_training_progress"]
    log = tmp_path / "training.log"
    assert progress(log)["completed_steps_from_log"] is None
    log.write_text("epoch step total_loss velocity_loss active_loss direction_loss\n1 1 1 1 0 0\n1 2 1 1 0 0\n1 1 1 1 0 0\n")
    report = progress(log)
    assert report["completed_steps_from_log"] == 1
    assert report["max_logged_step"] == 2
    assert report["logged_optimizer_updates"] == 3
    assert report["strictly_increasing_steps"] is False
