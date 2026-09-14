"""Capture current workspace source and benchmark hashes after runs finish."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ("basinflow", "examples", "tools", "configs", "tests", "docs")
SOURCE_SUFFIXES = {".py", ".ini", ".toml", ".yaml", ".yml", ".md"}
ARTIFACT_SUFFIXES = {".json", ".csv", ".log", ".ini", ".pt"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _training_progress(path: Path) -> dict:
    steps = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 6 and fields[0].isdigit() and fields[1].isdigit():
                steps.append(int(fields[1]))
    return {
        "source": str(path),
        "completed_steps_from_log": steps[-1] if steps else None,
        "max_logged_step": max(steps) if steps else None,
        "logged_optimizer_updates": len(steps),
        "strictly_increasing_steps": all(after > before for before, after in zip(steps, steps[1:])),
        "interpretation": "last_logged_step_not_configured_budget; resets_require_manual_review",
    }


def _environment() -> dict:
    packages = {}
    for name in ("torch", "torch-geometric", "ase", "numpy", "scipy", "basinflow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "scope": "capture_process_environment_not_proof_of_training_environment",
    }


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True).stdout


def capture_provenance(run_dirs: list[str | Path], output_dir: str | Path) -> dict:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    runs = [Path(directory).resolve() for directory in run_dirs]
    for directory in runs:
        if not directory.is_dir():
            raise FileNotFoundError(f"run directory not found: {directory}")
        if output == directory or output in directory.parents or directory in output.parents:
            raise ValueError("provenance output must be outside the captured run directories")
    paths = set()
    for directory in SOURCE_DIRECTORIES:
        paths.update(path for path in (REPO_ROOT / directory).rglob("*")
                     if path.is_file() and not path.is_symlink() and path.suffix in SOURCE_SUFFIXES
                     and "__pycache__" not in path.parts)
    paths.update(path for name in ("pyproject.toml", "README.md", "AGENTS.md")
                 if (path := REPO_ROOT / name).is_file())
    if any(output == REPO_ROOT / directory or REPO_ROOT / directory in output.parents
           for directory in SOURCE_DIRECTORIES):
        raise ValueError("provenance output must be outside source snapshot directories")
    started = _now()
    diff = _git("diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv")
    status = _git("status", "--porcelain=v1", "--untracked-files=all").decode()
    manifest = {
        "schema_version": 1,
        "capture_timing": "post_run_current_workspace_not_launch_snapshot",
        "capture_started_utc": started,
        "repository": str(REPO_ROOT),
        "environment": _environment(),
        "git": {
            "head": _git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(status.strip()),
            "status_porcelain": status,
            "diff_head_sha256": hashlib.sha256(diff).hexdigest(),
            "diff_scope": "tracked_worktree_and_index_against_HEAD; untracked_source_in_snapshot",
        },
        "source_files": {},
        "runs": [],
        "limitations": [
            "Snapshot records capture-time files; it does not establish which revisions ran earlier.",
            "Run artifacts are hashed in place; datasets, trajectories, plots, and checkpoint bytes are not copied.",
            "Missing artifacts and log resets are reported, not interpreted as successful completion.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".benchmark-provenance-", dir=output.parent) as temporary:
        prepared = Path(temporary) / "prepared"
        prepared.mkdir()
        for path in sorted(paths):
            relative = path.relative_to(REPO_ROOT)
            destination = prepared / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
            manifest["source_files"][relative.as_posix()] = _hash_file(destination)
        (prepared / "working_tree.patch").write_bytes(diff)
        for directory in runs:
            artifacts = {
                path.relative_to(directory).as_posix(): _hash_file(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file() and not path.is_symlink() and path.suffix in ARTIFACT_SUFFIXES
            }
            required = ("config.resolved.ini", "split_manifest.json", "checkpoint.pt", "training.log")
            manifest["runs"].append({
                "path": str(directory),
                "artifacts": artifacts,
                "missing_required_artifacts": [name for name in required if name not in artifacts],
                "training_progress": _training_progress(directory / "training.log"),
            })
        manifest["capture_finished_utc"] = _now()
        (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prepared.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="completed run directory; repeat to capture multiple runs")
    parser.add_argument("--output", required=True, help="new provenance directory outside source and run directories")
    args = parser.parse_args()
    manifest = capture_provenance(args.run_dir, args.output)
    print(f"Captured post-run provenance: {len(manifest['source_files'])} source files, {len(manifest['runs'])} runs at {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
