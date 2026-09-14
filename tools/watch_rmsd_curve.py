"""Track the product-RMSD of every epoch checkpoint while a run is in flight.

A run that only samples at the end hides its learning curve behind the loss,
and loss does not decide whether the benchmark target is met.  This watcher
scores each new epoch checkpoint on a fixed basin subset and appends the result
to ``rmsd_curve.csv`` next to the run, so the trajectory is visible without
waiting for the run to finish.

It never writes into the run's own artifacts: checkpoints are read-only and the
sampling output goes under its own directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIELDS = (
    "epoch", "steps", "checkpoint",
    "raw_mean", "raw_median",
    "clipped_mean", "clipped_median",
    "aligned_mean", "aligned_median",
    "best_raw_median", "best_aligned_median",
)


def _scored(output_root: Path) -> set[int]:
    scored = set()
    if not output_root.exists():
        return scored
    for record in output_root.glob("epoch_*/checkpoint_eval.json"):
        scored.add(int(json.loads(record.read_text())["checkpoint_epoch"]))
    return scored


def _score(script: Path, run_dir: Path, output_root: Path, checkpoint: Path,
           basins: int, trials: int, split: str, steps: int | None) -> dict:
    epoch = int(checkpoint.stem.split("_")[1])
    output = output_root / f"epoch_{epoch:04d}"
    command = [
        sys.executable, str(script),
        "--run-dir", str(run_dir),
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--split", split,
        "--max-basins", str(basins),
        "--trials-per-basin", str(trials),
    ]
    if steps is not None:
        command += ["--steps", str(steps)]
    subprocess.run(command, check=True, capture_output=True)
    record = json.loads((output / "checkpoint_eval.json").read_text())
    return {
        "epoch": record["checkpoint_epoch"],
        "steps": record["checkpoint_steps"],
        "checkpoint": record["checkpoint"],
        "raw_mean": record["raw_rmsd_angstrom"]["mean"],
        "raw_median": record["raw_rmsd_angstrom"]["median"],
        "clipped_mean": record["clipped_raw_rmsd_angstrom"]["mean"],
        "clipped_median": record["clipped_raw_rmsd_angstrom"]["median"],
        "aligned_mean": record["aligned_rmsd_angstrom"]["mean"],
        "aligned_median": record["aligned_rmsd_angstrom"]["median"],
        "best_raw_median": record["best_of_trials"]["raw_rmsd_angstrom"]["median"],
        "best_aligned_median": record["best_of_trials"]["aligned_rmsd_angstrom"]["median"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--basins", type=int, default=128, help="basins scored per checkpoint")
    parser.add_argument("--trials-per-basin", type=int, default=1)
    parser.add_argument("--split", default="val")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--interval", type=float, default=60.0, help="seconds between polls")
    parser.add_argument("--once", action="store_true", help="score what exists now and exit")
    args = parser.parse_args()

    script = Path(__file__).resolve().parent / "evaluate_checkpoint.py"
    output_root = args.run_dir / "rmsd_curve"
    output_root.mkdir(parents=True, exist_ok=True)
    curve_path = output_root / "rmsd_curve.csv"

    while True:
        checkpoints = sorted((args.run_dir / "epoch_checkpoints").glob("epoch_*.pt"))
        pending = [path for path in checkpoints if int(path.stem.split("_")[1]) not in _scored(output_root)]
        for checkpoint in pending:
            row = _score(script, args.run_dir, output_root, checkpoint, args.basins,
                         args.trials_per_basin, args.split, args.steps)
            new_file = not curve_path.exists()
            with curve_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            print(
                f"epoch {row['epoch']:>4} raw {row['raw_mean']:.4f}/{row['raw_median']:.4f} "
                f"clipped {row['clipped_mean']:.4f} "
                f"aligned {row['aligned_mean']:.4f}/{row['aligned_median']:.4f}",
                flush=True,
            )
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
