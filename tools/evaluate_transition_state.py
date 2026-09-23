"""Score R+P -> TS checkpoints, and optionally track the curve while training.

The R -> P scorer cannot be reused here: this task has no seeds, no basins and
no bond-change condition, so it has its own sampling and metric path in
``basinflow.evaluation.transition_state``. ``--watch`` keeps scoring each new
epoch checkpoint so the trajectory is visible before the run ends.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.sampling.ode import DEFAULT_METHOD, METHODS
from basinflow.workflows.dataset import load_configured_dataset
from basinflow.evaluation.transition_state import run_transition_state_trials
from basinflow.models.factory import build_model, model_cutoff, model_spec
from basinflow.workflows.config import read_config
from basinflow.workflows.transition_state import REQUIRED

FIELDS = ("epoch", "steps", "aligned_mean", "aligned_median", "clipped_mean", "source_median",
          "reactot_mean", "reactot_median", "reactot_uncapped_mean", "chirality_gain_mean",
          "best_of_trials_median", "ode_steps", "trials", "saved_points_mean", "integrator")


def _score(run_dir: Path, checkpoint: Path, output: Path, split: str, events: int | None = None,
           steps: int | None = None, trials: int | None = None, integrator: str | None = None,
           write_trajectories: bool = False, config_path: Path | None = None) -> dict:
    config = read_config(config_path or (run_dir / "config.resolved.ini"), REQUIRED)
    integrator = (integrator or config["evaluation"].get("integrator", DEFAULT_METHOD)).strip().lower()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_config" in saved:
        backend, model_config = saved.get("model_backend", "painn"), saved["model_config"]
    else:
        backend, model_config = model_spec(config["model"])
    model = build_model(backend, model_config)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    catalog, splitter, _ = load_configured_dataset(config, seed=int(config['training'].get('seed', 42)))
    evaluation = splitter.select(catalog, split)
    summary = run_transition_state_trials(
        model=model,
        catalog=evaluation,
        output_dir=output,
        cutoff=model_cutoff(model),
        num_steps=steps if steps is not None else config["evaluation"].getint("eval_num_steps"),
        source_scale=config["init"].getfloat("source_scale", fallback=0.0),
        trials_per_event=trials if trials is not None else 1,
        num_events=events,
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        time_distribution=config["init"].get("time_distribution", "beta").strip().lower(),
        beta_alpha=config["init"].getfloat("beta_alpha", fallback=0.8),
        time_epsilon=config["init"].getfloat("time_epsilon", fallback=0.0),
        solver_atol=config["evaluation"].getfloat("solver_atol", fallback=1e-6),
        solver_rtol=config["evaluation"].getfloat("solver_rtol", fallback=1e-3),
        integrator=integrator,
        write_trajectories=write_trajectories,
        device=torch.device("cpu"),
    )
    return {
        "epoch": saved.get("epoch"),
        "steps": saved.get("completed_steps"),
        "aligned_mean": summary["aligned_rmsd_angstrom"]["mean"],
        "aligned_median": summary["aligned_rmsd_angstrom"]["median"],
        "clipped_mean": summary["clipped_aligned_rmsd_angstrom"]["mean"],
        "reactot_mean": summary["reactot_reference_protocol"]["rmsd_angstrom"]["mean"],
        "reactot_median": summary["reactot_reference_protocol"]["rmsd_angstrom"]["median"],
        "reactot_uncapped_mean": summary["reactot_reference_protocol"]["uncapped_rmsd_angstrom"]["mean"],
        "chirality_gain_mean": summary["reactot_reference_protocol"]["chirality_gain_angstrom"]["mean"],
        "source_median": summary["source_rmsd_angstrom"]["median"],
        "best_of_trials_median": summary["best_of_trials"]["aligned_rmsd_angstrom"]["median"],
        "ode_steps": summary["num_steps"],
        "trials": summary["trials_per_event"],
        "saved_points_mean": summary["saved_points"]["mean"],
        "integrator": summary["integrator"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None,
                        help="config to score with; defaults to <run-dir>/config.resolved.ini. "
                             "Pass a migrated config to rescore an existing checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="defaults to <run-dir>/ts_curve")
    parser.add_argument("--checkpoint", type=Path, default=None, help="defaults to the newest epoch checkpoint")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-events", type=int, default=96,
                        help="cap the number of scored events; full split when omitted")
    parser.add_argument("--steps", type=int, default=None,
                        help="ODE steps for this scoring pass; defaults to eval_num_steps in the config")
    parser.add_argument("--integrator", choices=list(METHODS), default=None,
                        help="solver for this scoring pass; defaults to [evaluation] integrator in the config")
    parser.add_argument("--trials", type=int, default=None,
                        help="independent source draws per event; defaults to 1")
    parser.add_argument("--write-trajectories", action="store_true",
                        help="store every trial's integration path as extxyz")
    parser.add_argument("--watch", action="store_true", help="keep scoring new epoch checkpoints")
    parser.add_argument("--interval", type=float, default=600.0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)

    config_path = args.config or (args.run_dir / "config.resolved.ini")
    config = read_config(config_path, REQUIRED)
    split = args.split or config["evaluation"].get("split", "val")
    output_root = args.output or args.run_dir / ("ts_curve" if args.watch else "ts_eval")
    output_root.mkdir(parents=True, exist_ok=True)
    curve = output_root / "ts_curve.csv"

    while True:
        if args.checkpoint is not None:
            pending = [args.checkpoint]
        else:
            seen = set()
            if curve.exists():
                seen = {int(row["epoch"]) for row in csv.DictReader(curve.open())}
            pending = [path for path in sorted((args.run_dir / "epoch_checkpoints").glob("epoch_*.pt"))
                       if int(path.stem.split("_")[1]) not in seen]
        for checkpoint in pending:
            epoch = int(checkpoint.stem.split("_")[1])
            record = _score(args.run_dir, checkpoint, output_root / f"epoch_{epoch:04d}",
                            split, args.num_events, steps=args.steps, trials=args.trials,
                            integrator=args.integrator, write_trajectories=args.write_trajectories,
                            config_path=config_path)
            new_file = not curve.exists()
            with curve.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(record)
            print(f"epoch {record['epoch']:>4} aligned {record['aligned_mean']:.4f}/{record['aligned_median']:.4f} "
                  f"clipped {record['clipped_mean']:.4f} reactot {record['reactot_mean']:.4f} "
                  f"gain {record['chirality_gain_mean']:.4f} source {record['source_median']:.4f}", flush=True)
            if args.checkpoint is not None:
                return 0
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
