"""Score a BasinFlow ReactOT-bridge checkpoint without retraining.

The workflow scores once at the end of training. On a cluster a 600k-step run
spans several jobs, so this tool scores the newest checkpoint in a run directory
(``checkpoint.pt`` or the latest complete ``epoch_checkpoints/epoch_*.pt``).

    python tools/evaluate_bridge.py --run-dir runs/leftnet_sb --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from basinflow.data import load_events, load_split, partition
from basinflow.evaluation.bridge import run_bridge_trials
from basinflow.models.factory import build_model, model_spec
from basinflow.models.sb import SBSchedule
from basinflow.workflows.config import read_config


def _latest_checkpoint(run_dir: Path) -> Path:
    final = run_dir / "checkpoint.pt"
    if final.is_file():
        return final
    epoch_dir = run_dir / "epoch_checkpoints"
    candidates = sorted(epoch_dir.glob("epoch_*.pt"), reverse=True) if epoch_dir.is_dir() else []
    for candidate in candidates:
        state = torch.load(candidate, map_location="cpu", weights_only=False)
        if state.get("epoch_complete"):
            return candidate
    raise FileNotFoundError(f"no complete checkpoint under {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="defaults to <run>/config.resolved.ini")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-events", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="evaluate every Nth event (for a representative subset)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--integrator", default=None)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-xyz", action="store_true", help="write per-event pred/target/reactant/product/midpoint XYZ")
    args = parser.parse_args()

    from basinflow.workflows.transition_state import REQUIRED

    config = read_config(args.config or (args.run_dir / "config.resolved.ini"), REQUIRED)
    checkpoint_path = args.checkpoint or _latest_checkpoint(args.run_dir)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_config" in state:
        backend, model_config = state.get("model_backend", "leftnet_sb"), state["model_config"]
    else:
        backend, model_config = model_spec(config["model"], default_backend="leftnet_sb")
    model = build_model(backend, model_config)
    weights = "raw"
    if args.weights == "ema" and state.get("ema_model_state_dict") is not None:
        model.load_state_dict(state["ema_model_state_dict"], strict=True)
        weights = "ema"
    else:
        model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(args.device).eval()

    events = Path(config["data"]["events_dir"])
    catalog = load_events(events)
    split = load_split(events / "split.csv", catalog)
    split_name = args.split or config["evaluation"].get("split", "test").strip().lower()
    identifiers = partition(split, split_name)
    if not identifiers:
        raise SystemExit(f"split {split_name!r} is empty")
    evaluation_catalog = catalog.select(identifiers, name=split_name)

    bridge = config["bridge"] if config.has_section("bridge") else {}
    schedule = SBSchedule(
        timesteps=int(bridge.get("timesteps", 3000)),
        beta_max=float(bridge.get("beta_max", 0.3)),
        power=float(bridge.get("power", 0.5)),
        inv_power=float(bridge.get("inv_power", 1.0)),
    )
    output = args.output or (args.run_dir / f"score_{checkpoint_path.stem}")
    summary = run_bridge_trials(
        model=model,
        catalog=evaluation_catalog,
        output_dir=output,
        num_steps=int(args.steps or config["evaluation"].getint("eval_num_steps")),
        schedule=schedule,
        trials_per_event=int(args.trials or config["evaluation"].getint("trials_per_event")),
        num_events=args.num_events if args.num_events is not None else config["evaluation"].getint("num_events", fallback=None),
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        integrator=args.integrator or config["evaluation"].get("integrator", "midpoint").strip().lower(),
        solver_atol=config["evaluation"].getfloat("solver_atol", fallback=1e-2),
        solver_rtol=config["evaluation"].getfloat("solver_rtol", fallback=1e-2),
        t_eps=float(bridge.get("t_eps", 1e-5)),
        device=args.device,
        write_structures=bool(args.write_xyz),
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "weights": weights}, indent=2))
    print(json.dumps(summary["reactot_reference_protocol"]["rmsd_angstrom"], indent=2))
    print(json.dumps(summary["aligned_rmsd_angstrom"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
