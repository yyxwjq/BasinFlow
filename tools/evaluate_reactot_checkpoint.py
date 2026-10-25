"""Score the released ReactOT checkpoint with BasinFlow's bridge sampler.

This is the parity check for the port: the reference reports mean RMSD 0.103 A and
median 0.053 A on the 1073 set-aside reactions at ``nfe = 10``. Run it after
converting the official pickles (``tools/transition1x_reactot_to_events.py``).

    REACTOT_SRC=/path/to/react-ot python tools/evaluate_reactot_checkpoint.py \
        --events /path/to/reactot_events \
        --checkpoint /path/to/reactot-pretrained.ckpt \
        --output /tmp/reactot_eval --num-events 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.data import load_events, load_split, partition
from basinflow.evaluation.bridge import run_bridge_trials
from basinflow.models.reference import load_reactot_checkpoint
from basinflow.models.sb import DEFAULT_T_EPS, SBSchedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, required=True, help="converted events directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="reactot-pretrained.ckpt")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num-events", type=int, default=None)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--steps", type=int, default=11, help="saved ODE points (reference nfe + 1)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-xyz", action="store_true", help="write per-event pred/target/reactant/product/midpoint XYZ")
    parser.add_argument("--reference-src", type=Path, default=None, help="react-ot checkout (to unpickle the class)")
    args = parser.parse_args()

    catalog = load_events(args.events)
    split = load_split(args.events / "split.csv", catalog)
    identifiers = partition(split, args.split)
    if not identifiers:
        raise SystemExit(f"split {args.split!r} is empty")
    test_catalog = catalog.select(identifiers, name=args.split)

    model, hyper = load_reactot_checkpoint(
        args.checkpoint, reference_src=args.reference_src, device=args.device
    )
    schedule = SBSchedule(
        timesteps=int(hyper.get("timesteps", 3000)),
        beta_max=float(hyper.get("beta_max", 0.3)),
        power=float(hyper.get("power", 0.5)),
        inv_power=float(hyper.get("inv_power", 1.0)),
    )
    summary = run_bridge_trials(
        model=model,
        catalog=test_catalog,
        output_dir=args.output,
        num_steps=int(args.steps),
        schedule=schedule,
        trials_per_event=int(args.trials),
        num_events=args.num_events,
        integrator="midpoint",
        solver_atol=1e-2,
        solver_rtol=1e-2,
        t_eps=DEFAULT_T_EPS,
        device=args.device,
        write_structures=bool(args.write_xyz),
    )
    print(json.dumps(summary["reactot_reference_protocol"]["rmsd_angstrom"], indent=2))
    print(json.dumps(summary["aligned_rmsd_angstrom"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
