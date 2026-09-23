"""Score a saved checkpoint without retraining.

A long training run only samples at the very end. This tool points the same
evaluation protocol at any intermediate epoch checkpoint, so a fixed review
deadline does not force a choice between training longer and having a number.

It never writes into the run's own ``sampling/`` directory: pass a separate
``--output`` so an in-flight run cannot be clobbered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.data import event_target, select_events
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD
from basinflow.workflows.dataset import load_configured_dataset
from basinflow.evaluation.single_event_molecule import run_single_event_molecule_trials
from basinflow.models.factory import (
    build_model,
    load_model_checkpoint,
    model_cutoff,
    model_spec,
)
from basinflow.workflows.config import read_config
from basinflow.workflows.transition1x import REQUIRED


def single_event_catalog(catalog: EventCatalog) -> EventCatalog:
    """Keep only events whose product keeps the reactant's fragment structure.

    This is a scope decision about which benchmark is being reported, not a
    per-sample oracle: the filter reads each reference event once and the same ids
    are removed for every arm being compared.  See ``docs/22`` section 八 for why
    the dissociation records are excluded.
    """
    from basinflow.geometry.bonds import bond_adjacency, preserves_fragments

    keep = []
    for event_id, event in catalog.events.items():
        reactant = event.reactant
        target = event_target(event)
        if preserves_fragments(
            bond_adjacency(reactant.positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
            bond_adjacency(target.target_positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
        ):
            keep.append(event_id)
    return select_events(catalog, keep)


def latest_epoch_checkpoint(run_dir: Path) -> Path:
    directory = run_dir / "epoch_checkpoints"
    candidates = sorted(directory.glob("epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no epoch checkpoints in {directory}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="defaults to the highest-numbered epoch checkpoint")
    parser.add_argument("--output", required=True, type=Path,
                        help="separate directory for the sampling artifacts")
    parser.add_argument("--split", default=None, help="overrides [evaluation] split")
    parser.add_argument("--steps", type=int, default=None,
                        help="overrides [evaluation] eval_num_steps")
    parser.add_argument("--max-events", type=int, default=None,
                        help="score only the first N events (smoke checks)")
    parser.add_argument("--trials-per-event", type=int, default=None,
                        help="overrides [evaluation] trials_per_event")
    parser.add_argument("--bond-change-source", default=None, choices=("seed", "oracle"),
                        help="condition arm; defaults to [evaluation] bond_change_source")
    parser.add_argument("--single-event-only", action="store_true",
                        help="drop events whose product is a separated fragment pair")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    config = read_config(args.run_dir / "config.resolved.ini", REQUIRED)
    checkpoint_path = args.checkpoint or latest_epoch_checkpoint(args.run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Epoch checkpoints carry weights only, so the backbone is rebuilt from the
    # run's resolved config; the final checkpoint also records it.
    if "model_config" in checkpoint:
        model = load_model_checkpoint(checkpoint, device="cpu")
    else:
        backend, model_config = model_spec(config["model"])
        model = build_model(backend, model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    split_config = config["split"] if config.has_section("split") else {}
    catalog, split, _ = load_configured_dataset(config, seed=int(config['training'].get('seed', 42)))
    evaluation_split = (args.split or config["evaluation"].get("split", "test")).strip().lower()
    evaluation_catalog = split.select(catalog, evaluation_split)
    if not evaluation_catalog.event_ids:
        raise ValueError(f"evaluation partition {evaluation_split} is empty")
    if args.single_event_only:
        before = len(evaluation_catalog)
        evaluation_catalog = single_event_catalog(evaluation_catalog)
        if not evaluation_catalog.event_ids:
            raise ValueError("single-event scope removed every event")
        print(f"single-event scope: {before} -> {len(evaluation_catalog.event_ids)} events")
    if args.max_events is not None:
        evaluation_catalog = select_events(
            evaluation_catalog, evaluation_catalog.event_ids[: args.max_events]
        )

    bond_change_source = (
        args.bond_change_source
        or config["evaluation"].get("bond_change_source", "seed").strip().lower()
    )
    trials_per_event = args.trials_per_event or config["evaluation"].getint("trials_per_event")
    num_steps = args.steps or config["evaluation"].getint("eval_num_steps")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = run_single_event_molecule_trials(
        model=model,
        test_catalog=evaluation_catalog,
        output_dir=args.output,
        cutoff=model_cutoff(model),
        num_steps=num_steps,
        integrator=config["evaluation"].get("integrator", DEFAULT_METHOD).strip().lower(),
        solver_atol=config["evaluation"].getfloat("solver_atol", fallback=DEFAULT_ATOL),
        solver_rtol=config["evaluation"].getfloat("solver_rtol", fallback=DEFAULT_RTOL),
        gaussian_scale=config["init"].getfloat("gaussian_scale"),
        num_test_events=config["evaluation"].getint("num_test_events", fallback=None),
        sampling_seed=config["evaluation"].getint("sampling_seed"),
        selection_seed=config["evaluation"].getint("selection_seed"),
        device=torch.device("cpu"),
        bond_change_source=bond_change_source,
        trials_per_event=trials_per_event,
    )
    record = {
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_steps": checkpoint.get("completed_steps"),
        "evaluation_split": evaluation_split,
        "raw_rmsd_angstrom": summary["raw_rmsd_angstrom"],
        "aligned_rmsd_angstrom": summary["aligned_rmsd_angstrom"],
        "clipped_raw_rmsd_angstrom": summary["clipped_raw_rmsd_angstrom"],
        "best_of_trials": summary["best_of_trials"],
        "num_trials": summary["num_trials"],
        "num_test_events": summary["num_test_events"],
        "bond_change_source": bond_change_source,
        "trials_per_event": summary["trials_per_test_event"],
        "steps": num_steps,
        "single_event_only": bool(args.single_event_only),
    }
    (args.output / "checkpoint_eval.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
