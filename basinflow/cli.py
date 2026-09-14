"""Command line entry point, installed as ``basinflow``.

The ``examples/*.py`` scripts are thin wrappers over the same two functions, so
the CLI and the scripts cannot drift apart.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="basinflow", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train-transition1x",
        help="train and sample a molecular (Transition1x) R-to-P product flow",
    )
    train.add_argument("--config", required=True, help="path to a converted Transition1x INI config")

    train_ts = commands.add_parser(
        "train-transition1x-ts",
        help="train and sample a molecular (Transition1x) R+P-to-TS flow",
    )
    train_ts.add_argument("--config", required=True, help="path to a Transition1x INI config")

    benchmark = commands.add_parser(
        "benchmark",
        help="train and sample a periodic-system (Au/Pt) basin benchmark",
    )
    benchmark.add_argument("--config", required=True, help="path to a semantic-flow INI config")
    benchmark.add_argument("--system", default=None, help="system name recorded in trial_metrics.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train-transition1x":
        from basinflow.workflows import run_transition1x

        run_transition1x(args.config)
    elif args.command == "train-transition1x-ts":
        from basinflow.workflows import run_transition1x_ts

        run_transition1x_ts(args.config)
    elif args.command == "benchmark":
        from basinflow.workflows import run_semantic_benchmark

        run_semantic_benchmark(args.config, system=args.system)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
