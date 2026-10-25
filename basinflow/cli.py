"""Command line entry point, installed as ``basinflow``.

The workflow behind ``train-transition1x-ts`` is the single entry point; see
README.md for the data conversion step that has to run first.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="basinflow", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_ts = commands.add_parser(
        "train-transition1x-ts",
        help="train and sample an R+P-to-TS bridge model (leftnet_sb, painn or egnn)",
    )
    train_ts.add_argument("--config", required=True, help="path to an INI config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train-transition1x-ts":
        from basinflow.workflows import run_transition1x_ts

        run_transition1x_ts(args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
