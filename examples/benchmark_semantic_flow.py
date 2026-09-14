"""Train and evaluate direct-basin product flow with optional relaxation diagnostics.

Thin wrapper: the orchestration lives in ``basinflow.workflows`` so that this
script and the ``basinflow`` CLI cannot drift apart.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basinflow.workflows import run_semantic_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a semantic-flow INI config.")
    parser.add_argument("--system", default=None, help="System name recorded in trial_metrics.csv.")
    parsed = parser.parse_args()
    run_semantic_benchmark(parsed.config, system=parsed.system)


if __name__ == "__main__":
    main()
