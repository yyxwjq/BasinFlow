"""Train and sample converted Transition1x event files.

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

from basinflow.workflows import run_transition1x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a converted Transition1x INI config.")
    parsed = parser.parse_args()
    run_transition1x(parsed.config)


if __name__ == "__main__":
    main()
