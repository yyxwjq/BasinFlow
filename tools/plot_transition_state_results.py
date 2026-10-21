"""Figures for a finished R + P -> TS run.

Three families, mirroring ``sampling/plots`` of the R -> P runs:

* ``<event>_rmsd_40_trials.png`` — one figure per evaluated event: the per-trial
  RMSD trace and its histogram;
* ``rmsd_by_event.png`` — mean RMSD per event, sorted, for an at-a-glance view of
  which events the field still misses;
* ``rmsd_distribution.png`` — the summary over every trial, with the per-event
  best-of-N overlaid and the reference baselines marked.

Usage:
    python tools/plot_transition_state_results.py <run-dir> [--output DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/basinflow-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Heiti TC covers both CJK and the Angstrom sign; the Latin-only default
    # sans font would drop characters from the annotations.
    plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# Reference value for this task (full val split).
MIDPOINT_BRIDGE = 0.4870


def read_trials(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def per_event_figures(rows: list[dict[str, str]], plots_dir: Path, plt) -> list[tuple[str, float]]:
    by_event: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_event.setdefault(str(row["event_id"]), []).append(row)

    means: list[tuple[str, float]] = []
    for event_id, event_rows in by_event.items():
        event_rows.sort(key=lambda row: int(row["trial_index"]))
        trials = [int(row["trial_index"]) for row in event_rows]
        aligned = [float(row["aligned_rmsd_angstrom"]) for row in event_rows]
        source = [float(row["source_rmsd_angstrom"]) for row in event_rows]
        mean_value = float(np.mean(aligned))
        median_value = float(np.median(aligned))
        means.append((event_id, mean_value))

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(trials, aligned, "o-", markersize=2.5, linewidth=0.8,
                     color="tab:blue", label="aligned RMSD")
        axes[0].axhline(mean_value, color="tab:red", linestyle="--", linewidth=1,
                        label=f"mean={mean_value:.3f} Å")
        axes[0].axhline(MIDPOINT_BRIDGE, color="tab:orange", linestyle=":", linewidth=1,
                        label=f"midpoint bridge={MIDPOINT_BRIDGE:.3f} Å")
        axes[0].set_xlabel("Trial index")
        axes[0].set_ylabel("Aligned RMSD (Å)")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

        bins = min(20, max(5, len(aligned) // 3))
        axes[1].hist(aligned, bins=bins, alpha=0.75, color="tab:blue")
        axes[1].axvline(mean_value, color="tab:red", linestyle="--", linewidth=1.2,
                        label=f"mean={mean_value:.3f}")
        axes[1].axvline(median_value, color="tab:green", linestyle="-.", linewidth=1.2,
                        label=f"median={median_value:.3f}")
        axes[1].set_xlabel("Aligned RMSD (Å)")
        axes[1].set_ylabel("Trial count")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)

        figure.suptitle(
            f"{event_id} · {len(aligned)} trials · source RMSD {np.mean(source):.2f} Å", fontsize=11)
        figure.tight_layout()
        figure.savefig(plots_dir / f"{event_id.replace(':', '_')}_rmsd_{len(aligned)}_trials.png",
                       dpi=140)
        plt.close(figure)
    return means


def by_event_figure(means: list[tuple[str, float]], plots_dir: Path, plt) -> None:
    ordered = sorted(means, key=lambda item: item[1])
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(np.arange(len(ordered)), [value for _, value in ordered], "o", markersize=2.5,
              color="tab:blue")
    axis.axhline(MIDPOINT_BRIDGE, color="tab:orange", linestyle=":", linewidth=1.2,
                 label=f"midpoint bridge {MIDPOINT_BRIDGE:.3f} Å")
    axis.set_xlabel(f"Events, sorted by mean RMSD (n={len(ordered)})")
    axis.set_ylabel("Mean aligned RMSD (Å)")
    axis.set_title("Per-event mean aligned RMSD")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_dir / "rmsd_by_event.png", dpi=140)
    plt.close(figure)


def summary_figure(rows: list[dict[str, str]], plots_dir: Path, plt, *, split: str) -> dict:
    aligned = np.asarray([float(row["aligned_rmsd_angstrom"]) for row in rows], dtype=float)
    best: dict[str, float] = {}
    for row in rows:
        key = str(row["event_id"])
        best[key] = min(best.get(key, float("inf")), float(row["aligned_rmsd_angstrom"]))
    best_values = np.asarray(list(best.values()), dtype=float)

    mean_value = float(aligned.mean())
    median_value = float(np.median(aligned))
    bins = min(80, max(10, int(np.sqrt(aligned.size))))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].hist(aligned, bins=bins, alpha=0.75, color="tab:blue", label="each trial")
    axes[0].hist(best_values, bins=bins, alpha=0.75, color="tab:green",
                 label=f"best-of-{len(rows) // max(len(best), 1)} per event")
    axes[0].axvline(mean_value, color="tab:red", linestyle="--", linewidth=1.4,
                    label=f"mean={mean_value:.3f} Å")
    axes[0].axvline(median_value, color="black", linestyle="-.", linewidth=1.2,
                    label=f"median={median_value:.3f} Å")
    axes[0].axvline(MIDPOINT_BRIDGE, color="tab:orange", linestyle=":", linewidth=1.6,
                    label=f"midpoint bridge={MIDPOINT_BRIDGE:.3f} Å")
    axes[0].set_xlabel("Aligned RMSD (Å)")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title(f"All {aligned.size} samples on {split}")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.25)

    within = [0.1, 0.2, 0.3, 0.5, 1.0]
    fractions_trial = [float((aligned <= value).mean()) for value in within]
    fractions_best = [float((best_values <= value).mean()) for value in within]
    positions = np.arange(len(within))
    axes[1].bar(positions - 0.2, fractions_trial, width=0.4, color="tab:blue", label="each trial")
    axes[1].bar(positions + 0.2, fractions_best, width=0.4, color="tab:green", label="best per event")
    for position, value in zip(positions - 0.2, fractions_trial):
        axes[1].text(position, value + 0.015, f"{value:.0%}", ha="center", fontsize=8)
    for position, value in zip(positions + 0.2, fractions_best):
        axes[1].text(position, value + 0.015, f"{value:.0%}", ha="center", fontsize=8)
    axes[1].set_xticks(positions, [f"≤ {value} Å" for value in within])
    axes[1].set_ylabel("Fraction of samples")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_title("Cumulative accuracy")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)

    figure.suptitle(
        f"R + P → TS aligned RMSD · {len(best)} events × {len(rows) // max(len(best), 1)} trials",
        fontsize=12)
    figure.tight_layout()
    figure.savefig(plots_dir / "rmsd_distribution.png", dpi=160)
    plt.close(figure)

    return {
        "count": int(aligned.size),
        "mean": mean_value,
        "median": median_value,
        "min": float(aligned.min()),
        "max": float(aligned.max()),
        "fraction_within_angstrom": dict(zip((str(v) for v in within), fractions_trial)),
        "best_of_event": {
            "count": int(best_values.size),
            "mean": float(best_values.mean()),
            "median": float(np.median(best_values)),
            "fraction_within_angstrom": dict(zip((str(v) for v in within), fractions_best)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path,
                        help="run directory, or the trial_metrics.csv itself")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to <run-dir>/sampling/plots")
    parser.add_argument("--per-event", action="store_true",
                        help="also write one figure per evaluated event")
    args = parser.parse_args(argv)

    if args.run_dir.is_file():
        metrics = args.run_dir
        run_dir = metrics.parent.parent
    else:
        run_dir = args.run_dir
        metrics = run_dir / "sampling" / "trial_metrics.csv"
    if not metrics.is_file():
        raise FileNotFoundError(f"no trial metrics at {metrics}")
    plots_dir = args.output or run_dir / "sampling" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = read_trials(metrics)
    if not rows:
        raise ValueError(f"{metrics} has no rows")

    report_path = run_dir / "report.json"
    split = "val"
    if report_path.is_file():
        evaluation = json.loads(report_path.read_text(encoding="utf-8")).get("evaluation", {})
        split = str(evaluation.get("split", split))

    plt = _matplotlib()
    if args.per_event:
        means = per_event_figures(rows, plots_dir, plt)
        by_event_figure(means, plots_dir, plt)
        print(f"per-event figures: {len(means)}")
    summary = summary_figure(rows, plots_dir, plt, split=split)

    (plots_dir / "plot_summary.json").write_text(
        json.dumps({"split": split, "aligned_rmsd_angstrom": summary}, indent=2) + "\n",
        encoding="utf-8")
    print(f"split: {split}")
    print(f"samples: {summary['count']}  mean {summary['mean']:.4f}  median {summary['median']:.4f}")
    print(f"best-of-event median: {summary['best_of_event']['median']:.4f}")
    print(f"wrote figures to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
