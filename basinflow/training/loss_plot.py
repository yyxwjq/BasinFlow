"""Aggregate the step log by epoch and write a small loss plot."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/basinflow-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_loss_artifacts(training_log: Path, output_dir: Path) -> None:
    """Aggregate step log records by epoch and write the loss plot."""
    rows: list[dict[str, float]] = []
    with Path(training_log).open(encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=" ", skipinitialspace=True)
        for row in reader:
            if not row or row.get("epoch") in {None, ""}:
                continue
            rows.append({key: float(row[key]) for key in row if row[key] not in {None, ""}})

    by_epoch: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_epoch[int(row["epoch"])].append(row)
    aggregate_rows = [
        {
            "epoch": epoch,
            "loss": float(np.mean([row["loss"] for row in epoch_rows])),
        }
        for epoch, epoch_rows in sorted(by_epoch.items())
    ]
    with (output_dir / "loss_by_epoch.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        [row["epoch"] for row in aggregate_rows],
        [row["loss"] for row in aggregate_rows],
        label="bridge loss",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean training loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close(figure)
