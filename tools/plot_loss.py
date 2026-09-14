"""Plot total loss against epochs from a whitespace-separated training log."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


def parse_loss_points(text: str) -> list[tuple[float, float]]:
    """Return ``(epoch, total_loss)`` points from a whitespace table.

    ``epoch`` is preferred when present. For logs with only ``step`` and
    ``total_loss``, the step column is treated as the epoch axis.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("input is empty")

    header_index = None
    header: list[str] | None = None
    for index, line in enumerate(lines):
        columns = line.split()
        if "total_loss" in columns and ("epoch" in columns or "step" in columns):
            header_index = index
            header = columns
            break

    if header_index is None or header is None:
        raise ValueError("could not find a header with total_loss and epoch or step")

    x_name = "epoch" if "epoch" in header else "step"
    x_index = header.index(x_name)
    loss_index = header.index("total_loss")
    points: list[tuple[float, float]] = []

    for line in lines[header_index + 1 :]:
        columns = line.split()
        if len(columns) <= max(x_index, loss_index):
            continue
        try:
            x_value = float(columns[x_index])
            loss_value = float(columns[loss_index])
        except ValueError:
            continue
        points.append((x_value, loss_value))

    if not points:
        raise ValueError("found header, but no numeric loss rows")
    return points


def plot_loss(
    points: list[tuple[float, float]],
    output_path: str | Path,
    *,
    title: str = "Total Loss vs Epochs",
) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "basinflow_mpl"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [point[0] for point in points]
    losses = [point[1] for point in points]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs, losses, marker="o", linewidth=1.8, markersize=3.5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot total_loss against epochs from a whitespace-separated log file.",
    )
    parser.add_argument("input", help="Path to a loss table or training.log file.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <input-stem>_total_loss.png.",
    )
    parser.add_argument("--title", default="Total Loss vs Epochs", help="Plot title.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_total_loss.png")

    points = parse_loss_points(input_path.read_text(encoding="utf-8"))
    plot_loss(points, output_path, title=args.title)

    print(f"Saved total loss plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
