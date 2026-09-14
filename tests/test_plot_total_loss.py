import importlib.util
from pathlib import Path
import subprocess
import sys


def _load_plot_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "plot_loss.py"
    spec = importlib.util.spec_from_file_location("plot_loss", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_step_total_loss_table_as_epochs():
    plot_loss = _load_plot_module()
    text = "\n".join(
        [
            "step total_loss velocity_loss active_loss direction_loss",
            "1 3.54381013 3.45688605 0.86308980 0.00615050",
            "2 3.25437951 3.17370605 0.80492896 0.00180558",
            "3 2.95749784 2.88178444 0.75636530 0.00076818",
        ]
    )

    points = plot_loss.parse_loss_points(text)

    assert points == [(1.0, 3.54381013), (2.0, 3.25437951), (3.0, 2.95749784)]


def test_parse_epoch_step_total_loss_log_prefers_epoch_column():
    plot_loss = _load_plot_module()
    text = "\n".join(
        [
            "epoch step total_loss velocity_loss active_loss direction_loss",
            "1 1 0.50 0.40 0.90 0.01",
            "1 2 0.30 0.20 0.80 0.01",
            "2 3 0.20 0.10 0.70 0.01",
        ]
    )

    points = plot_loss.parse_loss_points(text)

    assert points == [(1.0, 0.50), (1.0, 0.30), (2.0, 0.20)]


def test_plot_loss_cli_writes_png(tmp_path):
    input_path = tmp_path / "loss.txt"
    output_path = tmp_path / "loss.png"
    input_path.write_text(
        "\n".join(
            [
                "step total_loss velocity_loss active_loss direction_loss",
                "1 3.54381013 3.45688605 0.86308980 0.00615050",
                "2 3.25437951 3.17370605 0.80492896 0.00180558",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/plot_loss.py",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert "Saved total loss plot" in result.stdout
