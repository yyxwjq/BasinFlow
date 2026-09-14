import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_painn_config_aliases_and_checkpoint_roundtrip():
    from basinflow.models.factory import build_model, load_model_checkpoint, model_cutoff, model_spec

    backend, config = model_spec({"backend": "painn", "hidden_dim": "8", "num_layers": "1", "cutoff": "3.5", "num_rbf": "6"})
    model = build_model(backend, config)
    assert config["num_features"] == 8
    assert config["num_radial_basis"] == 6
    assert model_cutoff(model) == 3.5
    restored = load_model_checkpoint({"model_backend": backend, "model_config": config, "model_state_dict": model.state_dict()})
    assert type(restored) is type(model)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_legacy_egnn_checkpoint_loads_without_backend():
    from basinflow.models.factory import build_model, load_model_checkpoint, model_cutoff

    config = {"hidden_dim": 8, "num_layers": 1, "cutoff": 3.0}
    model = build_model("egnn", config)
    restored = load_model_checkpoint({"model_config": config, "model_state_dict": model.state_dict()})
    assert type(restored) is type(model)
    assert model_cutoff(restored) == 3.0


def test_unknown_backend_is_rejected():
    from basinflow.models.factory import model_spec

    with pytest.raises(ValueError, match="backend"):
        model_spec({"backend": "unknown"})


def test_legacy_painn_checkpoint_retains_unscaled_predictions():
    from basinflow.models.factory import build_model, load_model_checkpoint
    from test_painn_product_flow import _batch

    config = {"num_features": 8, "num_layers": 2, "num_radial_basis": 8, "r_max": 2.2}
    model = build_model("painn", {**config, "stability_mode": "none"}).double()
    restored = load_model_checkpoint({"model_backend": "painn", "model_config": config, "model_state_dict": model.state_dict()}).double()
    torch.testing.assert_close(restored(_batch())["velocity"], model(_batch())["velocity"])
    assert not restored.backbone.messages[0].scaled


def test_transition1x_painn_runner_loads_split_and_samples_validation(tmp_path):
    import configparser
    import csv
    import json
    from pathlib import Path
    import subprocess
    import sys

    from test_stage3_au_scripts import _write_tiny_events

    events_dir = tmp_path / "events"
    events_dir.mkdir()
    _write_tiny_events(events_dir)
    table = events_dir / "basin_table.csv"
    with table.open() as source:
        rows = list(csv.DictReader(source))
    for index, row in enumerate(rows):
        row["dataset_kind"] = "transition1x_single_event_molecule"
        row["source_index"] = str(index)
    with table.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps({"train": ["b0"], "val": ["b1"], "test": [], "seed": 42}))
    config = configparser.ConfigParser()
    config.read(Path("configs/0910/smoke_transition1x.ini"))
    config["data"]["events_dir"] = str(events_dir)
    config["split"]["manifest"] = str(manifest_path)
    config["output"]["output_dir"] = str(tmp_path / "output")
    config["model"]["num_features"] = "8"
    config["model"]["num_layers"] = "1"
    config["runtime"]["cpu_threads"] = "1"
    config_path = tmp_path / "config.ini"
    with config_path.open("w") as destination:
        config.write(destination)
    result = subprocess.run([sys.executable, "examples/train_transition1x_product_flow.py", "--config", str(config_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    checkpoint = torch.load(tmp_path / "output" / "checkpoint.pt", weights_only=False)
    assert checkpoint["model_backend"] == "painn"
    assert checkpoint["evaluation_split"] == "val"
    assert checkpoint["history"][-1]["last_step"] == 2
    report = json.loads((tmp_path / "output" / "eval_metrics.json").read_text())
    assert report["sampling"]["num_test_basins"] == 1
    assert report["relaxation"] == "not_run"
