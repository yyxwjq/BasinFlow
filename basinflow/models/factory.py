"""Construct flow backends and restore explicit or legacy checkpoints."""
from __future__ import annotations

from collections.abc import Mapping

from .egnn import EGNNFlow
from .painn import PaiNN


def _flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def model_spec(config: Mapping, *, default_backend: str = "painn") -> tuple[str, dict]:
    backend = str(config.get("backend", default_backend)).strip().lower()
    if backend == "painn":
        parameters = {
            "num_features": int(config.get("num_features", config.get("hidden_dim", 128))),
            "num_layers": int(config.get("num_layers", 4)),
            "num_radial_basis": int(config.get("num_radial_basis", config.get("num_rbf", 32))),
            "num_elements": int(config.get("num_elements", 119)),
            "r_max": float(config.get("r_max", config.get("cutoff", 5.0))),
            "time_dim": int(config.get("time_dim", 32)),
            "radial_basis": str(config.get("radial_basis", "bessel")),
            "envelope": str(config.get("envelope", "polynomial")),
            "stability_mode": str(config.get("stability_mode", "scaled")),
            "bond_change_condition": _flag(config.get("bond_change_condition", False)),
            "bond_change_direction": _flag(config.get("bond_change_direction", False)),
            "endpoint_condition": _flag(config.get("endpoint_condition", False)),
            "endpoint_equivariant": _flag(config.get("endpoint_equivariant", False)),
        }
    elif backend == "egnn":
        parameters = {
            "hidden_dim": int(config.get("hidden_dim", 64)),
            "num_layers": int(config.get("num_layers", 3)),
            "cutoff": float(config.get("cutoff", 5.0)),
        }
        for name in ("max_atomic_number", "seed_type_vocab"):
            if name in config:
                parameters[name] = int(config[name])
    else:
        raise ValueError(f"unknown model backend: {backend}")
    return backend, parameters


def build_model(backend: str, model_config: Mapping):
    backend, parameters = model_spec({**model_config, "backend": backend})
    model_type = PaiNN if backend == "painn" else EGNNFlow
    return model_type(**parameters)


def model_cutoff(model) -> float:
    return float(model.r_max if isinstance(model, PaiNN) else model.cutoff)


def load_model_checkpoint(checkpoint: Mapping, *, device="cpu"):
    config = checkpoint["model_config"]
    inferred = "painn" if "num_features" in config or "r_max" in config else "egnn"
    backend = checkpoint.get("model_backend", inferred)
    if backend == "painn" and "stability_mode" not in config:
        config = {**config, "stability_mode": "none"}
    model = build_model(backend, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
