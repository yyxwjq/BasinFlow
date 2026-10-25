"""Construct one of the three interchangeable R + P -> TS bridge backends."""
from __future__ import annotations

from collections.abc import Mapping

from .egnn import EGNNFlow
from .leftnet.adapter import LeftNetBridge
from .painn import PaiNN


def _flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def model_spec(config: Mapping, *, default_backend: str = "leftnet_sb") -> tuple[str, dict]:
    """Return ``(backend, constructor kwargs)`` for a ``[model]`` config section."""
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
            "time_scale": float(config.get("time_scale", 1.0)),
            "time_per_layer": _flag(config.get("time_per_layer", False)),
        }
    elif backend == "egnn":
        parameters = {
            "hidden_dim": int(config.get("hidden_dim", 64)),
            "num_layers": int(config.get("num_layers", 3)),
            "cutoff": float(config.get("cutoff", 5.0)),
            "max_atomic_number": int(config.get("max_atomic_number", 100)),
        }
    elif backend == "leftnet_sb":
        # ReactOT's LEFTNet defaults (train_rpsb_ts1x.py:38-51).
        parameters = {
            "hidden_channels": int(config.get("num_features", config.get("hidden_channels", 196))),
            "num_layers": int(config.get("num_layers", 6)),
            "num_radial": int(config.get("num_radial", config.get("num_radial_basis", 96))),
            "cutoff": float(config.get("cutoff", config.get("r_max", 10.0))),
            "num_elements": int(config.get("num_elements", 5)),
            "reflect_equiv": _flag(config.get("reflect_equiv", True)),
            "legacy": _flag(config.get("legacy", True)),
            "update": _flag(config.get("update", True)),
            "single_layer_output": _flag(config.get("single_layer_output", True)),
            "object_aware": _flag(config.get("object_aware", True)),
        }
    else:
        raise ValueError(f"unknown model backend: {backend}")
    return backend, parameters


_MODEL_TYPES = {
    "painn": PaiNN,
    "egnn": EGNNFlow,
    "leftnet_sb": LeftNetBridge,
}


def build_model(backend: str, model_config: Mapping):
    backend, parameters = model_spec({**model_config, "backend": backend})
    try:
        model_type = _MODEL_TYPES[backend]
    except KeyError:
        raise ValueError(f"unknown model backend: {backend}") from None
    return model_type(**parameters)
