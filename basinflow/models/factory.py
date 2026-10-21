"""Construct flow backends and restore saved checkpoints."""
from __future__ import annotations

from collections.abc import Mapping

from .egnn import EGNNFlow
from .leftnet.adapter import LeftNetBridge
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
            # Time-channel controls: ``time_scale`` widens the Fourier basis and
            # ``time_per_layer`` re-injects the clock into every message block.
            "time_scale": float(config.get("time_scale", 1.0)),
            "time_per_layer": _flag(config.get("time_per_layer", False)),
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
    elif backend in {"leftnet", "leftnet_sb"}:
        # ReactOT's LEFTNet defaults (train_rpsb_ts1x.py:38-51). ``leftnet`` and
        # ``leftnet_sb`` build the same module; the second name selects the
        # Schrodinger-bridge objective.
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
    "leftnet": LeftNetBridge,
    "leftnet_sb": LeftNetBridge,
}


def build_model(backend: str, model_config: Mapping):
    backend, parameters = model_spec({**model_config, "backend": backend})
    try:
        model_type = _MODEL_TYPES[backend]
    except KeyError:
        raise ValueError(f"unknown model backend: {backend}") from None
    return model_type(**parameters)


def model_cutoff(model) -> float:
    r_max = getattr(model, "r_max", None)
    return float(r_max if r_max is not None else model.cutoff)


def load_model_checkpoint(checkpoint: Mapping, *, device="cpu"):
    config = checkpoint["model_config"]
    if "object_aware" in config or "hidden_channels" in config:
        inferred = "leftnet_sb"
    elif "num_features" in config or "r_max" in config:
        inferred = "painn"
    else:
        inferred = "egnn"
    backend = checkpoint.get("model_backend", inferred)
    if backend == "painn" and "stability_mode" not in config:
        config = {**config, "stability_mode": "none"}
    model = build_model(backend, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
