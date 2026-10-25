"""Load the released ReactOT Lightning checkpoint.

The public checkpoint (``reactot-pretrained.ckpt``) stores the LEFTNet weights as
``ddpm.dynamics.model.*`` plus the per-fragment ``encoders``/``decoders`` used by
``EGNNDynamics``. :class:`LeftNetBridge` keeps the same module tree, so the map is
a rename. ``torch.load`` also unpickles the reference's ``LEFTNet`` *class* stored
in the hyper-parameters, so the reference checkout has to be importable: pass
``reference_src`` (or set ``REACTOT_SRC``) to put it on ``sys.path`` first.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from basinflow.models.leftnet.adapter import LeftNetBridge


def _install_torch_scatter_stub() -> None:
    """Let ``torch.load`` unpickle the reference classes without compiling PyG.

    ``reactot.model`` imports ``torch_scatter`` at import time.  This module only
    reads the checkpoint's ``state_dict``/``hyper_parameters``; it never runs the
    reference forward, so a stub with the imported names is sufficient.
    """
    try:
        import torch_scatter  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    import types

    stub = types.ModuleType("torch_scatter")

    def unavailable(*args, **kwargs):
        raise RuntimeError("torch_scatter is not installed; the reference forward is not used")

    for name in (
        "scatter",
        "scatter_mean",
        "scatter_add",
        "scatter_max",
        "scatter_min",
        "segment_coo",
        "segment_csr",
    ):
        setattr(stub, name, unavailable)
    sys.modules["torch_scatter"] = stub

_MODEL_PREFIX = "ddpm.dynamics.model."
_DYNAMICS_PREFIX = "ddpm.dynamics."


def load_reactot_checkpoint(
    path: str | Path,
    *,
    reference_src: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[LeftNetBridge, dict]:
    """Return ``(model, hyper_parameters)`` for a released ReactOT checkpoint."""
    source = reference_src or os.environ.get("REACTOT_SRC")
    if source is not None and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    _install_torch_scatter_stub()
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint or "hyper_parameters" not in checkpoint:
        raise ValueError(f"{path} is not a ReactOT Lightning checkpoint")

    hyper = checkpoint["hyper_parameters"]
    model_config = hyper["model_config"]
    model = LeftNetBridge(
        hidden_channels=int(model_config["hidden_channels"]),
        num_layers=int(model_config["num_layers"]),
        num_radial=int(model_config["num_radial"]),
        cutoff=float(model_config["cutoff"]),
        num_elements=int(hyper["node_nfs"][0]) - 4,  # 9 - pos_dim - time - condition
        reflect_equiv=bool(model_config.get("reflect_equiv", True)),
        legacy=bool(model_config.get("legacy", True)),
        update=bool(model_config.get("update", True)),
        single_layer_output=bool(model_config.get("single_layer_output", True)),
        object_aware=bool(model_config.get("object_aware", True)),
    )

    mapped = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith(_MODEL_PREFIX):
            mapped["leftnet." + key[len(_MODEL_PREFIX):]] = value
        elif key.startswith(_DYNAMICS_PREFIX):
            mapped[key[len(_DYNAMICS_PREFIX):]] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the port: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.to(device).eval()
    return model, hyper
