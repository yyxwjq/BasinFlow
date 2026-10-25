"""The shared bridge objective: mean residual error over movable atoms.

Every backend emits ``output["scaled"]``, the reference regression target
``(x_t - x_0) / sigma_t`` (``reactot/diffusion/en_sb.py:245-250``). The bridge
loss is a plain MSE (the reference's ``l2``) or L1 over the movable atoms.
"""
from __future__ import annotations

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - optional dependency path
    _TORCH = None


def _torch():
    if _TORCH is None:
        raise ImportError("the bridge loss requires optional model dependency: install torch")
    return _TORCH


def bridge_loss(output, batch, *, norm: str = "mse"):
    """Mean bridge residual error over movable atoms."""
    torch = _torch()
    if norm not in {"mse", "l1"}:
        raise ValueError(f"loss norm must be 'mse' or 'l1', got {norm!r}")
    prediction = output["scaled"]
    target = batch.target_scaled
    if prediction.shape != target.shape:
        raise ValueError(
            f"bridge prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    movable = batch.movable_mask
    residual = prediction[movable] - target[movable]
    if not residual.numel():
        return prediction.sum() * 0.0
    if norm == "l1":
        return torch.mean(residual.abs())
    return torch.mean(residual**2)
