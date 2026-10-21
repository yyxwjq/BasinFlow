"""The shared objective.

The PaiNN/EGNN backends fit a velocity field; the ReactOT bridge fits the
``(x_t - x_0) / sigma_t`` residual. :func:`flow_loss` selects by output key.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - optional dependency path
    _TORCH = None


def _torch():
    if _TORCH is None:
        raise ImportError("the flow loss requires optional model dependency: install torch")
    return _TORCH


def _field(batch, name: str):
    """Read a batch entry from either a dict batch or an object batch."""
    return batch[name] if isinstance(batch, dict) else getattr(batch, name)


@dataclass(frozen=True)
class FlowLossWeights:
    """Velocity scale and norm. Every field is read by :func:`flow_loss`."""

    velocity: float = 1.0
    norm: str = "mse"  # "mse" (L2) or "l1"


def _maybe_field(batch, name: str):
    """Like :func:`_field` but ``None`` when the batch has no such entry."""
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def _bridge_loss(output: dict[str, Any], batch, weights: "FlowLossWeights", torch):
    """ReactOT's Schrodinger-bridge objective, selected by the ``"scaled"`` key.

    The reference fits ``net_out = (x_t - x_0) / sigma_t``
    (``reactot/diffusion/en_sb.py:245-250``) and then recovers ``x_0`` with
    ``x_t - sigma_t * net_out``; there is no constant velocity field for this
    task, so ``target_velocity`` is not involved. The target is produced by the
    bridge dataset view and is time-dependent.
    """
    prediction = output["scaled"]
    target = _field(batch, "target_scaled")
    if prediction.shape != target.shape:
        raise ValueError(
            f"bridge prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    movable = _maybe_field(batch, "movable_mask")
    if movable is None:
        residual = prediction - target
    else:
        residual = prediction[movable] - target[movable]
    scaled_loss = (
        torch.mean(residual.abs()) if weights.norm == "l1" else torch.mean(residual**2)
    )
    # ``SBModule.compute_loss`` returns ``EnSB.forward``'s plain
    # ``F.mse_loss(pred, label)``. The ``scales = [1, 2, 1]`` normalisation lives in
    # ``DDPMModule`` and is *not* applied to the Schrödinger-bridge objective.
    total = weights.velocity * scaled_loss
    return total, {"velocity_loss": scaled_loss.detach(), "scaled_loss": scaled_loss.detach()}


def flow_loss(
    output: dict[str, Any],
    batch,
    weights: FlowLossWeights | None = None,
):
    """Mean velocity error over movable atoms, and its detached parts.

    Errors are taken on ``movable_mask`` only: a fixed atom has no target
    displacement by construction, and including it would dilute the gradient with
    a term the model is not allowed to act on.

    A model that emits the bridge residual ``"scaled"`` is scored with
    :func:`_bridge_loss` instead; every existing velocity-only run is unaffected
    because its output has no such key.
    """
    torch = _torch()
    weights = weights or FlowLossWeights()
    if weights.norm not in {"mse", "l1"}:
        raise ValueError(f"loss norm must be 'mse' or 'l1', got {weights.norm!r}")
    if "scaled" in output:
        return _bridge_loss(output, batch, weights, torch)
    movable = _field(batch, "movable_mask")
    prediction = output["velocity"]  # the head always emits a [N, 3] field
    target = _field(batch, "target_velocity")
    if torch.any(movable):
        residual = prediction[movable] - target[movable]
        velocity_loss = (
            torch.mean(residual.abs()) if weights.norm == "l1" else torch.mean(residual**2)
        )
    else:
        velocity_loss = prediction.sum() * 0.0
    total = weights.velocity * velocity_loss
    return total, {"velocity_loss": velocity_loss.detach()}
