"""The shared flow-matching objective.

This lives beside the backbones rather than inside one of them: both the legacy
EGNN backend and the PaiNN backbone are trained with exactly this loss, so
keeping it inside the EGNN module made the PaiNN package import from a backend
it does not use.
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
    velocity: float = 1.0
    active: float = 0.0
    direction: float = 0.0
    active_pos_weight: float = 1.0
    norm: str = "mse"  # "mse" (L2) or "l1"


def flow_loss(
    output: dict[str, Any],
    batch,
    weights: FlowLossWeights | None = None,
):
    """Combined product/event flow loss.

    The velocity term is the only one enabled by default. The auxiliary heads
    are opt-in because a zero weight must never require a prediction head the
    backbone does not have.
    """
    torch = _torch()
    weights = weights or FlowLossWeights()
    movable = _field(batch, "movable_mask")

    if weights.norm not in {"mse", "l1"}:
        raise ValueError(f"loss norm must be 'mse' or 'l1', got {weights.norm!r}")

    velocity = output["velocity"]
    target_velocity = _field(batch, "target_velocity")
    if torch.any(movable):
        residual = velocity[movable] - target_velocity[movable]
        velocity_loss = (torch.mean(residual.abs()) if weights.norm == "l1"
                         else torch.mean(residual ** 2))
    else:
        velocity_loss = velocity.sum() * 0.0

    active_loss = velocity.sum() * 0.0
    if weights.active != 0.0 and torch.any(movable):
        target_active = _field(batch, "target_active_mask")
        active_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["active_logits"][movable],
            target_active[movable].to(dtype=velocity.dtype),
            pos_weight=torch.as_tensor(
                weights.active_pos_weight,
                dtype=velocity.dtype,
                device=velocity.device,
            ),
        )

    direction_loss = velocity.sum() * 0.0
    if weights.direction != 0.0:
        target_active = _field(batch, "target_active_mask")
        target_direction = _field(batch, "target_direction")
        direction_mask = movable & target_active & (
            torch.linalg.norm(target_direction, dim=1) > 1e-12
        )
        if torch.any(direction_mask):
            pred = output["direction"][direction_mask]
            target = target_direction[direction_mask]
            pred_norm = torch.nn.functional.normalize(pred, dim=1)
            target_norm = torch.nn.functional.normalize(target, dim=1)
            direction_loss = torch.mean(1.0 - torch.sum(pred_norm * target_norm, dim=1))

    total = (
        weights.velocity * velocity_loss
        + weights.active * active_loss
        + weights.direction * direction_loss
    )
    return total, {
        "velocity_loss": velocity_loss.detach(),
        "active_loss": active_loss.detach(),
        "direction_loss": direction_loss.detach(),
    }
