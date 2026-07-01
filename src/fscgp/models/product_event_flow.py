from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ProductEventFlowProtocol(Protocol):
    """Minimal product-event flow model interface."""

    def forward(self, flow_item: dict, t: float | np.ndarray) -> np.ndarray:
        """Return an atom-wise velocity field with shape (N, 3)."""
        ...


@dataclass(frozen=True)
class DummyProductEventFlow:
    """Smoke-test implementation for the ProductEventFlow interface."""

    mode: str = "zero"

    def forward(self, flow_item: dict, t: float | np.ndarray = 0.0) -> np.ndarray:
        target = np.asarray(flow_item["target_velocity"], dtype=float)
        movable_mask = np.asarray(flow_item["movable_mask"], dtype=bool)
        fixed_mask = ~movable_mask
        if self.mode == "zero":
            output = np.zeros_like(target)
        elif self.mode == "target_velocity":
            output = target.copy()
        else:
            raise ValueError(f"unknown dummy flow mode: {self.mode}")
        output[fixed_mask] = 0.0
        return output


def masked_velocity_mse(prediction, flow_item: dict) -> float:
    """Mean squared velocity error over movable atoms only."""
    pred = np.asarray(prediction, dtype=float)
    target = np.asarray(flow_item["target_velocity"], dtype=float)
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction shape must match target_velocity shape, got {pred.shape} and {target.shape}"
        )
    movable_mask = np.asarray(flow_item["movable_mask"], dtype=bool)
    if not movable_mask.any():
        return 0.0
    error = pred[movable_mask] - target[movable_mask]
    return float(np.mean(error**2))
