from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - exercised by optional-dependency callers
    _TORCH = None


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


def _torch():
    if _TORCH is None:
        raise ImportError(
            "torch model utilities require optional model dependency: install torch"
        )
    return _TORCH


def torch_masked_velocity_mse(prediction, batch: dict):
    """Torch MSE over movable atoms only."""
    torch = _torch()
    target = batch["target_velocity"]
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction shape must match target_velocity shape, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    movable_mask = batch["movable_mask"]
    if not torch.any(movable_mask):
        return prediction.sum() * 0.0
    error = prediction[movable_mask] - target[movable_mask]
    return torch.mean(error**2)


_ModuleBase = _TORCH.nn.Module if _TORCH is not None else object


class MinimalProductEventFlow(_ModuleBase):
    """Tiny trainable product-event flow used only for Stage 3 smoke tests."""

    def __init__(self, hidden_dim: int = 32) -> None:
        torch = _torch()
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(14, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 3),
        )

    def forward(self, batch: dict, t=0.0):
        torch = _torch()
        initial = batch["initial_positions"]
        reactant = batch["reactant_positions"]
        seed = batch["seed_displacement"]
        movable = batch["movable_mask"]
        if isinstance(t, (float, int)):
            time = torch.full(
                (initial.shape[0], 1),
                float(t),
                dtype=initial.dtype,
                device=initial.device,
            )
        else:
            time_tensor = torch.as_tensor(t, dtype=initial.dtype, device=initial.device)
            time = time_tensor.reshape(-1, 1)
            if time.shape[0] == 1:
                time = time.expand(initial.shape[0], 1)
        movable_float = movable.to(dtype=initial.dtype).reshape(-1, 1)
        features = torch.cat(
            [initial, reactant, seed, movable_float, time, initial - reactant],
            dim=1,
        )
        output = self.net(features)
        return output * movable_float
