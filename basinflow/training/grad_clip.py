"""ReactOT's adaptive gradient clipping.

The reference clips at ``1.5 * mean + 3 * std`` of a rolling window of recent
gradient norms, seeded with ``3000`` (``reactot/trainer/pl_trainer.py:455-485``),
and pushes the clipped value back into the window.
"""
from __future__ import annotations

import numpy as np
import torch


class GradNormQueue:
    """Most-recent-first window of gradient norms (reference ``training_tools.Queue``)."""

    def __init__(self, max_len: int = 50) -> None:
        if int(max_len) <= 0:
            raise ValueError("max_len must be a positive integer")
        self.max_len = int(max_len)
        self.items: list[float] = []

    def __len__(self) -> int:
        return len(self.items)

    def add(self, value: float) -> None:
        self.items.insert(0, float(value))
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self) -> float:
        return float(np.mean(self.items))

    def std(self) -> float:
        return float(np.std(self.items))


class AdaptiveGradClipper:
    """Clip norms at the reference's rolling bound and record the clipped value."""

    def __init__(self, *, max_len: int = 50, initial: float = 3000.0) -> None:
        self.queue = GradNormQueue(max_len=max_len)
        self.queue.add(initial)

    def bound(self) -> float:
        return 1.5 * self.queue.mean() + 3.0 * self.queue.std()

    def step(self, parameters) -> tuple[float, float]:
        bound = self.bound()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=bound, error_if_nonfinite=True
        )
        self.queue.add(min(float(norm), bound))
        return float(norm), bound
