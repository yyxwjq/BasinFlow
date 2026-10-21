"""ReactOT's ``node^2`` dynamic batch sampler.

The reference sizes a mini-batch by the sum of squared node counts rather than by
the number of molecules (``reactot/dataset/sampler.py:105``): ``max_num = 2800``
for a 16 GB GPU. It draws a fresh permutation each epoch from a generator seeded
once, so the order changes between epochs while remaining reproducible.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class NodeSquareBatchSampler(Sampler[list[int]]):
    """Yield index batches whose ``sum(n_atoms ** 2)`` stays under ``max_num``."""

    def __init__(
        self,
        sizes: Sequence[int],
        *,
        max_num: int = 2800,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if int(max_num) <= 0:
            raise ValueError("max_num must be a positive integer")
        costs = [int(size) ** 2 for size in sizes]
        if any(size <= 0 for size in costs):
            raise ValueError("every dataset item needs at least one atom")
        self.costs = costs
        self.max_num = int(max_num)
        self.drop_last = bool(drop_last)
        self._generator = torch.Generator().manual_seed(int(seed))
        # ``__len__`` is only meaningful after the first epoch; the training loop
        # uses an explicit step budget, so this is a lower bound before then.
        self._num_batches = 1

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(len(self.costs), generator=self._generator).tolist()
        batch: list[int] = []
        cost = 0
        produced = 0
        for index in order:
            item_cost = self.costs[index]
            if batch and cost + item_cost > self.max_num:
                yield batch
                batch = []
                cost = 0
                produced += 1
            batch.append(index)
            cost += item_cost
        # ``drop_last`` drops a trailing partial batch, as the reference does. If
        # the whole dataset fits in one batch there is no full batch to keep, so
        # that batch is yielded anyway: dropping it would silently train zero
        # steps on small datasets.
        if batch and (not self.drop_last or produced == 0):
            yield batch
            produced += 1
        self._num_batches = max(produced, 1)

    def __len__(self) -> int:
        return self._num_batches
