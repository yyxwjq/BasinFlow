"""Validation for the ReactOT bridge: ``val_ep_scaled_err``.

The reference monitors ``compute_scaled_err`` on the 1073 set-aside reactions
every epoch (``EnSB.forward`` / ``pl_trainer.py:993``): per batch it is
``mean(|(pred - label) / max|label||)``, and the epoch value is the unweighted
mean over the first ``limit_val_batches = 20`` validation batches
(``trainer/_metrics.py:5-20``).
"""
from __future__ import annotations

import numpy as np

from basinflow.training.batch_sampler import NodeSquareBatchSampler


def evaluate_bridge_scaled_error(
    model,
    dataset,
    *,
    device=None,
    max_num: int = 8400,
    max_batches: int | None = 20,
    seed: int = 42,
) -> float:
    """``val_ep_scaled_err`` over the first ``max_batches`` validation batches.

    ``max_num`` is the reference's ``3 * 2800`` validation node budget. The
    dataset should be built with ``lowest_step=1`` to match the reference's
    evaluation-time time range.
    """
    from torch_geometric.loader import DataLoader

    sizes = [event.reactant.n_atoms for event in dataset.catalog.events.values()]
    sampler = NodeSquareBatchSampler(sizes, max_num=max_num, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=sampler)

    was_training = model.training
    model.eval()
    values: list[float] = []
    with __import__("torch").no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= int(max_batches):
                break
            if device is not None:
                batch = batch.to(device)
            prediction = model(batch)["scaled"]
            target = batch.target_scaled
            scale = float(target.abs().max())
            if scale <= 0.0:
                continue
            values.append(float((prediction - target).abs().mean()) / scale)
    if was_training:
        model.train()
    return float(np.mean(values)) if values else float("nan")
