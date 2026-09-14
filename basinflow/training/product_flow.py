"""Small PyG-based Stage 3 product-flow training loop."""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from itertools import islice
import hashlib
import json

import numpy as np

from basinflow.data.catalog import EventCatalog
from basinflow.data.pyg import EventFlowDataset
from basinflow.models import FlowLossWeights, flow_loss


def _torch():
    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - optional model dependency
        raise ImportError("product-flow training requires basinflow[models]") from exc
    return torch, DataLoader


def active_pos_weight(catalog: EventCatalog, *, active_threshold: float = 0.1) -> float:
    """Return the movable-atom negative/positive label ratio for a catalog."""
    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    positives = negatives = 0
    for event_id in catalog.event_ids:
        target = catalog.event_target(event_id, active_threshold=active_threshold)
        movable = np.asarray(target.reactant.movable_mask, dtype=bool)
        positives += int(np.count_nonzero(target.active_mask & movable))
        negatives += int(np.count_nonzero(~target.active_mask & movable))
    return 1.0 if positives == 0 else float(negatives / positives)


def _as_event_flow_dataset(dataset, initializers, *, active_threshold, flow_time):
    # Any prepared dataset that carries a catalog and can advance its epoch is
    # already in the right form; only a bare catalog has to be wrapped.
    if hasattr(dataset, "catalog") and hasattr(dataset, "set_epoch"):
        return dataset
    if not isinstance(dataset, EventCatalog):
        raise TypeError("dataset must be an EventFlowDataset or EventCatalog")
    if initializers is None:
        raise TypeError("initializers are required when dataset is an EventCatalog")
    return EventFlowDataset(
        dataset,
        initializers,
        active_threshold=active_threshold,
        flow_time=flow_time,
    )


def _batch_size(value, dataset) -> int:
    if isinstance(value, str):
        if value.strip().lower() == "full":
            return max(len(dataset), 1)
        value = int(value)
    value = int(value)
    if value <= 0:
        raise ValueError("batch_size must be positive")
    return value


def _dataset_signature(dataset):
    digest = hashlib.sha256()
    digest.update(json.dumps({
        'event_ids': dataset.catalog.event_ids,
        'initializers': [repr(value) for value in getattr(dataset, 'initializers', ())],
        'seed': dataset.seed, 'flow_time': dataset.fixed_flow_time,
        'active_threshold': getattr(dataset, 'active_threshold', 0.0),
        'diagnostic_oracle': getattr(dataset, 'diagnostic_oracle', False),
    }, sort_keys=True).encode())
    for event_id in dataset.catalog.event_ids:
        event = dataset.catalog.events[event_id]
        digest.update(json.dumps(asdict(event), sort_keys=True, default=str).encode())
        for identifier in (event.reactant_structure_id, event.product_structure_id):
            structure = dataset.catalog.structures[identifier]
            digest.update(json.dumps(structure.species).encode())
            for values in (structure.positions, structure.cell, structure.pbc, structure.movable_mask):
                digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def train_product_flow_epoch(
    model,
    dataset,
    initializers_or_optimizer,
    optimizer=None,
    *,
    device=None,
    batch_size: int | str = 1,
    loss_weights: FlowLossWeights | None = None,
    start_step: int = 0,
    log_step=None,
    active_threshold: float = 0.1,
    t: float | None = None,
    max_steps: int | None = None,
    max_grad_norm: float | None = 10.0,
    scheduler=None,
) -> dict[str, float]:
    """Train one epoch from ``EventFlowDataset`` PyG samples."""
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_grad_norm is not None and (not np.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be finite and positive")
    torch, DataLoader = _torch()
    if optimizer is None:
        optimizer = initializers_or_optimizer
        initializers = None
    else:
        initializers = initializers_or_optimizer
    dataset = _as_event_flow_dataset(
        dataset,
        initializers,
        active_threshold=active_threshold,
        flow_time=t,
    )
    batch_size = _batch_size(batch_size, dataset)
    model.train()
    totals = {"loss": 0.0, "velocity_loss": 0.0, "active_loss": 0.0, "direction_loss": 0.0}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batches = 0
    num_items = 0
    for step, batch in enumerate(islice(loader, max_steps), start=1):
        if device is not None:
            batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss, parts = flow_loss(output, batch, loss_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite product-flow training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf") if max_grad_norm is None else float(max_grad_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batches += 1
        num_items += int(batch.num_graphs)
        values = {
            "loss": float(loss.detach().cpu()),
            "velocity_loss": float(parts["velocity_loss"].detach().cpu()),
            "active_loss": float(parts["active_loss"].detach().cpu()),
            "direction_loss": float(parts["direction_loss"].detach().cpu()),
        }
        for name, value in values.items():
            totals[name] += value
        if log_step is not None:
            log_step(start_step + step, values)
    if batches == 0:
        return {**totals, "num_items": 0, "num_batches": 0, "last_step": start_step}
    return {
        **{name: value / batches for name, value in totals.items()},
        "num_items": num_items,
        "num_batches": batches,
        "last_step": start_step + batches,
    }


def evaluate_product_flow_velocity(
    model,
    dataset,
    initializers=None,
    *,
    device=None,
    batch_size: int | str = 1,
    active_threshold: float = 0.1,
    t: float | None = None,
) -> dict[str, float]:
    """Evaluate fixed epoch-zero pairwise movable-coordinate velocity MSE."""
    torch, DataLoader = _torch()
    dataset = _as_event_flow_dataset(
        dataset, initializers, active_threshold=active_threshold, flow_time=t
    )
    if len(dataset) == 0:
        raise ValueError("validation dataset must contain at least one item")
    batch_size = _batch_size(batch_size, dataset)
    was_training = model.training
    previous_epoch = dataset.epoch
    squared_error = 0.0
    num_components = num_items = num_batches = 0
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    try:
        dataset.set_epoch(0)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if device is not None:
                    batch = batch.to(device)
                velocity = model(batch)["velocity"]
                if not torch.isfinite(velocity).all():
                    raise FloatingPointError("non-finite product-flow validation velocity")
                error = (velocity - batch.target_velocity)[batch.movable_mask]
                batch_error = error.double().square().sum()
                if not torch.isfinite(batch_error):
                    raise FloatingPointError("non-finite product-flow validation loss")
                squared_error += float(batch_error.cpu())
                num_components += error.numel()
                num_items += int(batch.num_graphs)
                num_batches += 1
    finally:
        dataset.set_epoch(previous_epoch)
        model.train(was_training)
    return {
        "velocity_loss": squared_error / num_components if num_components else 0.0,
        "num_components": num_components,
        "num_items": num_items,
        "num_batches": num_batches,
    }


def train_product_flow(
    model,
    dataset,
    initializers=None,
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
    log_to_stdout: bool = False,
    device=None,
    batch_size: int | str = 1,
    loss_weights: FlowLossWeights | None = None,
    active_threshold: float = 0.1,
    t: float | None = None,
    max_steps: int | None = None,
    max_grad_norm: float | None = 10.0,
    validation_dataset=None,
    validation_batch_size: int | str | None = None,
    epoch_checkpoint_dir: str | Path | None = None,
    optimizer_name: str = "adam",
    amsgrad: bool = False,
    weight_decay: float = 0.0,
    scheduler_name: str = "constant",
    warmup_fraction: float = 0.0,
    min_lr_fraction: float = 0.0,
    resume_from: str | Path | None = None,
) -> list[dict[str, float]]:
    """Train a model from an ``EventFlowDataset`` without dict conversion."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer_name must be adam or adamw")
    if not isinstance(amsgrad, bool):
        raise ValueError("amsgrad must be a boolean")
    if not np.isfinite(lr) or lr < 0:
        raise ValueError("lr must be finite and nonnegative")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    torch, _ = _torch()
    dataset = _as_event_flow_dataset(
        dataset,
        initializers,
        active_threshold=active_threshold,
        flow_time=t,
    )
    if validation_dataset is not None:
        validation_dataset = _as_event_flow_dataset(
            validation_dataset, initializers,
            active_threshold=active_threshold, flow_time=t,
        )
        if len(validation_dataset) == 0:
            raise ValueError("validation dataset must contain at least one item")
        validation_batch_size = _batch_size(
            batch_size if validation_batch_size is None else validation_batch_size,
            validation_dataset,
        )
    optimizer_type = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(
        model.parameters(), lr=float(lr), amsgrad=amsgrad,
        weight_decay=float(weight_decay),
    )
    scheduler = build_scheduler(
        optimizer, name=scheduler_name, total_steps=total_scheduler_steps(epochs, dataset, batch_size, max_steps),
        warmup_fraction=warmup_fraction, min_lr_fraction=min_lr_fraction,
    )
    history: list[dict[str, float]] = []
    completed_epochs = 0
    epoch_complete = False
    budget_truncated = False
    signature = _dataset_signature(dataset)
    resume_step = 0
    if resume_from is not None:
        saved = torch.load(resume_from, map_location='cpu', weights_only=False)
        if not saved.get('epoch_complete', False):
            raise ValueError('resume requires a complete epoch checkpoint; partial sampler state is unavailable')
        if saved.get('training_dataset_sha256') != signature:
            raise ValueError('resume dataset identity or training inputs differ')
        requested = {'lr': float(lr), 'optimizer_name': optimizer_name, 'amsgrad': amsgrad,
                     'weight_decay': float(weight_decay), 'batch_size': batch_size,
                     'max_grad_norm': max_grad_norm, 'scheduler_name': scheduler_name,
                     'warmup_fraction': float(warmup_fraction),
                     'min_lr_fraction': float(min_lr_fraction),
                     'loss_weights': asdict(loss_weights or FlowLossWeights())}
        if any(saved['config'].get(key) != value for key, value in requested.items()):
            raise ValueError('resume optimizer, batch, or loss configuration differs')
        completed_epochs = int(saved['completed_epochs'])
        resume_step = int(saved['completed_steps'])
        if epochs <= completed_epochs or (max_steps is not None and max_steps <= resume_step):
            raise ValueError('resume budget must exceed completed training')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        torch.set_rng_state(saved['torch_rng_state'])
        if torch.cuda.is_available() and saved.get('cuda_rng_state_all'):
            torch.cuda.set_rng_state_all(saved['cuda_rng_state_all'])
        history = saved['history']

    def save_checkpoint(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "history": history,
                "epoch": epoch,
                "completed_epochs": completed_epochs,
                "completed_steps": step,
                "epoch_complete": epoch_complete,
                "budget_truncated": budget_truncated,
                "training_dataset_sha256": signature,
                "config": {
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "optimizer_name": optimizer_name,
                    "amsgrad": amsgrad,
                    "weight_decay": float(weight_decay),
                    "batch_size": batch_size,
                    "active_threshold": float(active_threshold),
                    "max_steps": max_steps,
                    "max_grad_norm": max_grad_norm,
                    "scheduler_name": scheduler_name,
                    "warmup_fraction": float(warmup_fraction),
                    "min_lr_fraction": float(min_lr_fraction),
                    "loss_weights": asdict(loss_weights or FlowLossWeights()),
                    "diagnostic_oracle": dataset.diagnostic_oracle,
                    "dataset_seed": dataset.seed,
                    "completed_steps": step,
                    "validation_dataset_seed": None if validation_dataset is None else validation_dataset.seed,
                    "validation_epoch": None if validation_dataset is None else 0,
                    "validation_batch_size": validation_batch_size,
                },
            },
            path,
        )

    if scheduler is not None and resume_step:
        # A resumed run must continue its schedule, not restart it: the budget
        # is the new total, so the rate has to pick up where the last epoch
        # stopped.
        scheduler.last_epoch = resume_step

    output = None
    if log_path is not None:
        output_path = Path(log_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = output_path.open("w", encoding="utf-8")
        output.write("epoch step total_loss velocity_loss active_loss direction_loss\n")
    if log_to_stdout:
        print("epoch step total_loss velocity_loss active_loss direction_loss", flush=True)
    try:
        step = resume_step
        for epoch in range(completed_epochs + 1, epochs + 1):
            dataset.set_epoch(epoch)

            def log_step(step_index, values):
                line = (
                    f"{epoch} {step_index} {values['loss']:.8f} "
                    f"{values['velocity_loss']:.8f} {values['active_loss']:.8f} "
                    f"{values['direction_loss']:.8f}"
                )
                if output is not None:
                    output.write(line + "\n")
                    output.flush()
                if log_to_stdout:
                    print(line, flush=True)

            metrics = train_product_flow_epoch(
                model,
                dataset,
                optimizer,
                device=device,
                batch_size=batch_size,
                loss_weights=loss_weights,
                start_step=step,
                log_step=log_step,
                active_threshold=active_threshold,
                t=t,
                max_steps=None if max_steps is None else max_steps - step,
                max_grad_norm=max_grad_norm,
                scheduler=scheduler,
            )
            step = int(metrics["last_step"])
            if validation_dataset is not None:
                validation_metrics = evaluate_product_flow_velocity(
                    model, validation_dataset, device=device,
                    batch_size=validation_batch_size,
                )
                metrics.update({
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                })
            history.append(metrics)
            epoch_complete = metrics["num_items"] == len(dataset)
            completed_epochs += int(epoch_complete)
            budget_truncated = (
                max_steps is not None and step >= max_steps and completed_epochs < epochs
            )
            if epoch_checkpoint_dir is not None:
                save_checkpoint(Path(epoch_checkpoint_dir) / f"epoch_{epoch:04d}.pt")
            if max_steps is not None and step >= max_steps:
                break
    finally:
        if output is not None:
            output.close()
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path)
    return history


SCHEDULERS = ("constant", "cosine")


def total_scheduler_steps(epochs: int, dataset, batch_size: int | str, max_steps: int | None) -> int:
    """Optimizer steps the schedule has to cover for the requested budget."""
    per_epoch = max(_batch_size(batch_size, dataset), 1)
    steps = -(-len(dataset) // per_epoch) * int(epochs)
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    return max(int(steps), 1)


def build_scheduler(optimizer, *, name: str = "constant", total_steps: int = 1,
                    warmup_fraction: float = 0.0, min_lr_fraction: float = 0.0):
    """Linear warmup followed by a cosine decay, or ``None`` for a constant rate.

    MolGEN reaches its reported accuracy with a warmup/cosine schedule
    (``mdgen/equivariant_wrapper.py:175``), while a constant high rate makes the
    L1 gradient of this task oscillate: every element of an L1 gradient has
    magnitude one, so step noise does not shrink as the loss falls. The
    schedule is what lets a peak rate large enough to fit the training set
    coexist with a quiet finish.
    """
    if name not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {name!r}")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= min_lr_fraction <= 1.0:
        raise ValueError("min_lr_fraction must be in [0, 1]")
    if name == "constant":
        return None
    import torch

    total_steps = max(int(total_steps), 1)
    warmup_steps = int(round(warmup_fraction * total_steps))
    decay_steps = max(total_steps - warmup_steps, 1)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_fraction + (1.0 - min_lr_fraction) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
