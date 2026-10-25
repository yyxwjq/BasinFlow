"""The single PyG training loop for every R + P -> TS bridge backend."""
from __future__ import annotations

import copy
import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np

from basinflow.models import bridge_loss
from basinflow.training.batch_sampler import NodeSquareBatchSampler
from basinflow.training.grad_clip import AdaptiveGradClipper
from basinflow.training.validation import evaluate_bridge_scaled_error


def _torch():
    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - optional model dependency
        raise ImportError("bridge training requires basinflow[models]") from exc
    return torch, DataLoader


def build_ema(model):
    """A frozen, exponentially averaged copy of ``model``."""
    ema = copy.deepcopy(model)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    ema.eval()
    return ema


def update_ema(ema, model, decay: float) -> None:
    """One EMA step: ``ema <- decay * ema + (1 - decay) * model``."""
    import torch

    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def _batch_size(value, dataset) -> int:
    if isinstance(value, str):
        if value.strip().lower() == "full":
            return max(len(dataset), 1)
        value = int(value)
    value = int(value)
    if value <= 0:
        raise ValueError("batch_size must be positive")
    return value


def _check_dataset(dataset) -> None:
    if not hasattr(dataset, "catalog") or not hasattr(dataset, "set_epoch"):
        raise TypeError("dataset must be a prepared BridgeFlowDataset")


def _dataset_signature(dataset) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "event_ids": dataset.catalog.event_ids,
                "seed": dataset.seed,
                "lowest_step": dataset.lowest_step,
                "timesteps": dataset.schedule.timesteps,
                "swap_reactant_product": dataset.swap_reactant_product,
            },
            sort_keys=True,
        ).encode()
    )
    for event_id in dataset.catalog.event_ids:
        event = dataset.catalog.events[event_id]
        digest.update(
            json.dumps(
                {
                    "event_id": event.event_id,
                    "metadata": {key: str(value) for key, value in event.metadata.items()},
                    "has_transition_state": event.transition_state is not None,
                },
                sort_keys=True,
            ).encode()
        )
        for structure in (event.reactant, event.product, event.transition_state):
            if structure is None:
                continue
            digest.update(json.dumps(structure.species).encode())
            for values in (
                structure.positions,
                structure.cell,
                structure.pbc,
                structure.movable_mask,
            ):
                digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def train_epoch(
    model,
    dataset,
    optimizer,
    *,
    device=None,
    batch_size: int | str = 1,
    loss_norm: str = "mse",
    start_step: int = 0,
    log_step=None,
    max_steps: int | None = None,
    max_grad_norm: float | None = 10.0,
    scheduler=None,
    ema=None,
    ema_decay: float = 0.0,
    batch_sampler=None,
    grad_clipper=None,
) -> dict[str, float]:
    """Train one epoch, or ``max_steps`` optimizer steps."""
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_grad_norm is not None and (not np.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be finite and positive")
    torch, DataLoader = _torch()
    _check_dataset(dataset)
    batch_size = _batch_size(batch_size, dataset)
    model.train()
    totals = {"loss": 0.0}
    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batches = 0
    num_items = 0
    for step, batch in enumerate(islice(loader, max_steps), start=1):
        if device is not None:
            batch = batch.to(device)
        optimizer.zero_grad()
        loss = bridge_loss(model(batch), batch, norm=loss_norm)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite bridge training loss")
        loss.backward()
        if grad_clipper is not None:
            grad_clipper.step(model.parameters())
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf") if max_grad_norm is None else float(max_grad_norm),
                error_if_nonfinite=True,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            update_ema(ema, model, float(ema_decay))
        batches += 1
        num_items += int(batch.num_graphs)
        values = {"loss": float(loss.detach().cpu())}
        totals["loss"] += values["loss"]
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


def train_loop(
    model,
    dataset,
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
    log_to_stdout: bool = False,
    device=None,
    batch_size: int | str = 1,
    loss_norm: str = "mse",
    max_steps: int | None = None,
    max_grad_norm: float | None = 10.0,
    epoch_checkpoint_dir: str | Path | None = None,
    optimizer_name: str = "adam",
    amsgrad: bool = False,
    weight_decay: float = 0.0,
    scheduler_name: str = "constant",
    warmup_fraction: float = 0.0,
    min_lr_fraction: float = 0.0,
    resume_from: str | Path | None = None,
    ema_decay: float | None = None,
    batch_sampler: str | None = None,
    max_num: int = 2800,
    adaptive_clip: bool = False,
    max_batches_per_epoch: int | None = None,
    bridge_validation_dataset=None,
    validation_max_num: int = 8400,
    validation_max_batches: int | None = 20,
    early_stopping_patience: int | None = None,
    best_checkpoint_path: str | Path | None = None,
    save_every_n_epochs: int = 1,
) -> list[dict[str, float]]:
    """Train a bridge model, validating and checkpointing on the reference cadence."""
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
    if ema_decay is not None:
        if not np.isfinite(ema_decay) or not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError("ema_decay must be in [0, 1) or None")
        ema_decay = float(ema_decay)
    if batch_sampler not in {None, "node2"}:
        raise ValueError("batch_sampler must be None or 'node2'")
    if max_batches_per_epoch is not None and int(max_batches_per_epoch) <= 0:
        raise ValueError("max_batches_per_epoch must be a positive integer or None")
    max_batches_per_epoch = None if max_batches_per_epoch is None else int(max_batches_per_epoch)
    if early_stopping_patience is not None and int(early_stopping_patience) <= 0:
        raise ValueError("early_stopping_patience must be a positive integer or None")
    if validation_max_batches is not None and int(validation_max_batches) <= 0:
        raise ValueError("validation_max_batches must be a positive integer or None")
    if int(save_every_n_epochs) <= 0:
        raise ValueError("save_every_n_epochs must be a positive integer")

    torch, _ = _torch()
    _check_dataset(dataset)
    ema = build_ema(model) if ema_decay is not None else None
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    early_stopped = False
    if best_checkpoint_path is not None and Path(best_checkpoint_path).is_file():
        saved_best = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        best_score = float(saved_best.get("val_scaled_err", float("inf")))
        best_epoch = int(saved_best.get("epoch", 0))
        best_state = saved_best.get("model_state_dict")

    sampler = None
    if batch_sampler == "node2":
        sizes = [event.reactant.n_atoms for event in dataset.catalog.events.values()]
        sampler = NodeSquareBatchSampler(sizes, max_num=max_num)
    clipper = AdaptiveGradClipper() if adaptive_clip else None
    if bridge_validation_dataset is not None:
        _check_dataset(bridge_validation_dataset)
        if len(bridge_validation_dataset) == 0:
            raise ValueError("validation dataset must contain at least one item")

    optimizer_type = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(
        model.parameters(), lr=float(lr), amsgrad=amsgrad,
        weight_decay=float(weight_decay),
    )
    scheduler = build_scheduler(
        optimizer,
        name=scheduler_name,
        total_steps=total_scheduler_steps(
            epochs, dataset, batch_size, max_steps, max_batches_per_epoch
        ),
        warmup_fraction=warmup_fraction,
        min_lr_fraction=min_lr_fraction,
    )
    history: list[dict[str, float]] = []
    completed_epochs = 0
    epoch_complete = False
    budget_truncated = False
    signature = _dataset_signature(dataset)
    resume_step = 0
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=False)
        if not saved.get("epoch_complete", False):
            raise ValueError(
                "resume requires a complete epoch checkpoint; partial sampler state is unavailable"
            )
        if saved.get("training_dataset_sha256") != signature:
            raise ValueError("resume dataset identity or training inputs differ")
        requested = {
            "lr": float(lr),
            "optimizer_name": optimizer_name,
            "amsgrad": amsgrad,
            "weight_decay": float(weight_decay),
            "batch_size": batch_size,
            "max_grad_norm": max_grad_norm,
            "scheduler_name": scheduler_name,
            "warmup_fraction": float(warmup_fraction),
            "min_lr_fraction": float(min_lr_fraction),
            "batch_sampler": batch_sampler,
            "max_num": int(max_num),
            "adaptive_clip": bool(adaptive_clip),
            "max_batches_per_epoch": max_batches_per_epoch,
            "loss_norm": loss_norm,
        }
        if ema_decay is not None:
            requested["ema_decay"] = ema_decay
        if any(saved["config"].get(key) != value for key, value in requested.items()):
            raise ValueError("resume optimizer, batch, or loss configuration differs")
        completed_epochs = int(saved["completed_epochs"])
        resume_step = int(saved["completed_steps"])
        if epochs <= completed_epochs or (max_steps is not None and max_steps <= resume_step):
            raise ValueError("resume budget must exceed completed training")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        if ema is not None and saved.get("ema_model_state_dict") is not None:
            ema.load_state_dict(saved["ema_model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available() and saved.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        history = saved["history"]

    def save_checkpoint(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "history": history,
                "epoch": epoch,
                "completed_epochs": completed_epochs,
                "completed_steps": step,
                "epoch_complete": epoch_complete,
                "budget_truncated": budget_truncated,
                "early_stopped": early_stopped,
                "training_dataset_sha256": signature,
                "ema_model_state_dict": None if ema is None else ema.state_dict(),
                "ema_decay": ema_decay,
                "config": {
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "optimizer_name": optimizer_name,
                    "amsgrad": amsgrad,
                    "weight_decay": float(weight_decay),
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                    "max_grad_norm": max_grad_norm,
                    "scheduler_name": scheduler_name,
                    "warmup_fraction": float(warmup_fraction),
                    "min_lr_fraction": float(min_lr_fraction),
                    "loss_norm": loss_norm,
                    "batch_sampler": batch_sampler,
                    "max_num": int(max_num),
                    "adaptive_clip": bool(adaptive_clip),
                    "max_batches_per_epoch": max_batches_per_epoch,
                    "bridge_validation": bridge_validation_dataset is not None,
                    "validation_max_num": int(validation_max_num),
                    "validation_max_batches": validation_max_batches,
                    "early_stopping_patience": early_stopping_patience,
                    "save_every_n_epochs": int(save_every_n_epochs),
                    "best_val_scaled_err": None if best_score == float("inf") else best_score,
                    "best_epoch": best_epoch,
                    "early_stopped": early_stopped,
                    "dataset_seed": dataset.seed,
                    "ema_decay": ema_decay,
                    "completed_steps": step,
                },
            },
            path,
        )

    if scheduler is not None and resume_step:
        # A resumed run continues its schedule; the budget is the new total.
        scheduler.last_epoch = resume_step

    output = None
    if log_path is not None:
        output_path = Path(log_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resumed = resume_step > 0
        output = output_path.open("a" if resumed else "w", encoding="utf-8")
        if not resumed:
            output.write("epoch step loss\n")
    if log_to_stdout:
        print("epoch step loss", flush=True)
    try:
        step = resume_step
        for epoch in range(completed_epochs + 1, epochs + 1):
            dataset.set_epoch(epoch)

            def log_step(step_index, values):
                line = f"{epoch} {step_index} {values['loss']:.8f}"
                if output is not None:
                    output.write(line + "\n")
                    output.flush()
                if log_to_stdout:
                    print(line, flush=True)

            epoch_budget = None if max_steps is None else max_steps - step
            if max_batches_per_epoch is not None:
                epoch_budget = (
                    max_batches_per_epoch
                    if epoch_budget is None
                    else min(epoch_budget, max_batches_per_epoch)
                )
            metrics = train_epoch(
                model,
                dataset,
                optimizer,
                device=device,
                batch_size=batch_size,
                loss_norm=loss_norm,
                start_step=step,
                log_step=log_step,
                max_steps=epoch_budget,
                max_grad_norm=max_grad_norm,
                scheduler=scheduler,
                ema=ema,
                ema_decay=0.0 if ema_decay is None else float(ema_decay),
                batch_sampler=sampler,
                grad_clipper=clipper,
            )
            step = int(metrics["last_step"])
            early_stop = False
            if bridge_validation_dataset is not None:
                bridge_validation_dataset.set_epoch(epoch)
                validation_model = ema if ema is not None else model
                val_scaled_err = evaluate_bridge_scaled_error(
                    validation_model,
                    bridge_validation_dataset,
                    device=device,
                    max_num=validation_max_num,
                    max_batches=validation_max_batches,
                    seed=42 + epoch,
                )
                metrics["val_scaled_err"] = val_scaled_err
                if np.isfinite(val_scaled_err) and val_scaled_err < best_score:
                    best_score = float(val_scaled_err)
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in validation_model.state_dict().items()
                    }
                    if best_checkpoint_path is not None:
                        torch.save(
                            {
                                "model_state_dict": best_state,
                                "val_scaled_err": best_score,
                                "epoch": best_epoch,
                                "weights": "ema" if ema is not None else "raw",
                            },
                            best_checkpoint_path,
                        )
                elif (
                    early_stopping_patience is not None
                    and best_epoch > 0
                    and epoch - best_epoch >= int(early_stopping_patience)
                ):
                    early_stop = True
            history.append(metrics)
            epoch_complete = (
                True
                if sampler is not None or max_batches_per_epoch is not None
                else metrics["num_items"] == len(dataset)
            )
            completed_epochs += int(epoch_complete)
            budget_truncated = (
                max_steps is not None and step >= max_steps and completed_epochs < epochs
            )
            if epoch_checkpoint_dir is not None and epoch % int(save_every_n_epochs) == 0:
                save_checkpoint(Path(epoch_checkpoint_dir) / f"epoch_{epoch:04d}.pt")
            if max_steps is not None and step >= max_steps:
                break
            if early_stop:
                early_stopped = True
                break
    finally:
        if output is not None:
            output.close()
    if best_state is not None and bridge_validation_dataset is not None:
        # Report the checkpoint with the lowest val_ep_scaled_err.
        model.load_state_dict(best_state)
        if ema is not None:
            ema.load_state_dict(best_state)
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path)
    return history


SCHEDULERS = ("constant", "cosine")


def total_scheduler_steps(
    epochs: int,
    dataset,
    batch_size: int | str,
    max_steps: int | None,
    max_batches_per_epoch: int | None = None,
) -> int:
    """Optimizer steps the schedule has to cover for the requested budget."""
    if max_batches_per_epoch is not None:
        per_epoch = int(max_batches_per_epoch)
    else:
        per_epoch = max(_batch_size(batch_size, dataset), 1)
    steps = (
        (-(-len(dataset) // per_epoch) if max_batches_per_epoch is None else per_epoch)
        * int(epochs)
    )
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    return max(int(steps), 1)


def build_scheduler(
    optimizer,
    *,
    name: str = "constant",
    total_steps: int = 1,
    warmup_fraction: float = 0.0,
    min_lr_fraction: float = 0.0,
):
    """Linear warmup followed by cosine decay, or ``None`` for a constant rate."""
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
