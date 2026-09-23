from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basinflow.evaluation.basin_recall import movable_mic_rmsd
from basinflow.data import EventCatalog, EventFlowDataset, Structure, event_target
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD, sample_ode


def _torch():
    try:
        import torch
        from torch_geometric.data import Batch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "product-flow quality evaluation requires optional model dependency: install torch"
        ) from exc
    return torch, Batch


def _generator_items(init_generators) -> list[tuple[str, Any]]:
    if isinstance(init_generators, Mapping):
        return [(str(name), generator) for name, generator in init_generators.items()]
    return [
        (str(getattr(generator, "seed_type", f"init_{index}")), generator)
        for index, generator in enumerate(init_generators)
    ]


def _positions_record(reference: Structure, positions) -> Structure:
    """A copy of ``reference``'s constraint and cell at new coordinates."""
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().numpy()
    return Structure(
        species=list(reference.species),
        positions=np.asarray(positions, dtype=float),
        cell=reference.cell.copy(),
        pbc=reference.pbc.copy(),
        movable_mask=reference.movable_mask.copy(),
    )


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "num_items": 0,
            "mean_rmsd": None,
            "median_rmsd": None,
            "best_rmsd": None,
            "max_rmsd": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "num_items": int(array.size),
        "mean_rmsd": float(np.mean(array)),
        "median_rmsd": float(np.median(array)),
        "best_rmsd": float(np.min(array)),
        "max_rmsd": float(np.max(array)),
    }


def evaluate_init_rmsd_baseline(
    catalog: EventCatalog, init_generators
) -> dict[str, Any]:
    """Measure how close initial proposal geometries are to known products.

    This isolates proposal-initialization quality from model rollout quality. A
    product-displacement init should give zero RMSD and acts as an oracle
    upper-bound check for the data and metric path.
    """
    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    report: dict[str, Any] = {}
    for name, generator in _generator_items(init_generators):
        rows = []
        rmsds: list[float] = []
        dataset = EventFlowDataset(
            catalog,
            [generator],
            flow_time=0.0,
            diagnostic_oracle=bool(getattr(generator, "requires_target", False)),
        )
        for sample in dataset:
            target = event_target(catalog.events[sample.event_id])
            generated = _positions_record(target.reactant, sample.source_pos)
            rmsd = movable_mic_rmsd(
                generated,
                target.product,
                target.reactant,
            )
            if abs(rmsd) < 1e-7:
                rmsd = 0.0
            rows.append(
                {
                    "event_id": sample.event_id,
                    "rmsd": rmsd,
                }
            )
            rmsds.append(rmsd)
        report[name] = {
            "overall": _summary(rmsds),
            "events": rows,
        }
    return report


def rollout_product_flow_positions(
    model,
    batch,
    *,
    num_steps: int = 50,
    cutoff: float = 3.0,
    integrator: str = DEFAULT_METHOD,
    solver_atol: float = DEFAULT_ATOL,
    solver_rtol: float = DEFAULT_RTOL,
):
    """Integrate a PyG product-flow batch into generated product coordinates.

    Uses the same ``torchdiffeq`` solver as the sampler and as MolGEN, so a
    reported rollout RMSD and a sampled candidate come from the same numerics.
    """
    torch, _ = _torch()
    frozen = ~batch.movable_mask

    def field(state, time):
        batch.pos = state
        batch.flow_time = time.reshape(1, 1).expand(batch.num_graphs, 1)
        velocity = model(batch, t=batch.flow_time.squeeze(-1))["velocity"]
        return torch.where(frozen[:, None], torch.zeros_like(velocity), velocity)

    with torch.no_grad():
        positions, _ = sample_ode(
            field, batch.pos.clone(), num_steps=num_steps,
            atol=solver_atol, rtol=solver_rtol, method=integrator,
        )
    return positions


def evaluate_product_flow_rollout(
    model,
    catalog: EventCatalog,
    init_generators,
    *,
    num_steps: int = 50,
    cutoff: float = 3.0,
    integrator: str = DEFAULT_METHOD,
    solver_atol: float = DEFAULT_ATOL,
    solver_rtol: float = DEFAULT_RTOL,
    device=None,
) -> dict[str, Any]:
    """Evaluate model rollout RMSD against known pairwise products."""
    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    torch, Batch = _torch()
    was_training = bool(getattr(model, "training", False))
    model.eval()
    report: dict[str, Any] = {}
    with torch.no_grad():
        for name, generator in _generator_items(init_generators):
            rows = []
            rmsds: list[float] = []
            dataset = EventFlowDataset(
                catalog,
                [generator],
                flow_time=0.0,
                diagnostic_oracle=bool(getattr(generator, "requires_target", False)),
            )
            for sample in dataset:
                batch = Batch.from_data_list([sample])
                if device is not None:
                    batch = batch.to(device)
                positions = rollout_product_flow_positions(
                    model,
                    batch,
                    num_steps=num_steps,
                    cutoff=cutoff,
                    integrator=integrator,
                    solver_atol=solver_atol,
                    solver_rtol=solver_rtol,
                )
                target = event_target(catalog.events[sample.event_id])
                generated = _positions_record(target.reactant, positions)
                rmsd = movable_mic_rmsd(
                    generated,
                    target.product,
                    target.reactant,
                )
                rows.append(
                    {
                        "event_id": sample.event_id,
                        "rmsd": rmsd,
                    }
                )
                rmsds.append(rmsd)
            report[name] = {
                "overall": _summary(rmsds),
                "events": rows,
            }
    if was_training:
        model.train()
    return report


def evaluate_product_flow_splits(
    model,
    split_datasets: Mapping[str, Any],
    init_generators,
    *,
    num_steps: int = 50,
    cutoff: float = 3.0,
    integrator: str = DEFAULT_METHOD,
    solver_atol: float = DEFAULT_ATOL,
    solver_rtol: float = DEFAULT_RTOL,
    device=None,
) -> dict[str, Any]:
    """Evaluate model rollout RMSD for named train/val/test splits."""
    return {
        "evaluation_protocol": "pairwise_rollout_no_relaxation",
        "splits": {
            split_name: evaluate_product_flow_rollout(
                model,
                split_catalog,
                init_generators,
                num_steps=num_steps,
                cutoff=cutoff,
                integrator=integrator,
                solver_atol=solver_atol,
                solver_rtol=solver_rtol,
                device=device,
            )
            for split_name, split_catalog in split_datasets.items()
        }
    }
