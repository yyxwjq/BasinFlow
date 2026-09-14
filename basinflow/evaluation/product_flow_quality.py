from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basinflow.evaluation.basin_recall import movable_mic_rmsd
from basinflow.data.catalog import EventCatalog
from basinflow.data.pyg import EventFlowDataset
from basinflow.data.records import StructureRecord


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


def _positions_record(
    structure_id: str,
    reference: StructureRecord,
    positions,
) -> StructureRecord:
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().numpy()
    return StructureRecord(
        structure_id=structure_id,
        species=list(reference.species),
        positions=np.asarray(positions, dtype=float),
        cell=reference.cell.copy(),
        pbc=reference.pbc.copy(),
        movable_mask=reference.movable_mask.copy(),
        tags=reference.tags.copy() if reference.tags is not None else None,
        metadata={"reference_structure_id": reference.structure_id},
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
    catalog: EventCatalog,
    init_generators,
    *,
    active_threshold: float = 0.1,
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
            active_threshold=active_threshold,
            flow_time=0.0,
            diagnostic_oracle=bool(getattr(generator, "requires_target", False)),
        )
        for sample in dataset:
            target = catalog.event_target(
                sample.event_id,
                active_threshold=active_threshold,
            )
            generated = _positions_record(
                f"{sample.event_id}:{name}:init_initial",
                target.reactant,
                sample.source_pos,
            )
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
                    "basin_id": sample.basin_id,
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
    num_steps: int = 8,
    cutoff: float = 3.0,
    graph_update_interval: int = 1,
):
    """Integrate a PyG product-flow batch into generated product coordinates."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if graph_update_interval <= 0:
        raise ValueError("graph_update_interval must be positive")
    torch, _ = _torch()
    positions = batch.pos.clone()
    with torch.no_grad():
        for step in range(num_steps):
            batch.pos = positions
            output = model(batch, t=(step + 0.5) / float(num_steps))
            positions = positions + output["velocity"] / float(num_steps)
            fixed = ~batch.movable_mask
            positions[fixed] = batch.reactant_pos[fixed]
    return positions


def evaluate_product_flow_rollout(
    model,
    catalog: EventCatalog,
    init_generators,
    *,
    num_steps: int = 8,
    cutoff: float = 3.0,
    graph_update_interval: int = 1,
    device=None,
    active_threshold: float = 0.1,
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
                active_threshold=active_threshold,
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
                    graph_update_interval=graph_update_interval,
                )
                target = catalog.event_target(
                    sample.event_id,
                    active_threshold=active_threshold,
                )
                generated = _positions_record(
                    f"{sample.event_id}:{name}:rollout",
                    target.reactant,
                    positions,
                )
                rmsd = movable_mic_rmsd(
                    generated,
                    target.product,
                    target.reactant,
                )
                rows.append(
                    {
                        "event_id": sample.event_id,
                        "basin_id": sample.basin_id,
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
    num_steps: int = 8,
    cutoff: float = 3.0,
    graph_update_interval: int = 1,
    device=None,
    active_threshold: float = 0.1,
) -> dict[str, Any]:
    """Evaluate model rollout RMSD for named train/val/test splits."""
    return {
        "splits": {
            split_name: evaluate_product_flow_rollout(
                model,
                split_catalog,
                init_generators,
                num_steps=num_steps,
                cutoff=cutoff,
                graph_update_interval=graph_update_interval,
                device=device,
                active_threshold=active_threshold,
            )
            for split_name, split_catalog in split_datasets.items()
        }
    }
