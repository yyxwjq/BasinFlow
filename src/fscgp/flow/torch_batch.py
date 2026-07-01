from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "flow_item_to_torch_batch requires optional model dependency: install torch"
        ) from exc
    return torch


def flow_item_to_torch_batch(
    flow_item: dict[str, Any],
    *,
    dtype=None,
    device=None,
) -> dict[str, Any]:
    """Convert one product-flow item to a torch tensor batch.

    The returned mapping keeps the Stage 2.5 field names so minimal training
    code can consume the same contract as the NumPy flow item.
    """
    torch = _torch()
    float_dtype = dtype or torch.float32

    def tensor(name: str, *, tensor_dtype):
        return torch.as_tensor(flow_item[name], dtype=tensor_dtype, device=device)

    n_atoms = int(flow_item["reactant_positions"].shape[0])
    return {
        "event_id": flow_item["event_id"],
        "basin_id": flow_item["basin_id"],
        "reactant_positions": tensor("reactant_positions", tensor_dtype=float_dtype),
        "initial_positions": tensor("initial_positions", tensor_dtype=float_dtype),
        "product_positions": tensor("product_positions", tensor_dtype=float_dtype),
        "target_positions": tensor("target_positions", tensor_dtype=float_dtype),
        "seed_displacement": tensor("seed_displacement", tensor_dtype=float_dtype),
        "target_displacement": tensor("target_displacement", tensor_dtype=float_dtype),
        "target_velocity": tensor("target_velocity", tensor_dtype=float_dtype),
        "active_mask": tensor("active_mask", tensor_dtype=torch.bool),
        "movable_mask": tensor("movable_mask", tensor_dtype=torch.bool),
        "event_direction": tensor("event_direction", tensor_dtype=float_dtype),
        "cell": tensor("cell", tensor_dtype=float_dtype).unsqueeze(0),
        "pbc": tensor("pbc", tensor_dtype=torch.bool).unsqueeze(0),
        "batch": torch.zeros((n_atoms,), dtype=torch.long, device=device),
        "ptr": torch.tensor([0, n_atoms], dtype=torch.long, device=device),
        "metadata": dict(flow_item.get("metadata", {})),
    }
