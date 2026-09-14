from __future__ import annotations

import torch
from torch import nn

from basinflow.geometry.radius_graph import dual_radius_graph
from .painn import DualPaiNN


def _field(data, name, default=None):
    return data.get(name, default) if isinstance(data, dict) else getattr(data, name, default)


def prepare_flow_input(data, cutoff, time=None):
    """Adapt EventData to the tensor API, rebuilding both geometric neighborhoods.

    Positions use a shared unwrapped lift relative to the reactant. Applying a
    common lattice shift to either atom's two states preserves the prediction.
    """
    current = _field(data, "pos")
    reference = _field(data, "reactant_pos")
    if reference.shape != current.shape:
        raise ValueError("reactant_pos and pos must have matching [N,3] shapes")
    graph_ids = _field(data, "batch")
    if graph_ids is None:
        graph_ids = torch.zeros(len(current), dtype=torch.long, device=current.device)
    cells = _field(data, "cell").reshape(-1, 3, 3).to(current)
    periodic = _field(data, "pbc").reshape(-1, 3).bool()
    graph = dual_radius_graph(reference, current, graph_ids, cells, periodic, cutoff)
    receiver = graph["edge_index"][0]
    shifts = torch.einsum("ei,eij->ej", graph["cell_offsets"].to(current), cells[graph_ids[receiver]])
    if time is None:
        time = _field(data, "flow_time", 0.0)
    time = torch.as_tensor(time, dtype=current.dtype, device=current.device).reshape(-1)
    if time.numel() == 1:
        time = time.expand(len(current))
    elif time.numel() == len(cells):
        time = time[graph_ids]
    else:
        raise ValueError("flow time must be scalar or one value per graph")
    movable = _field(data, "movable_mask")
    active_prior = _field(data, "active_prior", torch.zeros(len(current), device=current.device))
    if movable.shape != (len(current),) or movable.dtype != torch.bool or active_prior.shape != (len(current),):
        raise ValueError("movable_mask must be bool [N] and active_prior [N]")
    prepared = {
        "positions_1": reference,
        "positions_2": current,
        "elements": _field(data, "z"),
        "edge_index": graph["edge_index"],
        "shifts": shifts,
        "time": time,
        "node_conditions": torch.stack([movable.to(current), active_prior.to(current)], dim=-1),
        "batch": graph_ids,
    }
    # The bond-change condition is a discrete graph-level label; it must survive
    # this adaptation or the backbone cannot see it.
    bond_change = _field(data, "bond_change", None)
    if bond_change is not None:
        prepared["bond_change"] = bond_change
    endpoint = _field(data, "endpoint_pos", None)
    if endpoint is not None:
        if endpoint.shape != current.shape:
            raise ValueError("endpoint_pos and pos must have matching [N,3] shapes")
        prepared["endpoint_pos"] = endpoint
    return prepared


class PaiNN(nn.Module):
    """BasinFlow flow adapter around the independent dual-geometry backbone."""

    builds_own_graph = True

    def __init__(self, num_features=128, num_radial_basis=32, num_layers=4, num_elements=119, r_max=5.0, time_dim=32, radial_basis="bessel", envelope="polynomial", stability_mode="scaled", bond_change_condition=False, bond_change_direction=False, endpoint_condition=False, endpoint_equivariant=False):
        super().__init__()
        self.r_max = float(r_max)
        self.cutoff = self.r_max
        # Recorded so that basin inference can build the matching dataset view
        # without the caller having to repeat the flag.
        self.bond_change_condition = bool(bond_change_condition)
        self.bond_change_direction = bool(bond_change_direction)
        self.endpoint_condition = bool(endpoint_condition)
        self.endpoint_equivariant = bool(endpoint_equivariant)
        self.backbone = DualPaiNN(num_features, num_radial_basis, num_layers, num_elements, r_max, time_dim, radial_basis, envelope, stability_mode, bond_change_condition, bond_change_direction,
            endpoint_condition, endpoint_equivariant)

    def forward(self, batch, t=None):
        output = self.backbone(prepare_flow_input(batch, self.r_max, t))
        output["velocity"] = output["velocity"] * _field(batch, "movable_mask")[:, None]
        return output
