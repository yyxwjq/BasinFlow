"""The PaiNN backbone: one graph builder, one bridge output contract."""
from __future__ import annotations

import torch
from torch import nn

from basinflow.geometry.graph import torch_neighbor_graph_from_batch
from basinflow.models.sb import remove_mean_batch
from .layers import DualMessageBlock, GaussianFourierBasis, GatedEquivariantBlock, RadialBasis, UpdateBlock


def prepare_flow_input(data, cutoff, time=None):
    """Adapt one :class:`EventData` batch to the PaiNN tensor API.

    The graph is the union of the reference (reactant) and current radii and is
    built once by :func:`torch_neighbor_graph_from_batch`, which also supplies the
    minimum-image edge vectors for both geometries.
    """
    current = data.pos
    reference = data.reactant_pos
    if reference.shape != current.shape:
        raise ValueError("reactant_pos and pos must have matching [N, 3] shapes")
    graph_ids = data.batch
    graph = torch_neighbor_graph_from_batch(
        data, cutoff=cutoff, reference_positions=reference
    )
    cells = data.cell.reshape(-1, 3, 3).to(current)
    shifts = torch.einsum(
        "ei,eij->ej",
        graph["cell_offsets"].to(current),
        cells[graph_ids[graph["edge_index"][0]]],
    )
    if time is None:
        time = data.flow_time
    time = torch.as_tensor(time, dtype=current.dtype, device=current.device).reshape(-1)
    num_graphs = int(graph_ids.max()) + 1 if graph_ids.numel() else 0
    if time.numel() == 1:
        time = time.expand(len(current))
    elif time.numel() == num_graphs:
        time = time[graph_ids]
    elif time.numel() != len(current):
        raise ValueError("flow time must be scalar, per graph or per node")
    return {
        "positions_1": reference,
        "positions_2": current,
        "elements": data.z,
        "edge_index": graph["edge_index"],
        "reference_vectors": graph["edge_vectors_2"],
        "current_vectors": graph["edge_vectors"],
        "endpoint_pos": data.endpoint_pos,
        "shifts": shifts,
        "time": time,
        "movable": data.movable_mask,
        "batch": graph_ids,
    }


class PaiNN(nn.Module):
    """Bridge adapter around the dual-geometry PaiNN backbone."""

    builds_own_graph = True
    bridge = True
    endpoint_condition = True
    endpoint_equivariant = True

    def __init__(self, num_features=128, num_radial_basis=32, num_layers=4,
                 num_elements=119, r_max=5.0, time_dim=32, radial_basis="bessel",
                 envelope="polynomial", stability_mode="scaled", time_scale=1.0,
                 time_per_layer=False):
        super().__init__()
        self.r_max = float(r_max)
        self.cutoff = self.r_max
        self.backbone = DualPaiNN(
            num_features, num_radial_basis, num_layers, num_elements, r_max,
            time_dim, radial_basis, envelope, stability_mode, time_scale,
            time_per_layer,
        )

    def forward(self, batch, t=None):
        prepared = prepare_flow_input(batch, self.r_max, t)
        net_out = self.backbone(prepared)["net_out"]
        return {"scaled": remove_mean_batch(net_out, batch.batch)}


class DualPaiNN(nn.Module):
    """PaiNN over three edge geometries: reactant, current state and product."""

    def __init__(self, num_features=128, num_radial_basis=32, num_layers=4,
                 num_elements=119, r_max=5.0, time_dim=32, radial_basis="bessel",
                 envelope="polynomial", stability_mode="scaled", time_scale=1.0,
                 time_per_layer=False):
        super().__init__()
        if stability_mode not in {"scaled", "bounded", "none"}:
            raise ValueError("stability_mode must be scaled, bounded or none")
        if num_features < 2 or num_layers < 1 or num_elements < 2:
            raise ValueError("features/elements must be at least two and layers positive")
        self.atom_embedding = nn.Embedding(num_elements, num_features)
        self.time_embedding = GaussianFourierBasis(time_dim, scale=float(time_scale))
        self.scalar_input = nn.Sequential(
            nn.Linear(num_features + time_dim + 1, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )
        self.vector_input = nn.Linear(1, num_features, bias=False)
        self.radial = RadialBasis(num_radial_basis, r_max, radial_basis, envelope)
        gate_features = time_dim if time_per_layer else 0
        self.messages = nn.ModuleList(
            [
                DualMessageBlock(num_features, num_radial_basis, stability_mode, gate_features)
                for _ in range(num_layers)
            ]
        )
        self.updates = nn.ModuleList(
            [UpdateBlock(num_features, stability_mode) for _ in range(num_layers)]
        )
        self.output = GatedEquivariantBlock(num_features, stability_mode)

    def forward(self, data):
        reference, current = data["positions_1"], data["positions_2"]
        receiver, sender = data["edge_index"]
        elements = data["elements"]
        if elements.dtype != torch.long or elements.shape != (len(current),):
            raise ValueError("elements must be int64 [N] atomic numbers")
        if elements.numel() and (
            elements.min() < 1 or elements.max() >= self.atom_embedding.num_embeddings
        ):
            raise ValueError("atomic number outside embedding vocabulary")

        reference_vector = data["reference_vectors"]
        current_vector = data["current_vectors"]
        reference_length = torch.linalg.vector_norm(reference_vector, dim=-1)
        current_length = torch.linalg.vector_norm(current_vector, dim=-1)
        reference_unit = reference_vector / reference_length.clamp_min(1e-8)[:, None]
        current_unit = current_vector / current_length.clamp_min(1e-8)[:, None]
        reference_radial, reference_cutoff = self.radial(reference_length)
        current_radial, current_cutoff = self.radial(current_length)

        endpoint_vector = (
            data["endpoint_pos"][sender] - data["endpoint_pos"][receiver] + data["shifts"]
        )
        endpoint_length = torch.linalg.vector_norm(endpoint_vector, dim=-1)
        endpoint_unit = endpoint_vector / endpoint_length.clamp_min(1e-8)[:, None]
        endpoint_radial, endpoint_cutoff = self.radial(endpoint_length)

        time_embedding = self.time_embedding(data["time"])
        conditions = data["movable"].to(current.dtype).unsqueeze(-1)
        scalar = self.scalar_input(
            torch.cat([self.atom_embedding(elements), time_embedding, conditions], dim=-1)
        )
        vector = self.vector_input((current - reference)[:, :, None])
        for message, update in zip(self.messages, self.updates):
            scalar, vector = message(
                scalar, vector, reference_radial, current_radial,
                reference_cutoff, current_cutoff, reference_unit, current_unit,
                endpoint_radial, endpoint_cutoff, endpoint_unit,
                data["edge_index"], time_embedding,
            )
            scalar, vector = update(scalar, vector)
        return {"net_out": self.output(scalar, vector)}
