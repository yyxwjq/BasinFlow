from __future__ import annotations

import torch
from torch import nn

from basinflow.geometry.bonds import BOND_CHANGE_CLASSES

from .layers import DualMessageBlock, GaussianFourierBasis, GatedEquivariantBlock, RadialBasis, UpdateBlock


class DualPaiNN(nn.Module):
    """Tensor backbone with LiFlow-style positions_1/positions_2 graph inputs.

    positions_1 is the unperturbed reactant, positions_2 the current flow state.
    Edge shifts are Cartesian, and edges point from receiver to sender.
    Time is atom-level [N]. No source-noise or target fields are consumed.
    """

    def __init__(self, num_features=128, num_radial_basis=32, num_layers=4, num_elements=119, r_max=5.0, time_dim=32, radial_basis="bessel", envelope="polynomial", stability_mode="scaled", bond_change_condition=False, bond_change_direction=False, endpoint_condition=False, endpoint_equivariant=False):
        super().__init__()
        if stability_mode not in {"scaled", "bounded", "none"}:
            raise ValueError("stability_mode must be scaled, bounded or none")
        if num_features < 2 or num_layers < 1 or num_elements < 2:
            raise ValueError("features/elements must be at least two and layers positive")
        self.atom_embedding = nn.Embedding(num_elements, num_features)
        self.time_embedding = GaussianFourierBasis(time_dim)
        self.scalar_input = nn.Sequential(nn.Linear(num_features + time_dim + 2, num_features), nn.SiLU(), nn.Linear(num_features, num_features))
        self.vector_input = nn.Linear(1, num_features, bias=False)
        if bond_change_direction and not bond_change_condition:
            raise ValueError("bond_change_direction requires bond_change_condition")
        self.bond_change_direction = bool(bond_change_direction)
        # Present only for the directional arm, so every checkpoint written
        # before it keeps its exact state dict.
        self.condition_vector_input = (
            nn.Linear(1, num_features, bias=False)
            if (bond_change_condition and bond_change_direction) else None
        )
        self.radial = RadialBasis(num_radial_basis, r_max, radial_basis, envelope)
        self.bond_change_condition = bool(bond_change_condition)
        if endpoint_equivariant and not endpoint_condition:
            raise ValueError("endpoint_equivariant requires endpoint_condition")
        self.endpoint_condition = bool(endpoint_condition)
        self.endpoint_equivariant = bool(endpoint_equivariant)
        self.messages = nn.ModuleList([DualMessageBlock(num_features, num_radial_basis, stability_mode,
                                                       self.bond_change_condition,
                                                       self.endpoint_condition,
                                                       self.endpoint_equivariant) for _ in range(num_layers)])
        self.updates = nn.ModuleList([UpdateBlock(num_features, stability_mode) for _ in range(num_layers)])
        self.output = GatedEquivariantBlock(num_features, stability_mode)

    def forward(self, data):
        reference, current = data["positions_1"], data["positions_2"]
        receiver, sender = data["edge_index"]
        elements = data["elements"]
        if elements.dtype != torch.long or elements.shape != (len(current),):
            raise ValueError("elements must be int64 [N] atomic numbers")
        if elements.numel() and (elements.min() < 1 or elements.max() >= self.atom_embedding.num_embeddings):
            raise ValueError("atomic number outside embedding vocabulary")
        reference_vector = reference[sender] - reference[receiver] + data["shifts"]
        current_vector = current[sender] - current[receiver] + data["shifts"]
        reference_length = torch.linalg.vector_norm(reference_vector, dim=-1)
        current_length = torch.linalg.vector_norm(current_vector, dim=-1)
        reference_unit = reference_vector / reference_length.clamp_min(1e-8)[:, None]
        current_unit = current_vector / current_length.clamp_min(1e-8)[:, None]
        reference_radial, reference_cutoff = self.radial(reference_length)
        current_radial, current_cutoff = self.radial(current_length)
        endpoint = None
        if self.endpoint_condition:
            # The conditioning endpoint (the product of an R+P -> TS task) is
            # lifted onto the same edge set, so its directions can enter the
            # vector message rather than only its distances.
            if "endpoint_pos" not in data:
                raise ValueError("endpoint_condition is enabled but no endpoint was passed")
            endpoint_positions = data["endpoint_pos"]
            endpoint_vector = endpoint_positions[sender] - endpoint_positions[receiver] + data["shifts"]
            endpoint_length = torch.linalg.vector_norm(endpoint_vector, dim=-1)
            endpoint_unit = endpoint_vector / endpoint_length.clamp_min(1e-8)[:, None]
            endpoint_radial, endpoint_cutoff = self.radial(endpoint_length)
            endpoint = (endpoint_radial, endpoint_cutoff, endpoint_unit)
        conditions = data.get("node_conditions", current.new_zeros((len(current), 2)))
        scalar = self.scalar_input(torch.cat([self.atom_embedding(elements), self.time_embedding(data["time"]), conditions], dim=-1))
        vector = self.vector_input((current - reference)[:, :, None])
        bond_change = None
        if self.bond_change_condition:
            bond_change = _bond_change_one_hot(data, receiver, sender, current.dtype)
            # The discrete condition says which bonds change; this says in which
            # direction the two atoms of such a bond have to move. At t = 0 the
            # current and reference geometries coincide, so without this the
            # initial vector features are exactly zero and the network has to
            # synthesise every direction from invariant inputs.
            if self.condition_vector_input is not None:
                vector = vector + self.condition_vector_input(
                    _bond_change_direction(reference, receiver, sender, bond_change)[:, :, None]
                )
        for message, update in zip(self.messages, self.updates):
            scalar, vector = message(scalar, vector, reference_radial, current_radial, reference_cutoff, current_cutoff, reference_unit, current_unit, data["edge_index"], bond_change, endpoint)
            scalar, vector = update(scalar, vector)
        return {"velocity": self.output(scalar, vector), "hidden": scalar}


def _bond_change_direction(reference, receiver, sender, bond_change):
    """Per-atom pull towards a forming bond and away from a breaking one.

    Returns an equivariant ``[N, 3]`` field built from the reference geometry
    and the discrete labels only, so no product coordinate enters.
    """
    unit = reference[sender] - reference[receiver]
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    signed = (bond_change[:, 1] - bond_change[:, 2])[:, None] * unit
    direction = reference.new_zeros((reference.shape[0], 3))
    direction = direction.index_add(0, receiver, signed)
    return direction.index_add(0, sender, -signed)


def _bond_change_one_hot(data, receiver, sender, dtype):
    """Per-edge ``[unchanged, formed, broken]`` one-hot from the condition tensor.

    ``data["bond_change"]`` is a padded ``[B, n_max, n_max]`` label matrix and
    carries no coordinate, so no product geometry reaches the model through this
    path.  It is an input condition: the supervised view takes it from the
    reactant/product difference, the basin view takes it from the seed.
    """
    labels = data["bond_change"]
    if labels.dim() != 3:
        raise ValueError(f"bond_change must be [B, n_max, n_max], got {tuple(labels.shape)}")
    graph_of_node = data["batch"]
    counts = torch.bincount(graph_of_node, minlength=labels.shape[0])
    offsets = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(len(graph_of_node), device=graph_of_node.device) - offsets[graph_of_node]
    graph_r, graph_s = graph_of_node[receiver], graph_of_node[sender]
    local_r, local_s = local[receiver], local[sender]
    pair_labels = labels[graph_r, local_r, local_s].long().clamp_(0, 2)
    return torch.nn.functional.one_hot(pair_labels, num_classes=BOND_CHANGE_CLASSES).to(dtype)
