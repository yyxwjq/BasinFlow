"""The assembled EGNN backbone (legacy Stage 3 compatibility path).

The equivariant layer lives in :mod:`basinflow.models.egnn.layers`; this module
only assembles embeddings, layers and output heads, mirroring how
:mod:`basinflow.models.painn.painn` relates to its own ``layers`` module.
"""
from __future__ import annotations

from typing import Any

from basinflow.models.egnn.layers import EGNNLayer, _ModuleBase
from basinflow.models.loss import _field, _torch


class EGNNFlow(_ModuleBase):
    """Seed-conditioned EGNN product/event flow for Stage 3."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        cutoff: float = 5.0,
        max_atomic_number: int = 100,
        seed_type_vocab: int = 8,
    ) -> None:
        torch = _torch()
        nn = torch.nn
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.cutoff = float(cutoff)
        self.atomic_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.seed_type_embedding = nn.Embedding(seed_type_vocab, hidden_dim)
        # masks + time + vector norms/displacement norms
        self.scalar_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim) for _ in range(self.num_layers)]
        )
        self.velocity_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.active_head = nn.Linear(hidden_dim, 1)
        self.direction_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def _seed_type_ids(self, batch, n_atoms: int, device):
        torch = _torch()
        if not isinstance(batch, dict) and hasattr(batch, "seed_type_id"):
            graph_ids = batch.seed_type_id.reshape(-1).to(dtype=torch.long, device=device)
            return graph_ids[_field(batch, "batch")]
        if isinstance(batch, dict) and "seed_type_ids" in batch:
            return batch["seed_type_ids"].to(dtype=torch.long, device=device)
        return torch.zeros((n_atoms,), dtype=torch.long, device=device)

    def _time_feature(self, t, batch, n_atoms: int, dtype, device):
        torch = _torch()
        if isinstance(t, (float, int)):
            return torch.full((n_atoms, 1), float(t), dtype=dtype, device=device)
        tensor = torch.as_tensor(t, dtype=dtype, device=device).reshape(-1, 1)
        if tensor.shape[0] == 1:
            return tensor.expand(n_atoms, 1)
        if tensor.shape[0] != n_atoms:
            return tensor[_field(batch, "batch")]
        return tensor

    def _graph(self, batch):
        keys = set(batch.keys())
        if {"edge_index", "edge_vectors", "edge_lengths"} <= keys:
            return {
                "edge_index": _field(batch, "edge_index"),
                "edge_vectors": _field(batch, "edge_vectors"),
                "edge_lengths": _field(batch, "edge_lengths"),
            }
        from basinflow.geometry.graph import torch_neighbor_graph_from_batch
        return torch_neighbor_graph_from_batch(
            batch,
            cutoff=self.cutoff,
            positions_key="pos" if not isinstance(batch, dict) else "initial_positions",
        )

    def forward(self, batch, t=None) -> dict[str, Any]:
        torch = _torch()
        initial = _field(batch, "pos" if not isinstance(batch, dict) else "initial_positions")
        reactant = _field(batch, "reactant_pos" if not isinstance(batch, dict) else "reactant_positions")
        seed = _field(batch, "seed_displacement")
        seed_direction = _field(batch, "seed_direction")
        movable = _field(batch, "movable_mask")
        active_prior = _field(batch, "active_prior")
        n_atoms = int(initial.shape[0])

        atomic_numbers = _field(batch, "z" if not isinstance(batch, dict) else "atomic_numbers").clamp(min=0, max=self.atomic_embedding.num_embeddings - 1)
        atom_embed = self.atomic_embedding(atomic_numbers)
        seed_embed = self.seed_type_embedding(self._seed_type_ids(batch, n_atoms, initial.device))
        current_disp = initial - reactant
        if t is None:
            t = _field(batch, "flow_time") if (not isinstance(batch, dict) or "flow_time" in batch) else 0.0
        time = self._time_feature(t, batch, n_atoms, initial.dtype, initial.device)
        scalar_features = torch.cat(
            [
                atom_embed,
                seed_embed,
                movable.to(dtype=initial.dtype).reshape(-1, 1),
                active_prior.to(dtype=initial.dtype).reshape(-1, 1),
                time,
                torch.linalg.norm(seed, dim=1, keepdim=True),
                torch.linalg.norm(seed_direction, dim=1, keepdim=True),
                torch.linalg.norm(current_disp, dim=1, keepdim=True),
            ],
            dim=1,
        )
        h = self.scalar_proj(scalar_features)
        vector_state = seed + current_disp
        graph = self._graph(batch)
        update_mask = movable.to(dtype=initial.dtype)
        for layer in self.layers:
            h, vector_state = layer(
                h,
                vector_state,
                graph["edge_index"],
                graph["edge_vectors"],
                graph["edge_lengths"],
                update_mask=update_mask,
            )

        bases = torch.stack([vector_state, seed, seed_direction], dim=1)
        velocity_weights = self.velocity_gate(h).unsqueeze(-1)
        velocity = (bases * velocity_weights).sum(dim=1)
        velocity = velocity * update_mask.reshape(-1, 1)

        direction_weights = self.direction_gate(h).unsqueeze(-1)
        direction = (bases * direction_weights).sum(dim=1)
        direction = direction * update_mask.reshape(-1, 1)
        return {
            "velocity": velocity,
            "active_logits": self.active_head(h).squeeze(-1),
            "direction": direction,
        }
