from __future__ import annotations

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - optional dependency path
    _TORCH = None

from basinflow.models.loss import _torch


_ModuleBase = _TORCH.nn.Module if _TORCH is not None else object


class EGNNLayer(_ModuleBase):
    """Small EGNN layer using explicit edge vectors.

    It consumes precomputed edge vectors, so periodic image handling stays in the
    shared graph builder rather than in the model.
    """

    def __init__(self, hidden_dim: int) -> None:
        torch = _torch()
        nn = torch.nn
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vector_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.xavier_uniform_(self.vector_mlp[-1].weight, gain=0.001)

    def forward(self, h, vector_state, edge_index, edge_vectors, edge_lengths,
                update_mask=None):
        torch = _torch()
        if edge_index.numel() == 0:
            return h, vector_state

        row, col = edge_index
        radial = edge_lengths.reshape(-1, 1)
        edge_feat = self.edge_mlp(torch.cat([h[row], h[col], radial], dim=1))

        aggregated = h.new_zeros((h.shape[0], edge_feat.shape[1]))
        aggregated.scatter_add_(0, row[:, None].expand_as(edge_feat), edge_feat)
        h = h + self.node_mlp(torch.cat([h, aggregated], dim=1))

        weights = self.vector_mlp(edge_feat)
        translation = edge_vectors * weights
        vector_update = vector_state.new_zeros(vector_state.shape)
        vector_update.scatter_add_(0, row[:, None].expand_as(translation), translation)
        if update_mask is not None:
            vector_update = vector_update * update_mask.reshape(-1, 1)
        return h, vector_state + vector_update
