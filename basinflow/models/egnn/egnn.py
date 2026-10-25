"""The assembled EGNN backbone for the shared R + P -> TS bridge contract."""
from __future__ import annotations

from basinflow.geometry.graph import torch_neighbor_graph_from_batch
from basinflow.models.egnn.layers import EGNNLayer, _ModuleBase
from basinflow.models.loss import _torch
from basinflow.models.sb import remove_mean_batch


class EGNNFlow(_ModuleBase):
    """Equivariant message passing over the reactant, current state and product."""

    builds_own_graph = True
    bridge = True
    endpoint_condition = True
    endpoint_equivariant = True

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        cutoff: float = 5.0,
        max_atomic_number: int = 100,
    ) -> None:
        torch = _torch()
        nn = torch.nn
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.cutoff = float(cutoff)
        self.atomic_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        # atom features + movable flag + time + two displacement norms
        self.scalar_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim) for _ in range(self.num_layers)]
        )
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def _time_nodes(self, t, batch, n_atoms: int, dtype, device):
        torch = _torch()
        if t is None:
            t = batch.flow_time
        tensor = torch.as_tensor(t, dtype=dtype, device=device).reshape(-1)
        graph_ids = batch.batch
        if tensor.numel() == 1:
            return tensor.expand(n_atoms, 1)
        if tensor.numel() == int(graph_ids.max()) + 1:
            return tensor[graph_ids].unsqueeze(-1)
        if tensor.numel() == n_atoms:
            return tensor.unsqueeze(-1)
        raise ValueError("flow time must be scalar, per graph or per node")

    def forward(self, batch, t=None) -> dict:
        torch = _torch()
        initial = batch.pos
        reactant = batch.reactant_pos
        endpoint = batch.endpoint_pos
        movable = batch.movable_mask
        n_atoms = int(initial.shape[0])

        atomic_numbers = batch.z.clamp(
            min=0, max=self.atomic_embedding.num_embeddings - 1
        )
        atom_embed = self.atomic_embedding(atomic_numbers)
        time = self._time_nodes(t, batch, n_atoms, initial.dtype, initial.device)
        reactant_disp = initial - reactant
        endpoint_disp = endpoint - reactant
        h = self.scalar_proj(
            torch.cat(
                [
                    atom_embed,
                    movable.to(dtype=initial.dtype).reshape(-1, 1),
                    time,
                    torch.linalg.norm(reactant_disp, dim=1, keepdim=True),
                    torch.linalg.norm(endpoint_disp, dim=1, keepdim=True),
                ],
                dim=1,
            )
        )

        graph = torch_neighbor_graph_from_batch(batch, self.cutoff, positions_key="pos")
        vector_state = reactant_disp + endpoint_disp
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

        bases = torch.stack([vector_state, reactant_disp, endpoint_disp], dim=1)
        weights = self.vector_gate(h).unsqueeze(-1)
        net_out = (bases * weights).sum(dim=1)
        return {"scaled": remove_mean_batch(net_out, batch.batch)}
