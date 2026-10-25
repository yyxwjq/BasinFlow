"""The object-aware LEFTNet backbone, following the ReactOT reference.

The tensor-in/tensor-out contract is identical to ``reactot/model/leftnet.py``
(``LEFTNet``, ``nn_vector``); the ``EventData`` adaptation lives in
:mod:`basinflow.models.leftnet.adapter`.

``torch_scatter`` is replaced by :func:`basinflow.models.leftnet.layers.scatter_sum`
inside ``EquiMessage.aggregate``; ``assert_rot_equiv`` is not ported. The switch
semantics (``reflect_equiv``, ``object_aware``, ``legacy``, ``update``,
``single_layer_output``) match the reference.

This backbone is molecular: the adapter builds a direct distance graph over the
three fragments, so periodic frames are rejected rather than wrapped. PaiNN and
EGNN share ``geometry.radius_graph`` and do support PBC.

Numerical deviation: ``EPS_SQ = 1e-12`` is added *inside* each normalisation square
root, where the reference adds ``EPS`` outside it. The forward value differs by at
most ``1e-6`` relative, and the ``inf``/``nan`` derivative of ``sqrt`` at an exactly
zero cross product disappears.
"""
from __future__ import annotations

import math
from math import pi

import torch
from torch import Tensor, nn

from basinflow.models.leftnet.layers import (
    CFConvS2V,
    EPS,
    EquiMessage,
    EquiOutput,
    EquiUpdate,
    GCLMessage,
    MLP,
    NeighborEmb,
    RBFEmb,
    remove_mean_batch,
    vector,
)

#: In-sqrt epsilon guarding the three cross-product normalisations (see the
#: module docstring): keeps the reference's forward value to ~1e-6 while
#: removing the ``inf``/``nan`` derivative of ``sqrt`` at exactly zero.
EPS_SQ = 1e-12


def nn_vector(dist: Tensor, edge_index: Tensor, pos: Tensor) -> Tensor:
    """Nearest-neighbour position, used to build the node frame.

    For every atom, the position of its nearest neighbour by ``dist``, following
    ``reactot/model/leftnet.py``. Only reached when ``legacy=False``.
    """
    ii, jj = edge_index
    vec = []
    pairs: dict[int, int] = {}
    for n in range(pos.size(0)):
        if n not in pairs:
            inds = torch.where(ii == n)[0]
            if not len(inds):
                nn_j = n
            else:
                min_ind = torch.argmin(dist[inds])
                nn_j = int(jj[inds][min_ind].item())
            pairs.update({nn_j: n})
        else:
            nn_j = pairs[n]
        vec.append(pos[nn_j])
    return torch.stack(vec)


class LEFTNet(nn.Module):
    """Object-aware LEFTNet scoring network.

    Args:
        cutoff: Interatomic cutoff used for the radial basis and the edge mask.
        num_layers: Number of message-passing blocks.
        hidden_channels: Scalar/vector feature width.
        num_radial: Number of radial basis functions.
        in_hidden_channels: Width of the input node features ``h``.
        reflect_equiv: When ``True`` the frame is reflection invariant (the
            reference's ``reflect_equiv``).
        legacy: Reference's ``legacy`` node-frame construction (nearest-neighbour
            frame); ``True`` in the ReactOT default config.
        update: Run the per-layer :class:`EquiUpdate`.
        single_layer_output: One gated equivariant output block instead of two.
        object_aware: Mask inter-fragment edges with ``subgraph_mask``.
    """

    def __init__(
        self,
        cutoff: float = 10.0,
        num_layers: int = 4,
        hidden_channels: int = 128,
        num_radial: int = 96,
        in_hidden_channels: int = 8,
        reflect_equiv: bool = True,
        legacy: bool = True,
        update: bool = True,
        single_layer_output: bool = True,
        object_aware: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.reflect_equiv = reflect_equiv
        self.legacy = legacy
        self.update = update
        self.object_aware = object_aware
        # ``r_max`` mirrors the PaiNN convention for tools that read one attribute.
        self.r_max = cutoff

        self.embedding = nn.Linear(in_hidden_channels, hidden_channels)
        self.embedding_out = nn.Linear(hidden_channels, in_hidden_channels)
        self.radial_emb = RBFEmb(num_radial, self.cutoff)
        self.neighbor_emb = NeighborEmb(hidden_channels, in_hidden_channels)
        self.s2v = CFConvS2V(hidden_channels)

        self.radial_lin = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.lin3 = nn.Sequential(
            nn.Linear(3, hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels // 4, 1),
        )
        self.pos_expansion = MLP(
            in_dim=3,
            out_dims=[hidden_channels // 2, hidden_channels],
            activation="swish",
            last_layer_no_activation=True,
            bias=False,
        )
        if self.legacy:
            self.distance_embedding = MLP(
                in_dim=num_radial,
                out_dims=[hidden_channels // 2, hidden_channels],
                activation="swish",
                bias=False,
            )
        self.gcl_layers = nn.ModuleList()
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcl_layers.append(GCLMessage(hidden_channels, num_radial, legacy=legacy))
            self.message_layers.append(EquiMessage(hidden_channels, num_radial, reflect_equiv))
            self.update_layers.append(EquiUpdate(hidden_channels, reflect_equiv))

        self.last_layer = nn.Linear(hidden_channels, 1)
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.out_pos = EquiOutput(
            hidden_channels, out_channels=1, single_layer_output=single_layer_output
        )
        self.vec = vector()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.radial_emb.reset_parameters()

    def scalarization(self, pos: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        i, j = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        coord_diff = pos[i] - pos[j]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        coord_cross = torch.cross(pos[i], pos[j], dim=-1)
        norm = torch.sqrt(radial + EPS_SQ) + EPS
        coord_diff = coord_diff / norm
        cross_norm = torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1) + EPS_SQ) + EPS
        coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross, dim=-1)
        return dist, coord_diff, coord_cross, coord_vertical

    @staticmethod
    def assemble_nodemask(edge_index: Tensor, pos: Tensor) -> Tensor:
        node_mask = torch.zeros(pos.size(0), device=pos.device)
        node_mask[:] = -1
        _i, _j = edge_index
        _ind = 0
        for center in range(pos.size(0)):
            if node_mask[center] > -1:
                continue
            _connected = _j[torch.where(_i == center)]
            _connected = torch.concat([_connected, torch.tensor([center], device=pos.device)])
            node_mask[_connected] = _ind
            _ind += 1
        return node_mask

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        subgraph_mask: Tensor | None = None,
    ):
        """Run the backbone on ``[R | TS | P]`` concatenated coordinates.

        ``subgraph_mask`` is one for edges inside a fragment and zero for
        cross-fragment edges; the latter only exist so that the per-sample frame
        can be assembled, and are masked out of the message passing.
        """
        if not self.object_aware:
            subgraph_mask = None

        i, j = edge_index
        z_emb = self.embedding(h)

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        inner_subgraph_mask = torch.zeros(edge_index.size(1), 1, device=dist.device)
        inner_subgraph_mask[torch.where(dist < self.cutoff)[0]] = 1
        all_edge_masks = inner_subgraph_mask
        if subgraph_mask is not None:
            all_edge_masks = all_edge_masks * subgraph_mask

        edge_index_w_cutoff = edge_index.T[torch.where(all_edge_masks > 0)[0]].T
        node_mask_w_cutoff = self.assemble_nodemask(edge_index=edge_index_w_cutoff, pos=pos)

        pos_frame = remove_mean_batch(pos.clone(), node_mask_w_cutoff.long())
        dist, coord_diff, coord_cross, coord_vertical = self.scalarization(pos_frame, edge_index)

        dist = dist * all_edge_masks.squeeze(-1)
        coord_diff = coord_diff * all_edge_masks
        coord_cross = coord_cross * all_edge_masks
        coord_vertical = coord_vertical * all_edge_masks

        frame = torch.cat(
            (coord_diff.unsqueeze(-1), coord_cross.unsqueeze(-1), coord_vertical.unsqueeze(-1)),
            dim=-1,
        )
        radial_emb = self.radial_emb(dist) * all_edge_masks

        f = self.radial_lin(radial_emb)
        rbounds = 0.5 * (torch.cos(dist * pi / self.cutoff) + 1.0)
        f = rbounds.unsqueeze(-1) * f

        s = self.neighbor_emb(h, z_emb, edge_index, f)

        NE1 = self.s2v(s, coord_diff.unsqueeze(-1), edge_index, f)
        scalrization1 = torch.sum(NE1[i].unsqueeze(2) * frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(NE1[j].unsqueeze(2) * frame.unsqueeze(-1), dim=1)
        if self.reflect_equiv:
            scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
            scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (
            self.lin3(torch.permute(scalrization1, (0, 2, 1)))
            + torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1)
        scalar4 = (
            self.lin3(torch.permute(scalrization2, (0, 2, 1)))
            + torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1)
        edgeweight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edgeweight = torch.cat((edgeweight, f), dim=-1)
        edgeweight = torch.cat((edgeweight, radial_emb), dim=-1)

        a = pos_frame
        if self.legacy:
            b = self.vec(pos_frame, edge_index)
        else:
            eff_edge_ij = torch.where(all_edge_masks.squeeze(-1) == 1)[0]
            eff_edge_index = edge_index[:, eff_edge_ij]
            eff_dist = dist[eff_edge_ij]
            b = nn_vector(eff_dist, eff_edge_index, pos_frame)

        x1 = (a - b) / (torch.sqrt(torch.sum((a - b) ** 2, 1).unsqueeze(1) + EPS_SQ) + EPS)
        y1 = torch.cross(a, b, dim=-1)
        normy = torch.sqrt(torch.sum(y1**2, 1).unsqueeze(1) + EPS_SQ) + EPS
        y1 = y1 / normy
        z1 = torch.cross(x1, y1, dim=-1)
        nodeframe = torch.cat((x1.unsqueeze(-1), y1.unsqueeze(-1), z1.unsqueeze(-1)), dim=-1)

        pos_prjt = torch.sum(pos_frame.unsqueeze(-1) * nodeframe, dim=1)

        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)
        for layer in range(self.num_layers):
            if self.legacy or layer == 0:
                s = s + self.pos_expansion(pos_prjt)
            s, edgeweight = self.gcl_layers[layer](s, edge_index, edgeweight)
            dx, dvec = self.message_layers[layer](
                s, vec, edge_index, radial_emb, edgeweight, coord_diff, coord_cross
            )
            s = s + dx
            vec = vec + dvec
            s = s * self.inv_sqrt_2
            if self.update:
                dx, dvec = self.update_layers[layer](s, vec, nodeframe)
                s = s + dx
                vec = vec + dvec

        _, dpos = self.out_pos(s, vec)
        pos = pos + dpos
        return self.embedding_out(s), pos, None
