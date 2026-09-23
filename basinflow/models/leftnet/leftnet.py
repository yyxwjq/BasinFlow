"""The object-aware LEFTNet backbone, ported from the ReactOT reference.

Provenance: ``reactot/model/leftnet.py:576-890`` (``LEFTNet``) and
``:428-457`` (``nn_vector``). The tensor-in/tensor-out contract is kept
identical to the reference so the port can be compared against
``reactot/model/leftnet.py`` line by line; the ``EventData`` adaptation lives in
:mod:`basinflow.models.leftnet.adapter` instead.

``torch_scatter`` is replaced by :func:`basinflow.models.leftnet.layers.scatter_sum`
inside ``EquiMessage.aggregate``; ``assert_rot_equiv`` (a debug helper) is not
ported. The switch semantics (``reflect_equiv``, ``object_aware``, ``legacy``,
``update``, ``single_layer_output``, ``pos_grad``) are preserved exactly.

One deliberate numerical deviation
----------------------------------
The reference normalises cross products with ``sqrt(sum(v ** 2)) + EPS``, i.e.
the epsilon sits *outside* the square root while ``EPS = 1e-6`` is added to the
denominator. Whenever a cross product is exactly zero -- which happens for
parallel atom pairs in the edge frame and for collinear ``(a, b)`` in the node
frame -- ``sqrt`` has derivative ``inf`` at its input and the backward pass
returns ``nan``. That is a property of the reference formulas, not of this port:
it reproduces with untouched reference code and zeroes the gradients of an
otherwise well-conditioned batch.

The fix here is minimal and semantic-free: ``EPS_SQ = 1e-12`` is added *inside*
each of those square roots. The forward value changes by at most ``1e-6``
relative, and the singular derivative disappears. This must be quoted next to
any claimed bitwise equivalence with ``reactot/model/leftnet.py``.
"""
from __future__ import annotations

import math
from math import pi
from typing import Optional

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
    """Nearest-neighbour position, used to build the node frame (non-legacy).

    Port of ``reactot/model/leftnet.py:428-457``: for every atom pick the
    position of its nearest neighbour (by ``dist``) as the reference direction.
    Kept as an explicit loop for parity with the reference; it is only reached
    when ``legacy=False``.
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
            reference's ``reflect_equiv``), which matches scoring after a
            chirality-ignoring RMSD.
        legacy: Reference's legacy node-frame construction (nearest-neighbour
            frame). The ReactOT default config uses ``True``.
        update: Run the per-layer :class:`EquiUpdate`.
        single_layer_output: One gated equivariant output block instead of two.
        object_aware: Mask inter-fragment edges with ``subgraph_mask``.
    """

    def __init__(
        self,
        pos_require_grad: bool = False,
        cutoff: float = 10.0,
        num_layers: int = 4,
        hidden_channels: int = 128,
        num_radial: int = 96,
        in_hidden_channels: int = 8,
        reflect_equiv: bool = True,
        legacy: bool = True,
        update: bool = True,
        pos_grad: bool = False,
        single_layer_output: bool = True,
        for_conf: bool = False,
        ff: bool = False,
        object_aware: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.pos_require_grad = pos_require_grad
        self.reflect_equiv = reflect_equiv
        self.legacy = legacy
        self.update = update
        self.pos_grad = pos_grad
        self.for_conf = for_conf
        self.ff = ff
        self.object_aware = object_aware
        # ``r_max`` mirrors the PaiNN convention so ``models/factory.model_cutoff``
        # can read a single attribute for every backend.
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
        if self.pos_grad:
            self.dynamic_mlp_modules = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_channels // 2, 3),
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
        coord_cross = torch.cross(pos[i], pos[j])
        norm = torch.sqrt(radial + EPS_SQ) + EPS
        coord_diff = coord_diff / norm
        cross_norm = torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1) + EPS_SQ) + EPS
        coord_cross = coord_cross / cross_norm
        coord_vertical = torch.cross(coord_diff, coord_cross)
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
        edge_attr: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
        edge_mask: Optional[Tensor] = None,
        update_coords_mask: Optional[Tensor] = None,
        subgraph_mask: Optional[Tensor] = None,
        pbc: bool = False,
        distance_vectors: Optional[Tensor] = None,
    ):
        if not self.object_aware:
            subgraph_mask = None

        i, j = edge_index
        z_emb = self.embedding(h)

        if not pbc:
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
            inner_subgraph_mask = torch.zeros(edge_index.size(1), 1, device=dist.device)
            inner_subgraph_mask[torch.where(dist < self.cutoff)[0]] = 1
            all_edge_masks = inner_subgraph_mask
            if subgraph_mask is not None:
                all_edge_masks = all_edge_masks * subgraph_mask
        else:
            all_edge_masks = subgraph_mask

        edge_index_w_cutoff = edge_index.T[torch.where(all_edge_masks > 0)[0]].T
        node_mask_w_cutoff = self.assemble_nodemask(edge_index=edge_index_w_cutoff, pos=pos)

        pos_frame = remove_mean_batch(pos.clone(), node_mask_w_cutoff.long())

        if not pbc:
            dist, coord_diff, coord_cross, coord_vertical = self.scalarization(pos_frame, edge_index)
        else:
            coord_diff = distance_vectors
            dist = distance_vectors.norm(dim=-1)
            coord_diff = coord_diff / (dist.unsqueeze(1) + EPS)
            coord_cross = torch.cross(pos[i], pos[j])
            coord_cross = coord_cross / (
                torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1) + EPS_SQ) + EPS
            )
            coord_vertical = torch.cross(coord_diff, coord_cross)

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
        y1 = torch.cross(a, b)
        normy = torch.sqrt(torch.sum(y1**2, 1).unsqueeze(1) + EPS_SQ) + EPS
        y1 = y1 / normy
        z1 = torch.cross(x1, y1)
        nodeframe = torch.cat((x1.unsqueeze(-1), y1.unsqueeze(-1), z1.unsqueeze(-1)), dim=-1)

        pos_prjt = torch.sum(pos_frame.unsqueeze(-1) * nodeframe, dim=1)

        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)
        gradient = torch.zeros(s.size(0), 3, device=s.device)
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
            if self.pos_grad:
                dynamic_coff = self.dynamic_mlp_modules(s)
                basis_mix = (
                    dynamic_coff[:, :1] * x1 + dynamic_coff[:, 1:2] * y1 + dynamic_coff[:, 2:3] * z1
                )
                gradient = gradient + basis_mix / self.num_layers

        if self.for_conf:
            return s

        _, dpos = self.out_pos(s, vec)
        if update_coords_mask is not None:
            dpos = update_coords_mask * dpos
        pos = pos + dpos + gradient

        if self.ff:
            return s, dpos

        h = self.embedding_out(s)
        if node_mask is not None:
            h = h * node_mask
        return h, pos, None
