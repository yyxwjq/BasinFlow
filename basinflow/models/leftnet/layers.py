"""LEFTNet layers: a faithful port of the reference ReactOT message-passing stack.

Provenance (read, not copied wholesale): ``reactot/model/leftnet.py`` classes
``RBFEmb``, ``NeighborEmb``, ``CFConvS2V``, ``GCLMessage``, ``EquiMessage``,
``EquiUpdate``, ``vector``, ``EquiOutput``, ``GatedEquivariantBlock``; and
``reactot/model/core.py`` (``ACTIVATION_MAPPING``, ``OneLayerActivation``,
``MLP``).

Two deliberate adaptations, both required by this repository's rules:

* ``torch_scatter.scatter`` / ``scatter_mean`` and
  ``util_funcs.unsorted_segment_sum`` are re-implemented locally
  (:func:`scatter_sum`, :func:`scatter_mean`, :func:`unsorted_segment_sum`), so
  BasinFlow keeps its "no ``torch_scatter``" guarantee while the reference repo
  still runs against the PyG shim in ``tools/reactot_compat``.
* ``MessagePassing`` is kept from ``torch_geometric`` (already a ``models``
  extra) instead of re-deriving PyG's source/target conventions by hand, which
  is the part of the reference that is easiest to get subtly wrong. Only the
  aggregation is overridden, where the reference calls ``torch_scatter``.

Numerical note: segment reductions are exact; only the floating-point summation
order can differ from the compiled ``torch_scatter`` kernels. That difference is
irrelevant here because these weights are trained from scratch, not loaded from
the reference checkpoint.
"""
from __future__ import annotations

import math
from math import pi
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

try:  # pragma: no cover - the dependency is present in `models` installs
    from torch_geometric.nn.conv import MessagePassing
except ImportError as exc:  # pragma: no cover
    raise ImportError("LEFTNet requires basinflow[models] (torch_geometric)") from exc

EPS = 1e-6


# --------------------------------------------------------------------------- #
# segment reductions (replacements for torch_scatter / unsorted_segment_sum)
# --------------------------------------------------------------------------- #
def _segment_ids(index: Tensor) -> Tensor:
    return index.to(torch.long).reshape(-1)


def _dim_size(index: Tensor, dim_size: Optional[int]) -> int:
    if dim_size is not None:
        return int(dim_size)
    return int(index.max()) + 1 if index.numel() else 0


def scatter_sum(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """``torch_scatter.scatter`` / ``scatter_sum`` with ``reduce="sum"``."""
    index = _segment_ids(index)
    size = _dim_size(index, dim_size)
    moved = src.movedim(dim, 0)
    out = torch.zeros((size,) + tuple(moved.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, moved)
    return out.movedim(0, dim)


def scatter_mean(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """``torch_scatter.scatter_mean``."""
    index = _segment_ids(index)
    size = _dim_size(index, dim_size)
    moved = src.movedim(dim, 0)
    out = torch.zeros((size,) + tuple(moved.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, moved)
    counts = torch.zeros(size, dtype=src.dtype, device=src.device)
    counts.index_add_(0, index, torch.ones(moved.shape[0], dtype=src.dtype, device=src.device))
    out = out / counts.clamp(min=1.0).reshape((size,) + (1,) * (moved.dim() - 1))
    return out.movedim(0, dim)


def unsorted_segment_sum(
    data: Tensor,
    segment_ids: Tensor,
    num_segments: int,
    normalization_factor: float = 1.0,
    aggregation_method: str = "sum",
) -> Tensor:
    """Port of ``reactot/model/util_funcs.py:29-47`` (sum or mean)."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor
    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


# --------------------------------------------------------------------------- #
# small helpers (reactot/model/leftnet.py:18-29)
# --------------------------------------------------------------------------- #
def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def com(x: Tensor) -> Tensor:
    return x - torch.mean(x, dim=0)


def remove_mean_batch(x: Tensor, indices: Tensor) -> Tensor:
    mean = scatter_mean(x, indices, dim=0)
    return x - mean[indices]


# --------------------------------------------------------------------------- #
# MLP (reactot/model/core.py)
# --------------------------------------------------------------------------- #
ACTIVATION_MAPPING = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
}


class OneLayerActivation(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, activation=None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = ACTIVATION_MAPPING[activation] if activation is not None else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.linear(inputs))


class MLP(nn.Module):
    """Multi-layer perceptron, identical layout to the reference ``MLP``.

    The parameter names (``mlp.0.linear.weight`` ...) match the reference so a
    state dict stays auditable against ``reactot/model/core.py``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dims: list[int],
        bias: bool = True,
        activation: Union[list[Union[str, None]], str, None] = "swish",
        last_layer_no_activation: bool = False,
    ) -> None:
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(out_dims), "activation and out_dims must have the same length"
        if last_layer_no_activation:
            activation[-1] = None
        for _activation in activation:
            assert (_activation is None) or (
                _activation in ACTIVATION_MAPPING
            ), f"activation {activation} not avail."

        module_list = []
        for index in range(len(out_dims)):
            module_list.append(
                OneLayerActivation(
                    in_dim=input_dim,
                    out_dim=out_dims[index],
                    bias=bias,
                    activation=activation[index],
                )
            )
            input_dim = out_dims[index]
        self.mlp = nn.Sequential(*module_list)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.mlp(inputs)


# --------------------------------------------------------------------------- #
# LEFTNet building blocks
# --------------------------------------------------------------------------- #
class RBFEmb(nn.Module):
    """Radial basis function embedding (reactot/model/leftnet.py:32-69)."""

    def __init__(self, num_rbf: int, rbound_upper: float, rbf_trainable: bool = False) -> None:
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()
        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self) -> tuple[Tensor, Tensor]:
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        rbounds = 0.5 * (torch.cos(dist * pi / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).float()
        return rbounds * torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class NeighborEmb(MessagePassing):
    """Node features from neighbours (reactot/model/leftnet.py:72-89)."""

    def __init__(self, hid_dim: int, in_hidden_channels: int = 5) -> None:
        super().__init__(aggr="add")
        self.embedding = nn.Linear(in_hidden_channels, hid_dim)
        self.hid_dim = hid_dim
        self.ln_emb = nn.LayerNorm(hid_dim, elementwise_affine=False)

    def forward(self, z: Tensor, s: Tensor, edge_index: Tensor, embs: Tensor) -> Tensor:
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)
        return s + s_neighbors

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, self.hid_dim) * x_j


class CFConvS2V(MessagePassing):
    """Scalar-to-vector block (reactot/model/leftnet.py:92-125)."""

    def __init__(self, hid_dim: int) -> None:
        super().__init__(aggr="add")
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=False),
            nn.SiLU(),
        )

    def forward(self, s: Tensor, v: Tensor, edge_index: Tensor, emb: Tensor) -> Tensor:
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v
        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class GCLMessage(nn.Module):
    """Gated message passing on scalar node features (leftnet.py:128-183)."""

    def __init__(
        self,
        hidden_channels: int,
        num_radial: int,
        act_fn: str = "swish",
        legacy: bool = False,
    ) -> None:
        super().__init__()
        self.edge_mlp = MLP(
            in_dim=hidden_channels * 2 + 3 * hidden_channels + num_radial,
            out_dims=[hidden_channels, hidden_channels],
            activation=act_fn,
        )
        self.node_mlp = MLP(
            in_dim=hidden_channels + hidden_channels,
            out_dims=[hidden_channels, hidden_channels],
            activation=act_fn,
            last_layer_no_activation=True if legacy else False,
        )
        self.edge_out_trans = MLP(
            in_dim=hidden_channels,
            out_dims=[3 * hidden_channels + num_radial],
            activation=act_fn,
        )
        self.att_mlp = MLP(hidden_channels, [1], activation=act_fn)
        self.x_layernorm = nn.LayerNorm(hidden_channels)
        self.x_layernorm.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
        xh = self.x_layernorm(x)
        edgeh = weight
        ii, jj = edge_index
        m_ij = self.edge_message(xh[ii], xh[jj], edgeh)
        xh = self.node_message(xh, edge_index, m_ij)
        edgeh = edgeh + self.edge_out_trans(m_ij)
        return xh, edgeh

    def edge_message(self, xh_i: Tensor, xh_j: Tensor, edgeh: Tensor) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([xh_i, xh_j, edgeh], dim=1))
        return m_ij * self.att_mlp(m_ij)

    def node_message(self, xh: Tensor, edge_index: Tensor, m_ij: Tensor) -> Tensor:
        ii, _ = edge_index
        agg = unsorted_segment_sum(
            m_ij,
            ii,
            num_segments=xh.size(0),
            normalization_factor=1,
            aggregation_method="mean",
        )
        agg = torch.cat([xh, agg], dim=1)
        return xh + self.node_mlp(agg)


class EquiMessage(MessagePassing):
    """Equivariant scalar/vector message (reactot/model/leftnet.py:186-277)."""

    def __init__(self, hidden_channels: int, num_radial: int, reflect_equiv: bool) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.reflect_equiv = reflect_equiv
        self.dir_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_channels + self.num_radial, self.hidden_channels * 3),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3),
        )
        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3, bias=False),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3, bias=False)
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.x_layernorm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        edge_rbf: Tensor,
        weight: Tensor,
        edge_vector: Tensor,
        edge_cross: Tensor,
    ) -> tuple[Tensor, Tensor]:
        xh = self.x_proj(self.x_layernorm(x))
        rbfh = self.rbf_proj(edge_rbf)
        weight = self.dir_proj(weight)
        rbfh = rbfh * weight
        return self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            edge_cross=edge_cross,
            size=None,
        )

    def message(self, xh_j, xh_i, vec_j, rbfh_ij, r_ij, edge_cross) -> tuple[Tensor, Tensor]:
        x, xh2, xh3 = torch.split((xh_j + xh_i) * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        if not self.reflect_equiv:
            vec = vec + x.unsqueeze(1) * edge_cross.unsqueeze(2)
        vec = vec * self.inv_sqrt_h
        return x, vec

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        x, vec = features
        x = scatter_sum(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter_sum(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs


class EquiUpdate(nn.Module):
    """Scalar/vector update via the node frame (leftnet.py:280-335)."""

    def __init__(self, hidden_channels: int, reflect_equiv: bool = True) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3, bias=False),
        )
        self.lin3 = nn.Sequential(
            nn.Linear(3, 48),
            nn.SiLU(inplace=True),
            nn.Linear(48, 8),
            nn.SiLU(inplace=True),
            nn.Linear(8, 1),
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.reflect_equiv = reflect_equiv
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)

    def forward(self, x: Tensor, vec: Tensor, nodeframe: Tensor) -> tuple[Tensor, Tensor]:
        vec = self.vec_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)
        scalrization = torch.sum(vec1.unsqueeze(2) * nodeframe.unsqueeze(-1), dim=1)
        if self.reflect_equiv:
            scalrization[:, 1, :] = torch.abs(scalrization[:, 1, :].clone())
        scalar = self.lin3(torch.permute(scalrization, (0, 2, 1))).squeeze(-1)
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h
        x_vec_h = self.xvec_proj(torch.cat([x, scalar], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)
        dx = (xvec1 + xvec2 + vec_dot) * self.inv_sqrt_2
        dvec = xvec3.unsqueeze(1) * vec2
        return dx, dvec


class vector(MessagePassing):
    """Mean-aggregated neighbour position (leftnet.py:418-425)."""

    def __init__(self) -> None:
        super().__init__(aggr="mean")

    def forward(self, vector: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=vector)


class GatedEquivariantBlock(nn.Module):
    """Gated equivariant block (reactot/model/leftnet.py:528-573).

    Note: the reference output head deliberately has no normalisation; the
    ``bounded`` stabilisation idea from ``docs/18`` section 12 is *not* applied
    here so this stays a faithful port. It is instead enforced by the training
    config (gradient clipping) until a numerical audit justifies a change.
    """

    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )
        self.act = nn.Identity()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)
        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2
        return self.act(x), v


class EquiOutput(nn.Module):
    """Equivariant output head (reactot/model/leftnet.py:497-525)."""

    def __init__(self, hidden_channels: int, out_channels: int = 1, single_layer_output: bool = True) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        if single_layer_output:
            self.output_network = nn.ModuleList([GatedEquivariantBlock(hidden_channels, out_channels)])
        else:
            self.output_network = nn.ModuleList(
                [
                    GatedEquivariantBlock(hidden_channels, hidden_channels // 2),
                    GatedEquivariantBlock(hidden_channels // 2, out_channels),
                ]
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x: Tensor, vec: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return x, vec.squeeze()
