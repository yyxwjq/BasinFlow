from __future__ import annotations

import math

from basinflow.geometry.bonds import BOND_CHANGE_CLASSES

import torch
from torch import nn


class GaussianFourierBasis(nn.Module):
    def __init__(self, num_basis: int, scale: float = 1.0):
        super().__init__()
        if num_basis < 2 or num_basis % 2:
            raise ValueError("num_basis must be positive and even")
        self.register_buffer("frequencies", torch.randn(num_basis // 2) * 2 * math.pi * scale)

    def forward(self, values):
        phases = values.reshape(-1, 1) * self.frequencies
        return torch.cat([phases.sin(), phases.cos()], dim=-1)


class BesselBasis(nn.Module):
    def __init__(self, num_basis: int, r_max: float):
        super().__init__()
        if num_basis < 1 or r_max <= 0:
            raise ValueError("num_basis and r_max must be positive")
        self.r_max = float(r_max)
        self.register_buffer("orders", torch.arange(1, num_basis + 1, dtype=torch.float32))

    def forward(self, distances):
        frequencies = self.orders * math.pi / self.r_max
        return math.sqrt(2 / self.r_max) * frequencies * torch.sinc(distances[..., None] * self.orders / self.r_max)


class GaussianBasis(nn.Module):
    def __init__(self, num_basis: int, r_max: float):
        super().__init__()
        if num_basis < 2 or r_max <= 0:
            raise ValueError("Gaussian basis requires at least two centers and a positive cutoff")
        self.register_buffer("centers", torch.linspace(0, r_max, num_basis))
        self.width = float(r_max) / (num_basis - 1)

    def forward(self, distances):
        return torch.exp(-0.5 * ((distances[..., None] - self.centers) / self.width).square())


class PolynomialEnvelope(nn.Module):
    def __init__(self, r_max: float, exponent: int = 5):
        super().__init__()
        if r_max <= 0 or exponent < 2:
            raise ValueError("r_max must be positive and exponent at least two")
        self.r_max = float(r_max)
        self.exponent = int(exponent)

    def forward(self, distances):
        fraction = (distances / self.r_max).clamp(0, 1)
        exponent = self.exponent
        coefficient_a = -(exponent + 1) * (exponent + 2) / 2
        coefficient_b = exponent * (exponent + 2)
        coefficient_c = -exponent * (exponent + 1) / 2
        values = 1 + coefficient_a * fraction.pow(exponent) + coefficient_b * fraction.pow(exponent + 1) + coefficient_c * fraction.pow(exponent + 2)
        return torch.where(distances < self.r_max, values, torch.zeros_like(values))


class CosineEnvelope(nn.Module):
    def __init__(self, r_max: float):
        super().__init__()
        self.r_max = float(r_max)

    def forward(self, distances):
        fraction = (distances / self.r_max).clamp(0, 1)
        return torch.where(distances < self.r_max, 0.5 * (1 + torch.cos(math.pi * fraction)), torch.zeros_like(distances))


class RadialBasis(nn.Module):
    def __init__(self, num_basis: int, r_max: float, basis: str = "bessel", envelope: str = "polynomial"):
        super().__init__()
        basis_types = {"bessel": BesselBasis, "gaussian": GaussianBasis}
        envelope_types = {"polynomial": PolynomialEnvelope, "cosine": CosineEnvelope}
        if basis not in basis_types or envelope not in envelope_types:
            raise ValueError("basis must be bessel/gaussian and envelope polynomial/cosine")
        self.basis = basis_types[basis](num_basis, r_max)
        self.envelope = envelope_types[envelope](r_max)

    def forward(self, distances):
        return self.basis(distances), self.envelope(distances)


def _bound_vector(vector: torch.Tensor, cap: float) -> torch.Tensor:
    """Soft per-node cap on the vector stream.

    The rescaling factor is a per-node scalar derived from that node's vector
    norms, so it commutes with rotations and reflections and therefore preserves
    E(3) equivariance. A constant divisor cannot bound a stream whose magnitude
    grows across layers; this can.
    """
    rms = torch.sqrt(vector.square().sum(dim=1).mean(dim=1, keepdim=True) + 1e-12)
    return vector * (cap / (cap + rms))[:, :, None]


class DualMessageBlock(nn.Module):
    """PaiNN scalar/vector messages using separate filters for both geometries."""

    def __init__(self, features: int, num_radial_basis: int, stability_mode: str = "scaled",
                 bond_change_condition: bool = False, endpoint_condition: bool = False,
                 endpoint_equivariant: bool = False):
        super().__init__()
        self.scaled = stability_mode in {"scaled", "bounded"}
        self.bounded = stability_mode == "bounded"
        # An extra conditioning geometry (the product endpoint of an R+P -> TS
        # task). ``endpoint_equivariant`` decides whether it contributes edge
        # unit vectors to the vector message or only distances to the scalar
        # filter; the two settings are the ablation the user asked for.
        if endpoint_equivariant and not endpoint_condition:
            raise ValueError("endpoint_equivariant requires endpoint_condition")
        self.endpoint_condition = bool(endpoint_condition)
        self.endpoint_equivariant = bool(endpoint_equivariant)
        # The bond-change condition is a discrete reactant-side label, so it
        # enters the reference filter as extra one-hot channels rather than
        # through the geometry. Off by default: enabling it changes the filter
        # shape, hence the checkpoint shape.
        self.bond_change_condition = bool(bond_change_condition)
        filter_dim = num_radial_basis + (BOND_CHANGE_CLASSES if self.bond_change_condition else 0)
        self.vector_scale = math.sqrt(features)
        self.vector_cap = math.sqrt(features)
        self.scalar_norm = nn.LayerNorm(features, elementwise_affine=False) if self.scaled else nn.Identity()
        self.source = nn.Sequential(nn.Linear(features, features), nn.SiLU(), nn.Linear(features, features * 4))
        self.reference_filter = nn.Linear(filter_dim, features * 4)
        self.current_filter = nn.Linear(num_radial_basis, features * 4)
        # The endpoint always contributes invariantly through the scalar
        # filter; the optional separate projection feeds its edge directions
        # into the vector message.
        self.endpoint_filter = (
            nn.Linear(num_radial_basis, features * 4) if self.endpoint_condition else None
        )
        self.endpoint_vector_filter = (
            nn.Linear(num_radial_basis, features)
            if (self.endpoint_condition and self.endpoint_equivariant) else None
        )

    def forward(self, scalar, vector, reference_radial, current_radial, reference_cutoff, current_cutoff, reference_unit, current_unit, edge_index, bond_change=None, endpoint=None):
        receiver, sender = edge_index
        if self.bond_change_condition:
            if bond_change is None:
                raise ValueError("bond_change_condition is enabled but no bond_change was passed")
            reference_radial = torch.cat([reference_radial, bond_change], dim=-1)
        filters = self.reference_filter(reference_radial) * reference_cutoff[:, None]
        filters = filters + self.current_filter(current_radial) * current_cutoff[:, None]
        endpoint_radial = endpoint_cutoff = endpoint_unit = None
        if self.endpoint_condition:
            if endpoint is None:
                raise ValueError("endpoint_condition is enabled but no endpoint was passed")
            endpoint_radial, endpoint_cutoff, endpoint_unit = endpoint
            filters = filters + self.endpoint_filter(endpoint_radial) * endpoint_cutoff[:, None]
        scalar_message, vector_gate, reference_gate, current_gate = (self.source(self.scalar_norm(scalar)[sender]) * filters).chunk(4, dim=-1)
        if self.scaled:
            vector_gate = vector_gate / math.sqrt(3)
        vector_message = vector[sender] * vector_gate[:, None, :]
        vector_message = vector_message + reference_unit[:, :, None] * reference_gate[:, None, :] * reference_cutoff[:, None, None]
        vector_message = vector_message + current_unit[:, :, None] * current_gate[:, None, :] * current_cutoff[:, None, None]
        if self.endpoint_equivariant:
            # The endpoint edge direction is a first-class equivariant message,
            # not just a distance in the scalar filter.
            endpoint_gate = self.endpoint_vector_filter(endpoint_radial)
            vector_message = vector_message + endpoint_unit[:, :, None] * endpoint_gate[:, None, :] * endpoint_cutoff[:, None, None]
        if self.scaled:
            vector_message = vector_message / self.vector_scale
        scalar = scalar + torch.zeros_like(scalar).index_add(0, receiver, scalar_message)
        vector = vector + torch.zeros_like(vector).index_add(0, receiver, vector_message)
        if self.scaled:
            scalar = scalar / math.sqrt(2)
        if self.bounded:
            vector = _bound_vector(vector, self.vector_cap)
        return scalar, vector


class UpdateBlock(nn.Module):
    def __init__(self, features: int, stability_mode: str = "scaled"):
        super().__init__()
        self.scaled = stability_mode in {"scaled", "bounded"}
        self.bounded = stability_mode == "bounded"
        self.dot_scale = math.sqrt(features)
        self.dot_bound = math.sqrt(features)
        self.vector_cap = math.sqrt(features)
        self.gate_norm = nn.LayerNorm(features * 2, elementwise_affine=False) if self.scaled else nn.Identity()
        self.vector_mix = nn.Linear(features, features * 2, bias=False)
        self.gates = nn.Sequential(nn.Linear(features * 2, features), nn.SiLU(), nn.Linear(features, features * 3))

    def forward(self, scalar, vector):
        vector_left, vector_right = self.vector_mix(vector).chunk(2, dim=-1)
        magnitude = torch.sqrt(vector_right.square().sum(dim=1) + 1e-8)
        vector_gate, scalar_gate, scalar_shift = self.gates(self.gate_norm(torch.cat([scalar, magnitude], dim=-1))).chunk(3, dim=-1)
        vector_dot = (vector_left * vector_right).sum(dim=1)
        if self.scaled:
            vector_dot = vector_dot / self.dot_scale
        if self.bounded:
            vector_dot = self.dot_bound * torch.tanh(vector_dot / self.dot_bound)
        scalar_delta = scalar_shift + scalar_gate * vector_dot
        if self.scaled:
            scalar_delta = scalar_delta / math.sqrt(2)
        scalar = scalar + scalar_delta
        vector = vector + vector_gate[:, None, :] * vector_left
        if self.bounded:
            vector = _bound_vector(vector, self.vector_cap)
        return scalar, vector


class GatedEquivariantBlock(nn.Module):
    """Output head. PaiNN's gate is where the last amplification can happen.

    ``UpdateBlock`` normalizes ``cat([scalar, magnitude])`` before its gate MLP;
    this head historically did not, so it consumed the raw scalar stream. Under
    ``bounded`` the scalar stream is *not* capped (only the vector stream and the
    inner product are), so this head is the one place where an unbounded scalar
    could still be turned into a huge velocity. Normalization is applied only for
    ``bounded`` so that ``none`` and ``scaled`` keep their exact behaviour and
    already-published runs stay reproducible.
    """

    def __init__(self, features: int, stability_mode: str = "scaled"):
        super().__init__()
        self.normalize_gate = stability_mode == "bounded"
        self.vector_mix = nn.Linear(features, features + 1, bias=False)
        # elementwise_affine=False introduces no parameters, so checkpoints stay loadable.
        self.gate_norm = (nn.LayerNorm(2 * features, elementwise_affine=False)
                          if self.normalize_gate else nn.Identity())
        self.gate = nn.Sequential(nn.Linear(2 * features, features), nn.SiLU(), nn.Linear(features, 1))

    def forward(self, scalar, vector):
        mixed = self.vector_mix(vector)
        magnitude = torch.sqrt(mixed[:, :, 1:].square().sum(dim=1) + 1e-8)
        gate = self.gate(self.gate_norm(torch.cat([scalar, magnitude], dim=-1)))
        return mixed[:, :, 0] * gate
