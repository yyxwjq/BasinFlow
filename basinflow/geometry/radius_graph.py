"""Tensor-level radius graphs in pure PyTorch.

These build graphs directly from batched tensors, with no ``StructureRecord``
and no ASE, because the PaiNN backbone rebuilds its own graph at every
integration step. The case-preserving reference implementation lives in
:mod:`basinflow.geometry.graph`.

Unlike :mod:`basinflow.geometry.graph`, this module requires torch at import
time; it is only imported by model and sampling code.
"""
from __future__ import annotations

import itertools
import math

import torch


@torch.no_grad()
def radius_graph(positions, batch, cell, pbc, cutoff: float, pair_chunk_size: int = 128):
    """Exact directed radius graph with all periodic images, in pure PyTorch.

    Integer offsets use the ASE convention: r_sender-r_receiver+offset@cell.
    Topology is discrete; use edge_geometry for differentiable edge vectors.
    Pair chunks bound temporary memory. Complexity is quadratic per structure.
    """
    if cutoff <= 0 or not math.isfinite(cutoff) or pair_chunk_size < 1:
        raise ValueError("cutoff and pair_chunk_size must be positive and finite")
    if positions.ndim != 2 or positions.shape[1] != 3 or not torch.isfinite(positions).all():
        raise ValueError("positions must be finite [N, 3]")
    if batch.shape != (len(positions),) or batch.dtype != torch.long:
        raise ValueError("batch must be int64 [N]")
    if cell.ndim != 3 or cell.shape[1:] != (3, 3) or pbc.shape != cell.shape[:1] + (3,):
        raise ValueError("cell and pbc must have shapes [B,3,3] and [B,3]")
    if not torch.isfinite(cell).all():
        raise ValueError("cell must be finite, including for nonperiodic structures")
    if len(batch) and (batch.min() < 0 or batch.max() >= len(cell)):
        raise ValueError("batch contains out-of-range graph ids")
    edge_parts, offset_parts = [], []
    for graph_id in torch.unique(batch).tolist():
        atom_ids = torch.where(batch == graph_id)[0]
        local = positions[atom_ids]
        lattice = cell[graph_id].to(positions)
        periodic = pbc[graph_id].bool()
        atom_images = torch.zeros_like(local, dtype=torch.long)
        bounds = [0, 0, 0]
        if periodic.any():
            periodic_lattice = lattice[periodic]
            if not torch.isfinite(lattice).all() or torch.linalg.matrix_rank(periodic_lattice) != len(periodic_lattice):
                raise ValueError("periodic cell vectors must be finite and linearly independent")
            reciprocal = torch.linalg.pinv(periodic_lattice)
            atom_images[:, periodic] = torch.floor(local @ reciprocal).long()
            periodic_bounds = torch.ceil(cutoff * reciprocal.norm(dim=0)).long()
            for axis, bound in zip(torch.where(periodic)[0].tolist(), periodic_bounds.tolist()):
                bounds[axis] = int(bound)
        wrapped = local - atom_images.to(local) @ lattice
        for image in itertools.product(*(range(-bound, bound + 1) for bound in bounds)):
            image_offset = torch.tensor(image, dtype=torch.long, device=positions.device)
            shift = image_offset.to(local) @ lattice
            for start in range(0, len(local), pair_chunk_size):
                vectors = wrapped[None, :, :] - wrapped[start:start + pair_chunk_size, None, :] + shift
                squared = vectors.square().sum(-1)
                within = squared < cutoff * cutoff
                if image == (0, 0, 0):
                    diagonal = torch.arange(start, min(start + pair_chunk_size, len(local)), device=positions.device)
                    within[torch.arange(len(diagonal), device=positions.device), diagonal] = False
                receiver, sender = torch.where(within)
                receiver = receiver + start
                if receiver.numel():
                    edge_parts.append(torch.stack([atom_ids[receiver], atom_ids[sender]]))
                    offset_parts.append(image_offset + atom_images[receiver] - atom_images[sender])
    return {
        "edge_index": torch.cat(edge_parts, dim=1) if edge_parts else batch.new_empty((2, 0)),
        "cell_offsets": torch.cat(offset_parts) if offset_parts else batch.new_empty((0, 3)),
    }


def edge_geometry(positions, edge_index, cell_offsets, cell, batch):
    receiver, sender = edge_index
    shifts = torch.einsum("ei,eij->ej", cell_offsets.to(positions), cell[batch[receiver]].to(positions))
    vectors = positions[sender] - positions[receiver] + shifts
    return vectors, torch.linalg.vector_norm(vectors, dim=-1)


def dual_radius_graph(reference, current, batch, cell, pbc, cutoff):
    """Union of the reference and current radius graphs, including image ids."""
    graphs = [radius_graph(positions, batch, cell, pbc, cutoff) for positions in (reference, current)]
    rows = torch.cat([torch.cat([graph["edge_index"].T, graph["cell_offsets"]], dim=1) for graph in graphs])
    rows = torch.unique(rows, dim=0)
    return {"edge_index": rows[:, :2].T.contiguous(), "cell_offsets": rows[:, 2:]}
