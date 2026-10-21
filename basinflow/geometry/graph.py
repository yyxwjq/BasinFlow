"""Batched tensor neighbor graphs, one API for every backbone.

:func:`torch_neighbor_graph_from_batch` returns the same five keys for every
backbone:

* EGNN integrates on one geometry and asks for the radius graph of ``pos``.
* PaiNN passes ``reference_positions`` and gets the union of the radius graphs over
  the reference and the current state, with per-state edge vectors.

The builder is pure torch (:mod:`basinflow.geometry.radius_graph`) and
differentiable with respect to positions.
"""
from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "torch graph utilities require optional model dependency: install torch"
        ) from exc
    return torch


def _batch_field(batch, name: str):
    if isinstance(batch, dict):
        return batch[name]
    return getattr(batch, name)


def torch_neighbor_graph_from_batch(
    batch,
    cutoff: float,
    *,
    positions_key: str = "pos",
    reference_positions=None,
) -> dict[str, Any]:
    """Build the neighbor graph a flow model integrates on.

    Args:
        batch: PyG ``Data``/``Batch`` or a plain dict holding ``cell`` ``[B,3,3]``,
            ``pbc`` ``[B,3]`` and a ``batch`` ``[N]`` index.
        cutoff: Edge cutoff in the same length unit as the positions.
        positions_key: Field holding the geometry to build from.
        reference_positions: Optional second geometry (the unperturbed reactant).
            When given, the graph is the union of both radius graphs and the
            returned ``edge_vectors_2`` belongs to that geometry, which is what a
            dual-geometry backbone consumes.

    Returns:
        ``edge_index`` ``[2, E]`` (row 0 receiver, row 1 sender),
        ``cell_offsets`` ``[E, 3]`` (the image used, ASE convention),
        ``edge_vectors``/``edge_lengths`` for the primary geometry and, when
        ``reference_positions`` is given, ``edge_vectors_2``/``edge_lengths_2`` for
        it. Vector convention is ``sender - receiver + offset @ cell``, so
        ``edge_vectors[k].norm(dim=-1) == edge_lengths[k]``.
    """
    torch = _torch()
    from basinflow.geometry.radius_graph import edge_geometry, radius_graph

    positions = _batch_field(batch, positions_key)
    graph_ids = _batch_field(batch, "batch")
    cell = _batch_field(batch, "cell")
    pbc = _batch_field(batch, "pbc")
    if reference_positions is not None and reference_positions.shape != positions.shape:
        raise ValueError("reference_positions and the primary positions must match in shape")

    graph = radius_graph(positions, graph_ids, cell, pbc, cutoff=float(cutoff))
    edge_index, cell_offsets = graph["edge_index"], graph["cell_offsets"]
    if reference_positions is not None:
        secondary = radius_graph(
            reference_positions, graph_ids, cell, pbc, cutoff=float(cutoff)
        )
        rows = torch.cat(
            [
                torch.cat([edge_index.T, cell_offsets], dim=1),
                torch.cat([secondary["edge_index"].T, secondary["cell_offsets"]], dim=1),
            ]
        )
        rows = torch.unique(rows, dim=0)
        edge_index, cell_offsets = rows[:, :2].T.contiguous(), rows[:, 2:]

    vectors, lengths = edge_geometry(positions, edge_index, cell_offsets, cell, graph_ids)
    result = {
        "edge_index": edge_index,
        "cell_offsets": cell_offsets,
        "edge_vectors": vectors,
        "edge_lengths": lengths,
    }
    if reference_positions is not None:
        reference_vectors, reference_lengths = edge_geometry(
            reference_positions, edge_index, cell_offsets, cell, graph_ids
        )
        result["edge_vectors_2"] = reference_vectors
        result["edge_lengths_2"] = reference_lengths
    return result
