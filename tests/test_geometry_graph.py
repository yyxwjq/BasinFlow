"""Graph construction and minimum-image handling, pinned on known geometries."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from basinflow.geometry.graph import torch_neighbor_graph_from_batch
from basinflow.geometry.radius_graph import radius_graph
from basinflow.models.egnn import EGNNFlow
from basinflow.models.painn import PaiNN


def _batch(pos, *, cell=None, pbc=None):
    positions = torch.tensor(pos, dtype=torch.float32)
    count = len(positions)
    cell = torch.eye(3, dtype=torch.float32) * 10.0 if cell is None else torch.as_tensor(cell, dtype=torch.float32)
    pbc = torch.zeros(3, dtype=torch.bool) if pbc is None else torch.as_tensor(pbc, dtype=torch.bool)
    return SimpleNamespace(
        pos=positions,
        reactant_pos=positions.clone(),
        endpoint_pos=positions.clone(),
        z=torch.ones(count, dtype=torch.long),
        batch=torch.zeros(count, dtype=torch.long),
        movable_mask=torch.ones(count, dtype=torch.bool),
        flow_time=torch.zeros(1, 1),
        cell=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
        num_graphs=1,
    )


def _graph(batch, cutoff=1.0, *, reference_positions=None):
    return torch_neighbor_graph_from_batch(
        batch, cutoff, reference_positions=reference_positions
    )


def test_nonperiodic_distances_are_not_wrapped():
    batch = _batch([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], pbc=[False, False, False])
    graph = _graph(batch)
    assert graph["edge_index"].shape == (2, 0)


def test_periodic_pair_uses_the_minimum_image():
    batch = _batch([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], pbc=[True, True, True])
    graph = _graph(batch)
    assert graph["edge_index"].shape == (2, 2)
    assert torch.allclose(graph["edge_lengths"], torch.full((2,), 0.5), atol=1e-6)
    # Edge vector convention is sender - receiver + offset @ cell.
    assert torch.allclose(graph["edge_vectors"].norm(dim=-1), graph["edge_lengths"])


def test_only_periodic_axes_are_wrapped():
    axial = _batch([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], pbc=[True, False, False])
    transverse = _batch([[0.0, 0.0, 0.0], [0.0, 9.5, 0.0]], pbc=[True, False, False])
    assert _graph(axial)["edge_index"].shape == (2, 2)
    assert _graph(transverse)["edge_index"].shape == (2, 0)


def test_atoms_outside_the_cell_are_wrapped_before_the_search():
    batch = _batch([[20.0, 0.0, 0.0], [30.5, 0.0, 0.0]], pbc=[True, True, True])
    graph = _graph(batch)
    assert graph["edge_index"].shape == (2, 2)
    assert torch.allclose(graph["edge_lengths"], torch.full((2,), 0.5), atol=1e-6)


def test_reference_geometry_gets_its_own_edge_vectors():
    batch = _batch([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], pbc=[True, True, True])
    batch.reactant_pos = torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    graph = _graph(batch, reference_positions=batch.reactant_pos)
    assert torch.allclose(graph["edge_lengths"], torch.full((2,), 0.5), atol=1e-6)
    assert torch.allclose(graph["edge_lengths_2"], torch.full((2,), 1.0), atol=1e-6)


def test_radius_graph_rejects_a_periodic_direction_without_a_cell():
    positions = torch.zeros(2, 3)
    batch = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="cell"):
        radius_graph(positions, batch, torch.zeros(1, 3, 3), torch.ones(1, 3, dtype=torch.bool), 1.0)


@pytest.mark.parametrize("model", [PaiNN(num_features=16, num_layers=1, num_radial_basis=4, time_dim=8),
                                   EGNNFlow(hidden_dim=16, num_layers=1, cutoff=1.0)])
def test_backends_accept_periodic_frames(model):
    batch = _batch([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], pbc=[True, True, True])
    output = model(batch)["scaled"]
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.mean(dim=0), torch.zeros(3), atol=1e-5)
