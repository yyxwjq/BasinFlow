import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _ase_edge_rows(positions, cell, pbc, cutoff):
    """The edge set ASE's own neighbour list produces, as (receiver, sender, offset) rows."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms("H" * len(positions), positions=positions, cell=cell, pbc=pbc)
    # ASE returns (center i, neighbour j, image S); an edge is stored as
    # (receiver, sender, offset), so i is the receiver and j the sender.
    center, neighbour, offsets = neighbor_list("ijS", atoms, cutoff=cutoff, self_interaction=False)
    return {
        (int(i), int(j), tuple(int(value) for value in offset))
        for i, j, offset in zip(center, neighbour, offsets)
    }


@pytest.mark.parametrize("periodic", [[False, False, False], [True, True, True], [True, False, True]])
def test_tensor_graph_matches_ase_in_skew_cell(periodic):
    from basinflow.geometry.radius_graph import radius_graph

    cell = np.array([[2.1, 0, 0], [1.1, 2.3, 0], [0.4, 0.8, 2.5]])
    positions = np.array([[0.1, 0.2, 0.1], [4.8, 0.3, 0.2], [1.0, 1.3, 1.2]])
    expected = {
        "edge_index": np.array(
            [[r, s] for r, s, _ in _ase_edge_rows(positions, cell, periodic, 2.8)]
        ).T,
        "cell_offsets": np.array(
            [[*offset] for _, _, offset in _ase_edge_rows(positions, cell, periodic, 2.8)]
        ),
    }
    actual = radius_graph(torch.tensor(positions), torch.zeros(3, dtype=torch.long), torch.tensor(cell)[None], torch.tensor(periodic)[None], 2.8)
    expected_rows = np.concatenate([expected["edge_index"].T, expected["cell_offsets"]], axis=1)
    actual_rows = torch.cat([actual["edge_index"].T, actual["cell_offsets"]], dim=1).numpy()
    assert set(map(tuple, expected_rows)) == set(map(tuple, actual_rows))


def test_tensor_graph_handles_noncontiguous_batch_and_position_gradient():
    from basinflow.geometry.radius_graph import radius_graph, edge_geometry

    positions = torch.tensor([[0., 0., 0.], [10., 0., 0.], [0.5, 0., 0.], [10.4, 0., 0.]], requires_grad=True)
    graph_ids = torch.tensor([0, 1, 0, 1])
    cells = torch.zeros(2, 3, 3)
    graph = radius_graph(positions, graph_ids, cells, torch.zeros(2, 3, dtype=torch.bool), 1.)
    assert set(map(tuple, graph["edge_index"].T.tolist())) == {(0, 2), (2, 0), (1, 3), (3, 1)}
    vectors, lengths = edge_geometry(positions, graph["edge_index"], graph["cell_offsets"], cells, graph_ids)
    lengths.sum().backward()
    assert positions.grad.abs().sum() > 0
    assert vectors.shape == (4, 3)


def test_nonperiodic_graph_rejects_nonfinite_cell():
    from basinflow.geometry.radius_graph import radius_graph

    with pytest.raises(ValueError, match="finite"):
        radius_graph(torch.tensor([[0., 0., 0.], [0.5, 0., 0.]]), torch.zeros(2, dtype=torch.long), torch.full((1, 3, 3), float("nan")), torch.zeros(1, 3, dtype=torch.bool), 1.)


def _batch(positions, *, cell=None, pbc=None, key="pos"):
    import torch

    n = len(positions)
    values = torch.tensor(positions, dtype=torch.float32)
    return {
        key: values,
        "z": torch.ones(n, dtype=torch.long),
        "cell": torch.tensor(
            cell if cell is not None else np.eye(3) * 10.0, dtype=torch.float32
        ).reshape(1, 3, 3),
        "pbc": torch.tensor(
            pbc if pbc is not None else [False, False, False], dtype=torch.bool
        ).reshape(1, 3),
        "batch": torch.zeros(n, dtype=torch.long),
    }


def test_batch_graph_rebuilds_when_positions_change():
    """The solver moves atoms every step, so the graph must be rebuilt per call."""
    from basinflow.geometry.graph import torch_neighbor_graph_from_batch

    batch = _batch([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [2.0, 0.0, 0.0]])

    graph_a = torch_neighbor_graph_from_batch(batch, cutoff=0.6)
    batch["pos"][2] = torch.tensor([0.5, 0.0, 0.0])
    graph_b = torch_neighbor_graph_from_batch(batch, cutoff=0.6)

    assert graph_a["edge_index"].shape[1] == 2
    assert graph_b["edge_index"].shape[1] == 6


def test_batch_graph_edge_vectors_use_the_periodic_image():
    """A wrapped pair must still see its own image, not the raw separation."""
    from basinflow.geometry.graph import torch_neighbor_graph_from_batch

    batch = _batch(
        [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]],
        pbc=[True, False, False],
    )

    graph = torch_neighbor_graph_from_batch(batch, cutoff=0.5)

    assert graph["edge_index"].shape == (2, 2)
    assert graph["edge_vectors"].shape == (2, 3)
    assert torch.allclose(
        torch.sort(graph["edge_lengths"]).values,
        torch.tensor([0.2, 0.2], dtype=graph["edge_lengths"].dtype),
        atol=1e-6,
    )
    # The raw separation would be 9.8 A; the image edge is 0.2 A.
    raw = batch["pos"][1] - batch["pos"][0]
    assert not torch.allclose(graph["edge_vectors"][0], raw)


def test_batch_graph_positions_key_selects_the_tensor():
    from basinflow.geometry.graph import torch_neighbor_graph_from_batch

    batch = _batch([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], key="initial_positions")

    graph = torch_neighbor_graph_from_batch(
        batch, cutoff=0.6, positions_key="initial_positions"
    )

    assert graph["edge_index"].shape[1] == 2





def test_dual_geometry_graph_is_the_union_and_reports_both_vectors():
    """PaiNN's graph must cover both states, with per-state edge vectors."""
    from basinflow.geometry.graph import torch_neighbor_graph_from_batch

    batch = _batch([[0.0, 0.0, 0.0], [0.40, 0.0, 0.0]])
    reference = torch.tensor([[0.0, 0.0, 0.0], [0.50, 0.0, 0.0]])

    graph = torch_neighbor_graph_from_batch(
        batch, cutoff=0.45, reference_positions=reference
    )

    # 0-1 is inside the cutoff for the reference only, so the union has both
    # directions once each and the same topology serves both states.
    assert graph["edge_index"].shape[1] == 2
    current_vectors, reference_vectors = graph["edge_vectors"], graph["edge_vectors_2"]
    assert current_vectors.shape == reference_vectors.shape == (2, 3)
    assert torch.allclose(
        torch.sort(graph["edge_lengths"]).values, torch.tensor([0.40, 0.40])
    )
    assert torch.allclose(
        torch.sort(graph["edge_lengths_2"]).values, torch.tensor([0.50, 0.50])
    )
    # Vector and length agree for each state, and the two states differ.
    assert torch.allclose(
        torch.linalg.vector_norm(current_vectors, dim=-1), graph["edge_lengths"], atol=1e-6
    )
    assert torch.allclose(
        torch.linalg.vector_norm(reference_vectors, dim=-1), graph["edge_lengths_2"], atol=1e-6
    )
    assert not torch.allclose(current_vectors, reference_vectors)


def test_dual_geometry_graph_rejects_mismatched_shapes():
    from basinflow.geometry.graph import torch_neighbor_graph_from_batch

    with pytest.raises(ValueError, match="match in shape"):
        torch_neighbor_graph_from_batch(
            _batch([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
            cutoff=0.6,
            reference_positions=torch.zeros(3, 3),
        )
