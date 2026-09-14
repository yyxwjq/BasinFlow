import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("periodic", [[False, False, False], [True, True, True], [True, False, True]])
def test_tensor_graph_matches_ase_in_skew_cell(periodic):
    from basinflow.geometry.radius_graph import radius_graph
    from basinflow.data.records import StructureRecord
    from basinflow.geometry.graph import build_neighbor_graph

    cell = np.array([[2.1, 0, 0], [1.1, 2.3, 0], [0.4, 0.8, 2.5]])
    positions = np.array([[0.1, 0.2, 0.1], [4.8, 0.3, 0.2], [1.0, 1.3, 1.2]])
    expected = build_neighbor_graph(StructureRecord("test", ["H"] * 3, positions, cell=cell, pbc=periodic), cutoff=2.8)
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
