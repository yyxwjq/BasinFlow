import numpy as np
import pytest

from basinflow.data.records import StructureRecord


def test_torch_graph_from_structure_uses_pbc_edge_vectors():
    torch = pytest.importorskip("torch")
    from basinflow.geometry.graph import torch_neighbor_graph_from_structure

    structure = StructureRecord(
        "pbc",
        ["H", "H"],
        [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]],
        cell=np.eye(3) * 10.0,
        pbc=[True, False, False],
    )

    graph = torch_neighbor_graph_from_structure(structure, cutoff=0.5)

    assert graph["edge_index"].shape == (2, 2)
    assert graph["edge_vectors"].shape == (2, 3)
    assert torch.allclose(
        torch.sort(graph["edge_lengths"]).values,
        torch.tensor([0.2, 0.2], dtype=graph["edge_lengths"].dtype),
        atol=1e-6,
    )
    assert not torch.allclose(
        graph["edge_vectors"][0],
        torch.as_tensor(structure.positions[1] - structure.positions[0], dtype=graph["edge_vectors"].dtype),
    )


def test_torch_batch_neighbor_graph_rebuilds_when_positions_change():
    torch = pytest.importorskip("torch")
    from basinflow.geometry.graph import torch_neighbor_graph_from_batch

    batch = {
        "reactant_positions": torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "cell": torch.eye(3).unsqueeze(0) * 10.0,
        "pbc": torch.zeros((1, 3), dtype=torch.bool),
        "batch": torch.zeros(3, dtype=torch.long),
        "atomic_numbers": torch.tensor([1, 1, 1], dtype=torch.long),
    }

    graph_a = torch_neighbor_graph_from_batch(batch, cutoff=0.6)
    batch["reactant_positions"][2] = torch.tensor([0.5, 0.0, 0.0])
    graph_b = torch_neighbor_graph_from_batch(batch, cutoff=0.6)

    assert graph_a["edge_index"].shape[1] == 2
    assert graph_b["edge_index"].shape[1] == 6


def test_torch_graph_builders_are_importable_from_the_geometry_submodule():
    pytest.importorskip("torch")
    from basinflow.geometry.graph import torch_neighbor_graph_from_batch

    assert callable(torch_neighbor_graph_from_batch)
