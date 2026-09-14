import numpy as np

from basinflow.data.records import StructureRecord
from basinflow.geometry.graph import build_neighbor_graph


def test_nonperiodic_graph_edges_do_not_include_periodic_image_edges():
    structure = StructureRecord(
        structure_id="mol",
        species=["H", "H"],
        positions=[[0.1, 0, 0], [9.9, 0, 0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
    )

    graph = build_neighbor_graph(structure, cutoff=0.5)

    assert graph["edge_index"].shape == (2, 0)
    assert graph["cell_offsets"].shape == (0, 3)


def test_periodic_graph_edges_include_correct_cell_offsets():
    structure = StructureRecord(
        structure_id="pbc",
        species=["H", "H"],
        positions=[[0.1, 0, 0], [9.9, 0, 0]],
        cell=np.eye(3) * 10,
        pbc=[True, False, False],
    )

    graph = build_neighbor_graph(structure, cutoff=0.5)

    assert graph["edge_index"].shape[1] == 2
    offsets = {tuple(offset) for offset in graph["cell_offsets"].astype(int).tolist()}
    assert offsets == {(-1, 0, 0), (1, 0, 0)}
    assert np.allclose(graph["edge_lengths"], 0.2)


def test_moving_atoms_across_boundary_changes_edge_offsets():
    before = StructureRecord(
        structure_id="before",
        species=["H", "H"],
        positions=[[0.1, 0, 0], [9.9, 0, 0]],
        cell=np.eye(3) * 10,
        pbc=[True, False, False],
    )
    after = StructureRecord(
        structure_id="after",
        species=["H", "H"],
        positions=[[9.9, 0, 0], [0.1, 0, 0]],
        cell=np.eye(3) * 10,
        pbc=[True, False, False],
    )

    offsets_before = {tuple(offset) for offset in build_neighbor_graph(before, cutoff=0.5)["cell_offsets"].astype(int)}
    offsets_after = {tuple(offset) for offset in build_neighbor_graph(after, cutoff=0.5)["cell_offsets"].astype(int)}

    assert offsets_before == {(-1, 0, 0), (1, 0, 0)}
    assert offsets_after == {(-1, 0, 0), (1, 0, 0)}
    assert not np.array_equal(
        build_neighbor_graph(before, cutoff=0.5)["edge_vectors"],
        build_neighbor_graph(after, cutoff=0.5)["edge_vectors"],
    )


def test_edge_vectors_and_lengths_are_finite_and_consistent():
    structure = StructureRecord(
        structure_id="tri",
        species=["H", "H", "H"],
        positions=[[0, 0, 0], [0.4, 0, 0], [0, 0.3, 0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
    )

    graph = build_neighbor_graph(structure, cutoff=0.51)

    assert np.isfinite(graph["edge_vectors"]).all()
    assert np.isfinite(graph["edge_lengths"]).all()
    assert np.allclose(np.linalg.norm(graph["edge_vectors"], axis=1), graph["edge_lengths"])
