import numpy as np

from basinflow.evaluation import (
    cluster_candidates,
    evaluate_basin_recall,
    nearest_product_match,
)
from basinflow.data.catalog import EventCatalog
from basinflow.data.records import BasinRecord, EventRecord
from basinflow.data.records import CandidateRecord, StructureRecord


def _structure(sid, x):
    return StructureRecord(
        sid,
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [x, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
        constraints=[True, False, True],
    )


def _candidate(cid, sid):
    return CandidateRecord(
        candidate_id=cid,
        basin_id="b",
        reactant_structure_id="r",
        initial_seed_id=f"{cid}:seed",
        generated_structure_id=sid,
        model_checkpoint="test",
    )


def _catalog(reactant, products):
    event_ids = [f"e{index}" for index in range(1, len(products) + 1)]
    return EventCatalog(
        structures={
            reactant.structure_id: reactant,
            **{product.structure_id: product for product in products},
        },
        events={
            event_id: EventRecord(
                event_id,
                reactant.structure_id,
                product.structure_id,
                "b",
                metadata={"source_file": f"event_{index}.extxyz"},
            )
            for index, (event_id, product) in enumerate(zip(event_ids, products), start=1)
        },
        basins={"b": BasinRecord("b", reactant.structure_id, event_ids)},
    )


def test_cluster_candidates_merges_near_duplicates_and_separates_products():
    reactant = _structure("r", 0.8)
    structures = {
        "c1": _structure("c1", 1.00),
        "c2": _structure("c2", 1.05),
        "c3": _structure("c3", 0.45),
    }
    candidates = [_candidate("cand1", "c1"), _candidate("cand2", "c2"), _candidate("cand3", "c3")]

    clusters = cluster_candidates(
        candidates,
        structures,
        reactant,
        threshold=0.1,
    )

    assert len(clusters) == 2
    assert clusters[0].candidate_ids == ["cand1", "cand2"]
    assert clusters[1].candidate_ids == ["cand3"]


def test_evaluate_basin_recall_reports_recalled_events_and_duplicate_rate():
    reactant = _structure("r", 0.8)
    p1 = _structure("p1", 1.0)
    p2 = _structure("p2", 0.4)
    structures = {
        "c1": _structure("c1", 1.02),
        "c2": _structure("c2", 1.04),
        "c3": _structure("c3", 0.4),
    }
    candidates = [_candidate("cand1", "c1"), _candidate("cand2", "c2"), _candidate("cand3", "c3")]
    catalog = _catalog(reactant, [p1, p2])

    report = evaluate_basin_recall(
        catalog,
        "b",
        candidates,
        structures,
        cluster_threshold=0.1,
        match_threshold=0.08,
    )

    assert report["num_known_events"] == 2
    assert report["num_candidates"] == 3
    assert report["num_clusters"] == 2
    assert report["recalled_event_ids"] == ["e1", "e2"]
    assert report["recall"] == 1.0
    assert report["duplicate_rate"] == 1 / 3


def test_nearest_product_match_returns_event_source_and_rmsd():
    reactant = _structure("r", 0.8)
    p1 = _structure("p1", 1.0)
    p2 = _structure("p2", 0.4)
    catalog = _catalog(reactant, [p1, p2])

    match = nearest_product_match(_structure("candidate", 0.42), catalog, "b")

    assert match["event_id"] == "e2"
    assert match["source_file"] == "event_2.extxyz"
    assert match["product"] is p2
    assert np.isclose(match["rmsd_angstrom"], 0.02)
