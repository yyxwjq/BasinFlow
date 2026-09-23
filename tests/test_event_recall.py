import numpy as np
import pytest

from basinflow.evaluation import (
    cluster_candidates,
    evaluate_event_recall,
    nearest_product_match,
)
from basinflow.evaluation.basin_recall import comparison_events
from basinflow.data import Event, EventCatalog, Structure
from basinflow.sampling import Candidate


def _structure(x):
    return Structure(
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [x, 0.0, 0.0], [1.6, 0.0, 0.0]],
        cell=np.eye(3) * 10,
        pbc=[False, False, False],
        movable_mask=[False, True, False],
    )


def _candidate(cid, sid):
    return Candidate(
        candidate_id=cid,
        generated_structure_id=sid,
        seed_id=f"{cid}:seed",
        seed_type="gaussian_movable",
    )


def _catalog(reactant, products):
    event_ids = [str(index) for index in range(len(products))]
    return EventCatalog(
        events={
            event_id: Event(
                event_id, reactant, product,
                metadata={"source_file": f"event_{event_id}.extxyz"},
            )
            for event_id, product in zip(event_ids, products)
        },
    )


def test_cluster_candidates_merges_near_duplicates_and_separates_products():
    reactant = _structure(0.8)
    structures = {
        "c1": _structure(1.00),
        "c2": _structure(1.05),
        "c3": _structure(0.45),
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


def test_evaluate_event_recall_reports_recall_and_duplicate_rate():
    reactant = _structure(0.8)
    p1 = _structure(1.0)
    p2 = _structure(0.4)
    structures = {
        "c1": _structure(1.02),
        "c2": _structure(1.04),
        "c3": _structure(0.55),
    }
    candidates = [_candidate("cand1", "c1"), _candidate("cand2", "c2"), _candidate("cand3", "c3")]
    catalog = _catalog(reactant, [p1, p2])

    # One event is scored against its own known product: event "0" wants p1, and
    # both its clusters land within the match threshold.
    report = evaluate_event_recall(
        catalog,
        "0",
        candidates,
        structures,
        cluster_threshold=0.1,
        match_threshold=0.05,
    )
    assert report["event_id"] == "0"
    assert report["num_known_events"] == 1
    assert report["num_candidates"] == 3
    assert report["num_clusters"] == 2
    assert report["recalled_event_ids"] == ["0"]
    assert report["recall"] == 1.0
    assert report["duplicate_rate"] == 1 / 3

    # Event "1" is recalled by the one cluster that lands on its product p2, and
    # not by the two clusters that sit near p1.
    other = evaluate_event_recall(
        catalog, "1", candidates, structures, cluster_threshold=0.1, match_threshold=0.16
    )
    assert other["recalled_event_ids"] == ["1"]
    assert other["duplicate_rate"] == 1 / 3

    # A threshold nothing satisfies leaves the event unrecovered rather than
    # silently matching the nearest cluster.
    strict = evaluate_event_recall(
        catalog, "1", candidates, structures, cluster_threshold=0.1, match_threshold=0.1
    )
    assert strict["recalled_event_ids"] == []
    assert strict["recall"] == 0.0


def test_nearest_product_match_returns_event_source_and_rmsd():
    reactant = _structure(0.8)
    p1 = _structure(1.0)
    p2 = _structure(0.4)
    catalog = _catalog(reactant, [p1, p2])

    match = nearest_product_match(_structure(0.42), catalog.events["1"])

    assert match["event_id"] == "1"
    assert match["source_file"] == "event_1.extxyz"
    assert match["product"] is p2
    assert np.isclose(match["rmsd_angstrom"], 0.02)


def test_group_comparison_scores_against_every_event_of_a_basin():
    """A candidate for one event can be recalled by another event's product."""
    reactant = _structure(0.8)
    # Two events of one basin whose products sit on either side of the reactant.
    near = _structure(0.85)
    far = _structure(0.40)
    catalog = EventCatalog(
        events={
            "0": Event("0", reactant, far),
            "1": Event("1", reactant, near),
        },
    )
    # The candidate reproduces event 1's product, not the event it was sampled for.
    structures = {"c1": _structure(0.86)}
    candidates = [_candidate("cand1", "c1")]
    groups = {"0": "basin-a", "1": "basin-a"}

    event_level = evaluate_event_recall(
        catalog, "0", candidates, structures, cluster_threshold=0.05, match_threshold=0.02
    )
    basin_level = evaluate_event_recall(
        catalog,
        "0",
        candidates,
        structures,
        cluster_threshold=0.05,
        match_threshold=0.02,
        groups=groups,
    )

    # Scored against event 0 alone the candidate misses; against its basin the
    # known product of event 1 is found, and the report says which event matched.
    assert event_level["recalled_event_ids"] == []
    assert event_level["num_known_events"] == 1
    assert basin_level["recalled_event_ids"] == ["1"]
    assert basin_level["num_known_events"] == 2
    assert basin_level["recall"] == 0.5


def test_group_comparison_requires_the_event_to_be_in_the_table():
    reactant = _structure(0.8)
    catalog = _catalog(reactant, [_structure(1.0)])
    structures = {"c1": _structure(1.0)}
    candidates = [_candidate("cand1", "c1")]

    with pytest.raises(KeyError, match="missing from the grouping table"):
        evaluate_event_recall(
            catalog, "0", candidates, structures, groups={"9": "basin-a"}
        )


def test_comparison_events_default_to_the_event_itself():
    reactant = _structure(0.8)
    catalog = _catalog(reactant, [_structure(1.0)])

    events = comparison_events(catalog, "0")

    assert [event.event_id for event in events] == ["0"]
