from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from basinflow.data.catalog import EventCatalog
from basinflow.data.records import CandidateRecord, StructureRecord
from basinflow.geometry.mic import pairwise_displacements


@dataclass
class CandidateCluster:
    cluster_id: str
    candidate_ids: list[str] = field(default_factory=list)
    representative_candidate_id: str | None = None
    best_match_event_id: str | None = None
    best_match_rmsd: float | None = None


def movable_mic_rmsd(
    a: StructureRecord,
    b: StructureRecord,
    reference: StructureRecord,
) -> float:
    """RMSD over movable atoms using reference cell/PBC for MIC wrapping."""
    if a.n_atoms != b.n_atoms or a.n_atoms != reference.n_atoms:
        raise ValueError("structures must have the same atom count")
    movable = np.asarray(reference.movable_mask, dtype=bool)
    if not movable.any():
        movable = np.ones((reference.n_atoms,), dtype=bool)
    disp = pairwise_displacements(
        a.positions,
        b.positions,
        cell=reference.cell,
        pbc=reference.pbc,
    )
    return float(np.sqrt(np.mean(np.sum(disp[movable] ** 2, axis=1))))


def kabsch_aligned_rmsd(
    a: StructureRecord,
    b: StructureRecord,
    reference: StructureRecord,
    *,
    mirror_allowed: bool = False,
) -> float:
    """Movable-atom RMSD after optimal rigid alignment.

    ReactOT and MolGEN both score a proposal after Kabsch alignment
    (``rmsd_core`` in ``mdgen/equivariant_wrapper.py``), so this is the
    quantity that is comparable with their reported 0.1-0.2 A numbers.  The
    unaligned :func:`movable_mic_rmsd` is a different, stricter metric.

    ``mirror_allowed=False`` (default) is a proper-rotation fit: the optimal
    rotation is corrected to ``det = +1``, so an inverted structure cannot be
    scored as a good match and chirality is respected.  ``mirror_allowed=True``
    reproduces the reference implementation's ``ignore_chirality=True``
    (``OAReactDiff/oa_reactdiff/analyze/rmsd.py:95``), which reflects the
    structure and keeps whichever of the two fits is lower.  Only the second
    convention is directly comparable with the published TS numbers;
    ``docs/17_reference_protocol_audit.md`` requires both to be reported.
    """
    if a.n_atoms != b.n_atoms or a.n_atoms != reference.n_atoms:
        raise ValueError("structures must have the same atom count")
    movable = np.asarray(reference.movable_mask, dtype=bool)
    if not movable.any():
        movable = np.ones((reference.n_atoms,), dtype=bool)
    first = np.asarray(a.positions, dtype=float)[movable]
    second = np.asarray(b.positions, dtype=float)[movable]
    first = first - first.mean(axis=0)
    second = second - second.mean(axis=0)
    mirrored = second.copy()
    mirrored[:, -1] = -mirrored[:, -1]
    u, _, vt = np.linalg.svd(first.T @ second)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    rmsd = float(np.sqrt(np.mean(np.sum((first @ rotation.T - second) ** 2, axis=1))))
    if not mirror_allowed:
        return rmsd
    u, _, vt = np.linalg.svd(first.T @ mirrored)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    reflected = float(np.sqrt(np.mean(np.sum((first @ rotation.T - mirrored) ** 2, axis=1))))
    return min(rmsd, reflected)


def nearest_product_match(
    candidate: StructureRecord,
    catalog: EventCatalog,
    basin_id: str,
) -> dict[str, Any]:
    """Match a candidate to its closest known product in one reactant basin."""
    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    basin = catalog.basins[basin_id]
    event_ids = basin.known_event_ids
    if not event_ids:
        raise ValueError(f"basin {basin_id!r} has no known products")
    reactant = catalog.structures[basin.reactant_structure_id]
    events = [catalog.events[event_id] for event_id in event_ids]
    products = [catalog.structures[event.product_structure_id] for event in events]

    best_index = min(
        range(len(products)),
        key=lambda index: movable_mic_rmsd(candidate, products[index], reactant),
    )
    event = events[best_index]
    product = products[best_index]
    return {
        "event_id": event.event_id,
        "source_file": event.metadata.get("source_file"),
        "product": product,
        "rmsd_angstrom": movable_mic_rmsd(candidate, product, reactant),
    }


def cluster_candidates(
    candidates: list[CandidateRecord],
    structures: dict[str, StructureRecord],
    reactant: StructureRecord,
    threshold: float = 0.25,
) -> list[CandidateCluster]:
    """Greedy no-relaxation clustering by movable-atom MIC RMSD."""
    clusters: list[CandidateCluster] = []
    representatives: list[StructureRecord] = []
    for candidate in candidates:
        structure = structures[candidate.generated_structure_id]
        assigned = False
        for cluster, representative in zip(clusters, representatives):
            if movable_mic_rmsd(structure, representative, reactant) <= threshold:
                cluster.candidate_ids.append(candidate.candidate_id)
                assigned = True
                break
        if not assigned:
            cluster = CandidateCluster(
                cluster_id=f"cluster_{len(clusters)}",
                candidate_ids=[candidate.candidate_id],
                representative_candidate_id=candidate.candidate_id,
            )
            clusters.append(cluster)
            representatives.append(structure)
    return clusters


def evaluate_basin_recall(
    catalog: EventCatalog,
    basin_id: str,
    candidates: list[CandidateRecord],
    structures: dict[str, StructureRecord],
    *,
    cluster_threshold: float = 0.25,
    match_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate candidate product recall against known basin products."""
    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    basin = catalog.basins[basin_id]
    reactant = catalog.structures[basin.reactant_structure_id]
    events = [catalog.events[event_id] for event_id in basin.known_event_ids]
    products = [catalog.structures[event.product_structure_id] for event in events]
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    clusters = cluster_candidates(
        candidates,
        structures,
        reactant,
        threshold=cluster_threshold,
    )

    recalled: set[str] = set()
    cluster_reports: list[dict[str, Any]] = []
    nearest_rmsds: list[float] = []
    for cluster in clusters:
        representative_id = cluster.representative_candidate_id or cluster.candidate_ids[0]
        representative = structures[candidate_by_id[representative_id].generated_structure_id]
        best_event_id = None
        best_rmsd = None
        for event, product in zip(events, products):
            rmsd = movable_mic_rmsd(representative, product, reactant)
            if best_rmsd is None or rmsd < best_rmsd:
                best_rmsd = rmsd
                best_event_id = event.event_id
        if best_rmsd is not None:
            nearest_rmsds.append(float(best_rmsd))
        if best_rmsd is not None and best_rmsd <= match_threshold and best_event_id is not None:
            recalled.add(best_event_id)
            cluster.best_match_event_id = best_event_id
            cluster.best_match_rmsd = best_rmsd
        cluster_reports.append(
            {
                "cluster_id": cluster.cluster_id,
                "candidate_ids": list(cluster.candidate_ids),
                "representative_candidate_id": representative_id,
                "nearest_event_id": best_event_id,
                "nearest_event_rmsd": best_rmsd,
                "best_match_event_id": cluster.best_match_event_id,
                "best_match_rmsd": cluster.best_match_rmsd,
            }
        )

    num_known = len(events)
    num_candidates = len(candidates)
    num_clusters = len(clusters)
    return {
        "basin_id": basin_id,
        "num_known_events": num_known,
        "num_candidates": num_candidates,
        "num_clusters": num_clusters,
        "best_candidate_rmsd": min(nearest_rmsds) if nearest_rmsds else None,
        "mean_cluster_best_rmsd": (
            float(np.mean(nearest_rmsds)) if nearest_rmsds else None
        ),
        "recalled_event_ids": [
            event.event_id for event in events if event.event_id in recalled
        ],
        "recall": float(len(recalled) / num_known) if num_known else 0.0,
        "duplicate_rate": (
            float((num_candidates - num_clusters) / num_candidates)
            if num_candidates
            else 0.0
        ),
        "clusters": cluster_reports,
    }
