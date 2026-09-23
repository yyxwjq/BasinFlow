"""Per-event, target-free product proposal sampling.

This path must never see a known product: the samples come from
:class:`~basinflow.data.samples.EventProposalDataset`, which constructs no ``target_*``
field at all, so an oracle cannot reach the model even by accident. The one
exception is an explicitly labelled diagnostic (``bond_change_source="oracle"``),
which is gated below and must never be reported as a deployable result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from basinflow.data import EventCatalog, EventProposalDataset, Structure, event_target
from basinflow.sampling.ode import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_METHOD, sample_ode


def _torch():
    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("candidate sampling requires basinflow[models]") from exc
    return torch, DataLoader


@dataclass
class Candidate:
    """One generated proposal: what it is, where its geometry lives, how it was made.

    Deliberately not a data-layer record. Relaxation, clustering, scoring and
    validation status belong to the stages that produce them, and the earlier
    record carried fields for all of them that nothing ever wrote.
    """

    candidate_id: str
    generated_structure_id: str
    seed_id: str
    seed_type: str
    sampling_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSamplingResult:
    candidates: list[Candidate]
    generated_structures: dict[str, Structure]


@dataclass
class CandidateSampler:
    model: Any
    init_generators: Iterable
    cutoff: float = 5.0
    # Number of saved points on the t grid, as in MolGEN's ``--inference_steps``:
    # the adaptive solver picks its own step sizes. Not an Euler step count.
    num_steps: int = 50
    checkpoint_id: str = "uncheckpointed"
    device: Any = None
    bond_change_source: str = "seed"
    integrator: str = DEFAULT_METHOD
    solver_atol: float = DEFAULT_ATOL
    solver_rtol: float = DEFAULT_RTOL

    def __post_init__(self) -> None:
        self.init_generators = tuple(self.init_generators)
        if self.num_steps < 2:
            raise ValueError(
                "num_steps is the number of saved integration points and must be >= 2"
            )
        if self.bond_change_source not in {"seed", "oracle"}:
            raise ValueError("bond_change_source must be 'seed' or 'oracle'")

    def _build_graph(self, batch):
        from basinflow.geometry.graph import torch_neighbor_graph_from_batch

        return torch_neighbor_graph_from_batch(batch, cutoff=self.cutoff, positions_key="pos")

    def _oracle_bond_change(self, catalog: EventCatalog, event_id: str):
        """Diagnostic-only condition taken from the recorded product topology.

        This is the ceiling arm of the bond-change ablation: it answers "can the
        model hit the target if it already knows the event?" and must never be
        reported as a deployable event-level result.
        """
        if self.bond_change_source != "oracle":
            return None
        if not getattr(self.model, "bond_change_condition", False):
            return None
        event = catalog.events[event_id]
        from basinflow.data import require_pyg
        from basinflow.geometry.bonds import bond_adjacency, bond_change_labels

        require_pyg()
        import torch

        reactant = event.reactant
        labels = bond_change_labels(
            bond_adjacency(
                reactant.positions, reactant.atomic_numbers, reactant.cell, reactant.pbc
            ),
            bond_adjacency(
                event_target(event).target_positions,
                reactant.atomic_numbers,
                reactant.cell,
                reactant.pbc,
            ),
        ).astype(np.int8)
        padded = torch.zeros((1, reactant.n_atoms, reactant.n_atoms), dtype=torch.int8)
        padded[0] = torch.as_tensor(labels, dtype=torch.int8)
        return padded

    def sample(self, catalog: EventCatalog, event_id: str) -> CandidateSamplingResult:
        """Sample one candidate per initializer from one catalog event."""
        if not isinstance(catalog, EventCatalog):
            raise TypeError("sample requires an EventCatalog")
        if self.model is None:
            raise ValueError("CandidateSampler.sample requires a model")
        torch, DataLoader = _torch()
        local = catalog.select([event_id])
        dataset = EventProposalDataset(
            local,
            self.init_generators,
            bond_change_condition=bool(getattr(self.model, "bond_change_condition", False)),
        )
        reactant = local.events[event_id].reactant
        oracle_condition = self._oracle_bond_change(local, event_id)
        generated: dict[str, Structure] = {}
        candidates: list[Candidate] = []
        self.model.eval()

        with torch.no_grad():
            for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
                if self.device is not None:
                    batch = batch.to(self.device)
                if oracle_condition is not None and "bond_change" in batch:
                    batch.bond_change = oracle_condition.to(batch.bond_change.device)
                frozen = ~batch.movable_mask

                def field(state, time):
                    """One network query at the solver's requested (state, time)."""
                    batch.pos = state
                    batch.flow_time = time.reshape(1, 1).expand(batch.num_graphs, 1)
                    if not getattr(self.model, "builds_own_graph", False):
                        # EGNN does not build its own graph, so it is rebuilt here on
                        # *every* field evaluation -- which is what an adaptive solver
                        # requires, since its steps are neither uniform nor known ahead.
                        # PaiNN sets ``builds_own_graph`` and rebuilds internally, as
                        # MolGEN's backbone does.
                        for name, value in self._build_graph(batch).items():
                            setattr(batch, name, value)
                    velocity = self.model(batch, t=batch.flow_time.squeeze(-1))["velocity"]
                    if velocity.shape != state.shape:
                        raise ValueError("model velocity must have the same [N,3] shape as positions")
                    if not torch.isfinite(velocity).all():
                        raise FloatingPointError(
                            f"non-finite velocity for event {event_id}, candidate {index}, at t={float(time)}"
                        )
                    # Fixed atoms are held by zeroing their velocity inside the field,
                    # so the constraint holds at every solver stage instead of being
                    # repaired after a step.
                    return torch.where(frozen[:, None], torch.zeros_like(velocity), velocity)

                try:
                    positions, integration = sample_ode(
                        field,
                        batch.pos,
                        num_steps=self.num_steps,
                        atol=self.solver_atol,
                        rtol=self.solver_rtol,
                        method=self.integrator,
                    )
                except FloatingPointError as error:
                    raise FloatingPointError(
                        f"event {event_id}, candidate {index}: {error}"
                    ) from error
                if not torch.isfinite(positions).all():
                    raise FloatingPointError(
                        f"non-finite positions for event {event_id}, candidate {index}"
                    )

                structure_id = f"{event_id}:candidate_structure:{index}"
                final_positions = positions.detach().cpu().numpy().astype(float)
                final_positions[~reactant.movable_mask] = reactant.positions[~reactant.movable_mask]
                generated[structure_id] = Structure(
                    species=list(reactant.species),
                    positions=final_positions,
                    cell=reactant.cell.copy(),
                    pbc=reactant.pbc.copy(),
                    movable_mask=reactant.movable_mask.copy(),
                )
                candidates.append(
                    Candidate(
                        candidate_id=f"{event_id}:candidate:{index}",
                        generated_structure_id=structure_id,
                        seed_id=batch.seed_id[0],
                        seed_type=batch.seed_type[0],
                        sampling_config={
                            "num_steps": self.num_steps,
                            "cutoff": self.cutoff,
                            "integrator": integration["integrator"],
                            "solver_atol": integration["atol"],
                            "solver_rtol": integration["rtol"],
                            "field_evaluations": integration["field_evaluations"],
                            "checkpoint_id": self.checkpoint_id,
                            "bond_change_source": self.bond_change_source,
                            "evaluation_mode": "event_level_no_oracle_no_relaxation",
                        },
                    )
                )
        return CandidateSamplingResult(candidates=candidates, generated_structures=generated)
