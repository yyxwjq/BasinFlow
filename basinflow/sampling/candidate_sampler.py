"""Basin-level, target-free product proposal sampling."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

import numpy as np

from basinflow.data.pyg import BasinDataset
from basinflow.data.catalog import EventCatalog
from basinflow.data.records import CandidateRecord, StructureRecord
from basinflow.geometry.mic import derive_active_atoms, derive_event_direction, minimum_image_displacement


def _torch():
    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("candidate sampling requires basinflow[models]") from exc
    return torch, DataLoader


@dataclass
class CandidateSamplingResult:
    candidates: list[CandidateRecord]
    generated_structures: dict[str, StructureRecord]


@dataclass
class CandidateSampler:
    model: Any
    init_generators: Iterable
    cutoff: float = 5.0
    num_steps: int = 8
    graph_update_interval: int = 1
    checkpoint_id: str = "uncheckpointed"
    device: Any = None
    active_threshold: float = 0.1
    bond_change_source: str = "seed"

    def __post_init__(self) -> None:
        self.init_generators = tuple(self.init_generators)
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.graph_update_interval <= 0:
            raise ValueError("graph_update_interval must be positive")
        if not np.isfinite(self.active_threshold) or self.active_threshold < 0:
            raise ValueError("active_threshold must be finite and non-negative")
        if self.bond_change_source not in {"seed", "oracle"}:
            raise ValueError("bond_change_source must be 'seed' or 'oracle'")

    def _build_graph(self, batch):
        from basinflow.geometry.graph import torch_neighbor_graph_from_batch

        return torch_neighbor_graph_from_batch(
            batch,
            cutoff=self.cutoff,
            positions_key="pos",
        )

    def _oracle_bond_change(self, catalog: EventCatalog, basin):
        """Diagnostic-only condition taken from the recorded product topology.

        This is the ceiling arm of the bond-change ablation: it answers "can the
        model hit the target if it already knows the event?" and must never be
        reported as a deployable basin-level result.
        """
        if self.bond_change_source != "oracle":
            return None
        if not getattr(self.model, "bond_change_condition", False):
            return None
        if len(basin.known_event_ids) != 1:
            raise ValueError("oracle bond-change diagnostics require one known event per basin")
        from basinflow.geometry.bonds import bond_adjacency, bond_change_labels
        from basinflow.data.pyg import _require_pyg

        _require_pyg()
        import torch

        target = catalog.event_target(basin.known_event_ids[0])
        reactant = target.reactant
        n_atoms = reactant.n_atoms
        labels = bond_change_labels(
            bond_adjacency(reactant.positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
            bond_adjacency(target.target_positions, reactant.atomic_numbers, reactant.cell, reactant.pbc),
        ).astype(np.int8)
        padded = torch.zeros((1, n_atoms, n_atoms), dtype=torch.int8)
        padded[0] = torch.as_tensor(labels, dtype=torch.int8)
        return padded

    def sample(self, catalog: EventCatalog, basin_id: str) -> CandidateSamplingResult:
        """Sample one candidate per initializer from a catalog basin.

        ``BasinDataset`` is the only construction path for inference,
        so target/product fields cannot reach the model.
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError("sample requires an EventCatalog")
        if self.model is None:
            raise ValueError("CandidateSampler.sample requires a model")
        torch, DataLoader = _torch()
        local_catalog = catalog.subset([basin_id])
        dataset = BasinDataset(
            local_catalog,
            self.init_generators,
            bond_change_condition=bool(getattr(self.model, "bond_change_condition", False)),
        )
        basin = local_catalog.basins[basin_id]
        reactant = local_catalog.structures[basin.reactant_structure_id]
        oracle_condition = self._oracle_bond_change(local_catalog, basin)
        generated: dict[str, StructureRecord] = {}
        candidates: list[CandidateRecord] = []
        self.model.eval()

        with torch.no_grad():
            for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
                if self.device is not None:
                    batch = batch.to(self.device)
                if oracle_condition is not None and "bond_change" in batch:
                    batch.bond_change = oracle_condition.to(batch.bond_change.device)
                positions = batch.pos
                last_output = None
                for step in range(self.num_steps):
                    batch.pos = positions
                    if not getattr(self.model, "builds_own_graph", False):
                        if step % self.graph_update_interval == 0:
                            for name, value in self._build_graph(batch).items():
                                setattr(batch, name, value)
                        source, target = batch.edge_index
                        edge_cells = batch.cell[batch.batch[source]]
                        offsets = torch.einsum(
                            "ei,eij->ej", batch.cell_offsets.to(positions.dtype), edge_cells
                        )
                        batch.edge_vectors = positions[target] - positions[source] + offsets
                        batch.edge_lengths = torch.linalg.norm(batch.edge_vectors, dim=1)
                    last_output = self.model(batch, t=step / self.num_steps)
                    if last_output["velocity"].shape != positions.shape:
                        raise ValueError("model velocity must have the same [N,3] shape as positions")
                    if not torch.isfinite(last_output["velocity"]).all():
                        raise FloatingPointError(f"non-finite velocity for basin {basin_id}, candidate {index}, step {step}")
                    positions = positions + last_output["velocity"] / float(self.num_steps)
                    if not torch.isfinite(positions).all():
                        raise FloatingPointError(f"non-finite positions for basin {basin_id}, candidate {index}, step {step}")
                    positions[~batch.movable_mask] = batch.reactant_pos[~batch.movable_mask]

                structure_id = f"{basin_id}:candidate_structure:{index}"
                final_positions = positions.detach().cpu().numpy().astype(float)
                final_positions[~reactant.movable_mask] = reactant.positions[~reactant.movable_mask]
                generated[structure_id] = StructureRecord(
                    structure_id=structure_id,
                    species=list(reactant.species),
                    positions=final_positions,
                    cell=reactant.cell.copy(),
                    pbc=reactant.pbc.copy(),
                    movable_mask=reactant.movable_mask.copy(),
                    tags=reactant.tags.copy() if reactant.tags is not None else None,
                    metadata={"basin_id": basin_id, "seed_id": batch.seed_id[0]},
                )
                displacement = minimum_image_displacement(
                    generated[structure_id].positions,
                    reactant.positions,
                    cell=reactant.cell,
                    pbc=reactant.pbc,
                )
                geometric_active = derive_active_atoms(displacement, self.active_threshold) & reactant.movable_mask
                if "active_logits" in last_output:
                    active_scores = torch.sigmoid(last_output["active_logits"]).detach().cpu().numpy()
                    active_scores[~reactant.movable_mask] = 0.0
                    active_source = "model_active_head"
                else:
                    active_scores = geometric_active.astype(float)
                    active_source = "final_mic_displacement"
                if "direction" in last_output:
                    event_direction = last_output["direction"].detach().cpu().numpy()
                    event_direction[~reactant.movable_mask] = 0.0
                    direction_source = "model_direction_head"
                else:
                    event_direction = derive_event_direction(displacement, active_mask=geometric_active)
                    direction_source = "final_mic_displacement"
                candidates.append(
                    CandidateRecord(
                        candidate_id=f"{basin_id}:candidate:{index}",
                        basin_id=basin_id,
                        reactant_structure_id=reactant.structure_id,
                        initial_seed_id=batch.seed_id[0],
                        generated_structure_id=structure_id,
                        model_checkpoint=self.checkpoint_id,
                        sampling_config={
                            "num_steps": self.num_steps,
                            "cutoff": self.cutoff,
                            "graph_update_interval": self.graph_update_interval,
                            "integrator": "euler_left_endpoint",
                            "active_threshold": self.active_threshold,
                            "evaluation_mode": "basin_level_no_oracle_no_relaxation",
                        },
                        active_atom_scores=active_scores,
                        predicted_active_atoms=[
                            int(atom_id) for atom_id in np.where(active_scores >= 0.5)[0]
                        ],
                        event_direction=event_direction,
                        metadata={
                            "seed_id": batch.seed_id[0],
                            "seed_type": batch.seed_type[0],
                            "seed_metadata": json.loads(batch.seed_metadata_json[0]),
                            "seed_displacement": batch.seed_displacement.detach().cpu().tolist(),
                            "active_atoms_source": active_source,
                            "event_direction_source": direction_source,
                        },
                    )
                )
        return CandidateSamplingResult(candidates=candidates, generated_structures=generated)
