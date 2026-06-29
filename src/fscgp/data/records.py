from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import CalculatorError, PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers

from fscgp.geometry.mic import normalize_cell, normalize_pbc


@dataclass
class StructureRecord:
    """Unified molecular/periodic structure record.

    ``cell`` and ``pbc`` are always normalized to ``(3, 3)`` and ``(3,)``
    arrays.  Periodic behavior must be decided from ``pbc``, not from the
    presence of a nonzero cell.

    ``constraints`` is a per-atom boolean mask where ``True`` marks atoms
    that are fixed (e.g. a structural skeleton).  Only unconstrained atoms
    should be displaced during generation or relaxation.
    """

    structure_id: str
    species: list[str]
    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    charge: int | None = None
    constraints: np.ndarray | None = None
    tags: np.ndarray | None = None
    energy: float | None = None
    forces: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.structure_id = str(self.structure_id)
        self.species = [str(s) for s in self.species]
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (N, 3), got {self.positions.shape}"
            )
        if len(self.species) != self.positions.shape[0]:
            raise ValueError(
                "species length must match number of positions"
            )

        self.cell = normalize_cell(self.cell)
        self.pbc = normalize_pbc(self.pbc)

        if self.tags is not None:
            self.tags = np.asarray(self.tags, dtype=int)
            if self.tags.shape != (self.n_atoms,):
                raise ValueError(
                    f"tags must have shape ({self.n_atoms},), "
                    f"got {self.tags.shape}"
                )
        if self.constraints is not None:
            self.constraints = np.asarray(self.constraints, dtype=bool)
            if self.constraints.shape != (self.n_atoms,):
                raise ValueError(
                    f"constraints must have shape ({self.n_atoms},), "
                    f"got {self.constraints.shape}"
                )
        if self.forces is not None:
            self.forces = np.asarray(self.forces, dtype=float)
            if self.forces.shape != self.positions.shape:
                raise ValueError(
                    f"forces must have shape {self.positions.shape}, "
                    f"got {self.forces.shape}"
                )

        self.metadata = dict(self.metadata or {})

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def atomic_numbers(self) -> np.ndarray:
        try:
            return np.asarray(
                [atomic_numbers[symbol] for symbol in self.species], dtype=int
            )
        except KeyError as exc:
            raise ValueError(f"unknown chemical symbol {exc.args[0]!r}") from exc

    def to_ase(self) -> Atoms:
        atoms = Atoms(
            symbols=self.species,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
        )
        if self.tags is not None:
            atoms.set_tags(self.tags)
        if self.constraints is not None and self.constraints.any():
            from ase.constraints import FixAtoms

            fixed_indices = np.where(self.constraints)[0]
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
        atoms.info.update(self.metadata)

        calc_kwargs: dict[str, Any] = {}
        if self.energy is not None:
            calc_kwargs["energy"] = float(self.energy)
        if self.forces is not None:
            calc_kwargs["forces"] = np.asarray(self.forces, dtype=float)
        if calc_kwargs:
            atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
        return atoms

    @classmethod
    def from_ase(
        cls,
        atoms: Atoms,
        structure_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> "StructureRecord":
        """Create a `StructureRecord` from an ASE `Atoms` object.

        Energy and forces are read from the attached calculator when
        available.  ASE ``FixAtoms`` constraints are converted to a
        per-atom boolean mask (``True`` = atom is fixed).
        ``tags`` are preserved when present.
        """
        merged_metadata = dict(metadata or {})
        merged_metadata.update(atoms.info)

        try:
            energy = float(atoms.get_potential_energy())
        except (RuntimeError, CalculatorError, PropertyNotImplementedError):
            energy = None

        try:
            forces = np.asarray(
                atoms.get_forces(apply_constraint=False), dtype=float
            )
        except (RuntimeError, CalculatorError, PropertyNotImplementedError):
            forces = None

        tags = atoms.get_tags() if "tags" in atoms.arrays else None

        # Convert ASE FixAtoms constraints to a per-atom boolean mask
        constraints: np.ndarray | None = None
        if atoms.constraints:
            fixed = np.zeros(len(atoms), dtype=bool)
            for c in atoms.constraints:
                if hasattr(c, "get_indices"):
                    fixed[c.get_indices()] = True
            if fixed.any():
                constraints = fixed

        return cls(
            structure_id=structure_id,
            species=list(atoms.get_chemical_symbols()),
            positions=atoms.get_positions(),
            cell=atoms.get_cell().array,
            pbc=atoms.get_pbc(),
            tags=tags,
            constraints=constraints,
            energy=energy,
            forces=forces,
            metadata=merged_metadata,
        )


@dataclass
class EventRecord:
    """A known event from one reactant basin to one product basin.

    ``active_atoms`` and ``event_direction`` are derived automatically
    from reactant/product displacements when available.  They may also
    be set manually for curated datasets.
    """

    event_id: str
    reactant_structure_id: str
    product_structure_id: str
    basin_id: str
    transition_state_structure_id: str | None = None
    atom_mapping: list[int] | None = None
    active_atoms: list[int] | None = None
    event_direction: np.ndarray | None = None
    validation_status: str = "known_valid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_id = str(self.event_id)
        self.reactant_structure_id = str(self.reactant_structure_id)
        self.product_structure_id = str(self.product_structure_id)
        self.basin_id = str(self.basin_id)
        if self.atom_mapping is not None:
            self.atom_mapping = [int(i) for i in self.atom_mapping]
        if self.active_atoms is not None:
            self.active_atoms = [int(i) for i in self.active_atoms]
        if self.event_direction is not None:
            self.event_direction = np.asarray(self.event_direction, dtype=float)
        self.metadata = dict(self.metadata or {})


@dataclass
class BasinRecord:
    """Groups all known events that originate from the same reactant basin."""

    basin_id: str
    reactant_structure_id: str
    known_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.basin_id = str(self.basin_id)
        self.reactant_structure_id = str(self.reactant_structure_id)
        self.known_event_ids = [str(eid) for eid in self.known_event_ids]
        self.metadata = dict(self.metadata or {})


@dataclass
class CandidateRecord:
    """Generated candidate event proposal, before and after relaxation."""

    candidate_id: str
    basin_id: str
    reactant_structure_id: str
    initial_seed_id: str
    generated_structure_id: str
    model_checkpoint: str
    sampling_config: dict[str, Any] = field(default_factory=dict)
    relaxed_structure_id: str | None = None
    cluster_id: str | None = None
    relaxation_config: dict[str, Any] | None = None
    active_atom_scores: np.ndarray | None = None
    predicted_active_atoms: list[int] | None = None
    event_direction: np.ndarray | None = None
    priority_score: float | None = None
    uncertainty: float | None = None
    status: str = "generated"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.candidate_id = str(self.candidate_id)
        self.basin_id = str(self.basin_id)
        self.reactant_structure_id = str(self.reactant_structure_id)
        self.initial_seed_id = str(self.initial_seed_id)
        self.generated_structure_id = str(self.generated_structure_id)
        if self.active_atom_scores is not None:
            self.active_atom_scores = np.asarray(self.active_atom_scores, dtype=float)
        if self.predicted_active_atoms is not None:
            self.predicted_active_atoms = [int(i) for i in self.predicted_active_atoms]
        if self.event_direction is not None:
            self.event_direction = np.asarray(self.event_direction, dtype=float)
        self.sampling_config = dict(self.sampling_config or {})
        self.metadata = dict(self.metadata or {})
