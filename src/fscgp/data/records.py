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

    ``movable_mask`` is a per-atom boolean mask where ``True`` marks atoms
    that may be displaced during generation or relaxation.  Legacy
    ``constraints`` input is still accepted as ``True`` = fixed and converted
    to ``movable_mask``.
    """

    structure_id: str
    species: list[str]
    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    charge: int | None = None       # user-populated; not auto-extracted by from_ase()
    movable_mask: np.ndarray | None = None
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
        if self.movable_mask is not None:
            self.movable_mask = np.asarray(self.movable_mask, dtype=bool)
            if self.movable_mask.shape != (self.n_atoms,):
                raise ValueError(
                    f"movable_mask must have shape ({self.n_atoms},), "
                    f"got {self.movable_mask.shape}"
                )
        if self.constraints is not None:
            legacy_constraints = np.asarray(self.constraints, dtype=bool)
            if legacy_constraints.shape != (self.n_atoms,):
                raise ValueError(
                    f"constraints must have shape ({self.n_atoms},), "
                    f"got {legacy_constraints.shape}"
                )
            legacy_movable = ~legacy_constraints
            if self.movable_mask is not None and not np.array_equal(
                self.movable_mask,
                legacy_movable,
            ):
                raise ValueError("movable_mask must equal ~constraints when both are provided")
            self.movable_mask = legacy_movable
        if self.movable_mask is None:
            self.movable_mask = np.ones((self.n_atoms,), dtype=bool)
        self.constraints = ~self.movable_mask
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
        fixed_mask = ~np.asarray(self.movable_mask, dtype=bool)
        if fixed_mask.any():
            # Lazy import — FixAtoms is only needed for ASE export, not at module load.
            from ase.constraints import FixAtoms

            fixed_indices = np.where(fixed_mask)[0]
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
        # Copy to avoid mutating self.metadata when ASE later modifies atoms.info.
        atoms.info.update(dict(self.metadata))

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
        available.  ``move_mask`` arrays are treated as ``True`` = movable.
        ASE ``FixAtoms`` constraints are used as a fallback and converted to
        ``movable_mask``. ``tags`` are preserved when present.
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

        movable_mask: np.ndarray | None = None
        if "move_mask" in atoms.arrays:
            movable_mask = np.asarray(atoms.arrays["move_mask"], dtype=bool)

        # Convert ASE FixAtoms constraints to a movable per-atom boolean mask
        if atoms.constraints:
            fixed = np.zeros(len(atoms), dtype=bool)
            for c in atoms.constraints:
                if hasattr(c, "get_indices"):
                    fixed[c.get_indices()] = True
            constraint_movable = ~fixed
            if movable_mask is not None and not np.array_equal(
                movable_mask,
                constraint_movable,
            ):
                raise ValueError("move_mask and FixAtoms constraints disagree")
            movable_mask = constraint_movable

        return cls(
            structure_id=structure_id,
            species=list(atoms.get_chemical_symbols()),
            positions=atoms.get_positions(),
            cell=atoms.get_cell().array,
            pbc=atoms.get_pbc(),
            tags=tags,
            movable_mask=movable_mask,
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
