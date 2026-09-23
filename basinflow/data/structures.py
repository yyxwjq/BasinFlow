"""The in-memory frame type of the data layer.

:class:`Structure` holds one molecular or periodic frame: chemical species,
positions, cell, PBC flags and the atom-wise movement mask. It is built from an
:class:`ase.Atoms` object by :meth:`Structure.from_ase`, written back to one by
:meth:`Structure.to_ase`, and constructed directly wherever an intermediate
geometry exists only as arrays (model output, seed, relaxed frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from basinflow.geometry.mic import normalize_cell, normalize_pbc


@dataclass
class Structure:
    """One molecular or periodic frame.

    Attributes:
        species: Chemical symbols, in the same atom order as ``positions``.
        positions: Atomic coordinates, shape ``(N, 3)``.
        cell: Cell vectors in ASE's row-vector convention, shape ``(3, 3)``.
            A missing cell is normalized to a zero matrix, which is what a
            molecular frame carries.
        pbc: Per-axis periodicity, shape ``(3,)``, bool. A missing value is
            normalized to ``[False, False, False]``. Periodicity is decided by
            this field alone, never by whether ``cell`` is zero.
        movable_mask: Atom-wise flag, shape ``(N,)``, bool; ``True`` means the
            atom is free to move and ``False`` means it is fixed. ``None`` is
            normalized to all ``True``. It records a given constraint (an ASE
            ``FixAtoms``), not a learning target: the model, loss and sampler
            read it as the single source of movement semantics. How a frame
            read from disk resolves it is described in :meth:`from_ase`.

    Raises:
        ValueError: If ``positions`` is not ``(N, 3)``, if ``species`` and
            ``positions`` disagree on ``N``, if ``cell`` or ``pbc`` has an
            unusable shape, or if ``movable_mask`` is not ``(N,)``. These are
            reachable from the direct construction sites as well as from
            :meth:`from_ase`, so they stay here rather than in one caller.
    """

    species: list[str]
    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    movable_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.species = [str(symbol) for symbol in self.species]
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {self.positions.shape}")
        if len(self.species) != self.positions.shape[0]:
            raise ValueError("species length must match number of positions")
        self.cell = normalize_cell(self.cell)
        self.pbc = normalize_pbc(self.pbc)
        if self.movable_mask is None:
            self.movable_mask = np.ones((self.n_atoms,), dtype=bool)
        else:
            self.movable_mask = np.asarray(self.movable_mask, dtype=bool)
            if self.movable_mask.shape != (self.n_atoms,):
                raise ValueError(
                    f"movable_mask must have shape ({self.n_atoms},), got {self.movable_mask.shape}"
                )

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the frame."""
        return int(self.positions.shape[0])

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers of ``species``, shape ``(N,)``, dtype ``int``.

        Raises:
            ValueError: If a symbol is not a known chemical symbol.
        """
        try:
            return np.asarray([atomic_numbers[symbol] for symbol in self.species], dtype=int)
        except KeyError as exc:
            raise ValueError(f"unknown chemical symbol {exc.args[0]!r}") from exc

    def to_ase(self) -> Atoms:
        """Return this frame as an :class:`ase.Atoms` object.

        The returned object carries a ``FixAtoms`` constraint over the atoms
        whose ``movable_mask`` entry is ``False``, and no calculator.
        """
        atoms = Atoms(
            symbols=self.species, positions=self.positions, cell=self.cell, pbc=self.pbc
        )
        fixed_mask = ~self.movable_mask
        if fixed_mask.any():
            # Imported here because FixAtoms is needed only on this path.
            from ase.constraints import FixAtoms

            atoms.set_constraint(FixAtoms(indices=np.where(fixed_mask)[0]))
        return atoms

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "Structure":
        """Build a ``Structure`` from an :class:`ase.Atoms` object.

        Args:
            atoms: Frame to normalize.

        Returns:
            A ``Structure`` holding a copy of the species, positions, cell, PBC
            flags and the movement mask resolved as described above.

        Returns:
            A ``Structure`` whose ``movable_mask`` is all-``True`` when the frame
            carries no constraint.

        Raises:
            ValueError: If the resulting fields fail the ``Structure`` checks.
        """
        movable_mask: np.ndarray | None = None
        if atoms.constraints:
            fixed = np.zeros(len(atoms), dtype=bool)
            for constraint in atoms.constraints:
                if hasattr(constraint, "get_indices"):
                    fixed[constraint.get_indices()] = True
            movable_mask = ~fixed
        return cls(
            species=list(atoms.get_chemical_symbols()),
            positions=atoms.get_positions(),
            cell=atoms.get_cell().array,
            pbc=atoms.get_pbc(),
            movable_mask=movable_mask,
        )
