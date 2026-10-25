"""The in-memory frame type of the data layer.

:class:`Structure` holds one molecular or periodic frame: chemical species,
positions, cell, PBC flags and the atom-wise movement mask. It is built from an
:class:`ase.Atoms` object by :meth:`Structure.from_ase`, or directly from arrays.
"""
from __future__ import annotations
from dataclasses import dataclass
from ase import Atoms
from ase.data import atomic_numbers
import numpy as np


def _frame_array(name: str, value, default: np.ndarray, dtype) -> np.ndarray:
    """``value`` as an array shaped and typed like ``default``, else ``default``."""
    if value is None:
        return default
    array = np.asarray(value, dtype=dtype)
    if array.shape != default.shape:
        raise ValueError(f"{name} must have shape {default.shape}, got {array.shape}")
    return array


@dataclass
class Structure:
    """One molecular or periodic frame.

    Attributes:
        species: Chemical symbols, in the same atom order as ``positions``.
        positions: Atomic coordinates, shape ``(N, 3)``.
        cell: Cell vectors in ASE's row-vector convention, shape ``(3, 3)``. A
            missing cell is normalized to a zero matrix; an explicit cell must
            already have that shape.
        pbc: Per-axis periodicity, shape ``(3,)``, bool. A missing value is
            normalized to ``[False, False, False]``; an explicit value must
            already have that shape.
        movable_mask: Atom-wise flag, shape ``(N,)``, bool; ``True`` means the
            atom is free to move and ``False`` means it is fixed.
    """

    species: list[str]
    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    movable_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.species = list(self.species)
        self.positions = np.asarray(self.positions, dtype=float)
        self.cell = _frame_array("cell", self.cell, np.zeros((3, 3)), float)
        self.pbc = _frame_array("pbc", self.pbc, np.zeros(3, dtype=bool), bool)
        self.movable_mask = (
            np.ones(self.n_atoms, dtype=bool)
            if self.movable_mask is None
            else np.asarray(self.movable_mask, dtype=bool)
        )

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the frame."""
        return int(self.positions.shape[0])

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers of ``species``, shape ``(N,)``, dtype ``int``.
        """
        try:
            return np.asarray([atomic_numbers[symbol] for symbol in self.species], dtype=int)
        except KeyError as exc:
            raise ValueError(f"unknown chemical symbol {exc.args[0]!r}") from exc

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "Structure":
        """Build a ``Structure`` from an :class:`ase.Atoms` object.

        Args:
            atoms: Frame to normalize.

        Returns:
            A ``Structure`` holding the species, positions, cell and PBC flags,
            with ``movable_mask`` read off ``atoms.constraints`` and all-``True``
            when the frame carries no constraint.

        Raises:
            ValueError: If the resulting fields fail the ``Structure`` checks.
        """
        movable_mask: np.ndarray | None = None
        if atoms.constraints:
            fixed = np.zeros(len(atoms), dtype=bool)
            fixed[atoms.constraints[0].get_indices()] = True
            movable_mask = ~fixed
        return cls(
            species=list(atoms.get_chemical_symbols()),
            positions=atoms.get_positions(),
            cell=atoms.get_cell().array,
            pbc=atoms.get_pbc(),
            movable_mask=movable_mask,
        )
