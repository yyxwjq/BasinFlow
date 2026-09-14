"""Structure relaxation through any ASE calculator.

Relaxation deliberately goes through ``ase.calculators`` instead of naming a
potential format. The caller supplies a configured calculator, so exactly the
same code path relaxes with an empirical potential (EAM, EMT), a DFT code
(GPAW, VASP, ...), or a machine-learned potential (MACE, CHGNet, ...); this
module only drives the optimizer and converts the result back to a
:class:`~basinflow.data.records.StructureRecord`.

FixAtoms is not set here: ``StructureRecord.to_ase`` already derives it from
``movable_mask``, so fixed atoms stay fixed for every calculator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from basinflow.data.records import StructureRecord

OPTIMIZERS = ("FIRE", "BFGS", "LBFGS")


@dataclass(frozen=True)
class RelaxationConfig:
    """Optimizer settings. The potential itself comes from the caller."""

    fmax: float = 0.05
    steps: int = 200
    optimizer: str = "FIRE"

    def __post_init__(self) -> None:
        if self.fmax <= 0.0:
            raise ValueError("fmax must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")


def relax(
    structure: StructureRecord,
    calculator,
    config: RelaxationConfig | None = None,
    *,
    label: str = "relaxed",
) -> tuple[StructureRecord, bool, int]:
    """Relax ``structure`` with any ASE calculator.

    Returns ``(relaxed_structure, converged, optimizer_steps)``. The calculator
    is attached to a fresh ``Atoms`` object, so a factory is only needed when
    the calculator itself is stateful across trials.
    """
    from ase.optimize import BFGS, FIRE, LBFGS

    config = config or RelaxationConfig()
    optimizers = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}

    atoms = structure.to_ase()
    atoms.calc = calculator
    optimizer = optimizers[config.optimizer](atoms, logfile=None)
    converged = bool(optimizer.run(fmax=float(config.fmax), steps=int(config.steps)))
    steps = int(getattr(optimizer, "nsteps", 0))

    relaxed = StructureRecord.from_ase(
        atoms,
        structure_id=f"{structure.structure_id}:{label}",
        metadata={
            **structure.metadata,
            "relaxation": f"{config.optimizer.lower()}_via_ase",
            "relax_calculator": type(calculator).__name__,
            "relax_fmax": float(config.fmax),
            "relax_steps_limit": int(config.steps),
            "relax_steps": steps,
            "relax_converged": converged,
        },
    )
    return relaxed, converged, steps


def eam_calculator(potential_path: str | Path, form: str = "eam"):
    """Build an ASE EAM calculator.

    A convenience, not a restriction: pass any other calculator to :func:`relax`
    directly. ``form`` is one of ``eam``, ``alloy``, ``fs``, ``adp``.
    """
    from ase.calculators.eam import EAM

    if form not in {"eam", "alloy", "fs", "adp"}:
        raise ValueError(f"unsupported EAM form: {form!r}")
    return EAM(potential=str(potential_path), form=form)

def calculator_factory_of(calculator: Callable[[], Any] | None) -> Callable[[], Any] | None:
    """Normalise an optional calculator factory argument."""
    if calculator is not None and not callable(calculator):
        raise TypeError("calculator_factory must be callable")
    return calculator


def prepare_eam_potential(source_path: Path, output_dir: Path) -> Path:
    """Copy an EAM potential into a run directory after stripping inline comments.

    ASE's EAM reader rejects inline ``#`` comments, so a run-local copy is
    written before the potential is used.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(f"EAM potential not found: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / source_path.name
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    prepared_lines = [
        line if index < 3 else line.split("#", 1)[0].rstrip()
        for index, line in enumerate(source_lines)
    ]
    prepared_path.write_text("\n".join(prepared_lines) + "\n", encoding="utf-8")
    return prepared_path
