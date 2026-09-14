from __future__ import annotations

import sys
import types

import numpy as np

from basinflow.data import StructureRecord


def _fake_optimize(monkeypatch, calls):
    class FakeOptimizer:
        def __init__(self, atoms, logfile):
            calls["atoms"] = atoms
            calls["logfile"] = logfile
            self.nsteps = 7

        def run(self, *, fmax, steps):
            calls["fmax"] = fmax
            calls["steps"] = steps
            return True

    module = types.ModuleType("ase.optimize")
    module.FIRE = module.BFGS = module.LBFGS = FakeOptimizer
    monkeypatch.setitem(sys.modules, "ase.optimize", module)


class _ZeroCalculator:
    """Any object ASE accepts: it only has to expose energy and forces."""

    def get_potential_energy(self, atoms):
        return 0.0

    def get_forces(self, atoms):
        return np.zeros((len(atoms), 3))


def _two_atom_structure():
    return StructureRecord(
        "input",
        ["Au", "Au"],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=np.eye(3) * 8.0,
        pbc=[True, True, False],
        movable_mask=[False, True],
    )


def test_prepare_eam_potential_writes_clean_run_local_copy(tmp_path):
    from basinflow.relaxation import prepare_eam_potential

    source = tmp_path / "source.eam.alloy"
    source.write_text(
        "header # retained\nline two # retained\nline three # retained\n1 2 # removed\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "sampling" / "eam"

    prepared = prepare_eam_potential(source, output_dir)

    assert prepared == output_dir / source.name
    assert source.read_text(encoding="utf-8") == "header # retained\nline two # retained\nline three # retained\n1 2 # removed\n"
    assert prepared.read_text(encoding="utf-8") == "header # retained\nline two # retained\nline three # retained\n1 2\n"


def test_relax_drives_any_ase_calculator_and_keeps_fixed_atoms(monkeypatch):
    from basinflow.relaxation import RelaxationConfig, relax

    calls = {}
    _fake_optimize(monkeypatch, calls)
    structure = _two_atom_structure()

    relaxed, converged, steps = relax(
        structure,
        _ZeroCalculator(),
        RelaxationConfig(fmax=0.05, steps=200, optimizer="FIRE"),
    )

    assert calls["fmax"] == 0.05
    assert calls["steps"] == 200
    # FixAtoms comes from StructureRecord.to_ase, not from this module
    assert calls["atoms"].constraints[0].get_indices().tolist() == [0]
    assert np.array_equal(relaxed.movable_mask, structure.movable_mask)
    assert relaxed.metadata["relax_calculator"] == "_ZeroCalculator"
    assert relaxed.metadata["relax_converged"] is True
    assert converged is True
    assert steps == 7


def test_relaxation_config_rejects_bad_settings():
    from basinflow.relaxation import RelaxationConfig

    for kwargs in ({"optimizer": "NOPE"}, {"fmax": 0.0}, {"steps": 0}):
        try:
            RelaxationConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"RelaxationConfig accepted {kwargs}")


def test_eam_calculator_rejects_an_unknown_form():
    from basinflow.relaxation import eam_calculator

    try:
        eam_calculator("whatever.eam.alloy", form="nope")
    except ValueError:
        return
    raise AssertionError("eam_calculator accepted an unknown form")
