"""The bridge sampler must integrate the reference ODE to the reference metric.

The decisive property is checkable without training: a model whose residual is
identically zero produces a zero velocity field, so the integrator must return
the initial condition -- the midpoint ``(R + P) / 2`` -- and the reported
``aligned_rmsd_angstrom`` must equal the midpoint-versus-TS RMSD. Any sign or
time-scaling mistake in the reparameterised field changes that number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from basinflow.data import Event, EventCatalog, Structure
from basinflow.evaluation.bridge import run_bridge_trials
from basinflow.models.sb import SBSchedule

SYMBOLS = ["C", "H", "O", "H"]
SCHEDULE_KWARGS = dict(timesteps=32, beta_max=0.3, power=0.5, inv_power=1.0)


def _catalog() -> EventCatalog:
    rng = np.random.default_rng(3)
    built = {}
    for index in range(2):
        reactant = rng.normal(size=(len(SYMBOLS), 3))
        reactant -= reactant.mean(axis=0)
        product = reactant + 0.5
        product -= product.mean(axis=0)
        state = 0.5 * (reactant + product) + 0.08
        state -= state.mean(axis=0)
        built[str(index)] = Event(
            event_id=str(index),
            reactant=Structure(SYMBOLS, reactant),
            product=Structure(SYMBOLS, product),
            transition_state=Structure(SYMBOLS, state),
            metadata={"source_file": f"event_{index}.traj"},
        )
    return EventCatalog(events=built)


class _ZeroModel(torch.nn.Module):
    """Emits an identically zero bridge residual."""

    builds_own_graph = True
    bridge = True

    def forward(self, batch, t=None):
        return {"scaled": torch.zeros_like(batch.pos)}


def _schedule() -> SBSchedule:
    return SBSchedule(**SCHEDULE_KWARGS)


def _run(tmp_path: Path, model, **overrides) -> dict:
    parameters = dict(
        model=model,
        catalog=_catalog(),
        output_dir=tmp_path / "bridge",
        cutoff=10.0,
        num_steps=4,
        trials_per_event=1,
        sampling_seed=11,
        integrator="euler",
        solver_atol=1e-2,
        solver_rtol=1e-2,
        schedule=_schedule(),
    )
    parameters.update(overrides)
    return run_bridge_trials(**parameters)


def test_zero_field_returns_the_midpoint_prior(tmp_path: Path) -> None:
    summary = _run(tmp_path, _ZeroModel())
    assert summary["evaluation_protocol"] == "reactot_bridge_ode"
    assert summary["num_events"] == 2
    assert summary["num_trials"] == 2
    assert summary["integrator"] == "euler"
    # The field vanishes, so the proposal is the source and the two metrics agree.
    assert summary["aligned_rmsd_angstrom"]["mean"] == pytest.approx(
        summary["source_rmsd_angstrom"]["mean"], abs=1e-6
    )


def test_summary_reports_both_protocols(tmp_path: Path) -> None:
    summary = _run(tmp_path, _ZeroModel())
    block = summary["reactot_reference_protocol"]
    assert block["label"] == "reactot_reference_protocol"
    assert block["rmsd_angstrom"]["count"] == 2
    assert block["uncapped_rmsd_angstrom"]["count"] == 2
    # The strict column above is unchanged and reported next to the reference one.
    assert summary["clipped_aligned_rmsd_angstrom"]["mean"] == pytest.approx(
        min(summary["aligned_rmsd_angstrom"]["mean"], 1.0), abs=1e-6
    )


def test_artifacts_are_written(tmp_path: Path) -> None:
    _run(tmp_path, _ZeroModel())
    sampling = tmp_path / "bridge" / "sampling"
    assert (sampling / "summary.json").is_file()
    header = (sampling / "trial_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    for column in (
        "aligned_rmsd_angstrom",
        "reactot_reference_rmsd_angstrom",
        "chirality_gain_angstrom",
        "source_rmsd_angstrom",
    ):
        assert column in header
    summary = json.loads((sampling / "summary.json").read_text(encoding="utf-8"))
    assert summary["bridge"]["beta"] > 0


def test_movable_mask_zeroes_the_field(tmp_path: Path) -> None:
    """A fixed atom must not move even when the model predicts a displacement."""

    class _ConstantModel(torch.nn.Module):
        builds_own_graph = True
        bridge = True

        def forward(self, batch, t=None):
            return {"scaled": torch.ones_like(batch.pos)}

    catalog = _catalog()
    frozen = catalog.events["0"].reactant
    frozen.movable_mask = np.array([True, False, False, False])
    summary = _run(tmp_path, _ConstantModel(), catalog=catalog)
    assert summary["num_trials"] == 2
    assert summary["fixed_atom_max_motion_angstrom"] == pytest.approx(0.0, abs=1e-6)


def test_schedule_is_required(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        run_bridge_trials(
            model=_ZeroModel(),
            catalog=_catalog(),
            output_dir=tmp_path / "x",
            cutoff=10.0,
            num_steps=4,
        )
