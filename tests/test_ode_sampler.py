"""The shared ODE sampler is MolGEN's solver, called the same way.

These tests pin the properties that make a BasinFlow rollout comparable with a
MolGEN one: the same library, the same method, the same tolerances, and the same
meaning for the step argument (a *saved-point* grid, not a step count).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from basinflow.sampling.ode import (  # noqa: E402
    DEFAULT_ATOL,
    DEFAULT_METHOD,
    DEFAULT_RTOL,
    METHODS,
    sample_ode,
)


def test_defaults_match_molgen():
    """MolGEN: dopri5, atol=1e-6, rtol=1e-3 (mdgen/transport/transport.py:655)."""
    assert DEFAULT_METHOD == "dopri5"
    assert DEFAULT_ATOL == 1e-6
    assert DEFAULT_RTOL == 1e-3
    assert set(METHODS) == {"dopri5", "euler"}


def test_it_solves_a_constant_field_exactly():
    field = lambda state, t: torch.ones_like(state)  # noqa: E731
    final, report = sample_ode(field, torch.zeros(3), num_steps=5)

    assert torch.allclose(final, torch.ones(3), atol=1e-5)
    assert report["integrator"] == "torchdiffeq:dopri5"
    assert report["num_steps"] == 5
    assert report["field_evaluations"] >= 5


def test_field_is_queried_at_many_different_times():
    """The field must see t, or a time-dependent field cannot be integrated."""
    seen = []

    def field(state, t):
        seen.append(float(t))
        return torch.zeros_like(state)

    sample_ode(field, torch.zeros(1), num_steps=20)

    assert seen
    assert min(seen) >= 0.0
    # Many distinct times spanning the interval: a single query at t = 0 would be
    # the direct-regression regime this replaced.
    assert len(set(round(value, 6) for value in seen)) > 5
    assert max(seen) >= 0.9


def test_the_saved_grid_is_the_step_argument():
    """``num_steps`` is the output grid; the solver picks its own step sizes."""
    calls = {"n": 0}

    def field(state, t):
        calls["n"] += 1
        return -20.0 * state

    _, coarse = sample_ode(field, torch.ones(1), num_steps=3)
    calls["n"] = 0
    _, fine = sample_ode(field, torch.ones(1), num_steps=61)

    assert coarse["num_steps"] == 3 and fine["num_steps"] == 61
    assert fine["field_evaluations"] >= 61
    # An adaptive solver's step count is chosen at run time; torchdiffeq does not
    # report it, and none of these numbers is a step count.
    assert coarse["integration_steps"] is None
    assert fine["integration_steps"] is None


def test_adaptive_solves_to_the_requested_tolerance_and_euler_does_not():
    """dx/dt = 3t^2 has the exact solution x(1) = 1.

    With MolGEN's tolerances the solve carries ~1e-6 of error, which is two orders
    of magnitude below the ~0.1 A RMSD differences these runs are judged on; a
    ten-point Euler grid does not.
    """
    field = lambda state, t: 3.0 * t**2 * torch.ones_like(state)  # noqa: E731

    adaptive, _ = sample_ode(field, torch.zeros(1), num_steps=10, method="dopri5")
    euler, report = sample_ode(field, torch.zeros(1), num_steps=10, method="euler")

    assert abs(float(adaptive) - 1.0) < 1e-5
    assert abs(float(euler) - 1.0) > 1e-2
    assert report["integration_steps"] == 9


def test_record_returns_the_saved_trajectory():
    field = lambda state, t: torch.ones_like(state)  # noqa: E731
    final, report = sample_ode(field, torch.zeros(2), num_steps=6, record=True)

    assert len(report["states"]) == 6
    assert torch.allclose(report["states"][0], torch.zeros(2))
    assert torch.allclose(report["states"][-1], final)
    # Monotone in t on the saved grid.
    heights = [float(state[0]) for state in report["states"]]
    assert heights == sorted(heights)


def test_a_single_saved_point_is_rejected_rather_than_doing_nothing():
    """torchdiffeq would silently return the initial condition."""
    with pytest.raises(ValueError, match="must be >= 2"):
        sample_ode(lambda state, t: torch.zeros_like(state), torch.zeros(1), num_steps=1)


def test_bad_settings_are_rejected():
    field = lambda state, t: torch.zeros_like(state)  # noqa: E731
    with pytest.raises(ValueError, match="integrator"):
        sample_ode(field, torch.zeros(1), method="rk4")
    with pytest.raises(ValueError, match="interval"):
        sample_ode(field, torch.zeros(1), t0=0.5, t1=0.5)
    with pytest.raises(ValueError, match="atol and rtol"):
        sample_ode(field, torch.zeros(1), atol=0.0)


def test_a_rough_field_fails_with_a_diagnosable_error_not_a_library_assertion():
    """An untrained/rough field makes the adaptive controller give up.

    torchdiffeq signals that by asserting on its own step size (``underflow in
    dt``). Leaking that as an ``AssertionError`` from inside a solver reads like a
    bug in the caller; it is a statement about the field, so it is re-raised with
    the interval, the tolerances and a way out.
    """
    field = lambda state, t: 1e9 * (state - 10.0 * torch.sin(1000 * t))  # noqa: E731

    with pytest.raises(FloatingPointError, match="could not advance"):
        sample_ode(field, torch.zeros(1), num_steps=8, atol=1e-9, rtol=1e-9)


def test_the_solver_controls_reported_in_the_recipe_are_the_ones_used():
    field = lambda state, t: torch.zeros_like(state)  # noqa: E731
    _, report = sample_ode(field, torch.zeros(1), num_steps=7, atol=2e-6, rtol=4e-3)

    assert report["atol"] == 2e-6
    assert report["rtol"] == 4e-3
    assert report["t0"] == 0.0 and report["t1"] == 1.0


def test_the_solve_stays_on_the_state_dtype_and_device():
    """MPS cannot represent float64, and torchdiffeq's time dtype defaults to it."""
    field = lambda state, t: torch.zeros_like(state)  # noqa: E731
    final, _ = sample_ode(field, torch.zeros(2), num_steps=5)
    assert final.dtype == torch.float32

    if torch.backends.mps.is_available():
        state = torch.zeros(2, device="mps")
        final, _ = sample_ode(field, state, num_steps=5)
        assert final.device.type == "mps"
