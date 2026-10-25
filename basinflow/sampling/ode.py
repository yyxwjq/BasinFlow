"""ODE sampling through ``torchdiffeq``, called the way MolGEN calls it.

MolGEN's sampler is ``torchdiffeq.odeint`` with ``method="dopri5"`` over a
``linspace(t0, t1, num_steps)`` output grid (``mdgen/transport/integrators.py:77``
and ``mdgen/transport/transport.py:655``), with ``atol=1e-6``/``rtol=1e-3`` and
``--sampling_method`` defaulting to ``dopri5`` (``mdgen/parsing.py:78``). This
module calls the same library with the same defaults.

``num_steps`` is the number of *saved* points, as MolGEN's ``--inference_steps``:
the solver picks its own step sizes from ``atol``/``rtol`` and the output grid is
dense interpolation. It must be at least 2.
"""
from __future__ import annotations

from typing import Any, Callable

try:
    import torch
    from torchdiffeq import odeint
except ImportError:  # pragma: no cover - exercised in dependency-free installs
    torch = None
    odeint = None

DEFAULT_METHOD = "dopri5"
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-3
METHODS = ("dopri5", "euler", "midpoint")


def require_odeint() -> None:
    if odeint is None:
        raise ImportError(
            "ODE sampling requires the optional 'models' dependencies: install basinflow[models]"
        )


def sample_ode(
    field: Callable[[Any, Any], Any],
    initial: Any,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    num_steps: int = 50,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    method: str = DEFAULT_METHOD,
    max_num_steps: int = 100000,
) -> tuple[Any, dict[str, Any]]:
    """Integrate ``dx/dt = field(x, t)`` from ``t0`` to ``t1``.

    ``field`` receives the current state and the current time (a 0-dim tensor) and
    returns the state-shaped velocity. Returns ``(final_state, report)``.
    """
    require_odeint()
    if method not in METHODS:
        raise ValueError(f"integrator must be one of {METHODS}, got {method!r}")
    if int(num_steps) < 2:
        raise ValueError(
            f"num_steps is the number of saved points and must be >= 2, got {num_steps}; "
            "a single point is the initial condition and performs no integration"
        )
    if not 0.0 <= float(t0) < float(t1) <= 1.0:
        raise ValueError(f"integration interval must satisfy 0 <= t0 < t1 <= 1, got ({t0}, {t1})")
    if atol <= 0.0 or rtol <= 0.0:
        raise ValueError("atol and rtol must be positive")
    if max_num_steps <= 0:
        raise ValueError("max_num_steps must be positive")

    calls = {"n": 0}

    def wrapped(t, state):
        calls["n"] += 1
        return field(state, t)

    grid = torch.linspace(float(t0), float(t1), int(num_steps), device=_device(initial))
    try:
        # ``max_num_steps`` bounds a runaway adaptive solve; the fixed-step solvers
        # (``euler``, ``midpoint``) do not take it: their step count is the saved
        # grid by construction.
        #
        # ``dtype`` is the solver's *time* dtype (float64 by default), and float64 is
        # not representable on MPS: torchdiffeq would fail building its tolerance
        # tensors. Following the state's dtype keeps the solve on the device the
        # caller chose, and float32 is what these states carry anyway.
        options = (
            {"max_num_steps": int(max_num_steps), "dtype": _time_dtype(initial)}
            if method == "dopri5" else None
        )
        trajectory = odeint(
            wrapped,
            initial,
            grid,
            method=method,
            atol=float(atol),
            rtol=float(rtol),
            options=options,
        )
    except (AssertionError, RuntimeError) as error:
        # torchdiffeq's adaptive controller asserts on its own step size
        # (``underflow in dt``) once the local error estimate keeps rejecting
        # steps; report that as a field problem and name a fixed-step integrator.
        raise FloatingPointError(
            f"the adaptive {method} solver could not advance over [{float(t0)}, {float(t1)}] "
            f"(atol={float(atol):g}, rtol={float(rtol):g}, {calls['n']} field evaluations): "
            f"{error}. The learned field is most likely too stiff or too rough to integrate; "
            "check the checkpoint, or sample it with integrator='euler'."
        ) from error
    report = {
        "integrator": f"torchdiffeq:{method}",
        "atol": float(atol),
        "rtol": float(rtol),
        "num_steps": int(num_steps),
        "t0": float(t0),
        "t1": float(t1),
        "field_evaluations": calls["n"],
        # tol/interval are integration controls, not reports.
        "integration_steps": _integration_steps(t1, t0, num_steps, method),
    }
    return trajectory[-1], report


def _device(state):
    return state.device if hasattr(state, "device") else None


def _time_dtype(state):
    """Time dtype for the adaptive solver: float64 unless the state cannot hold it."""
    dtype = getattr(state, "dtype", torch.float32)
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


def _integration_steps(t1: float, t0: float, num_steps: int, method: str) -> int | None:
    """Solver steps actually taken: the saved-point count for a fixed-step method.

    An adaptive solver's step count is chosen at run time and is not reported by
    ``torchdiffeq``; ``num_steps`` must not be misread as it.
    """
    return int(num_steps) - 1 if method == "euler" else None
