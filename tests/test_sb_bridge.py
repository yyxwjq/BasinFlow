"""The Schrodinger-bridge algebra must match the ReactOT reference.

These tests pin the *equations*, not a trained run: the beta schedule shape, the
Gaussian-product coefficients, the forward/backward consistency of
``compute_label`` and ``compute_pred_x0``, the ``sigma = 0`` determinism, and the
sign/scale of the reparameterised ODE field that ``sample_ode`` integrates.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from basinflow.models.sb import (
    DEFAULT_BETA_MAX,
    DEFAULT_INV_POWER,
    DEFAULT_POWER,
    DEFAULT_T_EPS,
    DEFAULT_TIMESTEPS,
    SBSchedule,
    compute_gaussian_product_coef,
    compute_label,
    compute_pred_x0,
    make_beta_schedule,
    make_bridge_ode_field,
    posterior_step,
    q_sample,
    remove_mean_batch,
    space_indices,
)


def _schedule(**overrides) -> SBSchedule:
    parameters = dict(
        timesteps=64,
        beta_max=DEFAULT_BETA_MAX,
        power=DEFAULT_POWER,
        inv_power=DEFAULT_INV_POWER,
    )
    parameters.update(overrides)
    return SBSchedule(**parameters)


def test_reference_beta_schedule_is_constant() -> None:
    """With linear_start == linear_end the ramp collapses to a constant.

    This is what makes the reference's ``betas[:-1] == betas[1:]`` assertion in
    ``ode_sampling`` hold, and therefore what makes ``beta = betas[0] * T`` a
    valid scalar in the ODE field.
    """
    schedule = _schedule(timesteps=DEFAULT_TIMESTEPS)
    assert torch.allclose(schedule.betas, schedule.betas[:1].expand_as(schedule.betas))


def test_beta_schedule_is_symmetric_and_normalised() -> None:
    betas = make_beta_schedule(n_timestep=10, linear_start=1e-4, linear_end=1e-2, power=1.0)
    assert betas.shape == (10,)
    assert np.all(np.diff(betas) >= 0)


def test_gaussian_product_coefficients_sum_to_one() -> None:
    sigma1 = torch.linspace(0.1, 1.0, 7)
    sigma2 = torch.linspace(1.0, 0.1, 7)
    coef1, coef2, var = compute_gaussian_product_coef(sigma1, sigma2)
    assert torch.allclose(coef1 + coef2, torch.ones_like(coef1))
    assert torch.all(var > 0)


def test_schedule_forward_and_backward_variances_are_positive() -> None:
    schedule = _schedule()
    assert torch.all(schedule.std_fwd > 0)
    assert torch.all(schedule.std_bwd > 0)
    assert torch.all(schedule.std_sb > 0)
    assert schedule.timesteps == 64


def test_q_sample_endpoints_bracket_prior_and_target() -> None:
    """``step = 0`` leans to the target (TS), the last step to the prior (midpoint).

    The coefficients are not exactly 1/0 at the endpoints: ``mu_x0[0]`` is
    ``sigma_bwd**2 / (sigma_fwd**2 + sigma_bwd**2)``, so the residual is a small
    fraction of the ``x0 - x1`` gap rather than zero.
    """
    schedule = _schedule()
    x0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # recorded TS
    x1 = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])  # (R + P) / 2
    gap = (x1 - x0).abs().max()
    near_target = q_sample(schedule, torch.tensor(0), x0, x1)
    near_prior = q_sample(schedule, torch.tensor(schedule.timesteps - 1), x0, x1)
    assert (near_target - x0).abs().max() < (near_target - x1).abs().max()
    assert (near_prior - x1).abs().max() < (near_prior - x0).abs().max()
    assert (near_target - x0).abs().max() < 0.2 * gap
    assert (near_prior - x1).abs().max() < 0.2 * gap


def test_label_and_pred_x0_are_exact_inverses() -> None:
    schedule = _schedule()
    x0 = torch.randn(5, 3)
    x1 = torch.randn(5, 3)
    step = torch.tensor(17)
    xt = q_sample(schedule, step, x0, x1)
    label = compute_label(schedule, step, x0, xt)
    recovered = compute_pred_x0(schedule, step, xt, label)
    assert torch.allclose(recovered, x0, atol=1e-5)


def test_sigma_zero_bridge_is_deterministic() -> None:
    schedule = _schedule()
    x0 = torch.randn(4, 3)
    x1 = torch.randn(4, 3)
    step = torch.tensor(9)
    first = q_sample(schedule, step, x0, x1, ot_ode=True)
    second = q_sample(schedule, step, x0, x1, ot_ode=True)
    assert torch.equal(first, second)


def test_posterior_step_interpolates_between_x0_and_xn() -> None:
    schedule = _schedule()
    x_n = torch.randn(3, 3)
    x0 = torch.randn(3, 3)
    n, nprev = 40, 20
    std_n = schedule.std_fwd[n]
    std_nprev = schedule.std_fwd[nprev]
    std_delta = (std_n**2 - std_nprev**2).sqrt()
    coef0, coefn, _ = compute_gaussian_product_coef(std_nprev, std_delta)
    expected = coef0 * x0 + coefn * x_n
    assert torch.allclose(posterior_step(schedule, nprev, n, x_n, x0), expected)


def test_remove_mean_batch_centres_each_sample() -> None:
    x = torch.tensor([[1.0, 1.0], [3.0, 5.0], [10.0, 0.0], [20.0, 0.0]])
    index = torch.tensor([0, 0, 1, 1])
    centred = remove_mean_batch(x, index)
    assert torch.allclose(centred[0:2].mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(centred[2:4].mean(dim=0), torch.zeros(2), atol=1e-6)


def test_space_indices_spans_the_interval() -> None:
    assert space_indices(1000, 11)[0] == 0
    assert space_indices(1000, 11)[-1] == 999
    assert len(space_indices(1000, 11)) == 11


class _StubModel:
    """Returns a fixed residual so the field's sign and scale are checkable."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[float] = []

    def __call__(self, batch, t=None):
        self.calls.append(float(t.reshape(-1)[0]))
        return {"scaled": torch.full_like(batch.pos, self.value)}


class _StubBatch:
    def __init__(self, atoms: int) -> None:
        self.pos = torch.zeros(atoms, 3)
        self.num_graphs = 1
        self.flow_time = torch.zeros(1, 1)


def test_bridge_ode_field_flips_sign_and_uses_beta_over_time() -> None:
    schedule = _schedule()
    beta = schedule.betas[0] * schedule.timesteps
    model = _StubModel(0.5)
    batch = _StubBatch(4)
    field = make_bridge_ode_field(model, batch, beta=beta)
    progress = torch.tensor(0.0)
    velocity = field(torch.zeros(4, 3), progress)
    # progress = 0 -> t_sb = 1, and the field is -net_out * sqrt(beta / t_sb).
    expected = -0.5 * torch.sqrt(beta / torch.tensor(1.0))
    assert torch.allclose(velocity, expected.expand(4, 3), atol=1e-6)
    assert model.calls[0] == pytest.approx(1.0)
    assert batch.pos.shape == (4, 3)


def test_bridge_ode_field_clamps_bridge_time() -> None:
    schedule = _schedule()
    beta = schedule.betas[0] * schedule.timesteps
    model = _StubModel(1.0)
    batch = _StubBatch(2)
    field = make_bridge_ode_field(model, batch, beta=beta, t_eps=DEFAULT_T_EPS)
    field(torch.zeros(2, 3), torch.tensor(1.0))
    assert model.calls[0] == pytest.approx(DEFAULT_T_EPS)
