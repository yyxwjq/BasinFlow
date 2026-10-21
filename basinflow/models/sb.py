"""The Schrödinger-bridge schedule and bridge algebra used by ReactOT.

Follows ``reactot/diffusion/_schedule.py`` (``make_beta_schedule``,
``compute_gaussian_product_coef``, ``SBSchedule``), ``reactot/diffusion/_utils.py``
(``unsqueeze_xdim``, ``space_indices``) and ``reactot/diffusion/en_sb.py``
(``q_sample`` / ``compute_label`` / ``compute_pred_x0`` / ``p_posterior`` /
``ode_sampling.f``). This module carries the equations; the Lightning module, the
``EnSB`` wrapper and the multi-fragment tensor bookkeeping stay in the reference
repo.

Reference defaults (``reactot/trainer/train_rpsb_ts1x.py:128-141``):
``timesteps=3000``, ``beta_max=0.3``, ``power=0.5``, ``inv_power=1``,
``sigma=0``, ``ot_ode=True``. With ``linear_start == linear_end ==
beta_max/timesteps`` the reference beta schedule is constant.

Conventions (they differ between the two libraries and must not be mixed):

* ``t_sb`` in this module is ReactOT's normalised bridge time: ``t_sb = 1`` is
  the bridge prior ``(R + P) / 2`` and ``t_sb -> 0`` is the transition state.
  ``step = round(t_sb * timesteps)`` indexes :class:`SBSchedule`.
* ``progress`` is BasinFlow's integration variable for ``sample_ode``, which
  requires an increasing interval: ``progress = 1 - t_sb``. The bridge therefore
  integrates ``progress`` from ``0`` (midpoint) to ``1`` (TS).
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

#: ReactOT's reference bridge configuration (train_rpsb_ts1x.py:128-141).
DEFAULT_TIMESTEPS = 3000
DEFAULT_BETA_MAX = 0.3
DEFAULT_POWER = 0.5
DEFAULT_INV_POWER = 1.0
#: ``ode_sampling`` clamps the smallest bridge time to this value.
DEFAULT_T_EPS = 1e-5


def unsqueeze_xdim(z: Tensor, xdim) -> Tensor:
    """Broadcast ``z`` (shape ``[B]``) against a trailing shape (reactot/_utils.py:63)."""
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def space_indices(num_steps: int, count: int) -> list[int]:
    """``count`` evenly spaced integers in ``[0, num_steps - 1]`` (reactot/_utils.py:68)."""
    assert count <= num_steps
    frac_stride = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur_idx = 0.0
    taken = []
    for _ in range(count):
        taken.append(round(cur_idx))
        cur_idx += frac_stride
    return taken


def make_beta_schedule(
    n_timestep: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    power: float = 1.0,
    inv_power: float = 1.0,
) -> np.ndarray:
    """Beta ramp used by the bridge (reactot/_schedule.py:238-249)."""
    betas = torch.linspace(
        linear_start**inv_power, linear_end**inv_power, n_timestep, dtype=torch.float64
    ) ** power
    return betas.numpy()


def compute_gaussian_product_coef(sigma1, sigma2):
    """Product of two isotropic Gaussians (reactot/_schedule.py:252-266)."""
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


class SBSchedule:
    """Analytic bridge schedule: ``mu_x0``, ``mu_x1``, ``std_fwd``, ``std_sb``.

    Port of ``reactot/diffusion/_schedule.py:269-325``. Attributes are float32
    tensors exactly as in the reference.
    """

    def __init__(
        self,
        timesteps: int = DEFAULT_TIMESTEPS,
        beta_max: float = DEFAULT_BETA_MAX,
        power: float = DEFAULT_POWER,
        inv_power: float = DEFAULT_INV_POWER,
    ) -> None:
        if timesteps < 2:
            raise ValueError("timesteps must be >= 2")
        betas = make_beta_schedule(
            n_timestep=timesteps,
            linear_end=beta_max / timesteps,
            power=power,
            inv_power=inv_power,
        )
        betas = np.concatenate([betas[: timesteps // 2], np.flip(betas[: timesteps // 2])])
        betas = (beta_max / timesteps) / np.max(betas) * betas * 0.5

        self.timesteps = int(betas.shape[0])
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas)
        self.std_fwd = to_torch(std_fwd)
        self.std_bwd = to_torch(std_bwd)
        self.std_sb = to_torch(std_sb)
        self.mu_x0 = to_torch(mu_x0)
        self.mu_x1 = to_torch(mu_x1)

    @staticmethod
    def inflate_batch_array(array: Tensor, target: Tensor) -> Tensor:
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def get_std_fwd(self, step: Tensor, xdim=None) -> Tensor:
        step = step.to(self.mu_x0.device)
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def step_from_time(self, t_sb: Tensor) -> Tensor:
        """Bridge time in ``(0, 1]`` -> integer schedule index, clamped to range."""
        return torch.round(t_sb * self.timesteps).long().clamp(0, self.timesteps - 1)


def q_sample(
    schedule: SBSchedule,
    step: Tensor,
    x0: Tensor,
    x1: Tensor,
    *,
    ot_ode: bool = True,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """``q(x_t | x_0, x_1)`` -- reference ``EnSB.q_sample`` (en_sb.py:226-243).

    ``x0`` is the target (the recorded TS) and ``x1`` the bridge prior
    ``(R + P) / 2``, matching ReactOT's ``sample_batch`` for ``R+P->TS``.
    """
    assert x0.shape == x1.shape
    device = schedule.mu_x0.device
    step = step.to(device)
    mu_x0 = schedule.mu_x0[step].to(x0.device)
    mu_x1 = schedule.mu_x1[step].to(x0.device)
    xt = mu_x0 * x0 + mu_x1 * x1
    if not ot_ode:
        std_sb = schedule.std_sb[step].to(x0.device)
        xt = xt + std_sb * torch.randn_like(xt)
    if mask is not None:
        xt = remove_mean_batch(xt, mask)
    return xt


def compute_label(schedule: SBSchedule, step: Tensor, x0: Tensor, xt: Tensor) -> Tensor:
    """Regression target ``(x_t - x_0) / sigma_t`` (en_sb.py:245-250)."""
    std_fwd = schedule.get_std_fwd(step, xdim=x0.shape[1:]).to(x0.device)
    return (xt - x0) / std_fwd


def compute_pred_x0(
    schedule: SBSchedule,
    step: Tensor,
    xt: Tensor,
    net_out: Tensor,
    *,
    clip_denoise: bool = False,
    value: float = 10.0,
) -> Tensor:
    """Inverse of :func:`compute_label`: ``x_0 = x_t - sigma_t * net_out``."""
    std_fwd = schedule.get_std_fwd(step, xdim=xt.shape[1:]).to(xt.device)
    pred_x0 = xt - std_fwd * net_out
    if clip_denoise:
        pred_x0 = pred_x0.clamp_(-value, value)
    return pred_x0


def posterior_step(
    schedule: SBSchedule,
    nprev: int,
    n: int,
    x_n: Tensor,
    x0: Tensor,
    *,
    ot_ode: bool = True,
) -> Tensor:
    """``p(x_{nprev} | x_n, x_0)`` -- reference ``EnSB.p_posterior`` (en_sb.py:349-366)."""
    assert nprev < n
    std_n = schedule.std_fwd[n]
    std_nprev = schedule.std_fwd[nprev]
    std_delta = (std_n**2 - std_nprev**2).sqrt()
    mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)
    xt_prev = mu_x0 * x0 + mu_xn * x_n
    if not ot_ode and nprev > 0:
        xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)
    return xt_prev


def remove_mean_batch(x: Tensor, indices: Tensor) -> Tensor:
    """Per-sample mean removal, matching ``EnSB``'s ``remove_mean_batch``.

    Local implementation (index_add based) so the bridge does not pull in
    ``torch_scatter``.
    """
    index = indices.to(torch.long).reshape(-1)
    size = int(index.max()) + 1 if index.numel() else 0
    moved = x.movedim(0, 0)
    sums = torch.zeros((size,) + tuple(moved.shape[1:]), dtype=x.dtype, device=x.device)
    sums.index_add_(0, index, moved)
    counts = torch.zeros(size, dtype=x.dtype, device=x.device)
    counts.index_add_(0, index, torch.ones(moved.shape[0], dtype=x.dtype, device=x.device))
    mean = sums / counts.clamp(min=1.0).reshape((size,) + (1,) * (moved.dim() - 1))
    return x - mean[index]


def sigma_div_t(beta: Tensor, t_sb: Tensor) -> Tensor:
    """``sqrt(beta / t)``, the factor multiplying the network output in the ODE field.

    ``beta`` is ``schedule.betas[0] * schedule.timesteps`` (en_sb.py:412), i.e.
    the constant per-step variance of the reference schedule.
    """
    return torch.sqrt(beta / t_sb)


def make_bridge_ode_field(
    model: Callable,
    batch,
    *,
    beta: Tensor,
    t_eps: float = DEFAULT_T_EPS,
):
    """Build ``field(state, progress)`` for :func:`basinflow.sampling.ode.sample_ode`.

    BasinFlow integrates ``progress`` from ``0`` to ``1``; the bridge is defined
    in ``t_sb = 1 - progress`` and runs from ``1`` (midpoint prior) to ``0``
    (transition state). The sign flip below is the chain rule of that
    reparameterisation, so integrating ``progress`` upwards reproduces ReactOT's
    ``ode_sampling`` exactly (en_sb.py:398-452):

    ``dx/d(progress) = -sqrt(beta / t_sb) * net_out(x, t_sb)``.
    """

    def field(state: Tensor, progress: Tensor) -> Tensor:
        progress = progress.reshape(())
        t_sb = (1.0 - progress).clamp(min=t_eps)
        batch.pos = state
        batch.flow_time = t_sb.reshape(1, 1).expand(batch.num_graphs, 1)
        net_out = model(batch, t=batch.flow_time.squeeze(-1))["scaled"]
        return -net_out * sigma_div_t(beta.to(state.device), t_sb.to(state.device))

    return field
