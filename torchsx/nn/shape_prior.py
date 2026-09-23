"""Hierarchical prior on each observation's scatter distribution.

Every observation ``j`` gets the prior

    log p(theta_j) = -lambda KL(p_bar || p_j) + const = lambda E_{u ~ p_bar}[log p_j(u)] + const

on its predicted scatter density ``p_j``: the conjugate-prior form, ``lambda``
pseudo-observations of the scatter drawn from the population density
``p_bar`` next to the one real observation.  It is a KL between densities, so
it does not depend on how the head parametrises the shape (single two-width
family, mixture, directly predicted or conditioned).

The prior is the same for every observation; its effect is not.  A strong
observation measures its own multiplier almost directly and carries more
information about the shape than ``lambda <~ 1`` pseudo-observations, so the
data set its shape.  A weak one sees the multiplier through Poisson and
background noise, its likelihood is nearly flat in the shape, and the prior
takes it to ``p_bar`` -- the same value in every run and in both halves.

``p_bar`` is the empirical-Bayes estimate: the average of the predicted
densities over the sampled observations, held fixed (no gradient) within an
update.  Observations that the data leave at ``p_bar`` return ``p_bar`` to
that average, so its fixed point is set by the observations that carry
information about the shape.

Densities are of ``u = eps - log E[exp(eps)]``, the scatter of the
mean-normalised multiplier, so the prior sees shape and not the location that
``log K`` already carries.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.scatter.gengauss import gengauss_log_prob_from_obs
from torchsx.merging.scatter.mixture import N_BLOCKS, split_mix

__all__ = ["pack_single", "mixture_log_density", "shape_prior_kl"]

#: u grid in nats: the lower side reaches far (partial and near-missing
#: observations), the upper side is short
U_LO, U_HI, U_STEP = -40.0, 15.0, 0.2


def pack_single(log_alpha: Tensor, beta: Tensor, log_alpha_hi: Tensor | None = None) -> Tensor:
    """The single two-width family as a one-component packed mixture ``(n, 5)``."""
    zero = torch.zeros_like(log_alpha)
    hi = log_alpha if log_alpha_hi is None else log_alpha_hi
    return torch.stack([zero, zero, log_alpha, hi, beta.log()], dim=1)


def mixture_log_density(mix: Tensor, eps: Tensor) -> Tensor:
    """Log-density of each observation's packed mixture ``(n, 5K)`` at ``eps`` ``(n, G)``."""
    K = mix.shape[1] // N_BLOCKS
    log_pi, offset, la_lo, la_hi, log_beta = split_mix(mix, K)
    terms = [
        log_pi[:, k:k + 1]
        + gengauss_log_prob_from_obs(eps - offset[:, k:k + 1], la_lo[:, k], log_beta[:, k], la_hi[:, k])
        for k in range(K)
    ]
    return torch.logsumexp(torch.stack(terms, dim=0), dim=0)


def shape_prior_kl(mix: Tensor, log_e_s: Tensor, *, n_sample: int = 2048,
                   grid: Tensor | None = None) -> tuple[Tensor, float, float]:
    """``(mean_j KL(p_bar || p_j), p_bar's SD on the grid, p_bar's mass off it)``
    over a random sample of observations.

    ``mix`` is the packed ``(n, 5K)`` prediction and ``log_e_s`` its
    ``log E[exp(eps)]``.  The mean is an unbiased estimate of the per-observation
    mean over the batch; gradients reach ``p_j`` only.
    """
    n = mix.shape[0]
    if n > n_sample:
        idx = torch.randint(n, (n_sample,), device=mix.device)
        mix, log_e_s = mix.index_select(0, idx), log_e_s.index_select(0, idx)
    if grid is None:
        grid = torch.arange(U_LO, U_HI + 0.5 * U_STEP, U_STEP, device=mix.device, dtype=mix.dtype)
    du = float(grid[1] - grid[0])
    # Every density as bin masses: the grid cells plus one overflow bin for the
    # mass beyond the grid, so the KL is a proper discrete one -- non-negative,
    # zero when an observation has the population shape -- and a component
    # moved far off the grid (seen: alpha_lo ~ 1e3 carrying a quarter of the
    # weight once the grid alone was normalised) is visible to it.
    log_cell = mixture_log_density(mix, grid[None, :] + log_e_s[:, None]) + math.log(du)    # (m, G)
    on_grid = torch.logsumexp(log_cell, dim=1).exp()
    log_out = torch.log((1.0 - on_grid).clamp(min=1e-12))
    log_q = torch.log_softmax(torch.cat([log_cell, log_out[:, None]], dim=1), dim=1)       # (m, G + 1)
    with torch.no_grad():
        log_bar = torch.logsumexp(log_q, dim=0) - math.log(log_q.shape[0])
        w = log_bar.exp()
        wg = w[:-1] / w[:-1].sum()
        mean = (wg * grid).sum()
        sd = float(((wg * (grid - mean) ** 2).sum()).sqrt())
    kl = (w[None, :] * (log_bar[None, :] - log_q)).sum(dim=1)
    return kl.mean(), sd, float(w[-1])
