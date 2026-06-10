"""Unit tests for scatter / quadrature / likelihood modules."""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter import GenGaussLogScatter
from torchsx.merging.likelihood import (
    Poisson,
    PoissonWithBackgroundMixture,
)
from torchsx.merging.quadrature import TrapezoidalLogGrid


def _make_toy_params(n_hkl: int, *, alpha: float = 0.8, beta: float = 1.3) -> Parameters:
    return Parameters(
        log_I0_init=torch.full((n_hkl,), math.log(5.0)),
        log_alpha_init=torch.full((n_hkl,), math.log(alpha)),
        log_beta_init=torch.full((n_hkl,), math.log(beta)),
    )


def _make_toy_meta(hkl_idx: torch.Tensor) -> ObsInfo:
    n = hkl_idx.numel()
    return ObsInfo(
        crystal_id=torch.zeros(n, dtype=torch.long),
        s_sq=torch.zeros(n),
        hkl_idx=hkl_idx,
        n_crystals=1,
    )


# ─── ScatterModel ───────────────────────────────────────────────────────


def test_gengauss_log_prob_shape_and_normalisation() -> None:
    """At eps=0 the log-pdf equals the log-normalisation constant."""
    n_hkl = 3
    params = _make_toy_params(n_hkl, alpha=1.0, beta=2.0)  # β=2 ⇒ Gaussian
    meta = _make_toy_meta(torch.arange(n_hkl))
    scatter = GenGaussLogScatter()
    eps = torch.zeros(n_hkl, 4)
    lp = scatter.log_prob(eps, params, meta)
    assert lp.shape == (n_hkl, 4)
    # β=2, α=1 ⇒ log β − log 2α − lgamma(1/β) = log 2 − log 2 − lgamma(0.5)
    #         = − lgamma(0.5) = − (½ log π)
    expected = -0.5 * math.log(math.pi)
    assert torch.allclose(lp, torch.full_like(lp, expected), atol=1e-6)


def test_gengauss_gradient_flows() -> None:
    n_hkl = 5
    params = _make_toy_params(n_hkl)
    # Make requires_grad.
    params.log_alpha.requires_grad_(True)
    params.log_beta.requires_grad_(True)
    meta = _make_toy_meta(torch.arange(n_hkl))
    scatter = GenGaussLogScatter()
    eps = torch.randn(n_hkl, 3)
    loss = -scatter.log_prob(eps, params, meta).sum()
    loss.backward()
    assert params.log_alpha.grad is not None
    assert params.log_beta.grad is not None
    assert torch.isfinite(params.log_alpha.grad).all()
    assert torch.isfinite(params.log_beta.grad).all()


# ─── DetectionLikelihood ────────────────────────────────────────────────


def test_poisson_log_prob_per_grid_point_matches_closed_form() -> None:
    n = torch.tensor([3.0, 5.0])
    S = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    bg = torch.tensor([0.5, 1.0])
    lik = Poisson()
    got = lik.log_prob_given_signal(n, S, bg)
    # Direct computation: n·log(S+bg) − (S+bg) − lgamma(n+1)
    lam = S + bg[:, None]
    expected = n[:, None] * torch.log(lam) - lam - torch.lgamma(n + 1)[:, None]
    assert torch.allclose(got, expected, atol=1e-6)


def test_poisson_bg_mix_global_matches_logaddexp() -> None:
    n = torch.tensor([2.0, 4.0])
    bg = torch.tensor([1.0, 2.0])
    log_sig = torch.tensor([-3.0, -1.0])
    meta = _make_toy_meta(torch.tensor([0, 1], dtype=torch.long))
    params = _make_toy_params(2)
    lik = PoissonWithBackgroundMixture(mode="global")

    got = lik.combine_with_marginal(n, bg, log_sig, params, meta)
    log_p_bg = n * torch.log(bg) - bg - torch.lgamma(n + 1)
    expected = torch.logaddexp(log_sig, log_p_bg)
    assert torch.allclose(got, expected, atol=1e-6)


def test_poisson_bg_mix_per_hkl_uses_pi_logit() -> None:
    n = torch.tensor([2.0, 2.0, 2.0])
    bg = torch.tensor([1.0, 1.0, 1.0])
    log_sig = torch.tensor([-5.0, -5.0, -5.0])
    params = _make_toy_params(3)
    # Strongly push π toward 1 for obs 0 (should → bg), toward 0 for obs 2
    with torch.no_grad():
        params.pi_logit.copy_(torch.tensor([20.0, 0.0, -20.0]))
    meta = _make_toy_meta(torch.tensor([0, 1, 2], dtype=torch.long))
    lik = PoissonWithBackgroundMixture(mode="per_hkl")

    got = lik.combine_with_marginal(n, bg, log_sig, params, meta)
    log_p_bg = n * torch.log(bg) - bg - torch.lgamma(n + 1)
    # π≈1 ⇒ ≈ log_p_bg; π≈0 ⇒ ≈ log_sig
    assert torch.isclose(got[0], log_p_bg[0], atol=1e-5)
    assert torch.isclose(got[2], log_sig[2], atol=1e-5)


# ─── Quadrature ─────────────────────────────────────────────────────────


def test_trapezoidal_marginal_is_finite_and_differentiable() -> None:
    """Signal marginal returns a finite scalar per obs and backprops through
    ``log_mean``."""
    torch.manual_seed(0)
    n_obs, n_hkl = 8, 4
    hkl_idx = torch.randint(0, n_hkl, (n_obs,))
    params = _make_toy_params(n_hkl, alpha=0.8, beta=1.3)
    meta = _make_toy_meta(hkl_idx)
    scatter = GenGaussLogScatter()
    likelihood = Poisson()

    log_mean = torch.randn(n_obs, requires_grad=True)
    n_ph = torch.linspace(1.0, 10.0, n_obs)
    bg = torch.full((n_obs,), 0.5)
    sig_lo = torch.full_like(n_ph, 1e-4)
    quad = TrapezoidalLogGrid(51, 5.0, 0.01, sig_lo)
    log_p = quad.marginalize_signal(
        scatter, likelihood, log_mean, n_ph, bg, params, meta
    )
    assert log_p.shape == (n_obs,)
    assert torch.isfinite(log_p).all()
    log_p.sum().backward()
    assert log_mean.grad is not None
    assert torch.isfinite(log_mean.grad).all()


def test_trapezoidal_bg_mixture_recovers_bg_hypothesis_at_extreme_mean() -> None:
    """When log_mean → −∞, the signal marginal is ~0; bg mixture → log_p_bg."""
    n_hkl = 3
    hkl_idx = torch.arange(n_hkl)
    params = _make_toy_params(n_hkl, alpha=1.0, beta=1.5)
    meta = _make_toy_meta(hkl_idx)
    scatter = GenGaussLogScatter()
    likelihood = PoissonWithBackgroundMixture(mode="global")

    log_mean = torch.full((n_hkl,), -30.0)  # I_true → 0
    n_ph = torch.tensor([2.0, 4.0, 6.0])
    bg = torch.tensor([1.0, 2.0, 3.0])
    sig_lo = torch.full_like(n_ph, 1e-4)
    quad = TrapezoidalLogGrid(51, 5.0, 0.01, sig_lo)
    log_p_sig = quad.marginalize_signal(
        scatter, likelihood, log_mean, n_ph, bg, params, meta
    )
    log_p = likelihood.combine_with_marginal(n_ph, bg, log_p_sig, params, meta)
    log_p_bg = n_ph * torch.log(bg) - bg - torch.lgamma(n_ph + 1)
    # ``logaddexp(≫ negative, log_p_bg)`` ≈ log_p_bg.
    assert torch.allclose(log_p, log_p_bg, atol=1e-3)
