"""Unit tests for the scatter and likelihood modules."""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.scatter import GenGaussLogScatter
from torchsx.merging.likelihood import Poisson


def _scatter(log_alpha: torch.Tensor, log_beta: torch.Tensor) -> GenGaussLogScatter:
    s = GenGaussLogScatter()
    s.log_alpha_beta_obs = lambda params, meta: (log_alpha, log_beta)  # noqa: E731
    return s


# ─── ScatterModel ───────────────────────────────────────────────────────


def test_gengauss_log_prob_shape_and_normalisation() -> None:
    """At eps=0 the log-pdf equals the log-normalisation constant."""
    n_obs = 3
    # β=2 ⇒ Gaussian
    scatter = _scatter(torch.zeros(n_obs), torch.full((n_obs,), math.log(2.0)))
    eps = torch.zeros(n_obs, 4)
    lp = scatter.log_prob(eps, None, None)
    assert lp.shape == (n_obs, 4)
    # β=2, α=1 ⇒ log β − log 2α − lgamma(1/β) = log 2 − log 2 − lgamma(0.5)
    #         = − lgamma(0.5) = − (½ log π)
    expected = -0.5 * math.log(math.pi)
    assert torch.allclose(lp, torch.full_like(lp, expected), atol=1e-6)


def test_gengauss_gradient_flows() -> None:
    n_obs = 5
    log_alpha = torch.full((n_obs,), math.log(0.8), requires_grad=True)
    log_beta = torch.full((n_obs,), math.log(1.3), requires_grad=True)
    scatter = _scatter(log_alpha, log_beta)
    eps = torch.randn(n_obs, 3)
    loss = -scatter.log_prob(eps, None, None).sum()
    loss.backward()
    assert log_alpha.grad is not None
    assert log_beta.grad is not None
    assert torch.isfinite(log_alpha.grad).all()
    assert torch.isfinite(log_beta.grad).all()


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
