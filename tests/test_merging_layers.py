"""Unit tests for the scatter and likelihood modules."""

from __future__ import annotations

import math

import pytest
import torch
from torchsx.merging.scatter_shape import FreePerHKLShape

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter import GenGaussLogScatter
from torchsx.merging.likelihood import Poisson


def _make_toy_params(n_hkl: int, *, alpha: float = 0.8, beta: float = 1.3) -> Parameters:
    return Parameters(
        log_I0_init=torch.full((n_hkl,), math.log(5.0)),
        scatter_shape=FreePerHKLShape(
            torch.full((n_hkl,), math.log(alpha)),
            torch.full((n_hkl,), math.log(beta)),
        ),
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
