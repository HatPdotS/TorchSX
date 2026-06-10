"""Tests for ``WilsonAnisoPrior`` and ``ScaleBMeanRegularizer``."""

from __future__ import annotations

import math

import pytest
import torch

from torchref.base.math_torch import U_to_matrix
from torchref.base.reciprocal.basis import reciprocal_basis_matrix

from torchsx.merging.config import MergeConfig
from torchsx.merging.priors import WilsonAnisoPrior
from torchsx.merging.regularizers import ScaleBMeanRegularizer
from torchsx.merging.scale import ScaleBPerCrystal


def _generate_hkl_grid(d_min: float, cell: torch.Tensor) -> torch.Tensor:
    """Crude P1 HKL grid: enumerate within bounds where d_h > d_min."""
    a, b, c = cell[0].item(), cell[1].item(), cell[2].item()
    h_max = int(a / d_min) + 1
    k_max = int(b / d_min) + 1
    l_max = int(c / d_min) + 1
    hkl = torch.cartesian_prod(
        torch.arange(-h_max, h_max + 1),
        torch.arange(-k_max, k_max + 1),
        torch.arange(-l_max, l_max + 1),
    )
    hkl = hkl[(hkl != 0).any(dim=-1)]
    B = reciprocal_basis_matrix(cell.float())
    s = hkl.float() @ B
    s2 = (s * s).sum(-1)
    d = 1.0 / s2.clamp(min=1e-8).sqrt()
    return hkl[d > d_min]


def _build_prior_from_truth(
    cell: torch.Tensor,
    log_I_mean: float,
    U_diag: torch.Tensor,
    n_target: int = 6000,
    seed: int = 0,
) -> tuple[WilsonAnisoPrior, torch.Tensor, torch.Tensor]:
    """Construct a synthetic dataset and a prior wrapping its s_vec/centric flags."""
    torch.manual_seed(seed)
    hkl = _generate_hkl_grid(d_min=1.5, cell=cell)
    if len(hkl) > n_target:
        idx = torch.randperm(len(hkl))[:n_target]
        hkl = hkl[idx]
    B_orth = reciprocal_basis_matrix(cell.float())
    s_vec = hkl.float() @ B_orth.float()
    U_true = torch.diag(U_diag.float())
    sU = s_vec @ U_true
    log_Sigma = log_I_mean + (-2.0 * math.pi ** 2 * (sU * s_vec).sum(-1)).clamp(-10, 10)
    Sigma = torch.exp(log_Sigma)
    # Acentric Wilson: I ~ Exp(Sigma).
    u = torch.rand(len(hkl)).clamp_min(1e-9)
    I = -Sigma * torch.log(u)
    log_I0 = torch.log(I.clamp_min(1e-12))
    is_centric = torch.zeros(len(hkl), dtype=torch.bool)
    prior = WilsonAnisoPrior(
        s_vec=s_vec,
        is_centric=is_centric,
        log_I_mean_init=0.0,
        U_init=0.001 * torch.randn(6),
        lam_U_ridge=0.0,  # disable ridge for clean recovery
    )
    return prior, log_I0, log_Sigma


def _fit_prior(
    prior: WilsonAnisoPrior, log_I0: torch.Tensor, n_steps: int = 80
) -> None:
    """Optimise the prior's params with frozen log_I0 (Wilson plot fit)."""
    log_I0_frozen = log_I0.detach()
    opt = torch.optim.LBFGS(
        [prior.log_I_mean, prior.U],
        lr=1.0,
        max_iter=n_steps,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure():
        opt.zero_grad()
        loss = prior(log_I0_frozen).sum()
        loss.backward()
        return loss

    opt.step(closure)


def test_isotropic_recovery():
    cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
    log_I_mean_true = 5.0
    B_iso_true = 15.0
    u_iso = B_iso_true / (2.0 * math.pi ** 2)
    prior, log_I0, _ = _build_prior_from_truth(
        cell,
        log_I_mean=log_I_mean_true,
        U_diag=torch.tensor([u_iso, u_iso, u_iso]),
        n_target=6000,
        seed=1,
    )
    _fit_prior(prior, log_I0)

    eigs = torch.linalg.eigvalsh(U_to_matrix(prior.U.detach()))
    assert abs(prior.log_I_mean.item() - log_I_mean_true) < 0.10
    assert torch.allclose(eigs, torch.full((3,), u_iso), rtol=0.20)


def test_anisotropic_recovery():
    cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
    log_I_mean_true = 4.0
    U_diag = torch.tensor([10.0, 20.0, 5.0]) / (2.0 * math.pi ** 2)
    prior, log_I0, _ = _build_prior_from_truth(
        cell,
        log_I_mean=log_I_mean_true,
        U_diag=U_diag,
        n_target=8000,
        seed=2,
    )
    _fit_prior(prior, log_I0)

    eigs = torch.linalg.eigvalsh(U_to_matrix(prior.U.detach())).sort().values
    expected = U_diag.sort().values
    assert abs(prior.log_I_mean.item() - log_I_mean_true) < 0.10
    assert torch.allclose(eigs, expected, rtol=0.25, atol=0.05)


def test_centric_acentric_branch():
    s_vec = torch.tensor([[0.05, 0.0, 0.0], [0.0, 0.04, 0.0], [0.02, 0.02, 0.01]])
    is_centric = torch.tensor([False, True, False])
    prior = WilsonAnisoPrior(
        s_vec=s_vec,
        is_centric=is_centric,
        log_I_mean_init=2.0,
        U_init=torch.tensor([0.05, 0.05, 0.05, 0.0, 0.0, 0.0]),
        lam_U_ridge=0.0,
    )
    log_I0 = torch.tensor([1.5, 0.8, 2.2])
    nlp = prior(log_I0)
    log_Sigma = prior.log_sigma()
    J = torch.exp(log_I0 - log_Sigma)
    expected_acen = J + log_Sigma - log_I0
    expected_cent = 0.5 * J + 0.5 * log_Sigma - 0.5 * log_I0
    expected = torch.where(is_centric, expected_cent, expected_acen)
    assert torch.allclose(nlp, expected, atol=1e-6)


def test_gradcheck_aniso():
    torch.manual_seed(0)
    s_vec = 0.05 * torch.randn(8, 3, dtype=torch.float64)
    is_centric = torch.tensor([False, True] * 4)
    prior = WilsonAnisoPrior(
        s_vec=s_vec,
        is_centric=is_centric,
        log_I_mean_init=0.5,
        U_init=0.01 * torch.randn(6, dtype=torch.float64),
        lam_U_ridge=0.5,
    )
    prior = prior.double()
    log_I0 = torch.randn(8, dtype=torch.float64, requires_grad=True)
    prior.log_I_mean.requires_grad_(True)
    prior.U.requires_grad_(True)

    def fn(log_I0_, log_I_mean_, U_):
        prior.log_I_mean.data = log_I_mean_
        prior.U.data = U_
        return prior(log_I0_).sum()

    assert torch.autograd.gradcheck(
        fn,
        (log_I0, prior.log_I_mean.detach().clone(), prior.U.detach().clone()),
        eps=1e-6,
        atol=1e-4,
        nondet_tol=1e-7,
    )


def test_scale_b_mean_regularizer():
    n_crystals = 7
    scale = ScaleBPerCrystal(n_crystals=n_crystals)
    scale.B.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    reg = ScaleBMeanRegularizer(scale)
    val = reg(params=None)
    expected = float(scale.B.mean().item()) ** 2 * n_crystals
    assert abs(val.item() - expected) < 1e-6


def test_scale_b_mean_regularizer_pulls_to_zero():
    torch.manual_seed(0)
    scale = ScaleBPerCrystal(n_crystals=10)
    scale.B.data = 5.0 + 0.5 * torch.randn(10)
    initial_mean = scale.B.mean().item()
    reg = ScaleBMeanRegularizer(scale)
    opt = torch.optim.LBFGS([scale.B], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = reg(params=None)
        loss.backward()
        return loss

    opt.step(closure)
    # The mean should drop sharply; spread can also collapse since this reg
    # alone has no opinion about it. Just check the centering target.
    assert abs(scale.B.mean().item()) < 1e-3
    assert abs(initial_mean) > 1.0  # sanity: started off-center


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_init_from_data_recovers_intercept(seed: int):
    torch.manual_seed(seed)
    cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
    log_I_mean_true = 6.0
    B_iso_true = 12.0
    u_iso = B_iso_true / (2.0 * math.pi ** 2)
    hkl = _generate_hkl_grid(d_min=2.0, cell=cell)
    idx = torch.randperm(len(hkl))[:3000]
    hkl = hkl[idx]
    B_orth = reciprocal_basis_matrix(cell.float())
    s_vec = hkl.float() @ B_orth.float()
    s2 = (s_vec * s_vec).sum(-1)
    log_I_truth = log_I_mean_true - 2.0 * B_iso_true * s2
    # Tiny noise so lstsq is well-conditioned but truth is essentially exact.
    mu_init = log_I_truth + 0.05 * torch.randn(len(hkl))
    is_centric = torch.zeros(len(hkl), dtype=torch.bool)

    class _DummyPrep:
        pass

    prep = _DummyPrep()
    prep.hkl_valid = hkl
    prep.is_centric = is_centric
    prep.median_cell = cell
    prep.mu_init = mu_init

    cfg = MergeConfig(
        lam_wilson_aniso=1.0,
        lam_wilson_aniso_U_ridge=0.0,
        wilson_aniso_init_from_data=True,
    )
    prior = WilsonAnisoPrior.from_prep(prep, cfg)
    eigs = torch.linalg.eigvalsh(U_to_matrix(prior.U.detach()))
    assert abs(prior.log_I_mean.item() - log_I_mean_true) < 0.20
    # All three diagonal eigenvalues should be roughly u_iso since we seeded
    # U = u_iso · I.
    assert torch.allclose(eigs, torch.full((3,), u_iso), rtol=0.05, atol=1e-3)
