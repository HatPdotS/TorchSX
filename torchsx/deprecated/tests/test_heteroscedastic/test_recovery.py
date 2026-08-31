"""Test I0 and alpha recovery from synthetic heteroscedastic data.

Validates fit_I0_poisson (Poisson marginal MLE with softplus quadrature)
and fit_alpha on data drawn from the exact generative model.
"""

from __future__ import annotations

import pytest
import torch

from torchsx.deprecated.merge._dstep import fit_I0_poisson, fit_alpha, d_step
from torchsx.deprecated.merge._distribution import _build_adjacency

from .generate import generate_synthetic, generate_variable_alpha


# ── Helpers ──


def _dummy_adjacency(n_hkl):
    """Build a simple linear adjacency graph (h-1, h+1) for smoothing."""
    src, dst = [], []
    for h in range(n_hkl):
        if h > 0:
            src.append(h)
            dst.append(h - 1)
        if h < n_hkl - 1:
            src.append(h)
            dst.append(h + 1)
    edge_src = torch.tensor(src, dtype=torch.long)
    edge_dst = torch.tensor(dst, dtype=torch.long)
    n_neighbors = torch.zeros(n_hkl)
    n_neighbors.scatter_add_(0, edge_src, torch.ones_like(edge_src, dtype=torch.float))
    return edge_src, edge_dst, n_neighbors


def _relative_error(estimated, true, mask=None):
    """Median |estimated - true| / true."""
    if mask is None:
        mask = true > 0
    rel = (estimated[mask] - true[mask]).abs() / true[mask].clamp(min=1e-8)
    return rel.median().item()


def _correlation(a, b):
    """Pearson correlation."""
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


# ── Tests: I0 recovery via Poisson marginal ──


class TestI0Poisson:
    """Test fit_I0_poisson on synthetic data."""

    def test_basic_recovery(self):
        """I0 recovered with high correlation for 50 obs/HKL."""
        data = generate_synthetic(n_hkl=200, obs_per_hkl=50, alpha=0.3, beta=1.3)
        I0_init = data.I0_true * 1.2
        scale = torch.ones_like(data.I_obs)
        bg = data.sigma2_poi * 0  # zero background for synthetic data

        I0_est, sigma_I0 = fit_I0_poisson(
            data.I_obs, data.hkl_id, scale,
            data.alpha_true, data.beta_true,
            I0_init, data.obs_counts, data.n_hkl,
            bg=0.0, max_iter=50,
        )
        err = _relative_error(I0_est, data.I0_true)
        corr = _correlation(I0_est, data.I0_true)
        print(f"Poisson I0: median rel error = {err:.4f}, corr = {corr:.4f}")
        assert err < 0.20, f"Median relative error {err:.4f} > 0.20"
        assert corr > 0.95, f"Correlation {corr:.4f} < 0.95"

    def test_high_beta(self):
        """Near-Gaussian (beta=2.0) should be precise."""
        data = generate_synthetic(n_hkl=200, obs_per_hkl=50, alpha=0.3, beta=2.0)
        I0_init = data.I0_true * 1.2
        scale = torch.ones_like(data.I_obs)

        I0_est, _ = fit_I0_poisson(
            data.I_obs, data.hkl_id, scale,
            data.alpha_true, data.beta_true,
            I0_init, data.obs_counts, data.n_hkl,
            bg=0.0, max_iter=50,
        )
        err = _relative_error(I0_est, data.I0_true)
        corr = _correlation(I0_est, data.I0_true)
        print(f"Poisson I0 (beta=2.0): median rel error = {err:.4f}, corr = {corr:.4f}")
        assert err < 0.15, f"Gaussian case: median relative error {err:.4f} > 0.15"
        assert corr > 0.95


# ── Tests: alpha recovery ──


class TestAlphaRecovery:
    """Test fit_alpha on synthetic data."""

    def test_uniform_alpha(self):
        """All HKLs have alpha=0.3, should recover it."""
        data = generate_synthetic(n_hkl=200, obs_per_hkl=50, alpha=0.3, beta=1.3)

        alpha_est = fit_alpha(
            data.I_pos, data.hkl_id_pos, data.I0_true,
            data.beta_true, data.obs_counts_pos, data.n_hkl,
        )
        err = (alpha_est.median() - 0.3).abs() / 0.3
        print(f"Alpha recovery (uniform): median = {alpha_est.median():.4f}, "
              f"rel error = {err:.4f}")
        assert err < 0.15, f"Alpha median error {err:.4f} > 0.15"

    def test_variable_alpha(self):
        """Per-HKL alpha in [0.1, 0.8], should correlate well."""
        data = generate_variable_alpha(n_hkl=300, obs_per_hkl=80)

        alpha_est = fit_alpha(
            data.I_pos, data.hkl_id_pos, data.I0_true,
            data.beta_true, data.obs_counts_pos, data.n_hkl,
        )
        corr = _correlation(alpha_est, data.alpha_true)
        print(f"Alpha recovery (variable): corr = {corr:.4f}")
        assert corr > 0.85, f"Alpha correlation {corr:.4f} < 0.85"


# ── Tests: full d_step ──


class TestDStep:
    """Test the complete d_step (I0 + alpha + smoothing)."""

    def test_full_dstep(self):
        data = generate_variable_alpha(n_hkl=200, obs_per_hkl=50, seed=123)
        edge_src, edge_dst, n_neighbors = _dummy_adjacency(data.n_hkl)

        I0_init = data.I0_true * 1.5
        alpha_init = torch.full((data.n_hkl,), 0.5)

        result = d_step(
            I_all_corrected=data.I_obs,
            hkl_id_all=data.hkl_id,
            sigma2_poisson_all=data.sigma2_poi,
            obs_counts_all=data.obs_counts,
            I_pos_corrected=data.I_pos,
            hkl_id_pos=data.hkl_id_pos,
            obs_counts_pos=data.obs_counts_pos,
            I0_prev=I0_init,
            alpha_prev=alpha_init,
            beta=data.beta_true,
            n_valid=data.n_hkl,
            n_newton_iter=20,
            edge_src=edge_src,
            edge_dst=edge_dst,
            n_neighbors=n_neighbors,
            n_smooth_iter=5,
            smooth_lambda=0.2,
            I_obs_all=data.I_obs,
            scale_all=torch.ones_like(data.I_obs),
            bg=0.0,
        )

        I0_err = _relative_error(result.I0, data.I0_true)
        I0_corr = _correlation(result.I0, data.I0_true)
        alpha_corr = _correlation(result.alpha, data.alpha_true)

        print(f"D-step: I0 err={I0_err:.4f}, I0 corr={I0_corr:.4f}, "
              f"alpha corr={alpha_corr:.4f}")
        assert I0_err < 0.20
        assert I0_corr > 0.90
        assert alpha_corr > 0.3
