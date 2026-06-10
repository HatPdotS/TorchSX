"""Golden agreement: Triton Gaussian kernel vs eager log_prob_given_signal.

Both paths now consume a per-observation ``var`` (M,) tensor. σ² is
treated as constant across the K-grid during integration — see
:mod:`torchsx.merging.likelihood.gaussian_base` for the rationale.

Two checks:

1. **Forward agreement.** The Triton kernel and the eager evaluation
   of the marginal ``log_p`` must agree to floating-point tolerance
   across several σ²-recipes (homoscedastic constants, data-driven
   Poisson MLE, full polynomial-in-μ_obs).

2. **Grad-on-var.** The Triton backward returns ``∂log_p/∂var`` via
   the kernel's ``S2_var`` sufficient stat.  We cross-check it against
   finite differences on the eager path.
"""
from __future__ import annotations

import math
import os

import pytest
import torch


@pytest.fixture
def small_problem():
    """A small (M=8, K=33) GenGauss × Gaussian problem on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton kernel tests")
    torch.manual_seed(0)
    M, K = 8, 33
    log_mean = torch.randn(M, device="cuda", dtype=torch.float32) * 0.5
    n = (torch.rand(M, device="cuda") * 20 + 1).clamp(min=0.0)
    bg = (torch.rand(M, device="cuda") * 5 + 0.5)
    log_alpha = torch.randn(M, device="cuda") * 0.5 - 0.5
    log_beta = (torch.randn(M, device="cuda") * 0.2 + 0.8)
    return M, K, log_mean, n, bg, log_alpha, log_beta


def _build_var(recipe: str, n: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    """Build a (M,) σ² tensor according to ``recipe``."""
    mu_obs = torch.maximum(n, bg)
    if recipe == "const_1":
        return torch.full_like(n, 1.0)
    if recipe == "const_25":
        return torch.full_like(n, 25.0)
    if recipe == "poisson":          # σ² = max(n, bg) (Poisson moment-match)
        return mu_obs.clamp(min=1e-6)
    if recipe == "polynomial":       # gain-switching: g₀ + g₁μ + g₂μ²
        return (1.0 + 1.0 * mu_obs + 0.0025 * mu_obs * mu_obs).clamp(min=1e-6)
    raise ValueError(f"unknown var recipe: {recipe}")


def _eager_log_p(var, log_mean, n, bg, log_alpha_obs, log_beta_obs, K):
    """Eager marginal log_p with per-obs σ² constant in S."""
    from torchsx.merging.scatter.gengauss import gengauss_log_prob_from_obs
    log_S = torch.linspace(math.log(1e-3), math.log(1e6), K,
                           device=log_mean.device, dtype=log_mean.dtype)
    S = log_S.exp()
    eps = log_S[None, :] - log_mean[:, None]
    log_scatter = gengauss_log_prob_from_obs(eps, log_alpha_obs, log_beta_obs)
    mu = (S[None, :].expand(log_mean.shape[0], K) + bg[:, None]).clamp(min=1e-6)
    diff = n[:, None] - mu
    log_lik = -0.5 * (
        math.log(2 * math.pi)
        + var[:, None].log()
        + diff * diff / var[:, None]
    )
    dlog_S = (math.log(1e6) - math.log(1e-3)) / (K - 1)
    log_w = torch.full((K,), math.log(dlog_S),
                       device=log_mean.device, dtype=log_mean.dtype)
    log_w[0] -= math.log(2.0)
    log_w[-1] -= math.log(2.0)
    return torch.logsumexp(log_scatter + log_lik + log_w[None, :], dim=1)


@pytest.mark.parametrize("recipe", [
    "const_1",
    "const_25",
    "poisson",
    "polynomial",
])
def test_triton_vs_eager_logp(small_problem, recipe):
    """Forward log_p agreement between Triton kernel and eager path."""
    M, K, log_mean, n, bg, log_alpha, log_beta = small_problem

    from torchsx.merging.quadrature._fixed_log_grid_triton_gaussian import (
        marginalize_signal_triton_gaussian,
    )

    var = _build_var(recipe, n, bg)

    log_p_eager = _eager_log_p(var, log_mean, n, bg, log_alpha, log_beta, K)

    log_S = torch.linspace(math.log(1e-3), math.log(1e6), K,
                           device=log_mean.device, dtype=log_mean.dtype)
    dlog_S = (math.log(1e6) - math.log(1e-3)) / (K - 1)
    log_w = torch.full((K,), math.log(dlog_S),
                       device=log_mean.device, dtype=log_mean.dtype)
    log_w[0] -= math.log(2.0)
    log_w[-1] -= math.log(2.0)

    log_p_triton = marginalize_signal_triton_gaussian(
        log_mean, n, bg, var,
        log_alpha, log_beta,
        log_S, log_w,
    )

    diff = (log_p_triton - log_p_eager).abs().max().item()
    rel = (
        (log_p_triton - log_p_eager).abs() / log_p_eager.abs().clamp(min=1e-6)
    ).max().item()
    print(f"recipe={recipe}  max abs diff = {diff:.3e}  rel = {rel:.3e}")
    assert torch.isfinite(log_p_triton).all(), "Triton log_p has NaN/Inf"
    assert diff < 5e-4 or rel < 5e-5, (
        f"Triton ≠ eager: max abs {diff:.3e} rel {rel:.3e}\n"
        f"triton: {log_p_triton}\n eager: {log_p_eager}"
    )


@pytest.mark.parametrize("recipe", ["poisson", "polynomial"])
def test_triton_grad_on_var(small_problem, recipe):
    """``d log_p / d var`` from Triton backward vs finite differences on the eager path."""
    M, K, log_mean, n, bg, log_alpha, log_beta = small_problem

    from torchsx.merging.quadrature._fixed_log_grid_triton_gaussian import (
        marginalize_signal_triton_gaussian,
    )

    var0 = _build_var(recipe, n, bg).detach()
    log_S = torch.linspace(math.log(1e-3), math.log(1e6), K,
                           device=log_mean.device, dtype=log_mean.dtype)
    dlog_S = (math.log(1e6) - math.log(1e-3)) / (K - 1)
    log_w = torch.full((K,), math.log(dlog_S),
                       device=log_mean.device, dtype=log_mean.dtype)
    log_w[0] -= math.log(2.0)
    log_w[-1] -= math.log(2.0)

    # Autograd via Triton backward
    var = var0.clone().requires_grad_(True)
    log_p = marginalize_signal_triton_gaussian(
        log_mean, n, bg, var,
        log_alpha, log_beta,
        log_S, log_w,
    )
    log_p.sum().backward()
    g_auto = var.grad.detach().clone()

    # Finite difference on the eager path, one component at a time.
    eps = 1e-3
    g_fd = torch.zeros_like(var0)
    for i in range(M):
        v_plus = var0.clone()
        v_plus[i] = v_plus[i] + eps
        v_minus = var0.clone()
        v_minus[i] = v_minus[i] - eps
        lp_plus = _eager_log_p(v_plus, log_mean, n, bg, log_alpha, log_beta, K)
        lp_minus = _eager_log_p(v_minus, log_mean, n, bg, log_alpha, log_beta, K)
        # ∂(sum log_p)/∂var[i] picks up only the i-th component:
        g_fd[i] = (lp_plus[i] - lp_minus[i]) / (2 * eps)

    abs_diff = (g_auto - g_fd).abs().max().item()
    rel = ((g_auto - g_fd).abs() / g_fd.abs().clamp(min=1e-6)).max().item()
    print(f"recipe={recipe}  grad-on-var: max abs {abs_diff:.3e}  rel {rel:.3e}")
    assert abs_diff < 5e-4 or rel < 5e-3, (
        f"grad-on-var mismatch: max abs {abs_diff:.3e} rel {rel:.3e}\n"
        f"auto: {g_auto}\n fd  : {g_fd}"
    )


if __name__ == "__main__":
    # Allow running directly via python tests/test_gaussian_kernel.py for sbatch
    import sys
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)
    torch.manual_seed(0)
    M, K = 8, 33
    log_mean = torch.randn(M, device="cuda", dtype=torch.float32) * 0.5
    n = (torch.rand(M, device="cuda") * 20 + 1).clamp(min=0.0)
    bg = (torch.rand(M, device="cuda") * 5 + 0.5)
    log_alpha = torch.randn(M, device="cuda") * 0.5 - 0.5
    log_beta = (torch.randn(M, device="cuda") * 0.2 + 0.8)
    sp = (M, K, log_mean, n, bg, log_alpha, log_beta)
    for recipe in ("const_1", "const_25", "poisson", "polynomial"):
        try:
            test_triton_vs_eager_logp(sp, recipe)
            print(f"  PASS  log_p {recipe}")
        except AssertionError as e:
            print(f"  FAIL  log_p {recipe}")
            print(f"  {e}")
    for recipe in ("poisson", "polynomial"):
        try:
            test_triton_grad_on_var(sp, recipe)
            print(f"  PASS  grad {recipe}")
        except AssertionError as e:
            print(f"  FAIL  grad {recipe}")
            print(f"  {e}")
