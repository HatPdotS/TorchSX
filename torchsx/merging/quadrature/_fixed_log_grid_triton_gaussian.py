"""Triton fast path for ``FixedLogGrid.marginalize_signal`` with a
Gaussian-noise likelihood (``GaussianLikelihood`` and any subclass).

Variance polynomial::

    σ²(μ, bg) = g₀(bg) + g₁ · μ + g₂ · μ²

where ``μ = S + bg`` and ``(g₀, g₁, g₂)`` are supplied via
``GaussianLikelihood.coefficients(bg)``.  All concrete Gaussian-shape
likelihoods (``GaussianPoisson``, ``GaussianPoissonExcess``,
``GaussianProportional``, ``GaussianPolynomialVariance``) route through
this single kernel.

The ``g`` symbols here refer to *noise* (detector gain-chain)
coefficients — distinct from the per-HKL GenGauss-prior shape parameters
``log_alpha``, ``log_beta`` (which are also kernel inputs but live in
``GenGaussLogScatter``).

Per integrand evaluation the kernel emits a single per-observation
``log_p`` plus six auxiliary statistics:

    S1            — for d/d log_mean   (GenGauss prior, same as Poisson kernel)
    S2_mu         — for d/d bg through μ  (Gaussian-specific)
    S2_g0         — for d/d g0_obs        (Gaussian-specific)
    S3, S4        — for d/d log_alpha_obs, d/d log_beta_obs (GenGauss prior)

Tuning starts at the Poisson kernel's A100 winners (BLOCK_M=64,
BLOCK_K=16); re-tune empirically if needed.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                    # pragma: no cover
    HAVE_TRITON = False


# Share the tuning lookup with the Poisson kernel; the data path is the
# same shape (per-block M loop, K-tiled inner loop) so the winning tile
# sizes should carry over.  If empirically off, add a separate JSON entry.
from torchsx.merging.quadrature._fixed_log_grid_triton import _lookup_config


# ── Kernel ─────────────────────────────────────────────────────────────


if HAVE_TRITON:

    @triton.jit
    def _fwd_kernel_gaussian(
        log_mean_ptr, n_ptr, bg_ptr, g0_obs_ptr,
        log_alpha_ptr, log_beta_ptr, log_norm_ptr,
        log_S_grid_ptr, S_grid_ptr, log_w_ptr,
        log_p_ptr, S1_ptr, S2_mu_ptr, S2_g0_ptr, S3_ptr, S4_ptr,
        M, K,
        g1: tl.constexpr,
        g2: tl.constexpr,
        var_floor: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        log_mean = tl.load(log_mean_ptr + offs_m, mask=mask_m, other=0.0)
        n = tl.load(n_ptr + offs_m, mask=mask_m, other=0.0)
        bg = tl.load(bg_ptr + offs_m, mask=mask_m, other=0.0)
        g0_obs = tl.load(g0_obs_ptr + offs_m, mask=mask_m, other=0.0)
        log_alpha = tl.load(log_alpha_ptr + offs_m, mask=mask_m, other=0.0)
        log_beta = tl.load(log_beta_ptr + offs_m, mask=mask_m, other=0.0)
        log_norm = tl.load(log_norm_ptr + offs_m, mask=mask_m, other=0.0)

        # GenGauss prior scale / shape (per-HKL → per-obs lookup).  Note:
        # these are scatter-prior α, β — NOT the σ² polynomial coefficients
        # g₀, g₁, g₂.
        alpha = tl.exp(log_alpha)
        alpha_safe = tl.where(alpha > 1e-8, alpha, 1e-8)
        inv_alpha_safe = 1.0 / alpha_safe
        beta = tl.exp(log_beta)
        log_scatter_const = log_norm - log_alpha
        # Constant in the Gaussian log-PDF: -½ · log(2π) = -0.91893853...
        gauss_const = -0.9189385332046727

        m_max = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
        T0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T2_mu = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T2_g0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T4 = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            log_S_k = tl.load(log_S_grid_ptr + offs_k, mask=mask_k, other=0.0)
            S_k = tl.load(S_grid_ptr + offs_k, mask=mask_k, other=0.0)
            log_w_k = tl.load(log_w_ptr + offs_k, mask=mask_k, other=0.0)

            # ── GenGauss prior on (log_S − log_mean) ──────────────────
            eps = log_S_k[None, :] - log_mean[:, None]
            z = tl.abs(eps) * inv_alpha_safe[:, None]
            z_eff = tl.where(z > 1e-12, z, 1e-12)
            log_z_eff = 0.6931471805599453 * tl.math.log2(z_eff)
            z_pow = tl.exp(beta[:, None] * log_z_eff)

            # ── Gaussian log-PDF for n | μ, σ²(μ) ─────────────────────
            mu = S_k[None, :] + bg[:, None]
            mu = tl.where(mu > 1e-6, mu, 1e-6)
            # σ² = g0_obs + g1 · μ + g2 · μ²
            var = g0_obs[:, None] + g1 * mu + g2 * mu * mu
            var = tl.where(var > var_floor, var, var_floor)
            inv_var = 1.0 / var
            log_var = 0.6931471805599453 * tl.math.log2(var)
            diff = n[:, None] - mu
            log_lik = gauss_const - 0.5 * log_var - 0.5 * diff * diff * inv_var

            log_int = (
                -z_pow
                + log_scatter_const[:, None]
                + log_lik
                + log_w_k[None, :]
            )
            log_int = tl.where(mask_k[None, :], log_int, -1e30)

            m_chunk = tl.max(log_int, axis=1)
            m_new = tl.maximum(m_max, m_chunk)
            scale = tl.exp(m_max - m_new)
            T0 = T0 * scale
            T1 = T1 * scale
            T2_mu = T2_mu * scale
            T2_g0 = T2_g0 * scale
            T3 = T3 * scale
            T4 = T4 * scale

            e_shift = tl.exp(log_int - m_new[:, None])

            # GenGauss-prior gradient bits (T1, T3, T4): same as Poisson kernel.
            sign_eps = tl.where(eps > 0, 1.0, tl.where(eps < 0, -1.0, 0.0))
            grad_lm = sign_eps * (z_pow / z_eff) * inv_alpha_safe[:, None]

            # ∂ log_lik / ∂ μ:  derived from log_lik = -½ log(var) - ½ (n-μ)²/var
            #   dlogp_dmu = (n-μ)/var − (g1 + 2 g2 μ)·[ 1/(2var) − (n-μ)²/(2var²) ]
            d_var_d_mu = g1 + 2.0 * g2 * mu
            dlogp_dmu = (
                diff * inv_var
                - 0.5 * d_var_d_mu * inv_var
                + 0.5 * d_var_d_mu * diff * diff * inv_var * inv_var
            )

            # ∂ log_lik / ∂ g0_obs:  g0_obs only affects var (∂var/∂g0 = 1).
            #   dlogp_dg0 = (n-μ)²/(2 var²) − 1/(2 var)
            dlogp_dg0 = 0.5 * diff * diff * inv_var * inv_var - 0.5 * inv_var

            T0 += tl.sum(e_shift, axis=1)
            T1 += tl.sum(e_shift * grad_lm, axis=1)
            T2_mu += tl.sum(e_shift * dlogp_dmu, axis=1)
            T2_g0 += tl.sum(e_shift * dlogp_dg0, axis=1)
            T3 += tl.sum(e_shift * z_pow, axis=1)
            T4 += tl.sum(e_shift * z_pow * log_z_eff, axis=1)

            m_max = m_new

        log_p = m_max + 0.6931471805599453 * tl.math.log2(T0)
        inv_T0 = 1.0 / T0

        tl.store(log_p_ptr + offs_m, log_p, mask=mask_m)
        tl.store(S1_ptr + offs_m, T1 * inv_T0, mask=mask_m)
        tl.store(S2_mu_ptr + offs_m, T2_mu * inv_T0, mask=mask_m)
        tl.store(S2_g0_ptr + offs_m, T2_g0 * inv_T0, mask=mask_m)
        tl.store(S3_ptr + offs_m, T3 * inv_T0, mask=mask_m)
        tl.store(S4_ptr + offs_m, T4 * inv_T0, mask=mask_m)


# ── Autograd.Function wrapper ──────────────────────────────────────────


class _MarginalizeSignalTritonGaussian(torch.autograd.Function):
    """Per-obs ``log_p`` for Gaussian likelihood with polynomial-in-μ variance.

    Gradients:

        d log_p / d log_mean      = +β · S1            (GenGauss prior)
        d log_p / d bg            = +S2_mu             (Gaussian, direct via μ)
        d log_p / d g0_obs        = +S2_g0             (Gaussian, via σ²)
        d log_p / d log_alpha_obs = +β · S3 − 1        (GenGauss prior)
        d log_p / d log_beta_obs  = −β · S4 + 1 + ψ(1/β) / β    (GenGauss prior)

    ``g0_obs`` and ``bg`` are independent kernel inputs; autograd
    upstream handles the chain rule when g0_obs is itself a function of bg
    (which it is for :class:`GaussianPoissonExcess` and any subclass that
    includes ``bg_var_factor·bg``).
    """

    @staticmethod
    def forward(
        ctx,
        log_mean: Tensor, n: Tensor, bg: Tensor, g0_obs: Tensor,
        log_alpha_obs: Tensor, log_beta_obs: Tensor,
        log_S_grid: Tensor, S_grid: Tensor, log_w: Tensor,
        g1: float, g2: float, var_floor: float,
    ) -> Tensor:
        if not HAVE_TRITON:
            raise RuntimeError("triton not available — caller should fall back")
        if not log_mean.is_cuda:
            raise RuntimeError("Triton path requires CUDA tensors")

        M = log_mean.shape[0]
        K = log_S_grid.shape[0]

        beta = log_beta_obs.exp()
        log_norm = torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)

        device = log_mean.device
        dtype = log_mean.dtype
        log_p = torch.empty(M, dtype=dtype, device=device)
        S1 = torch.empty(M, dtype=dtype, device=device)
        S2_mu = torch.empty(M, dtype=dtype, device=device)
        S2_g0 = torch.empty(M, dtype=dtype, device=device)
        S3 = torch.empty(M, dtype=dtype, device=device)
        S4 = torch.empty(M, dtype=dtype, device=device)

        cfg = _lookup_config(K)
        BLOCK_M, BLOCK_K = cfg["BLOCK_M"], cfg["BLOCK_K"]
        grid = (triton.cdiv(M, BLOCK_M),)
        _fwd_kernel_gaussian[grid](
            log_mean, n, bg, g0_obs,
            log_alpha_obs, log_beta_obs, log_norm,
            log_S_grid, S_grid, log_w,
            log_p, S1, S2_mu, S2_g0, S3, S4,
            M, K,
            g1=float(g1),
            g2=float(g2),
            var_floor=float(var_floor),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )

        ctx.save_for_backward(
            log_alpha_obs, log_beta_obs, S1, S2_mu, S2_g0, S3, S4,
        )
        return log_p

    @staticmethod
    def backward(ctx, grad_log_p: Tensor):
        log_alpha_obs, log_beta_obs, S1, S2_mu, S2_g0, S3, S4 = (
            ctx.saved_tensors
        )
        beta = log_beta_obs.exp()
        psi_recip_beta = torch.special.digamma(1.0 / beta)

        g_log_mean = grad_log_p * (beta * S1)
        g_bg = grad_log_p * S2_mu
        g_g0_obs = grad_log_p * S2_g0
        g_log_alpha_obs = grad_log_p * (beta * S3 - 1.0)
        g_log_beta_obs = grad_log_p * (-beta * S4 + 1.0 + psi_recip_beta / beta)

        return (
            g_log_mean, None, g_bg, g_g0_obs,
            g_log_alpha_obs, g_log_beta_obs,
            None, None, None,  # log_S_grid, S_grid, log_w (no grad)
            None, None, None,  # g1, g2, var_floor (constants)
        )


def marginalize_signal_triton_gaussian(
    log_mean: Tensor,
    n: Tensor,
    bg: Tensor,
    g0_obs: Tensor,
    log_alpha_obs: Tensor,
    log_beta_obs: Tensor,
    log_S_grid: Tensor,
    log_w: Tensor,
    g1: float,
    g2: float,
    var_floor: float = 1e-12,
) -> Tensor:
    """Return per-obs ``log_p`` of shape ``(M,)`` via the Gaussian Triton kernel.

    Variance model: ``σ²(μ, bg) = g0_obs + g1·μ + g2·μ²``.

    Parameters
    ----------
    log_mean : (M,)      per-obs MAP log-signal estimate.
    n : (M,)             observed photon counts (or "effective photons" at corrected gain).
    bg : (M,)            background per observation.
    g0_obs : (M,)        per-obs constant variance term σ²(μ=0) ≥ 0.
    log_alpha_obs : (M,) GenGauss prior α(h) → obs lookup, log-space.
    log_beta_obs : (M,)  GenGauss prior β(h) → obs lookup, log-space.
    log_S_grid : (K,)    log of the trapezoidal quadrature grid in S.
    log_w : (K,)         log of the trapezoidal weights.
    g1 : float           linear (Poisson) coefficient — variance per photon.
    g2 : float           quadratic (gain-switching) coefficient — γ_g².
    var_floor : float    hard minimum on σ² (default 1e-12).
    """
    S_grid = log_S_grid.exp()
    return _MarginalizeSignalTritonGaussian.apply(
        log_mean, n, bg, g0_obs,
        log_alpha_obs, log_beta_obs,
        log_S_grid, S_grid, log_w,
        float(g1), float(g2), float(var_floor),
    )
