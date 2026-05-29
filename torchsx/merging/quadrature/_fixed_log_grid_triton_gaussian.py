"""Triton fast path for ``FixedLogGrid.marginalize_signal`` with a
Gaussian-noise likelihood (``GaussianLikelihood`` and any subclass).

σ² is a **per-observation** input — a ``(M,)`` tensor, *constant across
the K-grid* during integration.  Physically σ is the detector noise on
the specific measurement; it is set by the data (or external
calibration) and does not depend on the latent-signal hypothesis being
integrated over.  See :mod:`torchsx.merging.likelihood.gaussian_base`
for the conceptual rationale.

This kernel is reusable across every Gaussian-shape likelihood
(``GaussianPolynomialVariance`` and any future variant): the caller
computes σ² however it likes (polynomial, calibration lookup, copied
from a stream file, …) and passes the ``(M,)`` tensor in.

Per integrand evaluation the kernel emits a single per-observation
``log_p`` plus five auxiliary statistics:

    S1            — for d/d log_mean       (GenGauss prior, same as Poisson kernel)
    S2_mu         — for d/d bg through μ   (Gaussian, ``(n − μ)/σ²``)
    S2_var        — for d/d var            (closed form, ``(n−μ)²/(2σ⁴) − 1/(2σ²)``)
    S3, S4        — for d/d log_alpha_obs, d/d log_beta_obs (GenGauss prior)
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


# Share the tuning lookup with the Poisson kernel.
from torchsx.merging.quadrature._fixed_log_grid_triton import _lookup_config


# ── Kernel ─────────────────────────────────────────────────────────────


if HAVE_TRITON:

    @triton.jit
    def _fwd_kernel_gaussian(
        log_mean_ptr, n_ptr, bg_ptr, var_ptr,
        log_alpha_ptr, log_beta_ptr, log_norm_ptr,
        log_S_grid_ptr, S_grid_ptr, log_w_ptr,
        log_p_ptr, S1_ptr, S2_mu_ptr, S2_var_ptr, S3_ptr, S4_ptr,
        M, K,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        log_mean = tl.load(log_mean_ptr + offs_m, mask=mask_m, other=0.0)
        n = tl.load(n_ptr + offs_m, mask=mask_m, other=0.0)
        bg = tl.load(bg_ptr + offs_m, mask=mask_m, other=0.0)
        var = tl.load(var_ptr + offs_m, mask=mask_m, other=1.0)
        log_alpha = tl.load(log_alpha_ptr + offs_m, mask=mask_m, other=0.0)
        log_beta = tl.load(log_beta_ptr + offs_m, mask=mask_m, other=0.0)
        log_norm = tl.load(log_norm_ptr + offs_m, mask=mask_m, other=0.0)

        # GenGauss prior scale / shape (per-HKL → per-obs lookup).
        alpha = tl.exp(log_alpha)
        alpha_safe = tl.where(alpha > 1e-8, alpha, 1e-8)
        inv_alpha_safe = 1.0 / alpha_safe
        beta = tl.exp(log_beta)

        # Per-m constants: σ² is independent of S, so log(σ²) and 1/σ²
        # are computed once per m (outside the K-loop) and added in.
        inv_var = 1.0 / var
        log_var = 0.6931471805599453 * tl.math.log2(var)
        # Constant in the Gaussian log-PDF: -½ · log(2π) = -0.91893853...
        # Fold log_norm (GenGauss), -log_alpha (GenGauss), -½ log(2π),
        # and -½ log(σ²) into a single per-m additive constant on log_int.
        log_const = log_norm - log_alpha - 0.9189385332046727 - 0.5 * log_var

        m_max = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
        T0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T2_mu = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T2_var = tl.zeros((BLOCK_M,), dtype=tl.float32)
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

            # ── Gaussian log-PDF for n | μ, σ² ────────────────────────
            mu = S_k[None, :] + bg[:, None]
            mu = tl.where(mu > 1e-6, mu, 1e-6)
            diff = n[:, None] - mu
            # log_lik = -½(log 2π + log σ² + (n-μ)²/σ²)
            # The −½(log 2π + log σ²) constant is folded into log_const above.
            log_lik_var = -0.5 * diff * diff * inv_var[:, None]

            log_int = (
                -z_pow
                + log_const[:, None]
                + log_lik_var
                + log_w_k[None, :]
            )
            log_int = tl.where(mask_k[None, :], log_int, -1e30)

            m_chunk = tl.max(log_int, axis=1)
            m_new = tl.maximum(m_max, m_chunk)
            scale = tl.exp(m_max - m_new)
            T0 = T0 * scale
            T1 = T1 * scale
            T2_mu = T2_mu * scale
            T2_var = T2_var * scale
            T3 = T3 * scale
            T4 = T4 * scale

            e_shift = tl.exp(log_int - m_new[:, None])

            # GenGauss-prior gradient bits (T1, T3, T4): same as Poisson kernel.
            sign_eps = tl.where(eps > 0, 1.0, tl.where(eps < 0, -1.0, 0.0))
            grad_lm = sign_eps * (z_pow / z_eff) * inv_alpha_safe[:, None]

            # ∂log_lik/∂μ at fixed σ²:  (n − μ) / σ²
            dlogp_dmu = diff * inv_var[:, None]

            # ∂log_lik/∂var  (σ² treated as the parameter):
            #   (n − μ)² / (2 σ⁴) − 1 / (2 σ²)
            dlogp_dvar = (
                0.5 * diff * diff * inv_var[:, None] * inv_var[:, None]
                - 0.5 * inv_var[:, None]
            )

            T0 += tl.sum(e_shift, axis=1)
            T1 += tl.sum(e_shift * grad_lm, axis=1)
            T2_mu += tl.sum(e_shift * dlogp_dmu, axis=1)
            T2_var += tl.sum(e_shift * dlogp_dvar, axis=1)
            T3 += tl.sum(e_shift * z_pow, axis=1)
            T4 += tl.sum(e_shift * z_pow * log_z_eff, axis=1)

            m_max = m_new

        log_p = m_max + 0.6931471805599453 * tl.math.log2(T0)
        inv_T0 = 1.0 / T0

        tl.store(log_p_ptr + offs_m, log_p, mask=mask_m)
        tl.store(S1_ptr + offs_m, T1 * inv_T0, mask=mask_m)
        tl.store(S2_mu_ptr + offs_m, T2_mu * inv_T0, mask=mask_m)
        tl.store(S2_var_ptr + offs_m, T2_var * inv_T0, mask=mask_m)
        tl.store(S3_ptr + offs_m, T3 * inv_T0, mask=mask_m)
        tl.store(S4_ptr + offs_m, T4 * inv_T0, mask=mask_m)


# ── Autograd.Function wrapper ──────────────────────────────────────────


class _MarginalizeSignalTritonGaussian(torch.autograd.Function):
    """Per-obs ``log_p`` for Gaussian likelihood with per-observation σ².

    Gradients (closed form via sufficient stats; σ² is treated as a
    per-obs parameter, constant in S during the integration):

        d log_p / d log_mean      = +β · S1            (GenGauss prior)
        d log_p / d bg            = +S2_mu             ((n − μ)/σ², direct via μ)
        d log_p / d var           = +S2_var            ((n−μ)²/2σ⁴ − 1/2σ², via σ²)
        d log_p / d log_alpha_obs = +β · S3 − 1        (GenGauss prior)
        d log_p / d log_beta_obs  = −β · S4 + 1 + ψ(1/β)/β   (GenGauss prior)

    Autograd handles the chain rule when ``var`` is itself a function
    of upstream parameters (e.g. ``GaussianPolynomialVariance``'s
    coefficients): the kernel returns ∂log_p/∂var honestly, and the
    upstream graph propagates from there.
    """

    @staticmethod
    def forward(
        ctx,
        log_mean: Tensor, n: Tensor, bg: Tensor, var: Tensor,
        log_alpha_obs: Tensor, log_beta_obs: Tensor,
        log_S_grid: Tensor, S_grid: Tensor, log_w: Tensor,
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
        S2_var = torch.empty(M, dtype=dtype, device=device)
        S3 = torch.empty(M, dtype=dtype, device=device)
        S4 = torch.empty(M, dtype=dtype, device=device)

        cfg = _lookup_config(K)
        BLOCK_M, BLOCK_K = cfg["BLOCK_M"], cfg["BLOCK_K"]
        grid = (triton.cdiv(M, BLOCK_M),)
        _fwd_kernel_gaussian[grid](
            log_mean, n, bg, var,
            log_alpha_obs, log_beta_obs, log_norm,
            log_S_grid, S_grid, log_w,
            log_p, S1, S2_mu, S2_var, S3, S4,
            M, K,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )

        ctx.save_for_backward(
            log_alpha_obs, log_beta_obs, S1, S2_mu, S2_var, S3, S4,
        )
        return log_p

    @staticmethod
    def backward(ctx, grad_log_p: Tensor):
        log_alpha_obs, log_beta_obs, S1, S2_mu, S2_var, S3, S4 = (
            ctx.saved_tensors
        )
        beta = log_beta_obs.exp()
        psi_recip_beta = torch.special.digamma(1.0 / beta)

        g_log_mean = grad_log_p * (beta * S1)
        g_bg = grad_log_p * S2_mu
        g_var = grad_log_p * S2_var
        g_log_alpha_obs = grad_log_p * (beta * S3 - 1.0)
        g_log_beta_obs = grad_log_p * (-beta * S4 + 1.0 + psi_recip_beta / beta)

        return (
            g_log_mean, None, g_bg, g_var,
            g_log_alpha_obs, g_log_beta_obs,
            None, None, None,   # log_S_grid, S_grid, log_w
        )


def marginalize_signal_triton_gaussian(
    log_mean: Tensor,
    n: Tensor,
    bg: Tensor,
    var: Tensor,
    log_alpha_obs: Tensor,
    log_beta_obs: Tensor,
    log_S_grid: Tensor,
    log_w: Tensor,
) -> Tensor:
    """Return per-obs ``log_p`` of shape ``(M,)`` via the Gaussian Triton kernel.

    The Gaussian likelihood is ``N(n; μ=S+bg, σ²=var[m])`` with σ²
    held constant in S during the integration — σ² is a property of
    the observation, not of the latent signal hypothesis.

    Parameters
    ----------
    log_mean : (M,)      per-obs log of the GenGauss prior mode (MAP iterate).
    n : (M,)             observed photon counts (or "effective photons" at corrected gain).
    bg : (M,)            background per observation.
    var : (M,)           per-observation σ², constant in S inside the kernel.
                         Caller is responsible for clamping above any floor.
    log_alpha_obs : (M,) GenGauss prior α(h) → obs lookup, log-space.
    log_beta_obs : (M,)  GenGauss prior β(h) → obs lookup, log-space.
    log_S_grid : (K,)    log of the trapezoidal quadrature grid in S.
    log_w : (K,)         log of the trapezoidal weights.
    """
    S_grid = log_S_grid.exp()
    return _MarginalizeSignalTritonGaussian.apply(
        log_mean, n, bg, var,
        log_alpha_obs, log_beta_obs,
        log_S_grid, S_grid, log_w,
    )
