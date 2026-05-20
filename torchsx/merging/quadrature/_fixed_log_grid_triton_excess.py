"""Triton fast path for ``FixedLogGrid.marginalize_signal`` under the
``PoissonWithExcessNoise`` likelihood — Poisson with a Gauss-Hermite
marginalised Gaussian perturbation on the background:

    n | bg_true ~ Poisson(S + bg_true)
    bg_true ~ Normal(bg_obs, σ²_excess),  σ²_excess = a · bg + read_var

The eager path lives in
:meth:`~torchsx.merging.likelihood.PoissonWithExcessNoise.log_prob_given_signal`
and was previously the bottleneck — the (M, K, n_gh) integrand never
fits comfortably in HBM. This kernel folds the n_gh dimension into the
existing K-loop logsumexp, so the integrand stays on-chip.

Structure mirrors :mod:`_fixed_log_grid_triton`: per (M, K) tile we
compute the n_gh-marginalised log-likelihood, multiply by the GenGauss
scatter density, accumulate the K-axis logsumexp via the running-max
trick, and emit a per-obs ``log_p`` plus the four sufficient statistics
``S1..S4`` for the analytic backward.
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

from torchsx.merging.quadrature._fixed_log_grid_triton import (
    _lookup_config,
)


if HAVE_TRITON:

    @triton.jit
    def _fwd_kernel_excess(
        log_mean_ptr, n_ph_ptr, bg_ptr, sigma_excess_ptr,
        log_alpha_ptr, log_beta_ptr, log_nfact_ptr, log_norm_ptr,
        log_S_grid_ptr, S_grid_ptr, log_w_ptr,
        gh_z_ptr, gh_logw_ptr,
        log_p_ptr, S1_ptr, S2_ptr, S3_ptr, S4_ptr,
        M, K, BG_VAR_FACTOR,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        N_GH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        log_mean = tl.load(log_mean_ptr + offs_m, mask=mask_m, other=0.0)
        n_ph = tl.load(n_ph_ptr + offs_m, mask=mask_m, other=0.0)
        bg = tl.load(bg_ptr + offs_m, mask=mask_m, other=0.0)
        sigma = tl.load(sigma_excess_ptr + offs_m, mask=mask_m, other=0.0)
        log_alpha = tl.load(log_alpha_ptr + offs_m, mask=mask_m, other=0.0)
        log_beta = tl.load(log_beta_ptr + offs_m, mask=mask_m, other=0.0)
        log_nfact = tl.load(log_nfact_ptr + offs_m, mask=mask_m, other=0.0)
        log_norm = tl.load(log_norm_ptr + offs_m, mask=mask_m, other=0.0)

        alpha = tl.exp(log_alpha)
        alpha_safe = tl.where(alpha > 1e-8, alpha, 1e-8)
        inv_alpha_safe = 1.0 / alpha_safe
        beta = tl.exp(log_beta)
        log_scatter_const = log_norm - log_alpha
        log_lik_const = -log_nfact

        # GH nodes/weights are loaded scalar-by-scalar inside the unrolled
        # inner loop below (tl.arange requires power-of-2 ranges, and
        # N_GH=5 in practice).

        m_max = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
        T0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T4 = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            log_S_k = tl.load(log_S_grid_ptr + offs_k, mask=mask_k, other=0.0)
            S_k = tl.load(S_grid_ptr + offs_k, mask=mask_k, other=0.0)
            log_w_k = tl.load(log_w_ptr + offs_k, mask=mask_k, other=0.0)

            # Scatter (GenGauss) — independent of GH point.
            eps = log_S_k[None, :] - log_mean[:, None]
            z = tl.abs(eps) * inv_alpha_safe[:, None]
            z_eff = tl.where(z > 1e-12, z, 1e-12)
            log_z_eff = 0.6931471805599453 * tl.math.log2(z_eff)
            z_pow = tl.exp(beta[:, None] * log_z_eff)

            # ── GH-marginalised log-likelihood per (m, k) ──
            # Manual 5-way unroll: the 5 log-likelihoods only differ by
            # σ·z_j in bg_true → no cross-j dependency, the compiler
            # pipelines all 5 (log2, lam, mul, exp) chains through the
            # SFU. Two-phase logsumexp (offline-max then sum-exp) avoids
            # the running-max sequential dep that bottlenecked the
            # previous version. Hardcoded to N_GH=5 (the only production
            # value); add more branches if larger n_gh is ever needed.
            tl.static_assert(N_GH == 5, "Triton excess kernel hardcoded to N_GH=5")
            sigma_safe = tl.where(sigma > 1e-12, sigma, 1e-12)
            dsigma_dbg = BG_VAR_FACTOR / (2.0 * sigma_safe)   # (BLOCK_M,)

            z0 = tl.load(gh_z_ptr + 0); logw0 = tl.load(gh_logw_ptr + 0)
            z1 = tl.load(gh_z_ptr + 1); logw1 = tl.load(gh_logw_ptr + 1)
            z2 = tl.load(gh_z_ptr + 2); logw2 = tl.load(gh_logw_ptr + 2)
            z3 = tl.load(gh_z_ptr + 3); logw3 = tl.load(gh_logw_ptr + 3)
            z4 = tl.load(gh_z_ptr + 4); logw4 = tl.load(gh_logw_ptr + 4)

            # Phase 1: independent per-j log_lik + chain-rule grad term.
            # Use a helper-shaped expression so each variable's dataflow
            # is independent → compiler issues SFU ops in parallel.
            br0 = bg + sigma * z0; c0 = br0 <= 0.0
            br1 = bg + sigma * z1; c1 = br1 <= 0.0
            br2 = bg + sigma * z2; c2 = br2 <= 0.0
            br3 = bg + sigma * z3; c3 = br3 <= 0.0
            br4 = bg + sigma * z4; c4 = br4 <= 0.0
            bt0 = tl.where(c0, 0.0, br0)
            bt1 = tl.where(c1, 0.0, br1)
            bt2 = tl.where(c2, 0.0, br2)
            bt3 = tl.where(c3, 0.0, br3)
            bt4 = tl.where(c4, 0.0, br4)
            lr0 = S_k[None, :] + bt0[:, None]; lc0 = lr0 <= 1e-6; lam0 = tl.where(lc0, 1e-6, lr0)
            lr1 = S_k[None, :] + bt1[:, None]; lc1 = lr1 <= 1e-6; lam1 = tl.where(lc1, 1e-6, lr1)
            lr2 = S_k[None, :] + bt2[:, None]; lc2 = lr2 <= 1e-6; lam2 = tl.where(lc2, 1e-6, lr2)
            lr3 = S_k[None, :] + bt3[:, None]; lc3 = lr3 <= 1e-6; lam3 = tl.where(lc3, 1e-6, lr3)
            lr4 = S_k[None, :] + bt4[:, None]; lc4 = lr4 <= 1e-6; lam4 = tl.where(lc4, 1e-6, lr4)
            LN2 = 0.6931471805599453
            base = log_lik_const[:, None]
            np_bm = n_ph[:, None]
            ll0 = np_bm * LN2 * tl.math.log2(lam0) - lam0 + base + logw0
            ll1 = np_bm * LN2 * tl.math.log2(lam1) - lam1 + base + logw1
            ll2 = np_bm * LN2 * tl.math.log2(lam2) - lam2 + base + logw2
            ll3 = np_bm * LN2 * tl.math.log2(lam3) - lam3 + base + logw3
            ll4 = np_bm * LN2 * tl.math.log2(lam4) - lam4 + base + logw4
            mk = mask_k[None, :]
            ll0 = tl.where(mk, ll0, -1e30)
            ll1 = tl.where(mk, ll1, -1e30)
            ll2 = tl.where(mk, ll2, -1e30)
            ll3 = tl.where(mk, ll3, -1e30)
            ll4 = tl.where(mk, ll4, -1e30)

            # d log Poisson / d bg per j; both clamps zero the chain.
            ch0 = tl.where(c0, 0.0, 1.0 + z0 * dsigma_dbg)
            ch1 = tl.where(c1, 0.0, 1.0 + z1 * dsigma_dbg)
            ch2 = tl.where(c2, 0.0, 1.0 + z2 * dsigma_dbg)
            ch3 = tl.where(c3, 0.0, 1.0 + z3 * dsigma_dbg)
            ch4 = tl.where(c4, 0.0, 1.0 + z4 * dsigma_dbg)
            g0 = tl.where(lc0, 0.0, (np_bm / lam0 - 1.0) * ch0[:, None])
            g1 = tl.where(lc1, 0.0, (np_bm / lam1 - 1.0) * ch1[:, None])
            g2 = tl.where(lc2, 0.0, (np_bm / lam2 - 1.0) * ch2[:, None])
            g3 = tl.where(lc3, 0.0, (np_bm / lam3 - 1.0) * ch3[:, None])
            g4 = tl.where(lc4, 0.0, (np_bm / lam4 - 1.0) * ch4[:, None])

            # Phase 2: tree-reduce max, then 5 independent exps + sum.
            m01 = tl.maximum(ll0, ll1)
            m23 = tl.maximum(ll2, ll3)
            m_lik = tl.maximum(tl.maximum(m01, m23), ll4)
            e0 = tl.exp(ll0 - m_lik)
            e1 = tl.exp(ll1 - m_lik)
            e2 = tl.exp(ll2 - m_lik)
            e3 = tl.exp(ll3 - m_lik)
            e4 = tl.exp(ll4 - m_lik)
            sum_lik = e0 + e1 + e2 + e3 + e4
            num_d_bg = e0 * g0 + e1 * g1 + e2 * g2 + e3 * g3 + e4 * g4
            log_lik_marg = m_lik + LN2 * tl.math.log2(sum_lik)
            d_loglik_d_bg = num_d_bg / sum_lik

            log_int = (
                -z_pow
                + log_scatter_const[:, None]
                + log_lik_marg
                + log_w_k[None, :]
            )
            log_int = tl.where(mask_k[None, :], log_int, -1e30)

            m_chunk = tl.max(log_int, axis=1)
            m_new = tl.maximum(m_max, m_chunk)
            scale = tl.exp(m_max - m_new)
            T0 = T0 * scale
            T1 = T1 * scale
            T2 = T2 * scale
            T3 = T3 * scale
            T4 = T4 * scale

            e_shift = tl.exp(log_int - m_new[:, None])

            sign_eps = tl.where(eps > 0, 1.0, tl.where(eps < 0, -1.0, 0.0))
            grad_lm = sign_eps * (z_pow / z_eff) * inv_alpha_safe[:, None]

            T0 += tl.sum(e_shift, axis=1)
            T1 += tl.sum(e_shift * grad_lm, axis=1)
            T2 += tl.sum(e_shift * d_loglik_d_bg, axis=1)
            T3 += tl.sum(e_shift * z_pow, axis=1)
            T4 += tl.sum(e_shift * z_pow * log_z_eff, axis=1)

            m_max = m_new

        log_p = m_max + 0.6931471805599453 * tl.math.log2(T0)
        inv_T0 = 1.0 / T0

        tl.store(log_p_ptr + offs_m, log_p, mask=mask_m)
        tl.store(S1_ptr + offs_m, T1 * inv_T0, mask=mask_m)
        tl.store(S2_ptr + offs_m, T2 * inv_T0, mask=mask_m)
        tl.store(S3_ptr + offs_m, T3 * inv_T0, mask=mask_m)
        tl.store(S4_ptr + offs_m, T4 * inv_T0, mask=mask_m)


# ── Autograd.Function wrapper ──────────────────────────────────────────


class _MarginalizeSignalTritonExcess(torch.autograd.Function):
    """Per-obs ``log_p`` for Poisson + GH-marginalised excess noise.

    Backward analytic formulas are identical in shape to the vanilla
    Poisson kernel — the only change is that S2 (the bg-gradient) is now
    the posterior-weighted ``d log p_lik / d bg`` *under* the inner GH
    marginal, accumulated inside the kernel:

        d log_p / d log_mean      = +β · S1
        d log_p / d bg            = +S2                ←  GH-marginalised
        d log_p / d log_alpha_obs = +β · S3 − 1
        d log_p / d log_beta_obs  = −β · S4 + 1 + ψ(1/β) / β
    """

    @staticmethod
    def forward(
        ctx,
        log_mean: Tensor, n_ph: Tensor, bg: Tensor, sigma_excess: Tensor,
        log_alpha_obs: Tensor, log_beta_obs: Tensor,
        log_S_grid: Tensor, S_grid: Tensor, log_w: Tensor, log_nfact: Tensor,
        gh_z: Tensor, gh_logw: Tensor, bg_var_factor: float,
    ) -> Tensor:
        if not HAVE_TRITON:
            raise RuntimeError("triton not available — caller should fall back")
        if not log_mean.is_cuda:
            raise RuntimeError("Triton path requires CUDA tensors")

        M = log_mean.shape[0]
        K = log_S_grid.shape[0]
        N_GH = gh_z.shape[0]

        beta = log_beta_obs.exp()
        log_norm = torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)

        device = log_mean.device
        dtype = log_mean.dtype
        log_p = torch.empty(M, dtype=dtype, device=device)
        S1 = torch.empty(M, dtype=dtype, device=device)
        S2 = torch.empty(M, dtype=dtype, device=device)
        S3 = torch.empty(M, dtype=dtype, device=device)
        S4 = torch.empty(M, dtype=dtype, device=device)

        cfg = _lookup_config(K)
        BLOCK_M, BLOCK_K = cfg["BLOCK_M"], cfg["BLOCK_K"]
        grid = (triton.cdiv(M, BLOCK_M),)
        _fwd_kernel_excess[grid](
            log_mean, n_ph, bg, sigma_excess,
            log_alpha_obs, log_beta_obs, log_nfact, log_norm,
            log_S_grid, S_grid, log_w,
            gh_z, gh_logw,
            log_p, S1, S2, S3, S4,
            M, K, float(bg_var_factor),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, N_GH=N_GH,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )

        ctx.save_for_backward(log_alpha_obs, log_beta_obs, S1, S2, S3, S4)
        return log_p

    @staticmethod
    def backward(ctx, grad_log_p: Tensor):
        log_alpha_obs, log_beta_obs, S1, S2, S3, S4 = ctx.saved_tensors
        beta = log_beta_obs.exp()
        psi_recip_beta = torch.special.digamma(1.0 / beta)

        g_log_mean = grad_log_p * (beta * S1)
        g_bg = grad_log_p * S2
        g_log_alpha_obs = grad_log_p * (beta * S3 - 1.0)
        g_log_beta_obs = grad_log_p * (-beta * S4 + 1.0 + psi_recip_beta / beta)

        return (
            g_log_mean, None, g_bg, None,
            g_log_alpha_obs, g_log_beta_obs,
            None, None, None, None, None, None, None,
        )


def marginalize_signal_triton_excess(
    log_mean: Tensor,
    n_ph: Tensor,
    bg: Tensor,
    sigma_excess: Tensor,
    log_alpha_obs: Tensor,
    log_beta_obs: Tensor,
    log_S_grid: Tensor,
    log_w: Tensor,
    gh_z: Tensor,
    gh_logw: Tensor,
    bg_var_factor: float,
    *,
    log_nfact: Tensor | None = None,
) -> Tensor:
    """Return per-obs ``log_p`` of shape ``(M,)`` via the Triton kernel
    for Poisson + GH-marginalised Gaussian excess noise on bg.

    ``bg_var_factor`` is the per-obs σ² slope (σ² = bg_var_factor·bg +
    read_var); the kernel uses it for the σ-via-bg chain-rule term in
    the bg gradient.
    """
    if log_nfact is None:
        log_nfact = torch.lgamma(n_ph + 1)
    S_grid = log_S_grid.exp()
    return _MarginalizeSignalTritonExcess.apply(
        log_mean, n_ph, bg, sigma_excess,
        log_alpha_obs, log_beta_obs,
        log_S_grid, S_grid, log_w, log_nfact,
        gh_z, gh_logw, float(bg_var_factor),
    )
