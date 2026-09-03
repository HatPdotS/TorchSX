"""Triton kernel for :class:`SplitLogGrid`'s window integral.

Same integrand, same online logsumexp and the same four gated aux statistics
as ``_fixed_log_grid_triton._fwd_kernel``; the differences are that the node
positions are built per observation from eight ``(M,)`` grid parameters, and
that the trapezoid weight enters as a multiplier of the exponential instead of
a log-weight in the exponent, so the non-uniform spacing costs no extra
transcendental.  The running maximum is taken over the unweighted integrand;
every weight is below one (spacings are fractions of a nat), so it remains an
upper bound and nothing can overflow.

The analytic tail is added outside, in torch, where autograd differentiates
the incomplete gamma.
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

from torchsx.merging.quadrature._fixed_log_grid_triton import _lookup_config

_LN2 = 0.6931471805599453


if HAVE_TRITON:

    @triton.jit
    def _node(kk, x_lo, h1, d_lo, h_d, d_hi, h2, n1, n1d):
        kf = kk.to(tl.float32)
        lo = x_lo + kf * h1
        mid = d_lo + (kf - n1.to(tl.float32)) * h_d
        hi = d_hi + (kf - n1d.to(tl.float32) + 1.0) * h2
        return tl.where(kk < n1, lo, tl.where(kk < n1d, mid, hi))

    @triton.jit
    def _split_fwd_kernel(
        log_mean_ptr, n_ph_ptr, bg_ptr,
        log_alpha_ptr, log_beta_ptr, logpeak_ptr, log_norm_ptr, sig_b2_ptr,
        log_alpha_hi_ptr,
        x_lo_ptr, h1_ptr, d_lo_ptr, h_d_ptr, d_hi_ptr, h2_ptr, n1_ptr, n1d_ptr,
        log_p_ptr, S1_ptr, S2_ptr, S3_ptr, S4_ptr, S5_ptr,
        M, K,
        ASYM: tl.constexpr,
        BG_VAR: tl.constexpr,
        NEED_MU: tl.constexpr,
        NEED_AB: tl.constexpr,
        NEED_BG: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        log_mean = tl.load(log_mean_ptr + offs_m, mask=mask_m, other=0.0)
        n_ph = tl.load(n_ph_ptr + offs_m, mask=mask_m, other=0.0)
        bg = tl.load(bg_ptr + offs_m, mask=mask_m, other=0.0)
        log_alpha = tl.load(log_alpha_ptr + offs_m, mask=mask_m, other=0.0)
        log_beta = tl.load(log_beta_ptr + offs_m, mask=mask_m, other=0.0)
        logpeak = tl.load(logpeak_ptr + offs_m, mask=mask_m, other=0.0)
        if BG_VAR:
            sig_b2 = tl.load(sig_b2_ptr + offs_m, mask=mask_m, other=0.0)
        log_norm = tl.load(log_norm_ptr + offs_m, mask=mask_m, other=0.0)
        x_lo = tl.load(x_lo_ptr + offs_m, mask=mask_m, other=0.0)
        h1 = tl.load(h1_ptr + offs_m, mask=mask_m, other=0.0)
        d_lo = tl.load(d_lo_ptr + offs_m, mask=mask_m, other=0.0)
        h_d = tl.load(h_d_ptr + offs_m, mask=mask_m, other=0.0)
        d_hi = tl.load(d_hi_ptr + offs_m, mask=mask_m, other=0.0)
        h2 = tl.load(h2_ptr + offs_m, mask=mask_m, other=0.0)
        n1 = tl.load(n1_ptr + offs_m, mask=mask_m, other=0)
        n1d = tl.load(n1d_ptr + offs_m, mask=mask_m, other=0)

        alpha = tl.exp(log_alpha)
        alpha_safe = tl.where(alpha > 1e-8, alpha, 1e-8)
        inv_alpha_safe = 1.0 / alpha_safe
        if ASYM:
            # two-width family: alpha below the centre, alpha_hi above; the
            # normaliser is 1/(alpha + alpha_hi) in place of 1/(2 alpha)
            log_alpha_hi = tl.load(log_alpha_hi_ptr + offs_m, mask=mask_m, other=0.0)
            alpha_hi = tl.exp(log_alpha_hi)
            alpha_hi_safe = tl.where(alpha_hi > 1e-8, alpha_hi, 1e-8)
            inv_alpha_hi_safe = 1.0 / alpha_hi_safe
            log_scatter_const = (log_norm + 0.6931471805599453
                                 - 0.6931471805599453 * tl.math.log2(alpha_safe + alpha_hi_safe))
        else:
            inv_alpha_hi_safe = inv_alpha_safe
            log_scatter_const = log_norm - log_alpha
        beta = tl.exp(log_beta)
        inv_n = 1.0 / tl.where(n_ph > 1e-12, n_ph, 1e-12)

        m_max = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
        T0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T4 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        T5 = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            kk = offs_k[None, :]
            x_k = _node(kk, x_lo[:, None], h1[:, None], d_lo[:, None], h_d[:, None],
                        d_hi[:, None], h2[:, None], n1[:, None], n1d[:, None])
            k_prev = tl.maximum(kk - 1, 0)
            k_next = tl.minimum(kk + 1, K - 1)
            x_prev = _node(k_prev, x_lo[:, None], h1[:, None], d_lo[:, None], h_d[:, None],
                           d_hi[:, None], h2[:, None], n1[:, None], n1d[:, None])
            x_next = _node(k_next, x_lo[:, None], h1[:, None], d_lo[:, None], h_d[:, None],
                           d_hi[:, None], h2[:, None], n1[:, None], n1d[:, None])
            w_k = 0.5 * (x_next - x_prev)
            S_k = tl.exp(x_k)

            eps = x_k - log_mean[:, None]
            inv_a_k = tl.where(eps > 0, inv_alpha_hi_safe[:, None], inv_alpha_safe[:, None])
            z = tl.abs(eps) * inv_a_k
            z_eff = tl.where(z > 1e-12, z, 1e-12)
            log_z_eff = 0.6931471805599453 * tl.math.log2(z_eff)
            z_pow = tl.exp(beta[:, None] * log_z_eff)

            lam = S_k + bg[:, None]
            lam = tl.where(lam > 1e-6, lam, 1e-6)
            neg_half_D = (n_ph[:, None] * 0.6931471805599453
                          * tl.math.log2(lam * inv_n[:, None])
                          - (lam - n_ph[:, None]))
            if BG_VAR:
                inv_lps2 = 1.0 / (lam + sig_b2[:, None])
                inv_phi = lam * inv_lps2
                neg_half_D = (neg_half_D * inv_phi
                              + 0.5 * 0.6931471805599453 * tl.math.log2(inv_phi))
            log_int = -z_pow + log_scatter_const[:, None] + neg_half_D + logpeak[:, None]
            log_int = tl.where(mask_k[None, :], log_int, -1e30)
            w_k = tl.where(mask_k[None, :], w_k, 0.0)

            m_chunk = tl.max(log_int, axis=1)
            m_new = tl.maximum(m_max, m_chunk)
            scale = tl.exp(m_max - m_new)
            T0 = T0 * scale
            if NEED_MU:
                T1 = T1 * scale
            if NEED_BG:
                T2 = T2 * scale
            if NEED_AB:
                T3 = T3 * scale
                T4 = T4 * scale
                T5 = T5 * scale

            e_shift = tl.exp(log_int - m_new[:, None]) * w_k
            T0 += tl.sum(e_shift, axis=1)
            if NEED_MU:
                sign_eps = tl.where(eps > 0, 1.0, tl.where(eps < 0, -1.0, 0.0))
                T1 += tl.sum(
                    e_shift * sign_eps * (z_pow / z_eff) * inv_a_k,
                    axis=1,
                )
            if NEED_BG:
                T2 += tl.sum(e_shift * (n_ph[:, None] / lam - 1.0), axis=1)
            if NEED_AB:
                if ASYM:
                    # split E[z^beta] by side: the lower width's gradient reads
                    # the eps <= 0 part, the upper width's the eps > 0 part
                    T3 += tl.sum(tl.where(eps > 0, 0.0, e_shift * z_pow), axis=1)
                    T5 += tl.sum(tl.where(eps > 0, e_shift * z_pow, 0.0), axis=1)
                else:
                    T3 += tl.sum(e_shift * z_pow, axis=1)
                T4 += tl.sum(e_shift * z_pow * log_z_eff, axis=1)
            m_max = m_new

        T0s = tl.where(T0 > 1e-38, T0, 1e-38)
        log_p = m_max + 0.6931471805599453 * tl.math.log2(T0s)
        inv_T0 = 1.0 / T0s
        tl.store(log_p_ptr + offs_m, log_p, mask=mask_m)
        if NEED_MU:
            tl.store(S1_ptr + offs_m, T1 * inv_T0, mask=mask_m)
        if NEED_BG:
            tl.store(S2_ptr + offs_m, T2 * inv_T0, mask=mask_m)
        if NEED_AB:
            tl.store(S3_ptr + offs_m, T3 * inv_T0, mask=mask_m)
            tl.store(S4_ptr + offs_m, T4 * inv_T0, mask=mask_m)
            if ASYM:
                tl.store(S5_ptr + offs_m, T5 * inv_T0, mask=mask_m)


class _MarginalizeSignalSplitTriton(torch.autograd.Function):
    """Window integral with the analytic backward of the fixed-grid kernel.

    The gradient formulas are unchanged because the nodes and weights depend on
    the data only:  d/d log_mean = beta S1, d/d bg = S2,
    d/d log_alpha = beta S3 - 1, d/d log_beta = -beta S4 + 1 + psi(1/beta)/beta.

    Two-width family (``log_alpha_hi_obs`` given): S3 / S5 are E[z^beta] over
    the eps <= 0 / eps > 0 nodes, and with w_side = alpha_side/(alpha+alpha_hi)
    d/d log_alpha = beta S3 - w_lo, d/d log_alpha_hi = beta S5 - w_hi (their sum
    is the symmetric formula when the widths coincide).
    """

    @staticmethod
    def forward(ctx, log_mean, n_ph, bg, log_alpha_obs, log_beta_obs, logpeak,
                x_lo, h1, d_lo, h_d, d_hi, h2, n1, n1d, K,
                sig_b2=None, need_mu=True, need_ab=True, need_bg=True,
                log_alpha_hi_obs=None):
        M = log_mean.shape[0]
        beta = log_beta_obs.exp()
        log_norm = torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)
        device, dtype = log_mean.device, log_mean.dtype
        log_p = torch.empty(M, dtype=dtype, device=device)
        asym = log_alpha_hi_obs is not None

        def _buf(need):
            return torch.empty(M if need else 1, dtype=dtype, device=device)

        S1, S2 = _buf(need_mu), _buf(need_bg)
        S3, S4, S5 = _buf(need_ab), _buf(need_ab), _buf(need_ab and asym)
        cfg = _lookup_config(K)
        BLOCK_M, BLOCK_K = cfg["BLOCK_M"], cfg["BLOCK_K"]
        grid = (triton.cdiv(M, BLOCK_M),)
        _split_fwd_kernel[grid](
            log_mean, n_ph, bg, log_alpha_obs, log_beta_obs, logpeak, log_norm,
            sig_b2 if sig_b2 is not None else logpeak,
            log_alpha_hi_obs if asym else log_alpha_obs,
            x_lo, h1, d_lo, h_d, d_hi, h2, n1, n1d,
            log_p, S1, S2, S3, S4, S5,
            M, K,
            ASYM=asym,
            BG_VAR=sig_b2 is not None,
            NEED_MU=need_mu, NEED_AB=need_ab, NEED_BG=need_bg,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )
        ctx.need = (need_mu, need_ab, need_bg, asym)
        ctx.save_for_backward(log_alpha_obs, log_beta_obs, S1, S2, S3, S4, S5,
                              log_alpha_hi_obs if asym else log_alpha_obs)
        return log_p

    @staticmethod
    def backward(ctx, grad_log_p):
        log_alpha_obs, log_beta_obs, S1, S2, S3, S4, S5, log_alpha_hi_obs = ctx.saved_tensors
        need_mu, need_ab, need_bg, asym = ctx.need
        beta = log_beta_obs.exp()
        g_log_mean = grad_log_p * (beta * S1) if need_mu else None
        g_bg = grad_log_p * S2 if need_bg else None
        g_la_hi = None
        if need_ab:
            psi = torch.special.digamma(1.0 / beta)
            g_lb = grad_log_p * (-beta * S4 + 1.0 + psi / beta)
            if asym:
                log_sum = torch.logaddexp(log_alpha_obs, log_alpha_hi_obs)
                w_lo, w_hi = (log_alpha_obs - log_sum).exp(), (log_alpha_hi_obs - log_sum).exp()
                g_la = grad_log_p * (beta * S3 - w_lo)
                g_la_hi = grad_log_p * (beta * S5 - w_hi)
            else:
                g_la = grad_log_p * (beta * S3 - 1.0)
        else:
            g_la = g_lb = None
        return (g_log_mean, None, g_bg, g_la, g_lb) + (None,) * 14 + (g_la_hi,)


def marginalize_signal_split_triton(
    log_mean: Tensor, n_ph: Tensor, bg: Tensor,
    log_alpha_obs: Tensor, log_beta_obs: Tensor, p: dict, K: int,
    *, sig_b2: Tensor | None = None, log_alpha_hi_obs: Tensor | None = None,
) -> Tensor:
    """Window part of ``log p`` (no tail), shape ``(M,)``."""
    if not HAVE_TRITON or not log_mean.is_cuda:
        raise RuntimeError("split-grid Triton path needs CUDA + triton")
    n64 = n_ph.double()
    logpeak = (n64 * n64.clamp(min=1e-300).log() - n64
               - torch.lgamma(n64 + 1)).to(log_mean.dtype)
    logpeak = torch.where(n_ph > 0, logpeak, torch.zeros_like(logpeak))
    on = torch.is_grad_enabled()
    need_bg = on and bg.requires_grad
    if need_bg and sig_b2 is not None:
        raise RuntimeError("bg requires grad while the background-variance "
                           "correction is on; the analytic d log_p/d bg does not cover it")
    return _MarginalizeSignalSplitTriton.apply(
        log_mean.contiguous(), n_ph.contiguous(), bg.contiguous(),
        log_alpha_obs.contiguous(), log_beta_obs.contiguous(), logpeak,
        p["x_lo"], p["h1"], p["d_lo"], p["h_d"], p["d_hi"], p["h2"], p["n1"], p["n1d"],
        int(K), sig_b2,
        on and log_mean.requires_grad,
        on and (log_alpha_obs.requires_grad or log_beta_obs.requires_grad
                or (log_alpha_hi_obs is not None and log_alpha_hi_obs.requires_grad)),
        need_bg,
        None if log_alpha_hi_obs is None else log_alpha_hi_obs.contiguous(),
    )
