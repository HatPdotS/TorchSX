"""Triton fast path for ``FixedLogGrid.marginalize_signal``.

For ``GenGaussLogScatter`` × ``Poisson`` only (the fastest configurations
in production). Other scatter/likelihood combinations fall back to the
eager PyTorch path in :mod:`fixed_log_grid`.

Per integrand evaluation the kernel emits a single per-observation
``log_p`` plus up to four auxiliary statistics ``S1..S4`` (weighted softmax
sums over the K grid). Backward then applies analytic gradient formulas
without recomputing the (M, K) integrand.

Each statistic is gated by a ``constexpr`` derived from whether the matching
input actually requires grad, because they are not cheap: an ablation probe
with four of the five accumulators removed ran 25.1 ms -> 13.5 ms at
M = 1.05e6, K = 2001, so together they are ~47 % of the kernel.  ``S2``
(``d log p/d bg``) is the clearest case — ``bg`` is data except under the
learned-background ``bg_mix`` mode — and the ``locate`` sweep in
:mod:`torchsx.nn.posterior_target` runs under ``no_grad`` and needs none of
them.  The gates read ``requires_grad``, so no caller has to opt in.

Tuning is read from ``_fixed_log_grid_tuning.json`` (in-package),
keyed by GPU name and K bucket. On unknown GPUs we fall back to the
A100 winners with a one-time warning. To extend the cache for a new
GPU, run a tuning sweep externally and add the entry; the measured
state and the traps to avoid are in ``docs/optimize_integration.md``
(the sweep scripts themselves are no longer in the repo).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                    # pragma: no cover
    HAVE_TRITON = False


_TUNING_PATH = Path(__file__).resolve().parent / "_fixed_log_grid_tuning.json"

# A100 winners from a 14-config autotune sweep on the v4 kernel. Used as
# the fallback when no per-GPU entry exists in the JSON cache.
_FALLBACK_CONFIG: dict[str, dict[str, int]] = {
    "K<=1024": {"BLOCK_M": 64, "BLOCK_K": 16, "num_warps": 4, "num_stages": 3},
    "K>1024":  {"BLOCK_M": 64, "BLOCK_K": 16, "num_warps": 4, "num_stages": 2},
}


# ── Tuning lookup ──────────────────────────────────────────────────────


def _bucket(K: int) -> str:
    return "K<=1024" if K <= 1024 else "K>1024"


def _gpu_name() -> str:
    if not torch.cuda.is_available():
        return "no_cuda"
    return torch.cuda.get_device_name(0)


_TUNING_MEMO: dict[tuple[str, int], dict[str, Any]] = {}


def _lookup_config(K: int) -> dict[str, Any]:
    """Return ``{BLOCK_M, BLOCK_K, num_warps, num_stages}`` for current GPU + K.

    Memoized — JSON is read at most once per (GPU, K) per process.
    """
    gpu = _gpu_name()
    memo_key = (gpu, K)
    if memo_key in _TUNING_MEMO:
        return _TUNING_MEMO[memo_key]

    bucket = _bucket(K)
    cache: dict[str, Any] = {}
    if _TUNING_PATH.exists():
        try:
            cache = json.loads(_TUNING_PATH.read_text())
        except json.JSONDecodeError:
            cache = {}

    hit = cache.get(gpu, {}).get(bucket)
    if hit is None and "A100" in gpu:
        # A100 variants share the same kernel-tuning sweet spot.
        for k, v in cache.items():
            if "A100" in k and bucket in v:
                hit = v[bucket]
                break

    if hit is None:
        hit = _FALLBACK_CONFIG[bucket]
        if not getattr(_lookup_config, "_warned", False):
            print(
                f"[torchsx fixed_log_grid] no tuned config for ({gpu!r}, "
                f"{bucket}); using fallback {hit}",
                flush=True,
            )
            _lookup_config._warned = True

    _TUNING_MEMO[memo_key] = dict(hit)
    return _TUNING_MEMO[memo_key]


# ── Kernel ─────────────────────────────────────────────────────────────


if HAVE_TRITON:

    @triton.jit
    def _fwd_kernel(
        log_mean_ptr, n_ph_ptr, bg_ptr,
        log_alpha_ptr, log_beta_ptr, logpeak_ptr, log_norm_ptr, sig_b2_ptr,
        log_S_grid_ptr, S_grid_ptr, log_w_ptr,
        log_p_ptr, S1_ptr, S2_ptr, S3_ptr, S4_ptr,
        M, K,
        BG_VAR: tl.constexpr,
        BG_DEVIANCE: tl.constexpr,
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

        alpha = tl.exp(log_alpha)
        alpha_safe = tl.where(alpha > 1e-8, alpha, 1e-8)
        inv_alpha_safe = 1.0 / alpha_safe
        beta = tl.exp(log_beta)
        log_scatter_const = log_norm - log_alpha
        # Reciprocal of n is per-OBSERVATION, so hoist it: as a division
        # inside the k loop it is an SFU op evaluated M*K times and cost 29 %
        # of the kernel; as a multiply against a hoisted reciprocal it is free.
        # Floor at 1e-12, not 1: counts are gain-scaled and 0.07 % of them fall
        # between 0 and 1, where flooring at 1 would silently use the wrong
        # ratio.  At n = 0 the term vanishes anyway (0 * log(lam*1e12) = 0) and
        # lam*1e12 stays inside float32 for any lam on this grid.
        inv_n = 1.0 / tl.where(n_ph > 1e-12, n_ph, 1e-12)

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

            eps = log_S_k[None, :] - log_mean[:, None]
            z = tl.abs(eps) * inv_alpha_safe[:, None]
            z_eff = tl.where(z > 1e-12, z, 1e-12)
            # tl.log compiles to a polynomial in Triton 3.5; tl.math.log2 maps
            # to lg2.approx.f32 (single SFU instruction). Multiply by ln(2)
            # to recover natural log.
            log_z_eff = 0.6931471805599453 * tl.math.log2(z_eff)
            z_pow = tl.exp(beta[:, None] * log_z_eff)

            lam = S_k[None, :] + bg[:, None]
            lam = tl.where(lam > 1e-6, lam, 1e-6)

            # Poisson factor about its own peak, NOT as
            # `n log(lam) - lam - log n!`.  Those three terms are each ~3e6 at
            # the real batch's brightest observation (n = 2.7e5) and cancel to
            # -6.4, so float32 -- which resolves 3e6 to +-0.2 -- leaves ~0.2-0.4
            # nats of absolute error right at the peak, where the integrand
            # carries all its mass.  Measured against a float64 reference: 0.457
            # nats median error at n = 2e6 before this change.  Writing it as
            # `n log(lam/n) - (lam - n) + logpeak(n)` cancels the large terms on
            # paper; `logpeak = n log n - n - log n!` is formed once per
            # observation in float64 on the host, where the cancellation is
            # harmless.
            # -D/2, the Poisson log-factor measured from its own peak
            neg_half_D = (n_ph[:, None] * 0.6931471805599453
                          * tl.math.log2(lam * inv_n[:, None])
                          - (lam - n_ph[:, None]))
            if BG_VAR:
                if BG_DEVIANCE:
                    # bg is ESTIMATED from the 9-11 px annulus and carries
                    # sigma_b^2 = (n_peak/n_bg)*bg, so the count's variance is
                    # lam + sigma_b^2, not lam.  Scaling the Poisson deviance by
                    # the dispersion phi = 1 + sigma_b^2/lam is the saddlepoint
                    # density of that overdispersed law: it keeps the mode at
                    # lam = n, matches the variance, collapses to the exact
                    # Poisson at sigma_b^2 = 0, and is bounded -- so it needs no
                    # clamp.  Measured against the exact marginal over the
                    # annulus's Gamma posterior on a real batch, median error
                    # 0.13 nats against the second-order form's 0.89.
                    inv_lps2 = 1.0 / (lam + sig_b2[:, None])
                    inv_phi = lam * inv_lps2
                    neg_half_D = (neg_half_D * inv_phi
                                  + 0.5 * 0.6931471805599453
                                  * tl.math.log2(inv_phi))
                else:
                    # Legacy second-order expansion, kept only to reproduce the
                    # earlier measurements: +0.5 sigma_b^2 [(n/lam-1)^2 - n/lam^2].
                    # It is unbounded as lam leaves n, so it needs the +-10 clamp
                    # -- which fires on 16 % of the grid points that carry mass
                    # and inflates the weak quartile's posterior mean 5.4x.
                    inv_lam = 1.0 / lam
                    rr = n_ph[:, None] * inv_lam
                    corr = 0.5 * sig_b2[:, None] * ((rr - 1.0) * (rr - 1.0)
                                                    - rr * inv_lam)
                    neg_half_D = neg_half_D + tl.minimum(
                        tl.maximum(corr, -10.0), 10.0)
            log_int = (
                -z_pow
                + log_scatter_const[:, None]
                + neg_half_D
                + logpeak[:, None]
                + log_w_k[None, :]
            )
            log_int = tl.where(mask_k[None, :], log_int, -1e30)

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

            e_shift = tl.exp(log_int - m_new[:, None])
            T0 += tl.sum(e_shift, axis=1)

            # Each aux statistic costs a full BLOCK_K cross-lane reduction, and
            # together they are ~47 % of this kernel's runtime (measured by an
            # ablation probe: 25.1 ms -> 13.5 ms at M=1.05e6, K=2001 with four
            # of the five accumulators removed).  Every one of them is dead
            # weight unless the matching input actually requires grad -- and
            # the posterior accumulator's `locate` sweep runs under no_grad,
            # while `bg` is data in every path except the learned-background
            # `bg_mix` mode, where the gate turns S2 back on by itself.
            if NEED_MU:
                sign_eps = tl.where(eps > 0, 1.0, tl.where(eps < 0, -1.0, 0.0))
                T1 += tl.sum(
                    e_shift * sign_eps * (z_pow / z_eff) * inv_alpha_safe[:, None],
                    axis=1,
                )
            if NEED_BG:
                T2 += tl.sum(e_shift * (n_ph[:, None] / lam - 1.0), axis=1)
            if NEED_AB:
                T3 += tl.sum(e_shift * z_pow, axis=1)
                T4 += tl.sum(e_shift * z_pow * log_z_eff, axis=1)

            m_max = m_new

        log_p = m_max + 0.6931471805599453 * tl.math.log2(T0)
        inv_T0 = 1.0 / T0

        tl.store(log_p_ptr + offs_m, log_p, mask=mask_m)
        if NEED_MU:
            tl.store(S1_ptr + offs_m, T1 * inv_T0, mask=mask_m)
        if NEED_BG:
            tl.store(S2_ptr + offs_m, T2 * inv_T0, mask=mask_m)
        if NEED_AB:
            tl.store(S3_ptr + offs_m, T3 * inv_T0, mask=mask_m)
            tl.store(S4_ptr + offs_m, T4 * inv_T0, mask=mask_m)


# ── Autograd.Function wrapper ──────────────────────────────────────────


class _MarginalizeSignalTriton(torch.autograd.Function):
    """Per-obs ``log_p`` with analytic backward through the four scatter /
    likelihood inputs. The (M, K) integrand never touches HBM.

    Gradient formulas (verified against autograd; see
    ``docs/optimize_integration.md``):

        d log_p / d log_mean      = +β · S1
        d log_p / d bg            = +S2
        d log_p / d log_alpha_obs = +β · S3 − 1
        d log_p / d log_beta_obs  = −β · S4 + 1 + ψ(1/β) / β
    """

    @staticmethod
    def forward(
        ctx,
        log_mean: Tensor, n_ph: Tensor, bg: Tensor,
        log_alpha_obs: Tensor, log_beta_obs: Tensor,
        log_S_grid: Tensor, S_grid: Tensor, log_w: Tensor, logpeak: Tensor,
        sig_b2: Tensor | None = None,
        need_mu: bool = True, need_ab: bool = True, need_bg: bool = True,
        bg_deviance: bool = True,
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
        # A statistic that will not be stored needs no buffer -- but the kernel
        # still takes a valid pointer, so give it a one-element placeholder.
        # This is where the memory goes in the posterior accumulator: 21 outer
        # nodes x 4 statistics x (M,) is 373 MB of saved activations at
        # M = 1.1e6, none of which the no-grad `locate` sweep ever reads.
        def _buf(need):
            return torch.empty(M if need else 1, dtype=dtype, device=device)

        S1, S2 = _buf(need_mu), _buf(need_bg)
        S3, S4 = _buf(need_ab), _buf(need_ab)

        cfg = _lookup_config(K)
        BLOCK_M, BLOCK_K = cfg["BLOCK_M"], cfg["BLOCK_K"]
        grid = (triton.cdiv(M, BLOCK_M),)
        _fwd_kernel[grid](
            log_mean, n_ph, bg,
            log_alpha_obs, log_beta_obs, logpeak, log_norm,
            sig_b2 if sig_b2 is not None else logpeak,
            log_S_grid, S_grid, log_w,
            log_p, S1, S2, S3, S4,
            M, K,
            BG_VAR=sig_b2 is not None,
            BG_DEVIANCE=bg_deviance,
            NEED_MU=need_mu, NEED_AB=need_ab, NEED_BG=need_bg,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )

        ctx.need = (need_mu, need_ab, need_bg)
        ctx.save_for_backward(log_alpha_obs, log_beta_obs, S1, S2, S3, S4)
        return log_p

    @staticmethod
    def backward(ctx, grad_log_p: Tensor):
        log_alpha_obs, log_beta_obs, S1, S2, S3, S4 = ctx.saved_tensors
        need_mu, need_ab, need_bg = ctx.need
        beta = log_beta_obs.exp()

        g_log_mean = grad_log_p * (beta * S1) if need_mu else None
        g_bg = grad_log_p * S2 if need_bg else None
        if need_ab:
            psi_recip_beta = torch.special.digamma(1.0 / beta)
            g_log_alpha_obs = grad_log_p * (beta * S3 - 1.0)
            g_log_beta_obs = grad_log_p * (
                -beta * S4 + 1.0 + psi_recip_beta / beta
            )
        else:
            g_log_alpha_obs = g_log_beta_obs = None

        return (
            g_log_mean, None, g_bg, g_log_alpha_obs, g_log_beta_obs,
            None, None, None, None, None, None, None, None, None,
        )


def marginalize_signal_triton(
    log_mean: Tensor,
    n_ph: Tensor,
    bg: Tensor,
    log_alpha_obs: Tensor,
    log_beta_obs: Tensor,
    log_S_grid: Tensor,
    log_w: Tensor,
    *,
    log_nfact: Tensor | None = None,
    sig_b2: Tensor | None = None,
    bg_deviance: bool = True,
) -> Tensor:
    """Return per-obs ``log_p`` of shape ``(M,)`` via the Triton kernel.

    ``log_nfact`` (cached ``lgamma(n_ph + 1)``) may be passed by callers
    that compute it once across the whole data set; otherwise it's
    computed here.
    """
    if log_nfact is None:
        log_nfact = torch.lgamma(n_ph + 1)
    # log of the Poisson factor at its own peak, formed in float64 so the
    # ~3e6-magnitude cancellation happens where there are digits to spare.
    n64 = n_ph.double()
    logpeak = (
        n64 * n64.clamp(min=1e-300).log() - n64 - log_nfact.double()
    ).to(log_mean.dtype)
    logpeak = torch.where(n_ph > 0, logpeak, torch.zeros_like(logpeak))
    S_grid = log_S_grid.exp()
    # Compute the gradient gates HERE, not inside forward(): autograd runs
    # Function.forward with grad mode off, so requires_grad is not a reliable
    # signal in there.
    on = torch.is_grad_enabled()
    need_bg = on and bg.requires_grad
    if need_bg and sig_b2 is not None:
        # S2 is d log_p/d bg for the PLAIN Poisson factor; neither background
        # correction has that derivative, and bg additionally enters through
        # sigma_b^2 = (n_peak/n_bg)*bg.  A learned background must not be
        # combined with the correction rather than silently get a wrong grad.
        raise RuntimeError(
            "bg requires grad while the background-variance correction is on; "
            "the analytic d log_p/d bg does not cover it (see bg_var_factor)"
        )
    return _MarginalizeSignalTriton.apply(
        log_mean, n_ph, bg, log_alpha_obs, log_beta_obs,
        log_S_grid, S_grid, log_w, logpeak, sig_b2,
        on and log_mean.requires_grad,
        on and (log_alpha_obs.requires_grad or log_beta_obs.requires_grad),
        need_bg, bg_deviance,
    )
