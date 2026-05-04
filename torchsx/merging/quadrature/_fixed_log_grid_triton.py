"""Triton fast path for ``FixedLogGrid.marginalize_signal``.

For ``GenGaussLogScatter`` × ``Poisson`` only (the fastest configurations
in production). Other scatter/likelihood combinations fall back to the
eager PyTorch path in :mod:`fixed_log_grid`.

Per integrand evaluation the kernel emits a single per-observation
``log_p`` plus four auxiliary statistics ``S1..S4`` (weighted softmax
sums over the K grid). Backward then applies analytic gradient formulas
without recomputing the (M, K) integrand — see ``_apply_backward``.

Tuning is read from ``_fixed_log_grid_tuning.json`` (in-package),
keyed by GPU name and K bucket. On unknown GPUs we fall back to the
A100 winners with a one-time warning. To extend the cache for a new
GPU, run a tuning sweep externally (see
``experiments/optimize_integration``) and add the entry.
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
        log_alpha_ptr, log_beta_ptr, log_nfact_ptr, log_norm_ptr,
        log_S_grid_ptr, S_grid_ptr, log_w_ptr,
        log_p_ptr, S1_ptr, S2_ptr, S3_ptr, S4_ptr,
        M, K,
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
        log_nfact = tl.load(log_nfact_ptr + offs_m, mask=mask_m, other=0.0)
        log_norm = tl.load(log_norm_ptr + offs_m, mask=mask_m, other=0.0)

        alpha = tl.exp(log_alpha)
        alpha_safe = tl.where(alpha > 1e-8, alpha, 1e-8)
        inv_alpha_safe = 1.0 / alpha_safe
        beta = tl.exp(log_beta)
        log_scatter_const = log_norm - log_alpha
        log_lik_const = -log_nfact

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
            inv_lam = 1.0 / lam

            log_int = (
                -z_pow
                + log_scatter_const[:, None]
                + n_ph[:, None] * 0.6931471805599453 * tl.math.log2(lam) - lam
                + log_lik_const[:, None]
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
            T2 += tl.sum(e_shift * (n_ph[:, None] * inv_lam - 1.0), axis=1)
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


class _MarginalizeSignalTriton(torch.autograd.Function):
    """Per-obs ``log_p`` with analytic backward through the four scatter /
    likelihood inputs. The (M, K) integrand never touches HBM.

    Gradient formulas (verified against autograd in
    ``experiments/optimize_integration/verify.py``):

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
        log_S_grid: Tensor, S_grid: Tensor, log_w: Tensor, log_nfact: Tensor,
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
        S2 = torch.empty(M, dtype=dtype, device=device)
        S3 = torch.empty(M, dtype=dtype, device=device)
        S4 = torch.empty(M, dtype=dtype, device=device)

        cfg = _lookup_config(K)
        BLOCK_M, BLOCK_K = cfg["BLOCK_M"], cfg["BLOCK_K"]
        grid = (triton.cdiv(M, BLOCK_M),)
        _fwd_kernel[grid](
            log_mean, n_ph, bg,
            log_alpha_obs, log_beta_obs, log_nfact, log_norm,
            log_S_grid, S_grid, log_w,
            log_p, S1, S2, S3, S4,
            M, K,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
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
            g_log_mean, None, g_bg, g_log_alpha_obs, g_log_beta_obs,
            None, None, None, None,
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
) -> Tensor:
    """Return per-obs ``log_p`` of shape ``(M,)`` via the Triton kernel.

    ``log_nfact`` (cached ``lgamma(n_ph + 1)``) may be passed by callers
    that compute it once across the whole data set; otherwise it's
    computed here.
    """
    if log_nfact is None:
        log_nfact = torch.lgamma(n_ph + 1)
    S_grid = log_S_grid.exp()
    return _MarginalizeSignalTriton.apply(
        log_mean, n_ph, bg, log_alpha_obs, log_beta_obs,
        log_S_grid, S_grid, log_w, log_nfact,
    )
