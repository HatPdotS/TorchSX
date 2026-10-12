"""Fused Triton kernel for the posterior accumulator's inner integrals.

``PosteriorMergeTarget.accumulate`` evaluates the same marginal at ``C`` outer
nodes per reflection, which the per-column path does as ``C`` separate launches
of ``_fixed_log_grid_triton``.  That leaves two structural inefficiencies on the
table, and this kernel takes both:

**The Poisson factor does not depend on the outer node.**  ``log_mean`` enters
only through the GenGauss, so ``n log(S+bg) - (S+bg) - log n!`` is recomputed
``C`` times for nothing.  Hoisting it out of the ``c`` loop removes one of the
four transcendentals per ``(m, c, k)``.

**The running max can be replaced by a closed form.**  The per-column kernel
carries an online-logsumexp maximum, which costs a rescale ``exp(m_old - m_new)``
on every accumulator every K-block, and forces the sum over ``k`` to be a
cross-lane reduction.  But the Poisson factor's maximum over the grid is known
analytically -- it peaks at ``S = n - bg``, clamped to the grid's lower end --
and the GenGauss factor is bounded above by 1.  So

    shift = log_norm - log_alpha + [n log(lam*) - lam* - log n!] + max_k log_w
    lam*  = max(n_ph, bg + S_min)

is an exact upper bound on the integrand, ``exp(log_int - shift) <= 1`` cannot
overflow, and the sum over ``k`` becomes a plain elementwise accumulation with
no reduction and no rescale.  The cost is dynamic range: anything more than
~87 nats below the bound underflows in float32, which is why ``T0`` is floored
before the log.  That regime contributes ``e^-87`` to the outer posterior.

The tile is ``(BLOCK_M, BLOCK_C)`` and ``k`` is the serial loop, so ``BLOCK_C``
should be ``next_pow2(k_coarse)``; a ``k_coarse`` that is already a power of two
(16, 32) wastes no lanes and is therefore *free* relative to a smaller
non-power-of-two value.

``BLOCK_M`` matters more than it looks.  The serial ``k`` loop means latency is
hidden by occupancy rather than by ILP within an iteration, so *small* tiles
win: in a tile sweep BLOCK_M = 16 with 4 warps beats the per-column path while
BLOCK_M = 32 with 4 warps is already slower than what it replaces, and large
tiles collapse.  Do not change the default without re-running an equivalent
tile sweep.
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

_LN2 = 0.6931471805599453   # host side only; Triton forbids non-constexpr globals


if HAVE_TRITON:

    @triton.jit
    def _fused_fwd(
        base_ptr, n_ph_ptr, bg_ptr, log_alpha_ptr, log_beta_ptr,
        log_nfact_ptr, log_norm_ptr, hkl_ptr, v_ptr,
        log_S_ptr, S_ptr, log_w_ptr, pmax_ptr, lam_hat_ptr,
        log_p_ptr, S1_ptr, S3_ptr, S4_ptr,
        M, K, C, max_log_w,
        NEED_MU: tl.constexpr, NEED_AB: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask = mask_m[:, None] & mask_c[None, :]

        base = tl.load(base_ptr + offs_m, mask=mask_m, other=0.0)
        n_ph = tl.load(n_ph_ptr + offs_m, mask=mask_m, other=0.0)
        bg = tl.load(bg_ptr + offs_m, mask=mask_m, other=0.0)
        log_alpha = tl.load(log_alpha_ptr + offs_m, mask=mask_m, other=0.0)
        log_beta = tl.load(log_beta_ptr + offs_m, mask=mask_m, other=0.0)
        log_norm = tl.load(log_norm_ptr + offs_m, mask=mask_m, other=0.0)
        pmax = tl.load(pmax_ptr + offs_m, mask=mask_m, other=0.0)
        lam_hat = tl.load(lam_hat_ptr + offs_m, mask=mask_m, other=1.0)
        hkl = tl.load(hkl_ptr + offs_m, mask=mask_m, other=0)

        alpha = tl.exp(log_alpha)
        inv_alpha = 1.0 / tl.where(alpha > 1e-8, alpha, 1e-8)
        beta = tl.exp(log_beta)

        # mu[m, c] = v[hkl_m, c] + base_m -- gathered once, held in registers
        mu = tl.load(v_ptr + hkl[:, None] * C + offs_c[None, :],
                     mask=mask, other=0.0) + base[:, None]

        shift = log_norm - log_alpha + pmax

        T0 = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
        T1 = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
        T3 = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
        T4 = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

        for k in range(0, K):
            log_S_k = tl.load(log_S_ptr + k)
            S_k = tl.load(S_ptr + k)
            log_w_k = tl.load(log_w_ptr + k)

            lam = S_k + bg
            lam = tl.where(lam > 1e-6, lam, 1e-6)
            # Hoisted out of the c loop: one transcendental per (m, k), not per
            # (m, c, k).  Formed as a RATIO against lam_hat rather than as
            # `n log(lam) - lam - log n!` minus a separately computed maximum:
            # at n = 2e6 and lam = e^34.5 those terms are 6.9e7 and 9.6e14, and
            # float32 carries ~4 nats of absolute error in the first alone.
            # The direct form leaves many entries wrong by more than a nat and
            # over-estimates by hundreds, which the per-column kernel
            # avoids only because its running max is built from the very same
            # expression so the error cancels.  log n! drops out exactly here.
            d = (n_ph * 0.6931471805599453 * tl.math.log2(lam / lam_hat)
                 - (lam - lam_hat) + log_w_k - max_log_w)

            eps = log_S_k - mu
            z = tl.abs(eps) * inv_alpha[:, None]
            z_eff = tl.where(z > 1e-12, z, 1e-12)
            log_z = 0.6931471805599453 * tl.math.log2(z_eff)
            z_pow = tl.exp(beta[:, None] * log_z)

            e = tl.exp(d[:, None] - z_pow)
            T0 += e
            if NEED_MU:
                sign_eps = tl.where(eps > 0, 1.0, tl.where(eps < 0, -1.0, 0.0))
                T1 += e * sign_eps * (z_pow / z_eff) * inv_alpha[:, None]
            if NEED_AB:
                T3 += e * z_pow
                T4 += e * z_pow * log_z

        # Underflow must floor log_p LOW, not at the smallest representable
        # float.  `shift` is an exact upper bound on the integrand, so T0
        # underflowing means the true log_p is at least ~80 nats below shift --
        # possibly hundreds.  Reporting `shift + log(1e-38)` = shift - 87.5
        # therefore over-estimates such a node by however much further down it
        # really was -- hundreds of nats against a float64 reference -- and
        # a node that should carry no posterior weight at all then carries a
        # lot.  That is invisible in log_Z and in I0_mean and shows up only
        # in the gradient direction.  Sending it to
        # -1e4 makes the node's weight exactly zero, which is the right limit.
        under = T0 <= 1e-38
        T0s = tl.where(under, 1.0, T0)
        log_p = tl.where(
            under,
            shift[:, None] - 1e4,
            shift[:, None] + 0.6931471805599453 * tl.math.log2(T0s),
        )
        inv_T0 = 1.0 / T0s
        off = offs_m[:, None] * C + offs_c[None, :]
        tl.store(log_p_ptr + off, log_p, mask=mask)
        if NEED_MU:
            tl.store(S1_ptr + off, T1 * inv_T0, mask=mask)
        if NEED_AB:
            tl.store(S3_ptr + off, T3 * inv_T0, mask=mask)
            tl.store(S4_ptr + off, T4 * inv_T0, mask=mask)


class _FusedPosterior(torch.autograd.Function):
    """``log p(n_j | v[h_j, c] + base_j)`` for all ``(j, c)``, shape ``(M, C)``.

    Gradient formulas are the per-column kernel's, summed over ``c`` in the
    backward because ``base``, ``log_alpha`` and ``log_beta`` are per
    observation while the upstream gradient is per ``(observation, node)``.
    """

    @staticmethod
    def forward(ctx, base, n_ph, bg, log_alpha, log_beta, hkl_id, v_nodes,
                log_S_grid, S_grid, log_w, log_nfact,
                need_mu=True, need_ab=True, block_m=16, num_warps=4,
                num_stages=2):
        M = base.shape[0]
        K = log_S_grid.shape[0]
        C = v_nodes.shape[1]
        dev, dt = base.device, base.dtype
        beta = log_beta.exp()
        log_norm = beta.log() - math.log(2.0) - torch.lgamma(1.0 / beta)

        BLOCK_C = max(16, 1 << (C - 1).bit_length())
        log_p = torch.empty(M, C, dtype=dt, device=dev)
        # pmax in float64 on the host: it is the same near-cancelling
        # expression, and it sets log_p's absolute level.  An error there is a
        # per-observation CONSTANT in c, so it cancels from logsumexp_c and
        # cannot move the posterior -- but there is no reason to carry it.
        S_min = log_S_grid[0].exp().double()
        lam_hat = torch.maximum(n_ph.double(), bg.double() + S_min).clamp(min=1e-6)
        pmax = (n_ph.double() * lam_hat.log() - lam_hat
                - torch.lgamma(n_ph.double() + 1) + float(log_w.max()))

        def _buf(need):
            return torch.empty((M, C) if need else (1, 1), dtype=dt, device=dev)

        S1, S3, S4 = _buf(need_mu), _buf(need_ab), _buf(need_ab)
        _fused_fwd[(triton.cdiv(M, block_m),)](
            base, n_ph, bg, log_alpha, log_beta, log_nfact, log_norm,
            hkl_id, v_nodes.contiguous(), log_S_grid, S_grid, log_w,
            pmax.to(dt), lam_hat.to(dt),
            log_p, S1, S3, S4,
            M, K, C, float(log_w.max()),
            NEED_MU=need_mu, NEED_AB=need_ab,
            BLOCK_M=block_m, BLOCK_C=BLOCK_C,
            num_warps=num_warps, num_stages=num_stages,
        )
        ctx.need = (need_mu, need_ab)
        ctx.save_for_backward(log_beta, S1, S3, S4)
        return log_p

    @staticmethod
    def backward(ctx, grad):
        log_beta, S1, S3, S4 = ctx.saved_tensors
        need_mu, need_ab = ctx.need
        beta = log_beta.exp()
        g_base = (grad * S1).sum(1) * beta if need_mu else None
        if need_ab:
            psi = torch.special.digamma(1.0 / beta)
            g_la = (grad * (beta[:, None] * S3 - 1.0)).sum(1)
            g_lb = (grad * (-beta[:, None] * S4 + 1.0 + psi[:, None]
                            / beta[:, None])).sum(1)
        else:
            g_la = g_lb = None
        return (g_base, None, None, g_la, g_lb, None, None,
                None, None, None, None, None, None, None, None, None)


def fused_posterior_logp(
    base: Tensor, n_ph: Tensor, bg: Tensor,
    log_alpha: Tensor, log_beta: Tensor,
    hkl_id: Tensor, v_nodes: Tensor,
    log_S_grid: Tensor, log_w: Tensor,
    *, log_nfact: Tensor | None = None, block_m: int = 16,
    num_warps: int = 4, num_stages: int = 2,
) -> Tensor:
    """``(M, C)`` marginal log-likelihood at every outer node.  See module doc."""
    if not HAVE_TRITON or not base.is_cuda:
        raise RuntimeError("fused posterior kernel needs Triton on CUDA")
    if log_nfact is None:
        log_nfact = torch.lgamma(n_ph + 1)
    on = torch.is_grad_enabled()
    return _FusedPosterior.apply(
        base, n_ph, bg, log_alpha, log_beta, hkl_id, v_nodes,
        log_S_grid, log_S_grid.exp(), log_w, log_nfact,
        on and base.requires_grad,
        on and (log_alpha.requires_grad or log_beta.requires_grad),
        block_m, num_warps, num_stages,
    )
