"""One split-grid pass for a mixture of fixed, two-width Gaussians.

Nodes, weights and the background-corrected Poisson factor are shared across
components. Each component uses a stable node reduction, adds its analytic
lower tail, then participates in the mixture reduction. No observation by
node tensor leaves the kernel. Backward scales four saved derivatives per
component; beta is fixed by the scatter model and has zero gradient here.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from torchsx.merging.quadrature._split_log_grid_triton import HAVE_TRITON

if HAVE_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from torchsx.merging.quadrature._split_log_grid_triton import _node

    @triton.jit
    def _log_erfc(z):
        # The same x=25 switch and asymptotic series as GenGaussLogCDF at
        # beta=2, retaining its numerical definition in the far lower tail.
        x = tl.minimum(z * z, 1.1420073898156842e26)  # exp(60)
        xa = tl.maximum(x, 25.0)
        inv = 1.0 / xa
        series = 1.0 - 0.5 * inv + 0.75 * inv * inv - 1.875 * inv * inv * inv
        asym = -0.5 * tl.log(xa) - xa - 0.5723649429247001 + tl.log(series)
        direct = tl.log(libdevice.erfc(tl.minimum(z, 5.0)))
        return tl.where(x > 25.0, asym, direct)

    @triton.jit
    def _mixture_fwd(
        Mu, Counts, Bg, Mix, Sig2, Peak, L0,
        Xlo, H1, Dlo, Hd, Dhi, H2, N1, N1d,
        Out, Aux, M,
        C: tl.constexpr, K: tl.constexpr, BG_VAR: tl.constexpr,
        GRAD: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
    ):
        m = tl.program_id(0) * BM + tl.arange(0, BM)
        valid = m < M
        mu = tl.load(Mu + m, valid, other=0.0)
        n = tl.load(Counts + m, valid, other=0.0).to(tl.float32)
        bg = tl.load(Bg + m, valid, other=1.0)
        peak = tl.load(Peak + m, valid, other=0.0)
        l0 = tl.load(L0 + m, valid, other=0.0)
        xlo = tl.load(Xlo + m, valid, other=0.0)
        h1 = tl.load(H1 + m, valid, other=1.0)
        dlo = tl.load(Dlo + m, valid, other=0.0)
        hd = tl.load(Hd + m, valid, other=1.0)
        dhi = tl.load(Dhi + m, valid, other=0.0)
        h2 = tl.load(H2 + m, valid, other=1.0)
        n1 = tl.load(N1 + m, valid, other=0)
        n1d = tl.load(N1d + m, valid, other=1)
        if BG_VAR:
            sig2 = tl.load(Sig2 + m, valid, other=0.0)
        cc = tl.arange(0, BC)[None, :]
        mask = valid[:, None] & (cc < C)
        pos = m[:, None] * (5 * C) + cc
        pi = tl.load(Mix + pos, mask, other=-float("inf"))
        offset = tl.load(Mix + pos + C, mask, other=0.0)
        la = tl.load(Mix + pos + 2 * C, mask, other=0.0)
        lah = tl.load(Mix + pos + 3 * C, mask, other=0.0)
        center = mu[:, None] + offset
        a, ah = tl.exp(la), tl.exp(lah)
        inv_a, inv_ah = 1.0 / tl.maximum(a, 1e-8), 1.0 / tl.maximum(ah, 1e-8)
        log_sum = tl.maximum(la, lah) + libdevice.log1p(tl.exp(-tl.abs(la - lah)))
        wlo, whi = tl.exp(la - log_sum), tl.exp(lah - log_sum)
        norm = 0.1207822376352452 - log_sum
        mx = tl.full((BM, BC), -float("inf"), tl.float32)
        total = tl.zeros((BM, BC), tl.float32)
        if GRAD:
            smu = tl.zeros((BM, BC), tl.float32)
            slo = tl.zeros((BM, BC), tl.float32)
            shi = tl.zeros((BM, BC), tl.float32)
        for start in range(0, K, BK):
            k = start + tl.arange(0, BK)[None, :]
            x = _node(k, xlo[:, None], h1[:, None], dlo[:, None], hd[:, None],
                      dhi[:, None], h2[:, None], n1[:, None], n1d[:, None])
            prev = _node(tl.maximum(k - 1, 0), xlo[:, None], h1[:, None], dlo[:, None],
                         hd[:, None], dhi[:, None], h2[:, None], n1[:, None], n1d[:, None])
            nxt = _node(tl.minimum(k + 1, K - 1), xlo[:, None], h1[:, None], dlo[:, None],
                        hd[:, None], dhi[:, None], h2[:, None], n1[:, None], n1d[:, None])
            weight = tl.where(k < K, tl.maximum(0.5 * (nxt - prev), 1e-30), 0.0)
            lam = tl.maximum(tl.exp(x) + bg[:, None], 1e-6)
            nn = tl.maximum(n[:, None], 1e-12)
            ratio = (lam - nn) / nn
            # Stable around the bright peak; the original deviance form
            # elsewhere avoids rounding ratio to -1 at tiny lambda/n.
            near = tl.abs(ratio) < 0.125
            small = tl.where(near, ratio, 0.0)
            # log(1+r)-r through r^10; the remainder is <1.3e-11
            # on this interval. Avoids both cancellation and a libdevice
            # log1p at every node (whose register cost limits occupancy).
            poly = tl.full((BM, BK), -0.1, tl.float32)
            for power in tl.static_range(9, 1, -1):
                poly = (1.0 if power % 2 else -1.0) / power + small * poly
            dev = tl.where(near, nn * small * small * poly,
                           nn * tl.log(lam / nn) - (lam - nn))
            dev = tl.where(n[:, None] > 0, dev, -lam)
            if BG_VAR:
                iphi = lam / (lam + sig2[:, None])
                dev = dev * iphi + 0.5 * tl.log(iphi)
            log_lik = dev + peak[:, None]
            eps = x[:, None, :] - center[:, :, None]
            ia = tl.where(eps > 0, inv_ah[:, :, None], inv_a[:, :, None])
            z2 = (eps * ia) * (eps * ia)
            log_int = tl.where(k[:, None, :] < K,
                                log_lik[:, None, :] + norm[:, :, None] - z2, -float("inf"))
            new_mx = tl.maximum(mx, tl.max(log_int, axis=2))
            scale = tl.exp(mx - new_mx)
            mass = tl.exp(log_int - new_mx[:, :, None]) * weight[:, None, :]
            total = total * scale + tl.sum(mass, axis=2)
            if GRAD:
                smu = smu * scale + tl.sum(mass * (2.0 * eps * ia * ia), axis=2)
                slo = slo * scale + 2.0 * tl.sum(tl.where(eps > 0, 0.0, mass * z2), axis=2)
                shi = shi * scale + 2.0 * tl.sum(tl.where(eps > 0, mass * z2, 0.0), axis=2)
            mx = new_mx
        total = tl.maximum(total, 1e-38)
        win = mx + tl.log(total)
        t = xlo[:, None] - center
        zt = tl.abs(t) * tl.exp(-tl.where(t > 0, lah, la))
        log_q = _log_erfc(zt)
        log_f = tl.where(t > 0,
            libdevice.log1p(-tl.minimum(tl.exp(lah - log_sum + log_q), 1.0 - 1e-7)),
            la - log_sum + log_q)
        tail = l0[:, None] + log_f
        high = tl.maximum(win, tail)
        lp = high + libdevice.log1p(tl.exp(-tl.abs(win - tail)))
        comp_lp = pi + lp
        if GRAD:
            pw, pt = tl.exp(win - lp), tl.exp(tail - lp)
            density = tl.exp(l0[:, None] + norm - tl.minimum(zt * zt, 1.1420073898156842e26) - lp)
            survival = tl.exp(l0[:, None] + lah - log_sum + log_q - lp)
            comp_mu = pw * smu / total - density
            comp_lo = (pw * (slo / total - wlo)
                       + tl.where(t > 0, wlo * survival, pt * whi - t * density))
            comp_hi = (pw * (shi / total - whi)
                       + tl.where(t > 0, -wlo * survival - t * density, -pt * whi))
        high = tl.max(comp_lp, axis=1)
        resp = tl.exp(comp_lp - high[:, None])
        total = tl.sum(resp, axis=1)
        tl.store(Out + m, high + tl.log(total), valid)
        if GRAD:
            resp = resp / total[:, None]
            pos = m[:, None] * (4 * C) + cc
            mask = valid[:, None] & (cc < C)
            tl.store(Aux + pos, resp, mask)
            tl.store(Aux + pos + C, resp * comp_mu, mask)
            tl.store(Aux + pos + 2 * C, resp * comp_lo, mask)
            tl.store(Aux + pos + 3 * C, resp * comp_hi, mask)

    @triton.jit
    def _mixture_bwd(Grad, Aux, Gmu, Gmix, M, C: tl.constexpr, BC: tl.constexpr,
                     MU: tl.constexpr, MIX: tl.constexpr, BM: tl.constexpr):
        m = tl.program_id(0) * BM + tl.arange(0, BM)
        c = tl.arange(0, BC)[None, :]
        mask = (m[:, None] < M) & (c < C)
        g = tl.load(Grad + m, m < M, other=0.0)[:, None]
        pos = m[:, None] * (4 * C) + c
        gm = tl.load(Aux + pos + C, mask, other=0.0)
        if MU:
            tl.store(Gmu + m, tl.sum(gm, axis=1) * tl.sum(g, axis=1), m < M)
        if MIX:
            dst = m[:, None] * (5 * C) + c
            tl.store(Gmix + dst, g * tl.load(Aux + pos, mask, other=0.0), mask)
            tl.store(Gmix + dst + C, g * gm, mask)
            tl.store(Gmix + dst + 2 * C, g * tl.load(Aux + pos + 2 * C, mask, other=0.0), mask)
            tl.store(Gmix + dst + 3 * C, g * tl.load(Aux + pos + 3 * C, mask, other=0.0), mask)
            tl.store(Gmix + dst + 4 * C, 0.0, mask)


class _GaussianMixture(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_mean, n_ph, mix, p, nodes, sig_b2, need_grad):
        n, columns = mix.shape
        components = columns // 5
        out = torch.empty_like(log_mean)
        aux = mix.new_empty((n, 4 * components) if need_grad else (0,))
        # Small node tiles keep the common detector work and component
        # accumulators in registers. Offline A100 tuning: no register spills.
        bm, bk = max(4, 32 // triton.next_power_of_2(components)), 8
        if n:
            _mixture_fwd[(triton.cdiv(n, bm),)](
                log_mean, n_ph, p["bg_model"], mix,
                sig_b2 if sig_b2 is not None else p["bg_model"], p["logpeak"], p["log_L0"],
                *(p[k] for k in ("x_lo", "h1", "d_lo", "h_d", "d_hi", "h2", "n1", "n1d")),
                out, aux, n, components, nodes, sig_b2 is not None, need_grad,
                bm, bk, triton.next_power_of_2(components),
                num_warps=4,
            )
        ctx.save_for_backward(aux)
        ctx.components = components
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (aux,) = ctx.saved_tensors
        n, c = grad.numel(), ctx.components
        mu, mix = ctx.needs_input_grad[0], ctx.needs_input_grad[2]
        gmu = grad.new_empty(n) if mu else None
        gmix = grad.new_empty(n, 5 * c) if mix else None
        if n:
            _mixture_bwd[(triton.cdiv(n, 128),)](
                grad.contiguous(), aux, gmu if mu else aux, gmix if mix else aux,
                n, c, triton.next_power_of_2(c), mu, mix, 128, num_warps=4,
            )
        return gmu, None, gmix, None, None, None, None


def marginalize_gaussian_mixture(log_mean, n_ph, mix, p, nodes, *, sig_b2=None):
    """CUDA float32, fixed beta=2, constant detection data; returns (M,) log p."""
    need = torch.is_grad_enabled() and (log_mean.requires_grad or mix.requires_grad)
    return _GaussianMixture.apply(log_mean.contiguous(), n_ph.contiguous(), mix.contiguous(),
                                   p, int(nodes), sig_b2, need)
