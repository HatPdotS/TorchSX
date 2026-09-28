"""All outer posterior nodes of a fixed Gaussian mixture in one split-grid pass.

``PosteriorMergeTarget.accumulate`` needs, for every reflection ``h`` and
outer node ``k``, ``S[h, k] = sum_{j in h} log p(n_j | log_mean = v[h, k] +
mu_j)`` -- the mixture marginal of ``_split_mixture_triton`` at ``K`` shifted
centres per observation, optionally mixed with the miss component.  Called
node by node, every call reloads the observation and rebuilds its detector
factor on the inner grid, although both depend on ``(n, bg)`` alone.  Here one
program takes a tile of observations, loads them once, and evaluates ``KO``
outer nodes per pass over the inner grid, so the detector factor is built
``ceil(K / KO)`` times instead of ``K``; the per-reflection sum is taken by
atomic adds into ``S`` (float32, order not deterministic).

Backward recomputes the same pass with the three gradient statistics of
``_mixture_fwd`` and contracts them with ``dL/dS`` in registers, so nothing of
size ``(M, K)`` is ever stored: the saved state is the inputs.  The integrand,
tails and derivatives are those of ``_mixture_fwd``, line for line.

Detector noise (``log_u = -log phi`` per observation, optional): the scaled
deviance is multiplied by ``u`` and ``1/2 log u`` joins the normaliser, at the
inner nodes and in ``log L(0)`` (tail and miss).  Its gradient is the
posterior mean over (node, tail, miss) of ``u * scaled deviance + 1/2``, with
the window part accumulated as one more mass-weighted statistic.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchsx.merging.quadrature._split_log_grid_triton import HAVE_TRITON

if HAVE_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from torchsx.merging.quadrature._split_log_grid_triton import _node
    from torchsx.merging.quadrature._split_mixture_triton import _log_erfc

    @triton.jit
    def _nodes_kernel(
        Mu, Counts, Bg, Mix, Sig2, Peak, L0, Lout,
        Xlo, H1, Dlo, Hd, Dhi, H2, N1, N1d,
        Hkl, V, S, G, Gmu, Gmix, Gz, Gr, LogU, D0, Gu,
        M, KN, l1m, lpi, sig_z, l1r, lr, sig_r,
        C: tl.constexpr, NI: tl.constexpr, BG_VAR: tl.constexpr, MISS: tl.constexpr,
        OUTLIER: tl.constexpr, GRAD: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
        KO: tl.constexpr, J: tl.constexpr, NOISE: tl.constexpr,
    ):
        # J = KO * BC columns: (outer node, component) pairs, node-major.
        l1m = l1m.to(tl.float32)
        lpi = lpi.to(tl.float32)
        sig_z = sig_z.to(tl.float32)
        l1r = l1r.to(tl.float32)
        lr = lr.to(tl.float32)
        sig_r = sig_r.to(tl.float32)
        m = tl.program_id(0) * BM + tl.arange(0, BM)
        valid = m < M
        mu = tl.load(Mu + m, valid, other=0.0)
        n = tl.load(Counts + m, valid, other=0.0).to(tl.float32)
        bg = tl.load(Bg + m, valid, other=1.0)
        peak = tl.load(Peak + m, valid, other=0.0)
        l0 = tl.load(L0 + m, valid, other=0.0)
        if OUTLIER:
            lo = tl.load(Lout + m, valid, other=0.0)
        xlo = tl.load(Xlo + m, valid, other=0.0)
        h1 = tl.load(H1 + m, valid, other=1.0)
        dlo = tl.load(Dlo + m, valid, other=0.0)
        hd = tl.load(Hd + m, valid, other=1.0)
        dhi = tl.load(Dhi + m, valid, other=0.0)
        h2 = tl.load(H2 + m, valid, other=1.0)
        n1 = tl.load(N1 + m, valid, other=0)
        n1d = tl.load(N1d + m, valid, other=1)
        hkl = tl.load(Hkl + m, valid, other=0).to(tl.int64)
        if BG_VAR:
            sig2 = tl.load(Sig2 + m, valid, other=0.0)
        if NOISE:
            lu = tl.load(LogU + m, valid, other=0.0)
            uu = tl.exp(lu)
            d0 = tl.load(D0 + m, valid, other=0.0)
            peak = peak + 0.5 * lu            # l0 already carries u (grid_params)
        nn = tl.maximum(n, 1e-12)

        j = tl.arange(0, J)
        ko = j // BC
        cc = j % BC
        cmask = valid[:, None] & (cc < C)[None, :]
        pos = m[:, None] * (5 * C) + cc[None, :]
        pi = tl.load(Mix + pos, cmask, other=-float("inf"))
        offset = tl.load(Mix + pos + C, cmask, other=0.0)
        la = tl.load(Mix + pos + 2 * C, cmask, other=0.0)
        lah = tl.load(Mix + pos + 3 * C, cmask, other=0.0)
        a, ah = tl.exp(la), tl.exp(lah)
        inv_a, inv_ah = 1.0 / tl.maximum(a, 1e-8), 1.0 / tl.maximum(ah, 1e-8)
        log_sum = tl.maximum(la, lah) + libdevice.log1p(tl.exp(-tl.abs(la - lah)))
        wlo, whi = tl.exp(la - log_sum), tl.exp(lah - log_sum)
        norm = 0.1207822376352452 - log_sum
        if GRAD:
            acc_pi = tl.zeros((BM, J), tl.float32)
            acc_mu = tl.zeros((BM, J), tl.float32)
            acc_lo = tl.zeros((BM, J), tl.float32)
            acc_hi = tl.zeros((BM, J), tl.float32)
            gz = tl.zeros((BM,), tl.float32)
            gr = tl.zeros((BM,), tl.float32)
            if NOISE:
                acc_u = tl.zeros((BM, J), tl.float32)
                gu = tl.zeros((BM,), tl.float32)

        for kb in range(0, KN, KO):
            kj = kb + ko
            vk = tl.load(V + hkl[:, None] * KN + kj[None, :],
                         valid[:, None] & (kj < KN)[None, :], other=0.0)
            center = mu[:, None] + vk + offset
            mx = tl.full((BM, J), -float("inf"), tl.float32)
            total = tl.zeros((BM, J), tl.float32)
            if GRAD:
                smu = tl.zeros((BM, J), tl.float32)
                slo = tl.zeros((BM, J), tl.float32)
                shi = tl.zeros((BM, J), tl.float32)
                if NOISE:
                    sdv = tl.zeros((BM, J), tl.float32)
            for start in range(0, NI, BK):
                k = start + tl.arange(0, BK)[None, :]
                x = _node(k, xlo[:, None], h1[:, None], dlo[:, None], hd[:, None],
                          dhi[:, None], h2[:, None], n1[:, None], n1d[:, None])
                prev = _node(tl.maximum(k - 1, 0), xlo[:, None], h1[:, None], dlo[:, None],
                             hd[:, None], dhi[:, None], h2[:, None], n1[:, None], n1d[:, None])
                nxt = _node(tl.minimum(k + 1, NI - 1), xlo[:, None], h1[:, None], dlo[:, None],
                            hd[:, None], dhi[:, None], h2[:, None], n1[:, None], n1d[:, None])
                weight = tl.where(k < NI, tl.maximum(0.5 * (nxt - prev), 1e-30), 0.0)
                lam = tl.maximum(tl.exp(x) + bg[:, None], 1e-6)
                ratio = (lam - nn[:, None]) / nn[:, None]
                near = tl.abs(ratio) < 0.125
                small = tl.where(near, ratio, 0.0)
                poly = tl.full((BM, BK), -0.1, tl.float32)
                for power in tl.static_range(9, 1, -1):
                    poly = (1.0 if power % 2 else -1.0) / power + small * poly
                dev = tl.where(near, nn[:, None] * small * small * poly,
                               nn[:, None] * tl.log(lam / nn[:, None]) - (lam - nn[:, None]))
                dev = tl.where(n[:, None] > 0, dev, -lam)
                if BG_VAR:
                    iphi = lam / (lam + sig2[:, None])
                    if NOISE:
                        dsc = dev * iphi
                        dev = dsc * uu[:, None] + 0.5 * tl.log(iphi)
                    else:
                        dev = dev * iphi + 0.5 * tl.log(iphi)
                elif NOISE:
                    dsc = dev
                    dev = dsc * uu[:, None]
                log_lik = dev + peak[:, None]
                eps = x[:, None, :] - center[:, :, None]
                ia = tl.where(eps > 0, inv_ah[:, :, None], inv_a[:, :, None])
                z2 = (eps * ia) * (eps * ia)
                log_int = tl.where(k[:, None, :] < NI,
                                   log_lik[:, None, :] + norm[:, :, None] - z2, -float("inf"))
                new_mx = tl.maximum(mx, tl.max(log_int, axis=2))
                scale = tl.exp(mx - new_mx)
                mass = tl.exp(log_int - new_mx[:, :, None]) * weight[:, None, :]
                total = total * scale + tl.sum(mass, axis=2)
                if GRAD:
                    smu = smu * scale + tl.sum(mass * (2.0 * eps * ia * ia), axis=2)
                    slo = slo * scale + 2.0 * tl.sum(tl.where(eps > 0, 0.0, mass * z2), axis=2)
                    shi = shi * scale + 2.0 * tl.sum(tl.where(eps > 0, mass * z2, 0.0), axis=2)
                    if NOISE:
                        sdv = sdv * scale + tl.sum(mass * dsc[:, None, :], axis=2)
                mx = new_mx
            # FLT_MIN, not 1e-38: a Python float below it becomes an fp64
            # constant and would promote everything downstream to fp64
            total = tl.maximum(total, 1.1754943508222875e-38)
            win = mx + tl.log(total)
            t = xlo[:, None] - center
            zt = tl.abs(t) * tl.exp(-tl.where(t > 0, lah, la))
            log_q = _log_erfc(zt)
            log_f = tl.where(t > 0,
                libdevice.log1p(-tl.minimum(tl.exp(lah - log_sum + log_q), 1.0 - 1e-7)),
                la - log_sum + log_q)
            tail = l0[:, None] + log_f
            high = tl.maximum(win, tail)
            lp_c = high + libdevice.log1p(tl.exp(-tl.abs(win - tail)))
            comp3 = tl.reshape(pi + lp_c, (BM, KO, BC))
            hi3 = tl.max(comp3, axis=2)
            e3 = tl.exp(comp3 - hi3[:, :, None])
            tot3 = tl.sum(e3, axis=2)
            lp = hi3 + tl.log(tot3)                           # (BM, KO) mixture marginal
            kk = kb + tl.arange(0, KO)
            omask = valid[:, None] & (kk < KN)[None, :]
            if MISS:
                lmiss = lpi + l0[:, None]
                lkeep = l1m + lp
                lpm = tl.maximum(lkeep, lmiss) + libdevice.log1p(tl.exp(-tl.abs(lkeep - lmiss)))
            else:
                lpm = lp
            if OUTLIER:
                # detector outliers: (1 - rho) p_miss + rho p_out, p_out per observation
                lin = l1r + lpm
                lout = lr + lo[:, None]
                lpo = tl.maximum(lin, lout) + libdevice.log1p(tl.exp(-tl.abs(lin - lout)))
            else:
                lpo = lpm
            if GRAD:
                gk = tl.load(G + hkl[:, None] * KN + kk[None, :], omask, other=0.0)
                if OUTLIER:
                    rin = tl.exp(lin - lpo)
                    gr += tl.sum(tl.where(omask, gk * (1.0 - rin - sig_r), 0.0), axis=1)
                    gk = gk * rin
                if MISS:
                    r1 = tl.exp(l1m + lp - lpm)
                    gz += tl.sum(tl.where(omask, gk * (1.0 - r1 - sig_z), 0.0), axis=1)
                    w = gk * r1
                    if NOISE:
                        # every class but the outlier carries 1/2 log u; the miss uses L(0)
                        gu += tl.sum(tl.where(omask, 0.5 * w + gk * (1.0 - r1)
                                              * (uu[:, None] * d0[:, None] + 0.5), 0.0), axis=1)
                else:
                    w = gk
                    if NOISE:
                        gu += tl.sum(tl.where(omask, 0.5 * w, 0.0), axis=1)
                resp = tl.reshape(e3 / tot3[:, :, None], (BM, J))
                wj = tl.reshape(tl.broadcast_to(w[:, :, None], (BM, KO, BC)), (BM, J))
                pw, pt = tl.exp(win - lp_c), tl.exp(tail - lp_c)
                density = tl.exp(l0[:, None] + norm - tl.minimum(zt * zt, 1.1420073898156842e26) - lp_c)
                survival = tl.exp(l0[:, None] + lah - log_sum + log_q - lp_c)
                comp_mu = pw * smu / total - density
                comp_lo = (pw * (slo / total - wlo)
                           + tl.where(t > 0, wlo * survival, pt * whi - t * density))
                comp_hi = (pw * (shi / total - whi)
                           + tl.where(t > 0, -wlo * survival - t * density, -pt * whi))
                wr = tl.where(cmask, wj * resp, 0.0)
                acc_pi += wr
                acc_mu += wr * comp_mu
                acc_lo += wr * comp_lo
                acc_hi += wr * comp_hi
                if NOISE:
                    acc_u += wr * (uu[:, None] * (pw * sdv / total + pt * d0[:, None]))
            else:
                tl.atomic_add(S + hkl[:, None] * KN + kk[None, :], lpo, mask=omask)

        if GRAD:
            g_pi = tl.sum(tl.reshape(acc_pi, (BM, KO, BC)), axis=1)
            g_mu = tl.sum(tl.reshape(acc_mu, (BM, KO, BC)), axis=1)
            g_lo = tl.sum(tl.reshape(acc_lo, (BM, KO, BC)), axis=1)
            g_hi = tl.sum(tl.reshape(acc_hi, (BM, KO, BC)), axis=1)
            tl.store(Gmu + m, tl.sum(g_mu, axis=1), valid)
            c2 = tl.arange(0, BC)[None, :]
            m2 = valid[:, None] & (c2 < C)
            dst = m[:, None] * (5 * C) + c2
            tl.store(Gmix + dst, g_pi, m2)
            tl.store(Gmix + dst + C, g_mu, m2)
            tl.store(Gmix + dst + 2 * C, g_lo, m2)
            tl.store(Gmix + dst + 3 * C, g_hi, m2)
            tl.store(Gmix + dst + 4 * C, tl.zeros((BM, BC), tl.float32), m2)
            if MISS:
                tl.atomic_add(Gz, tl.sum(gz, axis=0))
            if OUTLIER:
                tl.atomic_add(Gr, tl.sum(gr, axis=0))
            if NOISE:
                tl.store(Gu + m, tl.sum(acc_u, axis=1) + gu, valid)


#: ``(observations per program, outer nodes per inner-grid pass, warps)`` by the
#: node count.  Measured on A100 at 1.41 M observations: two warps win
#: throughout; 16 nodes per pass for the 31-node training grid (forward +
#: backward 371 ms, against 599 at (4, 8, 4)), 8 for the 9-node locate scan
#: (41 ms against 64), where 16 wastes most of the pass.
_CONFIG_MANY, _CONFIG_FEW = (2, 16, 2), (2, 8, 2)
#: benchmarking override, ``(BM, KO, warps)`` or None
_OVERRIDE = None


def _config(n_nodes: int) -> tuple[int, int, int]:
    if _OVERRIDE is not None:
        return _OVERRIDE
    return _CONFIG_MANY if n_nodes >= 16 else _CONFIG_FEW


def _launch(grad, mu, mix, v_nodes, hkl, n_ph, p, nodes, sig_b2, miss, outlier=None,
            S=None, G=None, noise=False):
    M, KN = mu.numel(), v_nodes.shape[1]
    C = mix.shape[1] // 5
    dummy = mu
    out = {}
    if grad:
        out = {"gmu": torch.empty_like(mu), "gmix": torch.empty_like(mix),
               "gz": torch.zeros(1, device=mu.device, dtype=torch.float32),
               "gr": torch.zeros(1, device=mu.device, dtype=torch.float32)}
        if noise:
            out["gu"] = torch.empty_like(mu)
    l1m, lpi, sig_z = miss if miss is not None else (0.0, 0.0, 0.0)
    (l1r, lr, sig_r), lout = outlier if outlier is not None else ((0.0, 0.0, 0.0), None)
    bc = triton.next_power_of_2(C)
    bm, ko, warps = _config(KN)
    if M:
        _nodes_kernel[(triton.cdiv(M, bm),)](
            mu, n_ph, p["bg_model"], mix,
            sig_b2 if sig_b2 is not None else dummy, p["logpeak"], p["log_L0"],
            lout if lout is not None else dummy,
            *(p[k] for k in ("x_lo", "h1", "d_lo", "h_d", "d_hi", "h2", "n1", "n1d")),
            hkl, v_nodes, S if S is not None else dummy, G if G is not None else dummy,
            out.get("gmu", dummy), out.get("gmix", dummy), out.get("gz", dummy),
            out.get("gr", dummy),
            p["log_u"] if noise else dummy, p["d0"] if noise else dummy, out.get("gu", dummy),
            M, KN, l1m, lpi, sig_z, l1r, lr, sig_r,
            C, int(nodes), sig_b2 is not None, miss is not None, outlier is not None, bool(grad),
            bm, 8, bc, ko, ko * bc, bool(noise),
            num_warps=warps,
        )
    return out


class _NodeSums(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, mix, miss_logit, outlier_logit, log_out, v_nodes, hkl, n_ph, p, nodes,
                sig_b2, n_hkl, log_u):
        def _mix(logit):
            return (float(F.logsigmoid(-logit)), float(F.logsigmoid(logit)), float(torch.sigmoid(logit)))
        miss = _mix(miss_logit) if miss_logit is not None else None
        outlier = ((_mix(outlier_logit), log_out.contiguous().float())
                   if outlier_logit is not None else None)
        S = torch.zeros(n_hkl, v_nodes.shape[1], device=mu.device, dtype=torch.float32)
        noise = log_u is not None
        _launch(False, mu, mix, v_nodes, hkl, n_ph, p, nodes, sig_b2, miss, outlier, S=S, noise=noise)
        ctx.save_for_backward(mu, mix, v_nodes, hkl, n_ph)
        ctx.p, ctx.nodes, ctx.sig_b2, ctx.miss, ctx.outlier = p, nodes, sig_b2, miss, outlier
        ctx.noise = noise
        return S

    @staticmethod
    @once_differentiable
    def backward(ctx, gS):
        mu, mix, v_nodes, hkl, n_ph = ctx.saved_tensors
        out = _launch(True, mu, mix, v_nodes, hkl, n_ph, ctx.p, ctx.nodes, ctx.sig_b2,
                      ctx.miss, ctx.outlier, G=gS.contiguous(), noise=ctx.noise)
        need = ctx.needs_input_grad
        return (out["gmu"] if need[0] else None, out["gmix"] if need[1] else None,
                out["gz"].reshape(()) if need[2] else None,
                out["gr"].reshape(()) if need[3] else None,
                None, None, None, None, None, None, None, None,
                out["gu"] if ctx.noise and need[12] else None)


def posterior_node_sums(mu, mix, miss_logit, v_nodes, hkl, n_ph, p, nodes, *,
                        sig_b2=None, n_hkl: int, outlier_logit=None, log_out=None, log_u=None):
    """``S[h, k] = sum_{j: hkl_j = h} log p_miss(n_j | v[h, k] + mu_j)``, ``(n_hkl, K)``.

    CUDA float32; ``mix`` is the packed ``(M, 5C)`` fixed-Gaussian mixture,
    ``p`` the split-grid parameters of ``(n_ph, bg)`` and ``miss_logit`` the
    objective's miss logit or ``None``.  ``outlier_logit`` with ``log_out``
    (``(M,)``, :meth:`MarginalNLL.log_outlier`) adds the detector-outlier
    component.  ``log_u`` (``(M,)``) is the detector noise; ``p`` must then be the
    grid built with the same values (``grid_params(..., log_u)``).  Differentiable
    in ``mu``, ``mix``, both logits and ``log_u``.  Summation into ``S`` uses atomics,
    so the last bits vary from run to run.
    """
    if log_u is not None and p.get("log_u") is None:
        raise ValueError("log_u given but the grid was built without it")
    if log_u is None and p.get("log_u") is not None:
        raise ValueError("the grid carries detector noise; pass the same log_u")
    return _NodeSums.apply(mu.contiguous(), mix.contiguous(), miss_logit, outlier_logit, log_out,
                           v_nodes.contiguous().float(), hkl.contiguous(), n_ph.contiguous(),
                           p, int(nodes), sig_b2, int(n_hkl), log_u)
