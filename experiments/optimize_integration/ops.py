"""Candidate implementations of the posterior accumulator, one signature.

    S[h, c] = sum_{j: hkl_j = h}  log p_j(c)

    log p_j(c) = log INT p_pois(n_j | S + bg_j)
                        . GG(log S - v[h,c] - base_j; alpha_j, beta_j) d log S

Every entry in :data:`CANDIDATES` takes the same arguments and returns
``(n_hkl, C)``, differentiable in ``base``, ``log_alpha`` and ``log_beta``.
Passing ``hkl_id = arange(M)`` and ``n_hkl = M`` makes ``S`` the per-observation
``log p`` table, which is what the accuracy harness does -- the group sum is
then exact by construction and cannot mask a per-observation error.

Accuracy note that decides the whole design: an error in ``S`` that is
**constant in c** cancels exactly from ``log Z_h = logsumexp_c(S + log w)``,
from the posterior weights, and hence from ``I0_mean``.  Only the c-profile is
observable downstream.  Score ``S - mean_c(S)``.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["reference", "current", "eager", "CANDIDATES"]


def _log_gg_norm(beta: Tensor) -> Tensor:
    return beta.log() - math.log(2.0) - torch.lgamma(1.0 / beta)


def _trap_log_w(K: int, dlog: float, device, dtype) -> Tensor:
    w = torch.full((K,), math.log(dlog), device=device, dtype=dtype)
    w[0] -= math.log(2.0)
    w[-1] -= math.log(2.0)
    return w


# ── ground truth ───────────────────────────────────────────────────────


def reference(
    base: Tensor, n_ph: Tensor, bg: Tensor,
    log_alpha: Tensor, log_beta: Tensor,
    hkl_id: Tensor, v_nodes: Tensor, n_hkl: int,
    *, n_points: int = 32001, lo: float = -60.0, hi: float = 60.0,
    chunk: int = 2048,
) -> Tensor:
    """float64 dense-grid truth.  Slow on purpose; run it once.

    Wider and 16x denser than the production grid in both directions, so its
    own truncation and discretisation errors are far below anything being
    measured.  ``bench.py --self-check`` verifies that by halving the density.
    """
    dev, C = base.device, v_nodes.shape[1]
    f64 = torch.float64
    log_S = torch.linspace(lo, hi, n_points, device=dev, dtype=f64)
    S = log_S.exp()
    log_w = _trap_log_w(n_points, float(log_S[1] - log_S[0]), dev, f64)

    b64 = base.to(f64)
    la, lb = log_alpha.to(f64), log_beta.to(f64)
    beta, alpha = lb.exp(), la.exp()
    log_norm = _log_gg_norm(beta) - la
    n64, bg64 = n_ph.to(f64), bg.to(f64)
    log_nfact = torch.lgamma(n64 + 1)

    out = base.new_zeros(n_hkl, C, dtype=f64)
    for s in range(0, base.shape[0], chunk):
        e = min(s + chunk, base.shape[0])
        lam = (S[None, :] + bg64[s:e, None]).clamp(min=1e-300)
        log_lik = n64[s:e, None] * lam.log() - lam - log_nfact[s:e, None]
        cols = []
        for c in range(C):
            mu = v_nodes[hkl_id[s:e], c].to(f64) + b64[s:e]
            z = (log_S[None, :] - mu[:, None]).abs() / alpha[s:e, None]
            log_gg = -z.clamp(min=1e-300).pow(beta[s:e, None]) + log_norm[s:e, None]
            cols.append(torch.logsumexp(log_gg + log_lik + log_w[None, :], dim=1))
        out.index_add_(0, hkl_id[s:e], torch.stack(cols, dim=1))
    return out


# ── production baseline ────────────────────────────────────────────────


def current(
    base: Tensor, n_ph: Tensor, bg: Tensor,
    log_alpha: Tensor, log_beta: Tensor,
    hkl_id: Tensor, v_nodes: Tensor, n_hkl: int,
    *, n_points: int = 2001, lo: float = -34.5, hi: float = 34.5,
) -> Tensor:
    """What ``PosteriorMergeTarget.accumulate`` runs today: C Triton launches
    on a shared fixed grid, one ``index_add_`` per column."""
    from torchsx.merging.quadrature._fixed_log_grid_triton import (
        marginalize_signal_triton,
    )
    dev, dt, C = base.device, base.dtype, v_nodes.shape[1]
    log_S = torch.linspace(lo, hi, n_points, device=dev, dtype=dt)
    log_w = _trap_log_w(n_points, float(log_S[1] - log_S[0]), dev, dt)
    log_nfact = torch.lgamma(n_ph + 1)

    cols = []
    for c in range(C):
        mu = v_nodes[:, c].index_select(0, hkl_id) + base
        lp = marginalize_signal_triton(
            mu, n_ph, bg, log_alpha, log_beta, log_S, log_w, log_nfact=log_nfact,
        )
        col = base.new_zeros(n_hkl)
        cols.append(col.index_add(0, hkl_id, lp))
    return torch.stack(cols, dim=1)


def eager(
    base: Tensor, n_ph: Tensor, bg: Tensor,
    log_alpha: Tensor, log_beta: Tensor,
    hkl_id: Tensor, v_nodes: Tensor, n_hkl: int,
    *, n_points: int = 2001, lo: float = -34.5, hi: float = 34.5,
    chunk: int = 65536,
) -> Tensor:
    """Same grid as :func:`current`, pure PyTorch.  Cross-checks the Triton
    kernel's *forward* independently of its hand-written backward."""
    dev, dt, C = base.device, base.dtype, v_nodes.shape[1]
    log_S = torch.linspace(lo, hi, n_points, device=dev, dtype=dt)
    S = log_S.exp()
    log_w = _trap_log_w(n_points, float(log_S[1] - log_S[0]), dev, dt)
    beta, alpha = log_beta.exp(), log_alpha.exp()
    log_norm = _log_gg_norm(beta) - log_alpha
    log_nfact = torch.lgamma(n_ph + 1)

    out = base.new_zeros(n_hkl, C)
    for s in range(0, base.shape[0], chunk):
        e = min(s + chunk, base.shape[0])
        lam = (S[None, :] + bg[s:e, None]).clamp(min=1e-30)
        log_lik = n_ph[s:e, None] * lam.log() - lam - log_nfact[s:e, None]
        acc = []
        for c in range(C):
            mu = v_nodes[hkl_id[s:e], c] + base[s:e]
            z = (log_S[None, :] - mu[:, None]).abs() / alpha[s:e, None]
            log_gg = -z.clamp(min=1e-30).pow(beta[s:e, None]) + log_norm[s:e, None]
            acc.append(torch.logsumexp(log_gg + log_lik + log_w[None, :], dim=1))
        out = out.index_add(0, hkl_id[s:e], torch.stack(acc, dim=1))
    return out


def _at(n_points: int, half: float = 34.5):
    """``current`` at a given inner node count / extent."""
    import functools
    return functools.partial(current, n_points=n_points, lo=-half, hi=half)


#: The real dark batch has median n_ph = 136, so a typical observation's
#: Poisson factor is 1/sqrt(136) = 0.086 wide in log S while the production
#: grid's spacing is 69/2000 = 0.0345 -- 2.5 nodes across the peak, and 0.5 at
#: the 99th percentile.  The synthetic problems used to pick the node count had
#: median counts of 26 and below, which is why they reported it as converged.
CANDIDATES = {
    "current": current,
    **{f"K={n}": _at(n) for n in (1001, 2001, 4001, 8001, 16001, 32001)},
    **{f"K=4001 half={h:.0f}": _at(4001, h) for h in (20.0, 25.0)},
    # The drop-in production change: same node count and runtime, narrower
    # extent.  The grid spans 69 units of log S but the real batch has bg
    # median 116 and n median 136, so the integrand lives near log 136 = 4.9
    # and most of that span holds nothing.  Narrowing trades GenGauss tail
    # coverage for resolution -- which way that lands is measured, not argued.
    **{f"K=2001 half={h:.0f}": _at(2001, h) for h in (12.0, 15.0, 20.0, 25.0)},
    "eager": eager,
}
