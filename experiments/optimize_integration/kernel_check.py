"""Correctness and speed of the gated Poisson kernel.

The gates must be invisible: turning a statistic off may not change ``log_p``
or any gradient that is still being asked for.  Reference is the pre-gate
kernel, restored from ``/tmp/_fixed_backup.py`` if present, else the eager path.
"""
from __future__ import annotations

import math
import time

import torch

from torchsx.merging.quadrature._fixed_log_grid_triton import (
    marginalize_signal_triton,
)


def _eager(log_mean, n_ph, bg, la, lb, log_S, log_w):
    beta, alpha = lb.exp(), la.exp()
    S = log_S.exp()
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    log_f = n_ph[:, None] * lam.log() - lam - torch.lgamma(n_ph[:, None] + 1)
    z = (log_S[None, :] - log_mean[:, None]).abs() / alpha[:, None].clamp(min=1e-8)
    lg = (-z.clamp(min=1e-12).pow(beta[:, None])
          + beta.log()[:, None] - math.log(2.0)
          - torch.lgamma(1.0 / beta)[:, None] - la[:, None])
    return torch.logsumexp(lg + log_f + log_w[None, :], dim=1)


def _inputs(M, K, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    log_S = torch.linspace(-34.5, 34.5, K, device=dev)
    log_w = torch.full((K,), math.log(69.0 / (K - 1)), device=dev)
    log_w[0] -= math.log(2.0)
    log_w[-1] -= math.log(2.0)
    return (
        torch.randn(M, device=dev, generator=g),
        (torch.rand(M, device=dev, generator=g) * 50).round(),
        torch.rand(M, device=dev, generator=g) * 5 + 0.5,
        torch.randn(M, device=dev, generator=g) * 0.2 + 0.8,
        torch.zeros(M, device=dev) + 0.35,
        log_S, log_w,
    )


def _bench(fn, reps=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps * 1e3


def main() -> None:
    dev = "cuda"
    print("── correctness: gates must not change any answer ──")
    M, K = 4096, 2001
    lm, n, bg, la, lb, log_S, log_w = _inputs(M, K, dev)
    ref = _eager(lm, n, bg, la, lb, log_S, log_w)
    got = marginalize_signal_triton(lm, n, bg, la, lb, log_S, log_w)
    print(f"  forward (no grad) vs eager      max abs {float((got-ref).abs().max()):.3e}")

    # every input live -- the pre-gate behaviour
    a = [t.clone().requires_grad_(True) for t in (lm, bg, la, lb)]
    out = marginalize_signal_triton(a[0], n, a[1], a[2], a[3], log_S, log_w)
    gseed = torch.randn(M, device=dev, generator=torch.Generator(device=dev).manual_seed(1))
    g_all = torch.autograd.grad((out * gseed).sum(), a)

    # bg no longer live -- S2 is skipped in the kernel
    b = [t.clone().requires_grad_(True) for t in (lm, la, lb)]
    out2 = marginalize_signal_triton(b[0], n, bg, b[1], b[2], log_S, log_w)
    g_sub = torch.autograd.grad((out2 * gseed).sum(), b)
    print(f"  log_p all-live vs bg-off        max abs "
          f"{float((out - out2).abs().max()):.3e}")
    for nm, x, y in (("d/d log_mean", g_sub[0], g_all[0]),
                     ("d/d log_alpha", g_sub[1], g_all[2]),
                     ("d/d log_beta", g_sub[2], g_all[3])):
        print(f"  {nm:<16} bg-off vs all-live    max abs "
              f"{float((x - y).abs().max()):.3e}")

    # only alpha/beta live -- S1 skipped too
    c = [t.clone().requires_grad_(True) for t in (la, lb)]
    out3 = marginalize_signal_triton(lm, n, bg, c[0], c[1], log_S, log_w)
    g_ab = torch.autograd.grad((out3 * gseed).sum(), c)
    print(f"  d/d log_alpha  ab-only vs all   max abs "
          f"{float((g_ab[0] - g_all[2]).abs().max()):.3e}")
    print(f"  d/d log_beta   ab-only vs all   max abs "
          f"{float((g_ab[1] - g_all[3]).abs().max()):.3e}")

    print("\n── speed: M = 1,048,576 ──")
    M = 1 << 20
    lm, n, bg, la, lb, log_S, log_w = _inputs(M, K, dev)
    lnf = torch.lgamma(n + 1)
    print(f"  {'gates':<34} {'fwd ms':>9} {'vs all':>8}")
    rows = []
    for name, live in (("all live (pre-gate behaviour)", ("mu", "ab", "bg")),
                       ("mu+ab live (bg is data)", ("mu", "ab")),
                       ("mu only", ("mu",)),
                       ("ab only", ("ab",)),
                       ("no grad (the `locate` sweep)", ())):
        x = lm.clone().requires_grad_("mu" in live)
        bb = bg.clone().requires_grad_("bg" in live)
        aa = la.clone().requires_grad_("ab" in live)
        ll = lb.clone().requires_grad_("ab" in live)
        ctx = torch.enable_grad() if live else torch.no_grad()

        def run(x=x, bb=bb, aa=aa, ll=ll, ctx=ctx):
            with ctx:
                marginalize_signal_triton(x, n, bb, aa, ll, log_S, log_w,
                                          log_nfact=lnf)
        t = _bench(run)
        rows.append(t)
        print(f"  {name:<34} {t:9.2f} {t/rows[0]:8.2f}x")


if __name__ == "__main__":
    main()
