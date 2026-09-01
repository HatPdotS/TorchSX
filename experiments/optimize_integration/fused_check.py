"""Fused kernel vs the per-column path: identical answers, and how much faster.

The fused kernel is not bit-exact against the per-column one and is not meant
to be -- it replaces the online-logsumexp maximum with a closed-form upper
bound, so the two sum the same terms in a different order and with a different
shift.  What must hold is agreement to float32 round-off on `log_p` and on
every gradient, which is what this checks.
"""
from __future__ import annotations

import math
import time

import torch

from torchsx.merging.quadrature._fixed_log_grid_triton import (
    marginalize_signal_triton,
)
from torchsx.merging.quadrature._fused_posterior_triton import (
    fused_posterior_logp,
)


def _percol(base, n_ph, bg, la, lb, hkl, v, log_S, log_w, lnf):
    return torch.stack([
        marginalize_signal_triton(
            v[:, c].index_select(0, hkl) + base, n_ph, bg, la, lb,
            log_S, log_w, log_nfact=lnf,
        ) for c in range(v.shape[1])
    ], dim=1)


def _inputs(M, K, C, n_hkl, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    log_S = torch.linspace(-34.5, 34.5, K, device=dev)
    log_w = torch.full((K,), math.log(69.0 / (K - 1)), device=dev)
    log_w[0] -= math.log(2.0)
    log_w[-1] -= math.log(2.0)
    hkl = torch.randint(0, n_hkl, (M,), device=dev, generator=g)
    v = (torch.randn(n_hkl, 1, device=dev, generator=g) * 2
         + torch.linspace(-6, 6, C, device=dev)[None, :])
    return dict(
        base=torch.randn(M, device=dev, generator=g) * 0.3,
        n_ph=(torch.rand(M, device=dev, generator=g) * 200).round(),
        bg=torch.rand(M, device=dev, generator=g) * 20 + 0.5,
        la=torch.randn(M, device=dev, generator=g) * 0.3 + 0.6,
        lb=torch.zeros(M, device=dev) + 0.35,
        hkl=hkl, v=v, log_S=log_S, log_w=log_w,
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
    dev, K, C = "cuda", 2001, 21
    print("── correctness (M = 8192) ──")
    d = _inputs(8192, K, C, 512, dev)
    lnf = torch.lgamma(d["n_ph"] + 1)
    args = (d["n_ph"], d["bg"])

    for live in (("mu", "ab"), ("mu",), ()):
        ctx = torch.enable_grad() if live else torch.no_grad()
        outs, grads = [], []
        for impl in ("percol", "fused"):
            b = d["base"].clone().requires_grad_("mu" in live)
            a = d["la"].clone().requires_grad_("ab" in live)
            l = d["lb"].clone().requires_grad_("ab" in live)
            with ctx:
                if impl == "percol":
                    o = _percol(b, *args, a, l, d["hkl"], d["v"],
                                d["log_S"], d["log_w"], lnf)
                else:
                    o = fused_posterior_logp(b, *args, a, l, d["hkl"], d["v"],
                                             d["log_S"], d["log_w"],
                                             log_nfact=lnf)
                outs.append(o)
                if live:
                    tgt = [p for p, k in ((b, "mu"), (a, "ab"), (l, "ab"))
                           if k in live]
                    torch.manual_seed(3)
                    gs = torch.randn_like(o)
                    grads.append(torch.autograd.grad((o * gs).sum(), tgt))
        rel = ((outs[1] - outs[0]).abs()
               / outs[0].abs().clamp(min=1e-6)).median()
        print(f"  live={str(live):<14} log_p  max abs "
              f"{float((outs[1]-outs[0]).abs().max()):.3e}   median rel {float(rel):.3e}")
        for i, nm in enumerate([n for n in ("base", "log_alpha", "log_beta")
                                if (n == "base") == ("mu" in live)
                                or (n != "base" and "ab" in live)][:len(grads[0]) if grads else 0]):
            x, y = grads[1][i], grads[0][i]
            r = ((x - y).abs() / y.abs().clamp(min=1e-6)).median()
            print(f"      d/d {nm:<10} max abs {float((x-y).abs().max()):.3e}"
                  f"   median rel {float(r):.3e}")

    print(f"\n── speed (M = 1,048,576, K = {K}, C = {C}) ──")
    d = _inputs(1 << 20, K, C, 14128, dev)
    lnf = torch.lgamma(d["n_ph"] + 1)
    rows = {}
    for name, live in (("grad (mu+ab)", ("mu", "ab")), ("no grad (scan)", ())):
        ctx = torch.enable_grad() if live else torch.no_grad()
        for impl in ("percol", "fused"):
            b = d["base"].clone().requires_grad_("mu" in live)
            a = d["la"].clone().requires_grad_("ab" in live)
            l = d["lb"].clone().requires_grad_("ab" in live)

            def run(impl=impl, b=b, a=a, l=l, ctx=ctx):
                with ctx:
                    if impl == "percol":
                        _percol(b, d["n_ph"], d["bg"], a, l, d["hkl"], d["v"],
                                d["log_S"], d["log_w"], lnf)
                    else:
                        fused_posterior_logp(b, d["n_ph"], d["bg"], a, l,
                                             d["hkl"], d["v"], d["log_S"],
                                             d["log_w"], log_nfact=lnf)
            rows[(name, impl)] = _bench(run, reps=3)
        p, f = rows[(name, "percol")], rows[(name, "fused")]
        print(f"  {name:<16} per-column {p:8.1f} ms   fused {f:8.1f} ms   "
              f"{p/f:.2f}x")

    n_scan, n_coarse = 9, C
    for impl in ("percol", "fused"):
        tot = (n_scan / C) * rows[("no grad (scan)", impl)] + \
              rows[("grad (mu+ab)", impl)]
        print(f"  -> full step ({n_scan} scan + {n_coarse} coarse), {impl}: "
              f"{tot:.0f} ms")


if __name__ == "__main__":
    main()
