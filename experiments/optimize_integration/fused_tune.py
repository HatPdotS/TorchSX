"""Does the fused kernel lose on tiling, or on its structure?

It does ~25 % fewer transcendentals per (m, c, k) than the per-column path and
no cross-lane reductions at all, yet measures slower.  The prime suspect is the
serial ``k`` loop: 2001 iterations with scalar grid loads, against 125
vectorised ones for the per-column kernel.  If that is the cause, more elements
per iteration (bigger BLOCK_M, more warps) should recover it; if no tiling
helps, the structure is wrong and the per-column path stays.
"""
from __future__ import annotations

import math
import time

import torch

from torchsx.merging.quadrature._fused_posterior_triton import (
    fused_posterior_logp,
)
from fused_check import _inputs, _percol, _bench


def main() -> None:
    dev, K, C, M = "cuda", 2001, 21, 1 << 20
    d = _inputs(M, K, C, 14128, dev)
    lnf = torch.lgamma(d["n_ph"] + 1)
    b = d["base"].clone().requires_grad_(True)
    a = d["la"].clone().requires_grad_(True)
    l = d["lb"].clone().requires_grad_(True)

    base_ms = _bench(lambda: _percol(b, d["n_ph"], d["bg"], a, l, d["hkl"],
                                     d["v"], d["log_S"], d["log_w"], lnf), reps=3)
    print(f"per-column baseline: {base_ms:.1f} ms\n")
    print(f"  {'BLOCK_M':>8} {'warps':>6} {'stages':>7} {'ms':>9} {'vs base':>9}")
    best = (None, 1e9)
    for bm in (16, 32, 64, 128, 256):
        for w in (2, 4, 8):
            for st in (1, 2, 3):
                try:
                    t = _bench(lambda bm=bm, w=w, st=st: fused_posterior_logp(
                        b, d["n_ph"], d["bg"], a, l, d["hkl"], d["v"],
                        d["log_S"], d["log_w"], log_nfact=lnf,
                        block_m=bm, num_warps=w, num_stages=st), reps=3)
                except Exception as exc:                      # noqa: BLE001
                    print(f"  {bm:>8} {w:>6} {st:>7}   {type(exc).__name__}")
                    continue
                flag = "  <-- best" if t < best[1] else ""
                if t < best[1]:
                    best = ((bm, w, st), t)
                print(f"  {bm:>8} {w:>6} {st:>7} {t:>9.1f} {base_ms/t:>8.2f}x{flag}",
                      flush=True)
    print(f"\nbest {best[0]} at {best[1]:.1f} ms = {base_ms/best[1]:.2f}x "
          f"the per-column path")


if __name__ == "__main__":
    main()
