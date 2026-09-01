"""Where the fused kernel should still win: node count.

The per-column path pays one launch per outer node, so its cost is linear in C.
The fused path holds all C nodes in one (BLOCK_M, BLOCK_C) tile with
BLOCK_C = next_pow2(C), so every C in (17..32] costs exactly the same.  If that
holds, k_coarse = 32 is free relative to 21 -- which is worth ~3x on gradient
accuracy for nothing.
"""
from __future__ import annotations

import torch

from torchsx.merging.quadrature._fused_posterior_triton import (
    fused_posterior_logp,
)
from fused_check import _inputs, _percol, _bench


def main() -> None:
    dev, K, M = "cuda", 2001, 1 << 20
    print(f"M = {M:,}  K = {K}   grad path (mu+ab live)\n")
    print(f"  {'C':>4} {'per-column ms':>14} {'fused ms':>10} {'ratio':>7} "
          f"{'BLOCK_C':>8}")
    for C in (9, 13, 17, 21, 25, 32, 33, 48, 64):
        d = _inputs(M, K, C, 14128, dev)
        lnf = torch.lgamma(d["n_ph"] + 1)
        b = d["base"].clone().requires_grad_(True)
        a = d["la"].clone().requires_grad_(True)
        l = d["lb"].clone().requires_grad_(True)
        p = _bench(lambda: _percol(b, d["n_ph"], d["bg"], a, l, d["hkl"],
                                   d["v"], d["log_S"], d["log_w"], lnf), reps=3)
        f = _bench(lambda: fused_posterior_logp(
            b, d["n_ph"], d["bg"], a, l, d["hkl"], d["v"], d["log_S"],
            d["log_w"], log_nfact=lnf), reps=3)
        print(f"  {C:>4} {p:>14.1f} {f:>10.1f} {p/f:>6.2f}x "
              f"{max(16, 1 << (C-1).bit_length()):>8}", flush=True)


if __name__ == "__main__":
    main()
