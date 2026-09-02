"""What does the background-variance branch cost the kernel?

The deviance form adds one reciprocal, one multiply and one log2 per grid point
inside the k loop; the second-order form adds one reciprocal and no
transcendental.  The roofline said this kernel runs at 17.5 % of SFU peak, so
the prediction is that neither is visible.  Measured, not assumed.
"""
from __future__ import annotations

import math
import time

import torch

from torchsx.merging.quadrature._fixed_log_grid_triton import (
    marginalize_signal_triton,
)

DEV = "cuda"


def bench(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    M, K = 1_050_000, 1355
    g = torch.Generator(device=DEV).manual_seed(0)
    log_S = torch.linspace(-34.5, 34.5, K, device=DEV)
    log_w = torch.full((K,), math.log(69.0 / (K - 1)), device=DEV)
    log_w[0] -= math.log(2.0); log_w[-1] -= math.log(2.0)
    n = (torch.rand(M, device=DEV, generator=g) * 250 + 5).round()
    bg = torch.rand(M, device=DEV, generator=g) * 200 + 10
    s2 = 1.1641 * bg
    la = torch.full((M,), math.log(1.65) - 0.263, device=DEV)
    lb = torch.full((M,), math.log(1.07), device=DEV)
    centre = (n - bg).clamp(min=1.0).log()

    print(f"M={M:,}  K={K}\n{'mode':<16} {'no-grad ms':>11} {'grad ms':>10}")
    for tag, kw in (("off", dict()),
                    ("deviance", dict(sig_b2=s2, bg_deviance=True)),
                    ("second_order", dict(sig_b2=s2, bg_deviance=False))):
        with torch.no_grad():
            t0 = bench(lambda: marginalize_signal_triton(
                centre, n, bg, la, lb, log_S, log_w, **kw))

        c = centre.clone().requires_grad_(True)
        A = la.clone().requires_grad_(True)
        B = lb.clone().requires_grad_(True)

        def fwd_bwd():
            lp = marginalize_signal_triton(c, n, bg, A, B, log_S, log_w, **kw)
            lp.sum().backward()

        t1 = bench(fwd_bwd, reps=10)
        print(f"{tag:<16} {t0:>11.2f} {t1:>10.2f}")


if __name__ == "__main__":
    main()
