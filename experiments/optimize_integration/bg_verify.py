"""Correctness of the background-variance branches in the Triton kernel.

Three things, in order of how badly a failure would mislead everything else:
  1. bg_var_factor = 0 must be BIT-IDENTICAL to before (the default path).
  2. Triton must agree with the eager implementation for each mode.
  3. The analytic gradients must agree with autograd through the eager path.
"""
from __future__ import annotations

import math
import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature._fixed_log_grid_triton import (
    marginalize_signal_triton,
)

DEV = "cuda"
M, K = 4096, 1355


def setup(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    log_S = torch.linspace(-34.5, 34.5, K, device=DEV)
    log_w = torch.full((K,), math.log(69.0 / (K - 1)), device=DEV)
    log_w[0] -= math.log(2.0)
    log_w[-1] -= math.log(2.0)
    n = (torch.rand(M, device=DEV, generator=g) * 400).round()
    bg = torch.rand(M, device=DEV, generator=g) * 250 + 2.0
    la = torch.full((M,), math.log(1.65) - 0.263, device=DEV)
    lb = torch.full((M,), math.log(1.07), device=DEV)
    centre = (n - bg).clamp(min=1.0).log()
    return log_S, log_w, n, bg, la, lb, centre


def eager(lik, log_S, log_w, n, bg, la, lb, centre):
    from torchsx.merging.scatter.gengauss import gengauss_log_prob_from_obs
    eps = log_S[None, :] - centre[:, None]
    ls = gengauss_log_prob_from_obs(eps, la, lb)
    ll = lik.log_prob_given_signal(n, log_S.exp()[None, :].expand(M, K), bg)
    return torch.logsumexp(ls + ll + log_w[None, :], dim=1)


def main():
    log_S, log_w, n, bg, la, lb, centre = setup()
    f = 149.0 / 128.0

    a = marginalize_signal_triton(centre, n, bg, la, lb, log_S, log_w)
    b = marginalize_signal_triton(centre, n, bg, la, lb, log_S, log_w,
                                  sig_b2=None, bg_deviance=False)
    print(f"1. bg_var off, both mode flags: max |diff| = "
          f"{float((a - b).abs().max()):.3e}   (must be 0)")

    print("\n2. triton vs eager (float32 grid, 1355 nodes)")
    print(f"{'mode':<16} {'max':>11} {'median':>11}")
    for tag, kw, lik in (
        ("off", dict(), Poisson()),
        ("deviance", dict(sig_b2=f * bg, bg_deviance=True),
         Poisson(bg_var_factor=f, bg_var_mode="deviance")),
        ("second_order", dict(sig_b2=f * bg, bg_deviance=False),
         Poisson(bg_var_factor=f, bg_var_mode="second_order")),
    ):
        t = marginalize_signal_triton(centre, n, bg, la, lb, log_S, log_w, **kw)
        e = eager(lik, log_S, log_w, n, bg, la, lb, centre)
        d = (t - e).abs()
        print(f"{tag:<16} {float(d.max()):>11.3e} {float(d.median()):>11.3e}")

    # the kernel's accumulators are fp32 by construction, so this is a fp32
    # comparison on both sides -- it checks the analytic formulas, not precision
    print("\n3. gradients, analytic (triton) vs autograd (eager)")
    log_S64, log_w64 = log_S, log_w
    n64, bg64 = n, bg
    for tag, kw, lik in (
        ("off", dict(), Poisson()),
        ("deviance", dict(sig_b2=f * bg64, bg_deviance=True),
         Poisson(bg_var_factor=f, bg_var_mode="deviance")),
        ("second_order", dict(sig_b2=f * bg64, bg_deviance=False),
         Poisson(bg_var_factor=f, bg_var_mode="second_order")),
    ):
        out = []
        for use_triton in (True, False):
            c = centre.clone().requires_grad_(True)
            A = la.clone().requires_grad_(True)
            B = lb.clone().requires_grad_(True)
            if use_triton:
                lp = marginalize_signal_triton(c, n64, bg64, A, B,
                                               log_S64, log_w64, **kw)
            else:
                lp = eager(lik, log_S64, log_w64, n64, bg64, A, B, c)
            lp.sum().backward()
            out.append((c.grad.clone(), A.grad.clone(), B.grad.clone()))
        (gc, ga, gb), (ec, ea, eb) = out
        rel = lambda x, y: float(((x - y).abs()
                                  / y.abs().clamp(min=1e-8)).median())  # noqa: E731
        print(f"{tag:<16} d/dlog_mean {rel(gc, ec):.2e}   "
              f"d/dlog_alpha {rel(ga, ea):.2e}   d/dlog_beta {rel(gb, eb):.2e}")


if __name__ == "__main__":
    main()
