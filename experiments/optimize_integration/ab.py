"""Localise the wired fused path's gradient error.

The kernel beats the per-column path against float64 in isolation, yet through
``PosteriorMergeTarget`` its gradient direction is ~600x worse.  So compare the
two paths at every stage -- the accumulated curve S, then log Z, then the
moments, then the gradients -- and see where they part company.
"""
from __future__ import annotations

import torch

from oracle import make_problem, oracle
from score import production, _mismatch, _relmed


def main() -> None:
    dev = "cuda"
    prob = make_problem(seed=0, device=dev, n_per_mult=32, mults=(128, 751),
                        w_range=(1.2, 2.4), beta_range=(1.3, 1.5),
                        sigma_ladder=(5.0, 50.0, 500.0, 5000.0))
    ref = oracle(prob)

    out, g, tgt = {}, {}, {}
    for f in (False, True):
        o, gr, t = production(prob, use_fused=f)
        out[f], g[f], tgt[f] = o, gr, t

    gref = (ref.grad_base, ref.grad_log_alpha, ref.grad_log_beta)
    print(f"{'quantity':<16} {'per-column':>12} {'fused':>12}   (vs oracle)")
    for nm, key in (("log_Z", "log_Z"), ("I0_mean", "I0_mean"),
                    ("I0_sd", "I0_sd")):
        r = getattr(ref, key)
        print(f"{nm:<16} {_relmed(getattr(out[False], key), r):>12.2e} "
              f"{_relmed(getattr(out[True], key), r):>12.2e}")
    for i, nm in enumerate(("1-cos g_base", "1-cos g_logalpha", "1-cos g_logbeta")):
        print(f"{nm:<16} {_mismatch(g[False][i], gref[i]):>12.2e} "
              f"{_mismatch(g[True][i], gref[i]):>12.2e}")

    print("\nper-column vs fused, directly:")
    for nm, key in (("log_Z", "log_Z"), ("I0_mean", "I0_mean")):
        a, b = getattr(out[False], key), getattr(out[True], key)
        print(f"  {nm:<10} max abs {float((a-b).abs().max()):.3e}   "
              f"median rel {_relmed(b, a):.3e}")
    for i, nm in enumerate(("g_base", "g_logalpha", "g_logbeta")):
        a, b = g[False][i], g[True][i]
        print(f"  {nm:<10} max abs {float((a-b).abs().max()):.3e}   "
              f"1-cos {_mismatch(b, a):.3e}")

    # the located grid is chosen by a no_grad sweep that ALSO runs through the
    # kernel under test -- if the two paths place different grids, everything
    # downstream differs for a reason that has nothing to do with the gradients
    c0, s0 = tgt[False]._loc
    c1, s1 = tgt[True]._loc
    print(f"\ngrid placement:  centre max |d| {float((c0-c1).abs().max()):.3e}"
          f"   sd max |d| {float((s0-s1).abs().max()):.3e}"
          f"   sd median {float(s0.median()):.3f}")


if __name__ == "__main__":
    main()
