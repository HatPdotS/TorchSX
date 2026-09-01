"""Does F = sqrt(E[I]) differ from E[sqrt(I)] enough to care?

Both come from the same posterior; the plug-in version is the delta-method
approximation, exact only to second order in the relative posterior width
r = sigma_I / I.  The gap goes as ~r^2/8, so it vanishes for well-determined
reflections and grows for weak ones -- which is where this merge struggles.
The exact version costs one extra weighted sum over a grid already built, so
the only question is whether it is worth the line.
"""
from __future__ import annotations

import torch

from oracle import REAL_LIKE, make_problem, oracle


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prob = make_problem(seed=0, device=dev, n_per_mult=16, **REAL_LIKE)
    o = oracle(prob)

    I = o.v.exp()
    w = o.w_post
    rootI = I.clamp(min=0).sqrt()
    F_exact = (w * rootI).sum(1)
    sF_exact = ((w * (rootI - F_exact[:, None]) ** 2).sum(1)).clamp(min=0).sqrt()

    F_plug = o.I0_mean.clamp(min=0).sqrt()
    sF_plug = o.I0_sd / (2 * F_plug.clamp(min=1e-12))

    r = o.I0_sd / o.I0_mean.clamp(min=1e-30)
    q = lambda t, p: float(t.quantile(torch.tensor(p, dtype=t.dtype, device=t.device)))
    mult = torch.zeros(prob.n_hkl, dtype=torch.long, device=dev)
    mult.index_add_(0, prob.hkl_id, torch.ones_like(prob.hkl_id))

    print(f"relative posterior width r = sigma_I/I:  median {float(r.median()):.3f}"
          f"  q90 {q(r, 0.9):.3f}  max {float(r.max()):.3f}\n")
    print(f"{'r range':>12} {'n':>5} {'F: |plug-exact|/exact':>22} "
          f"{'sigF: |plug-exact|/exact':>25}")
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1e9)):
        m = (r >= lo) & (r < hi)
        if not bool(m.any()):
            continue
        dF = (F_plug[m] - F_exact[m]).abs() / F_exact[m].clamp(min=1e-30)
        dS = (sF_plug[m] - sF_exact[m]).abs() / sF_exact[m].clamp(min=1e-30)
        lab = f"{lo:.1f}-{hi:.1f}" if hi < 1e8 else f">{lo:.1f}"
        print(f"{lab:>12} {int(m.sum()):>5} {float(dF.median()):>22.4f} "
              f"{float(dS.median()):>25.4f}")

    print(f"\n{'mult':>6} {'n':>5} {'median r':>10} {'median dF/F':>13} "
          f"{'median dsigF/sigF':>19}")
    for mm in sorted(int(x) for x in mult.unique()):
        m = mult == mm
        dF = (F_plug[m] - F_exact[m]).abs() / F_exact[m].clamp(min=1e-30)
        dS = (sF_plug[m] - sF_exact[m]).abs() / sF_exact[m].clamp(min=1e-30)
        print(f"{mm:>6} {int(m.sum()):>5} {float(r[m].median()):>10.3f} "
              f"{float(dF.median()):>13.4f} {float(dS.median()):>19.4f}")


if __name__ == "__main__":
    main()
