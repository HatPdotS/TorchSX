"""Is the bg-dependent excess a background error, or just positivity bias?

On real data, at fixed F_model, our merged intensity exceeds the model's by an
amount rising with background (~10-15 % of bg).  Two explanations:

  (a) the background is under-subtracted, so n - bg is too large;
  (b) the estimator cannot return a negative or zero intensity.  I0 = exp(v) > 0
      strictly, so for a reflection whose true intensity is near zero the
      posterior mean is necessarily positive, and the size of that positive
      bias is set by the likelihood width -- which is set by the noise, which
      goes as sqrt(bg).  So (b) ALSO produces an excess that grows with
      background, with no background error anywhere.

This settles it: generate from the model with KNOWN I0_true and an EXACT
background, integrate the posterior densely with the certified oracle, and bin
the recovered minus true intensity by bg at fixed I0_true.  Any bg-dependence
here is (b), because there is no background error to find.

The comparison is only fair at the real multiplicity: positivity bias shrinks
as the data constrain the reflection, so a simulation at mult 16 would
exaggerate it.  The dark stream averages ~865 observations per reflection.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from oracle import REAL_LIKE, make_problem, oracle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=int, default=865,
                    help="observations per reflection; the real dark stream "
                         "averages 865")
    ap.add_argument("--n-refl", type=int, default=384)
    ap.add_argument("--n-outer", type=int, default=2048)
    ap.add_argument("--bg-noise", action="store_true",
                    help="give the likelihood an ESTIMATED background, "
                         "sigma = sqrt(bg n_peak/n_bg), as the real data has")
    ap.add_argument("--bg-per-reflection", action="store_true",
                    help="background constant within a reflection, as it is on "
                         "a detector -- without this, averaging 865 i.i.d. "
                         "draws makes every reflection's mean bg identical and "
                         "the bg-dependence test has no leverage")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(REAL_LIKE)
    cfg["mults"] = (args.mult,)
    prob = make_problem(seed=0, device=dev, n_per_mult=args.n_refl,
                        bg_noise=args.bg_noise,
                        bg_per_reflection=args.bg_per_reflection, **cfg)
    tagb = ("ESTIMATED (sigma=sqrt(bg n_peak/n_bg))" if args.bg_noise
            else "EXACT by construction")
    print(f"simulated {prob.n_hkl} reflections x {args.mult} obs = "
          f"{prob.M:,} observations; background {tagb}"
          + ("; per-reflection" if args.bg_per_reflection else "; i.i.d. per obs"))
    q = lambda t, p: float(t.double().quantile(torch.tensor(p, dtype=torch.float64,
                                                            device=t.device)))
    print(f"  n_ph  med {q(prob.n_ph,.5):.1f}   bg med {q(prob.bg,.5):.1f}   "
          f"I0_true med {q(prob.I0_true,.5):.2f}")

    o = oracle(prob, n_outer=args.n_outer, need_grads=False)

    bg_mean = torch.zeros(prob.n_hkl, device=dev, dtype=torch.float64)
    cnt = torch.zeros(prob.n_hkl, device=dev, dtype=torch.float64)
    bg_mean.index_add_(0, prob.hkl_id, prob.bg_true)
    cnt.index_add_(0, prob.hkl_id, torch.ones_like(prob.bg))
    bg_mean = (bg_mean / cnt).cpu().numpy()
    It = prob.I0_true.cpu().numpy()
    Ie = o.I0_mean.cpu().numpy()

    print(f"\n  overall: median (est - true) = {np.median(Ie - It):+.3f}   "
          f"median est/true = {np.median(Ie / np.maximum(It, 1e-9)):.4f}")
    ib = np.quantile(It, np.linspace(0, 1, 4))
    print(f"\n{'I0_true band':>20} {'bg bin':>7} {'n':>5} {'med bg':>9} "
          f"{'med I_true':>11} {'med I_est':>10} {'est-true':>10} "
          f"{'(est-true)/bg':>14}")
    for a in range(3):
        band = (It >= ib[a]) & (It < ib[a+1] if a < 2 else It <= ib[a+1])
        if band.sum() < 30:
            continue
        bb, tt, ee = bg_mean[band], It[band], Ie[band]
        e = np.quantile(bb, np.linspace(0, 1, 4))
        idx = np.clip(np.digitize(bb, e) - 1, 0, 2)
        lab = f"[{ib[a]:.2f}, {ib[a+1]:.2f})"
        for b in range(3):
            s = idx == b
            if s.sum() < 10:
                continue
            d = np.median(ee[s] - tt[s])
            print(f"{lab if b == 0 else '':>20} {b:>7} {int(s.sum()):>5} "
                  f"{np.median(bb[s]):>9.1f} {np.median(tt[s]):>11.2f} "
                  f"{np.median(ee[s]):>10.2f} {d:>10.2f} "
                  f"{d/np.median(bb[s]):>14.4f}")
    print("\n  (est-true) rising with bg HERE too -> positivity bias, not a bg error")
    print("  (est-true) flat/zero here          -> the real-data effect is a bg error")


if __name__ == "__main__":
    main()
