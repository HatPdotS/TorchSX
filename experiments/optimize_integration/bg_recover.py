"""Ground-truth recovery: does each background treatment return I0_true?

The problem is generated from the model, so I0_true is known exactly, and the
background is ESTIMATED exactly as CrystFEL does it (sigma_b^2 = (n_peak/n_bg)*bg).
`exact-bg ref` regenerates the same problem with a noiseless background and is the
ceiling any treatment could reach.

Reported per variant: the ratio est/true (bias), the weak-band level (the
compression that shows up as `min F / median F` on real data), the log-log slope
over the weak half (1.0 = no compression), and the scatter about truth.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from oracle import REAL_LIKE, make_problem
from score import production
from torchsx.merging.likelihood import Poisson, GaussianPolynomialVariance

F = 149.0 / 128.0


def slope(est, true, mask):
    """log-log slope of est on true over `mask` -- 1.0 means no compression."""
    x = np.log(np.clip(true[mask], 1e-3, None))
    y = np.log(np.clip(est[mask], 1e-3, None))
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(REAL_LIKE)
    cfg["mults"] = (865,)
    nref = int(os.environ.get("NREFL", 384))
    prob = make_problem(seed=0, device=dev, n_per_mult=nref, bg_noise=True,
                        bg_per_reflection=True, **cfg)
    exact = make_problem(seed=0, device=dev, n_per_mult=nref, bg_noise=False,
                         bg_per_reflection=True, **cfg)
    tt = prob.I0_true.cpu().numpy()
    print(f"{prob.n_hkl} reflections x 865 obs, background ESTIMATED   "
          f"n_ph med {float(prob.n_ph.median()):.1f}  bg med {float(prob.bg.median()):.1f}")

    lo, hi = np.quantile(tt, [0.30, 0.75])
    band = (tt >= lo) & (tt < hi)
    weak = tt < np.quantile(tt, 0.50)

    variants = [
        ("exact-bg ref", exact, Poisson()),
        ("poisson", prob, Poisson()),
        ("+o2 (current)", prob, Poisson(bg_var_factor=F,
                                        bg_var_mode="second_order")),
        ("+deviance", prob, Poisson(bg_var_factor=F, bg_var_mode="deviance")),
        ("gaussian+bgvar", prob, GaussianPolynomialVariance(
            var_const=0.0, var_linear=1.0, var_quadratic=0.0, bg_var_factor=F)),
    ]
    print(f"\n{'variant':<16} {'med est/true':>12} {'med est-true':>12} "
          f"{'weak band':>10} {'slope':>7} {'rms log':>8}")
    print(f"{'':<16} {'':>12} {'':>12} {f'(true {np.median(tt[band]):.2f})':>10} "
          f"{'weak':>7} {'ratio':>8}")
    for lab, pr, lik in variants:
        out, _, _ = production(pr, likelihood=lik, grad=False)
        Ie = out.I0_mean.detach().cpu().numpy()
        t = pr.I0_true.cpu().numpy()
        r = np.log(np.clip(Ie, 1e-6, None)) - np.log(np.clip(t, 1e-6, None))
        print(f"{lab:<16} {np.median(Ie / np.maximum(t, 1e-9)):>12.4f} "
              f"{np.median(Ie - t):>12.3f} {np.median(Ie[band]):>10.3f} "
              f"{slope(Ie, t, weak):>7.3f} "
              f"{np.sqrt(np.mean(r[t > 0.05] ** 2)):>8.3f}")


if __name__ == "__main__":
    main()
