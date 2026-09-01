"""Does the Gaussian likelihood's bg-variance term remove the weak-end bias?

Ground truth known, background ESTIMATED as in reality.  Same problem, same
merge, two likelihoods:

  poisson  -- lam = S + bg, background taken as an exact constant
  gaussian -- sigma^2 = mu_obs + (n_peak/n_bg)*bg, i.e. Poisson variance plus
              the background estimate's own variance

If the missing variance term is the cause, the second should recover I0_true
where the first inflates it ~2.4x.
"""
from __future__ import annotations

import numpy as np
import torch

from oracle import REAL_LIKE, make_problem
from score import production


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(REAL_LIKE); cfg["mults"] = (865,)
    prob = make_problem(seed=0, device=dev, n_per_mult=int(__import__('os').environ.get('NREFL', 48)), bg_noise=True,
                        bg_per_reflection=True, **cfg)
    It = prob.I0_true.cpu().numpy()
    print(f"{prob.n_hkl} reflections x 865 obs, background ESTIMATED\n")
    print(f"{'likelihood':<12} {'med est/true':>13} {'med est-true':>13} "
          f"{'weak band est':>14} {'weak band true':>15}")
    lo, hi = np.quantile(It, [0.30, 0.75])
    band = (It >= lo) & (It < hi)
    exact = make_problem(seed=0, device=dev, n_per_mult=prob.n_hkl,
                         bg_noise=False, bg_per_reflection=True, **cfg)
    for lab, pr, lik in (("poisson", prob, "poisson"),
                         ("gaussian", prob, "gaussian"),
                         ("exact-bg ref", exact, "poisson")):
        out, _, _ = production(pr, likelihood=lik, grad=False)
        Ie = out.I0_mean.detach().cpu().numpy()
        tt = pr.I0_true.cpu().numpy()
        bd = (tt >= lo) & (tt < hi)
        print(f"{lab:<14} {np.median(Ie / np.maximum(tt, 1e-9)):>13.4f} "
              f"{np.median(Ie - tt):>13.3f} {np.median(Ie[bd]):>14.3f} "
              f"{np.median(tt[bd]):>15.3f}")


if __name__ == "__main__":
    main()
