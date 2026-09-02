"""Where does the residual bias of the deviance form come from?

Ground-truth recovery said: plain Poisson 3.05x too high, the clamped
second-order form 0.82x (overcorrected), deviance 1.20x, and a Gaussian with
sigma^2 = n_obs + (n_peak/n_bg)*bg essentially exact (1.03 against a 1.02
ceiling).  Yet pointwise against the exact marginal the deviance form is the
better approximation.  Two candidate explanations:

  width   -- the deviance form is still slightly too narrow, so scale sigma_b^2
  shape   -- what matters is that the Gaussian's variance does NOT depend on the
             grid point, so its penalty is symmetric in (n - lam)

The factor sweep separates them: if it is width, some factor lands on 1.0 and
the constant-dispersion variant behaves like the lambda-dependent one at the
same width.  If it is shape, no factor reaches the Gaussian's accuracy.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from oracle import REAL_LIKE, make_problem
from score import production
from torchsx.merging.likelihood import Poisson, GaussianPolynomialVariance

F = 149.0 / 128.0


class PoissonVariant(Poisson):
    """Research-only shapes, eager path.  `mode`:

      dev_phi0  -- deviance scaled by a CONSTANT phi = 1 + sigma_b^2/n
      nb_saddle -- saddlepoint of the negative binomial matched to
                   (mean lam, var lam + sigma_b^2); no lgamma
    """

    def __init__(self, bg_var_factor: float, mode: str) -> None:
        super().__init__(bg_var_factor=bg_var_factor, bg_var_mode="deviance")
        self.variant = mode

    def log_prob_given_signal(self, n: Tensor, S: Tensor, bg: Tensor) -> Tensor:
        lam = (S + bg[:, None]).clamp(min=1e-6)
        s2 = (self.bg_var_factor * bg.clamp(min=0.0))[:, None]
        nn = n.clamp(min=1e-12)[:, None]
        log_nf = torch.lgamma(n + 1)[:, None]
        if self.variant == "dev_phi0":
            neg_half_D = nn * torch.log(lam / nn) - (lam - nn)
            inv_phi = 1.0 / (1.0 + s2 / n.clamp(min=1.0)[:, None])
            logpeak = nn * torch.log(nn) - nn - log_nf
            return neg_half_D * inv_phi + 0.5 * torch.log(inv_phi) + logpeak
        if self.variant == "negbin":
            # Moment-matched negative binomial (mean lam, var lam + s2),
            # written as a CORRECTION to the exact Poisson so it reduces to it
            # at s2 = 0 and so the O(1e9) terms of the direct form never appear.
            # With p = n s2/lam^2, q = s2/lam and g(x) = log(1+x) - x, the
            # cancellation is exact because (lam^2/s2)(p - q) = n - lam.
            inv_lam = 1.0 / lam
            q = s2 * inv_lam
            pp = nn * s2 * inv_lam * inv_lam

            def g(x):                      # log(1+x) - x, series for small x
                c = [-0.5, 1 / 3, -0.25, 0.2, -1 / 6, 1 / 7, -0.125, 1 / 9]
                acc = torch.full_like(x, c[-1])
                for a in reversed(c[:-1]):
                    acc = acc * x + a
                return torch.where(x.abs() < 0.5, x * x * acc,
                                   torch.log1p(x) - x)

            delta = ((lam * lam / s2.clamp(min=1e-12)) * (g(pp) - g(q))
                     + nn * (torch.log1p(pp) - torch.log1p(q))
                     - 0.5 * torch.log1p(pp))
            neg_half_D = nn * torch.log(lam / nn) - (lam - nn)
            n64 = n.double()
            logpeak = torch.where(
                n > 0,
                (n64 * n64.clamp(min=1e-300).log() - n64
                 - torch.lgamma(n + 1).double()).to(n.dtype),
                torch.zeros_like(n))[:, None]
            return neg_half_D + logpeak + delta
        raise ValueError(self.variant)


def slope(est, true, mask):
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
    lo, hi = np.quantile(tt, [0.30, 0.75])
    band = (tt >= lo) & (tt < hi)
    weak = tt < np.quantile(tt, 0.50)
    print(f"{prob.n_hkl} reflections x 865 obs   weak band true "
          f"{np.median(tt[band]):.3f}")

    runs = [("exact-bg ref", exact, Poisson(), False),
            ("poisson", prob, Poisson(), False)]
    for m in (0.5, 1.0, 1.5, 2.0, 3.0):
        runs.append((f"deviance x{m:g}", prob,
                     Poisson(bg_var_factor=m * F, bg_var_mode="deviance"),
                     False))
    runs += [
        ("dev_phi0", prob, PoissonVariant(F, "dev_phi0"), True),
        ("negbin", prob, PoissonVariant(F, "negbin"), True),
        ("negbin x1.5", prob, PoissonVariant(1.5 * F, "negbin"), True),
        ("gaussian", prob, GaussianPolynomialVariance(
            var_const=0.0, var_linear=1.0, var_quadratic=0.0,
            bg_var_factor=F), False),
    ]

    # edge_mass is the outer grid's truncation sentinel.  The first real-data
    # deviance merge clipped 27.7 % of reflections against the plain Poisson's
    # 0.085 %, so report it here too: if the simulation does not reproduce that,
    # the cause is something the simulation lacks, not the likelihood.
    print(f"\n{'variant':<16} {'med est/true':>12} {'weak band':>10} "
          f"{'slope weak':>11} {'rms log':>8} {'edge>1%':>8} {'med edge':>9}")
    for lab, pr, lik, eager in runs:
        out, _, _ = production(pr, likelihood=lik, grad=False, eager=eager,
                               k_coarse=31, inner_points=1355)
        Ie = out.I0_mean.detach().cpu().numpy()
        t = pr.I0_true.cpu().numpy()
        r = np.log(np.clip(Ie, 1e-6, None)) - np.log(np.clip(t, 1e-6, None))
        em = out.edge_mass.detach().cpu().numpy()
        print(f"{lab:<16} {np.median(Ie / np.maximum(t, 1e-9)):>12.4f} "
              f"{np.median(Ie[band]):>10.3f} {slope(Ie, t, weak):>11.3f} "
              f"{np.sqrt(np.mean(r[t > 0.05] ** 2)):>8.3f} "
              f"{100 * float((em > 0.01).mean()):>7.2f}% {np.median(em):>9.2e}")


if __name__ == "__main__":
    main()
