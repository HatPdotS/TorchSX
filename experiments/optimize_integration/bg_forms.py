"""Which functional form actually approximates the background-marginalised Poisson?

The observed counts are n ~ Poisson(S + B) with B the peak-integrated background.
B is not known: CrystFEL estimates it from an n_bg = 128 px annulus, so the honest
posterior for B is Gamma(shape = A+1, scale = n_peak/n_bg) with A = annulus total.

The exact marginal is therefore

    p(n | S) = int_0^inf Poisson(n | S + B) Gamma(B; A+1, n_peak/n_bg) dB

which this script computes in float64 by direct quadrature.  Everything else is a
candidate cheap approximation to it.  We care about the SHAPE in S, not the
absolute level, so each form is compared after aligning at its own maximum, and
by the estimator it induces (posterior mean of S under a flat prior).
"""
from __future__ import annotations

import math
import numpy as np
import torch

F = 149.0 / 128.0          # n_peak / n_bg
N_BG, N_PEAK = 128.0, 149.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64


# ── exact reference ────────────────────────────────────────────────────
def exact_marginal(n, bg, S, nB=4001):
    """log int Pois(n | S+B) Gamma(B; A+1, n_peak/n_bg) dB, float64.

    n, bg: (M,)   S: (K,)   ->  (M, K)
    """
    A = N_BG * bg / N_PEAK                       # annulus total counts
    shape = A + 1.0
    scale = N_PEAK / N_BG
    mean, sd = shape * scale, torch.sqrt(shape) * scale
    lo = torch.clamp(mean - 10 * sd, min=1e-8)
    hi = mean + 12 * sd
    # log-spaced in B so the b^(k-1) end is handled for small shape too
    u = torch.linspace(0, 1, nB, device=n.device, dtype=DT)
    logB = torch.log(lo)[:, None] + u[None, :] * (torch.log(hi / lo))[:, None]
    B = torch.exp(logB)                                            # (M, nB)
    dlogB = (torch.log(hi / lo) / (nB - 1))[:, None]
    # log Gamma pdf + log B  (the dlogB Jacobian)
    logpB = ((shape - 1)[:, None] * logB - B / scale
             - torch.lgamma(shape)[:, None] - shape[:, None] * math.log(scale)
             + logB)
    out = torch.empty(n.shape[0], S.shape[0], device=n.device, dtype=DT)
    logn_f = torch.lgamma(n + 1)
    for j in range(S.shape[0]):                                    # memory, not speed
        lam = S[j] + B
        lp = n[:, None] * torch.log(lam) - lam - logn_f[:, None]
        w = lp + logpB
        m = w.max(dim=1, keepdim=True).values
        out[:, j] = (m.squeeze(1)
                     + torch.log((torch.exp(w - m) * dlogB).sum(dim=1)))
    return out


# ── candidate forms, all (M, K) ────────────────────────────────────────
def f_poisson(n, bg, S, s2):
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    return n[:, None] * torch.log(lam) - lam - torch.lgamma(n + 1)[:, None]


def _o2(n, bg, S, s2, clamp):
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    base = n[:, None] * torch.log(lam) - lam - torch.lgamma(n + 1)[:, None]
    inv = 1.0 / lam
    r = n[:, None] * inv
    corr = 0.5 * s2[:, None] * ((r - 1.0) ** 2 - r * inv)
    if clamp is not None:
        corr = corr.clamp(-clamp, clamp)
    return base + corr


def f_o2(n, bg, S, s2):
    return _o2(n, bg, S, s2, 10.0)


def f_o2_unclamped(n, bg, S, s2):
    return _o2(n, bg, S, s2, None)


def f_dev(n, bg, S, s2):
    """Saddlepoint / quasi-Poisson: scale the Poisson DEVIANCE by phi = 1+s2/lam.

    Reduces to the exact Poisson at s2 = 0, keeps the mode at lam = n, matches
    var = lam + s2 to second order, and is bounded with no clamp.
    """
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    nn = n.clamp(min=1e-12)[:, None]
    half_D = nn * torch.log(nn / lam) - (nn - lam)          # = D/2 >= 0
    phi = 1.0 + s2[:, None] / lam
    logpeak = nn * torch.log(nn) - nn - torch.lgamma(n + 1)[:, None]
    return -half_D / phi - 0.5 * torch.log(phi) + logpeak


def f_nb(n, bg, S, s2):
    """Negative binomial matched to mean lam, variance lam + s2."""
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    s2c = s2[:, None].clamp(min=1e-12)
    r = lam * lam / s2c
    return (torch.lgamma(n[:, None] + r) - torch.lgamma(r)
            - torch.lgamma(n + 1)[:, None]
            + r * torch.log(r / (r + lam)) + n[:, None] * torch.log(lam / (r + lam)))


def f_gauss(n, bg, S, s2):
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    v = lam + s2[:, None]
    return -0.5 * (n[:, None] - lam) ** 2 / v - 0.5 * torch.log(2 * math.pi * v)


FORMS = {"poisson": f_poisson, "o2(clamp10)": f_o2, "o2(unclamped)": f_o2_unclamped,
         "deviance": f_dev, "negbin": f_nb, "gaussian": f_gauss}


def f_dev_phi0(n, bg, S, s2):
    """Deviance scaled by a CONSTANT dispersion phi0 = 1 + s2/n (evaluated at the
    peak rather than at each grid point) -- the textbook quasi-likelihood."""
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    nn = n.clamp(min=1e-12)[:, None]
    half_D = nn * torch.log(nn / lam) - (nn - lam)
    phi = 1.0 + s2[:, None] / (n.clamp(min=1.0)[:, None])
    logpeak = nn * torch.log(nn) - nn - torch.lgamma(n + 1)[:, None]
    return -half_D / phi - 0.5 * torch.log(phi) + logpeak


def f_nb_saddle(n, bg, S, s2):
    """Saddlepoint form of the moment-matched negative binomial: same deviance
    the NB has, but without either lgamma."""
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    nn = n.clamp(min=1e-12)[:, None]
    r = lam * lam / s2[:, None].clamp(min=1e-12)
    half_D = nn * torch.log(nn / lam) - (nn + r) * torch.log((nn + r) / (lam + r))
    v = lam + lam * lam / r          # = lam + s2
    return -half_D - 0.5 * torch.log(2 * math.pi * nn * (1.0 + nn / r))


FORMS["dev(phi0)"] = f_dev_phi0
FORMS["nb-saddle"] = f_nb_saddle



# ── summaries ──────────────────────────────────────────────────────────
def summarise(lp, S, wS):
    """argmax S, posterior mean/sd of S under a flat prior."""
    k = lp.argmax(dim=1)
    smax = S[k]
    m = lp.max(dim=1, keepdim=True).values
    w = torch.exp(lp - m) * wS[None, :]
    w = w / w.sum(dim=1, keepdim=True)
    mu = (w * S[None, :]).sum(dim=1)
    sd = torch.sqrt(((w * (S[None, :] - mu[:, None]) ** 2).sum(dim=1)).clamp(min=0))
    return smax, mu, sd


def main():
    torch.manual_seed(0)
    import os
    if os.environ.get("SYNTH"):
        # deliberately wider than the real batch: low counts are where a
        # Gaussian or a saddlepoint form is most likely to break
        g = torch.Generator(device="cpu").manual_seed(1)
        bg = (10 ** (torch.rand(3000, generator=g) * 2.7)).to(DT)     # 1 .. 500
        n = torch.poisson(bg * (1 + torch.rand(3000, generator=g)),
                          generator=g).to(DT)
        n, bg = n.to(DEV), bg.to(DEV)
        print("SYNTHETIC sweep: bg 1..500 log-uniform")
    else:
        b = torch.load("batch.pt", map_location="cpu", weights_only=False)
        n_all = b["n_ph"].to(DT)
        bg_all = b["bg"].to(DT)
        idx = torch.randperm(n_all.numel())[:3000]
        n, bg = n_all[idx].to(DEV), bg_all[idx].clamp(min=1.0).to(DEV)
    s2 = F * bg

    # every form must collapse to the plain Poisson at sigma_b^2 = 0
    Schk = torch.linspace(0.0, 400.0, 257, device=DEV, dtype=DT)
    z = torch.zeros_like(bg)
    ref0 = f_poisson(n, bg, Schk, z)
    bad = [k for k, fn in FORMS.items()
           if k not in ("gaussian", "negbin", "nb-saddle")
           and float((fn(n, bg, Schk, z) - ref0).abs().max()) > 1e-8]
    print(f"reduce-to-Poisson at sigma_b^2=0: "
          f"{'OK' if not bad else 'FAILS for ' + ', '.join(bad)}")

    print(f"{n.numel()} real observations   n_ph med {n.median():.1f}  "
          f"bg med {bg.median():.1f}  sigma_b med {s2.median().sqrt():.2f}")

    # S grid: linear, wide enough to hold the whole likelihood for every obs
    top = float((n + 8 * torch.sqrt(n + s2)).max())
    K = 3001
    S = torch.linspace(0.0, top, K, device=DEV, dtype=DT)
    wS = torch.full((K,), top / (K - 1), device=DEV, dtype=DT)
    wS[0] *= 0.5; wS[-1] *= 0.5

    ex = exact_marginal(n, bg, S)
    ex_max, ex_mu, ex_sd = summarise(ex, S, wS)
    # "signal" the exact marginal implies, and the naive n-bg
    naive = (n - bg)

    print(f"\n{'form':<15} {'mode err':>10} {'E[S] err':>10} {'sd/exact':>9} "
          f"{'max|dlogp|':>11} {'med|dlogp|':>11}")
    print(f"{'':<15} {'(photons)':>10} {'(photons)':>10} {'':>9} "
          f"{'nats, within 15 of peak':>23}")
    rows = {}
    for name, fn in FORMS.items():
        lp = fn(n, bg, S, s2)
        sm, mu, sd = summarise(lp, S, wS)
        # align at each form's own max and compare shapes where mass lives
        d = (lp - lp.max(dim=1, keepdim=True).values) - (ex - ex.max(dim=1, keepdim=True).values)
        live = ex >= (ex.max(dim=1, keepdim=True).values - 15.0)
        d = torch.where(live, d.abs(), torch.zeros_like(d))
        rows[name] = (sm, mu, sd)
        print(f"{name:<15} {float((sm-ex_max).median()):>10.2f} "
              f"{float((mu-ex_mu).median()):>10.2f} "
              f"{float((sd/ex_sd).median()):>9.3f} "
              f"{float(d.max()):>11.2f} {float(d[live].median()):>11.4f}")

    print(f"\nreference: exact marginal   med mode {float(ex_max.median()):.2f}   "
          f"med E[S] {float(ex_mu.median()):.2f}   med sd {float(ex_sd.median()):.2f}")
    print(f"           plain n-bg        med {float(naive.median()):.2f}")

    # where does the clamp bite?
    lam = (S[None, :] + bg[:, None]).clamp(min=1e-6)
    inv = 1.0 / lam
    r = n[:, None] * inv
    corr = 0.5 * s2[:, None] * ((r - 1.0) ** 2 - r * inv)
    live = ex >= (ex.max(dim=1, keepdim=True).values - 15.0)
    print(f"\nclamp activity on grid points carrying mass (within 15 nats of peak):")
    print(f"  frac |corr| > 10 : {float((corr.abs() > 10)[live].to(DT).mean()):.4f}")
    print(f"  frac corr  > 2   : {float((corr > 2)[live].to(DT).mean()):.4f}")
    print(f"  median corr      : {float(corr[live].median()):.3f}")
    print(f"  max corr         : {float(corr[live].max()):.1f}")

    # weak subset: this is where the compression lives
    weak = naive < torch.quantile(naive, 0.25)
    print(f"\nweakest quartile (n-bg < {float(torch.quantile(naive,0.25)):.1f}), "
          f"n = {int(weak.sum())}")
    print(f"{'form':<15} {'med mode':>10} {'med E[S]':>10} {'med sd':>9}")
    print(f"{'exact':<15} {float(ex_max[weak].median()):>10.2f} "
          f"{float(ex_mu[weak].median()):>10.2f} {float(ex_sd[weak].median()):>9.2f}")
    for name, (sm, mu, sd) in rows.items():
        print(f"{name:<15} {float(sm[weak].median()):>10.2f} "
              f"{float(mu[weak].median()):>10.2f} {float(sd[weak].median()):>9.2f}")


if __name__ == "__main__":
    main()
