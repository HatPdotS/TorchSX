#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Plot fixed-grid vs obs-centered NLL landscapes for two HKLs.

Simple visual: where does each NLL curve have its minimum vs truth/median/IRLS?
"""

from __future__ import annotations
import math, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sample_gengauss(alpha, beta, size, gen=None):
    g = torch._standard_gamma(torch.full((size,), 1.0 / beta), generator=gen)
    signs = 2.0 * torch.bernoulli(torch.full((size,), 0.5), generator=gen) - 1.0
    return signs * alpha * g.pow(1.0 / beta)


def _quad_range(beta, coverage=0.999):
    a = torch.tensor(1.0 / beta)
    lo, hi = 1.0, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(torch.special.gammainc(a, torch.tensor(mid**beta))) < coverage:
            lo = mid
        else:
            hi = mid
    return min(hi, 15.0)


def _log_gg_norm(beta):
    return math.log(beta) - math.log(2) - float(torch.lgamma(torch.tensor(1.0 / beta)))


def nll_fixed(I_obs, alpha, beta, bg, I0_grid, n_quad=35):
    R = _quad_range(beta)
    K = max(n_quad, round(n_quad * R / 3.0))
    eps = torch.linspace(-R, R, K)
    d = float(eps[1] - eps[0])
    log_w = -(eps.abs() / alpha).pow(beta) + _log_gg_norm(beta) + math.log(d)
    n_ph = (I_obs + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)
    nll = torch.zeros(len(I0_grid))
    for g in range(len(I0_grid)):
        lam = (float(I0_grid[g]) * torch.exp(alpha * eps) + bg).clamp(min=1e-6)
        lp = n_ph[None, :] * torch.log(lam[:, None]) - lam[:, None] - log_nf[None, :]
        nll[g] = -torch.logsumexp(lp + log_w[:, None], dim=0).sum()
    return nll


def nll_obs_centered(I_obs, alpha, beta, bg, I0_grid, n_per_obs=31):
    n_ph = (I_obs + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)
    log_norm = _log_gg_norm(beta)
    sigma = n_ph.clamp(min=1).sqrt()
    lam_center = torch.maximum(n_ph, torch.full_like(n_ph, bg + 0.1))
    lam_lo = (lam_center - 4 * sigma).clamp(min=bg + 0.1)
    lam_hi = lam_center + 4 * sigma
    t = torch.linspace(0, 1, n_per_obs)
    lam = lam_lo[:, None] + (lam_hi - lam_lo)[:, None] * t[None, :]
    d_lam = (lam_hi - lam_lo) / (n_per_obs - 1)
    signal = lam - bg
    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_jac = -torch.log(signal)
    log_pj = log_poi + log_jac + torch.log(d_lam)[:, None]

    nll = torch.zeros(len(I0_grid))
    for g in range(len(I0_grid)):
        I0 = I0_grid[g]
        eps = torch.log(signal / I0)
        log_gg = -(eps.abs() / alpha).pow(beta) + log_norm
        nll[g] = -torch.logsumexp(log_gg + log_pj, dim=1).sum()
    return nll


def irls(I_obs, beta, n_irls=3):
    pos = I_obs[I_obs > 0]
    if len(pos) < 3:
        return float(I_obs.median().clamp(min=0.1))
    lp = torch.log(pos)
    mu = lp.median()
    for _ in range(n_irls):
        r = (lp - mu).abs().clamp(min=1e-8)
        w = r.pow(beta - 2); w = w / w.sum()
        mu = (w * lp).sum()
    return float(torch.exp(mu))


def main():
    alpha = 1.2
    beta = 1.3
    bg = 10.0
    n_obs = 100

    cases = [
        (10.0, 42, "I₀=10, seed A"),
        (10.0, 777, "I₀=10, seed B"),
        (3.0, 42, "I₀=3, seed A"),
        (3.0, 99, "I₀=3, seed B"),
    ]

    I0_grid = torch.logspace(-0.5, 3.0, 400)
    I0_np = I0_grid.numpy()

    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 18))

    for row, (I0_true, seed, label) in enumerate(cases):
        gen = torch.Generator().manual_seed(seed)
        eps = sample_gengauss(alpha, beta, n_obs, gen=gen)
        sig = (I0_true * torch.exp(eps)).clamp(min=0)
        I_obs = torch.poisson(sig + bg, generator=gen) - bg

        med = float(I_obs.median())
        irls_val = irls(I_obs, beta)
        n_neg = int((I_obs <= 0).sum())

        nll_f = nll_fixed(I_obs, alpha, beta, bg, I0_grid)
        nll_o = nll_obs_centered(I_obs, alpha, beta, bg, I0_grid)

        mle_f = float(I0_grid[nll_f.argmin()])
        mle_o = float(I0_grid[nll_o.argmin()])

        print(f"{label}: true={I0_true}, med={med:.1f}, irls={irls_val:.1f}, "
              f"mle_fix={mle_f:.1f}, mle_oc={mle_o:.1f}, neg={n_neg}")

        for col, (nll, title_extra) in enumerate([
            (nll_f, "Fixed-grid quadrature"),
            (nll_o, "Obs-centered quadrature"),
        ]):
            ax = axes[row, col]
            nll_r = (nll - nll.min()).numpy()
            ax.plot(I0_np, nll_r, "k-", lw=2)
            ax.axvline(I0_true, color="red", ls="--", lw=2, label=f"True={I0_true}")
            ax.axvline(med, color="tomato", ls="-", lw=1.5, label=f"Median={med:.1f}")
            ax.axvline(irls_val, color="seagreen", ls="-", lw=1.5, label=f"IRLS={irls_val:.1f}")
            mle_val = mle_f if col == 0 else mle_o
            ax.axvline(mle_val, color="darkorchid", ls="-", lw=1.5, label=f"MLE={mle_val:.1f}")
            ax.set_xscale("log")
            ax.set_xlabel("I₀")
            ax.set_ylabel("NLL − min")
            ax.set_ylim(0, min(float(nll_r.max()), 100))
            ax.set_title(f"{label} — {title_extra}\n({n_neg} neg obs)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2, which="both")

    fig.suptitle(f"NLL landscape: fixed-grid vs obs-centered  (α={alpha}, β={beta}, bg={bg}, n={n_obs})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/nll_compare.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
