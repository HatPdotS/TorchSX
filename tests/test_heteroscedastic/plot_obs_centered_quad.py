#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare fixed-grid vs observation-centered quadrature (λ-space sampling).

For each observation j with n_j photons:
  - Place 31 λ points spanning the Poisson peak: λ ∈ [n_j - 4√n, n_j + 4√n], clipped to λ > bg
  - Convert to ε: ε = log((λ - bg) / I0)
  - Include Jacobian: dε/dλ = 1/(λ - bg)
  - Evaluate GenGauss(ε) · Poisson(n; λ) · |dε/dλ| · Δλ at each point
  - logsumexp → log p(n_j | I0)
"""

from __future__ import annotations

import math
import torch
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


# ═══════════════════════════════════════════════════════════
# Fixed-grid quadrature (current approach)
# ═══════════════════════════════════════════════════════════

def nll_fixed_grid(I_obs, alpha, beta, bg, I0_grid, n_quad=35):
    R = _quad_range(beta)
    K = max(n_quad, round(n_quad * R / 3.0))
    eps = torch.linspace(-R, R, K)
    d_eps = float(eps[1] - eps[0])
    log_w = -(eps.abs() / alpha).pow(beta) + _log_gg_norm(beta) + math.log(d_eps)

    n_ph = (I_obs + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)

    nll = torch.zeros(len(I0_grid))
    for g in range(len(I0_grid)):
        lam = (float(I0_grid[g]) * torch.exp(alpha * eps) + bg).clamp(min=1e-6)
        log_poi = n_ph[None, :] * torch.log(lam[:, None]) \
                  - lam[:, None] - log_nf[None, :]
        log_joint = log_poi + log_w[:, None]
        nll[g] = -torch.logsumexp(log_joint, dim=0).sum()
    return nll


# ═══════════════════════════════════════════════════════════
# Observation-centered quadrature in λ-space
# ═══════════════════════════════════════════════════════════

def nll_obs_centered(I_obs, alpha, beta, bg, I0_grid, n_per_obs=31):
    """Per-observation quadrature sampling the Poisson peak in λ-space.

    For each observation j:
      1. n_j = I_obs_j + bg (photon count)
      2. Poisson(n_j; λ) peaked at λ = n_j, width σ = √n_j
      3. Sample 31 λ values in [max(bg+δ, n_j - 4σ), n_j + 4σ]
      4. Convert to ε = log((λ - bg) / I0) with Jacobian 1/(λ - bg)
    """
    N = len(I_obs)
    n_ph = (I_obs + bg).clamp(min=0)       # (N,) photon counts
    log_nf = torch.lgamma(n_ph + 1)        # (N,)
    log_norm = _log_gg_norm(beta)

    sigma = n_ph.clamp(min=1).sqrt()        # (N,) Poisson width
    lam_center = torch.maximum(n_ph, torch.full_like(n_ph, bg + 0.1))
    lam_lo = (lam_center - 4 * sigma).clamp(min=bg + 0.1)
    lam_hi = lam_center + 4 * sigma

    # (N, K) λ grid — per-observation
    t = torch.linspace(0, 1, n_per_obs)     # (K,)
    lam = lam_lo[:, None] + (lam_hi - lam_lo)[:, None] * t[None, :]  # (N, K)
    d_lam = (lam_hi - lam_lo) / (n_per_obs - 1)  # (N,) spacing

    signal = lam - bg                        # (N, K) always > 0 since lam > bg + 0.1

    nll = torch.zeros(len(I0_grid))
    for g in range(len(I0_grid)):
        I0 = I0_grid[g]
        eps = torch.log(signal / I0)        # (N, K)

        log_gg = -(eps.abs() / alpha).pow(beta) + log_norm
        log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
        log_jac = -torch.log(signal)         # Jacobian dε/dλ = 1/(λ-bg)

        log_integrand = log_gg + log_poi + log_jac + torch.log(d_lam)[:, None]
        log_p = torch.logsumexp(log_integrand, dim=1)  # (N,)
        nll[g] = -log_p.sum()

    return nll


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

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

    I0_grid = torch.logspace(-0.5, 3.5, 300)
    I0_np = I0_grid.numpy()

    fig, axes = plt.subplots(len(cases), 3, figsize=(18, 20))

    for row, (I0_true, seed, label) in enumerate(cases):
        gen = torch.Generator().manual_seed(seed)
        eps = sample_gengauss(alpha, beta, n_obs, gen=gen)
        signal = (I0_true * torch.exp(eps)).clamp(min=0)
        I_obs = torch.poisson(signal + bg, generator=gen) - bg

        med = float(I_obs.median())
        n_neg = int((I_obs <= 0).sum())
        print(f"{label}: med={med:.1f}, neg={n_neg}, "
              f"range=[{I_obs.min():.0f}, {I_obs.max():.0f}]")

        # Fixed-grid NLL
        print("  Fixed grid ...", end=" ", flush=True)
        nll_fix = nll_fixed_grid(I_obs, alpha, beta, bg, I0_grid)
        mle_fix = float(I0_grid[nll_fix.argmin()])
        print(f"MLE={mle_fix:.1f}")

        # Obs-centered NLL
        print("  Obs-centered ...", end=" ", flush=True)
        nll_oc = nll_obs_centered(I_obs, alpha, beta, bg, I0_grid)
        mle_oc = float(I0_grid[nll_oc.argmin()])
        print(f"MLE={mle_oc:.1f}")

        # (a) Histogram
        ax = axes[row, 0]
        obs_np = I_obs.numpy()
        hi_clip = max(200, I0_true * 10)
        bins = np.linspace(min(-bg, obs_np.min()), min(obs_np.max(), hi_clip), 50)
        ax.hist(obs_np[obs_np <= hi_clip], bins=bins, color="steelblue",
                alpha=0.7, edgecolor="white")
        n_clip = int((obs_np > hi_clip).sum())
        ax.axvline(I0_true, color="red", ls="--", lw=2, label=f"True={I0_true}")
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("I_obs")
        ax.set_title(f"{label} ({n_neg} neg, {n_clip} clipped)")
        ax.legend(fontsize=7)

        # (b) NLL landscapes overlaid
        ax = axes[row, 1]
        nll_fix_r = (nll_fix - nll_fix.min()).numpy()
        nll_oc_r = (nll_oc - nll_oc.min()).numpy()
        ax.plot(I0_np, nll_fix_r, "k-", lw=1.5, label=f"Fixed grid MLE={mle_fix:.1f}")
        ax.plot(I0_np, nll_oc_r, "darkorchid", lw=1.5, ls="--",
                label=f"Obs-centered MLE={mle_oc:.1f}")
        ax.axvline(I0_true, color="red", ls="--", lw=2, alpha=0.7, label=f"True={I0_true}")
        ax.axvline(med, color="seagreen", ls="-", lw=1.5, alpha=0.7, label=f"Median={med:.1f}")
        ax.set_xscale("log")
        ax.set_ylabel("NLL − min")
        ax.set_ylim(0, min(max(nll_fix_r.max(), nll_oc_r.max()), 150))
        ax.set_xlabel("I₀")
        ax.set_title("NLL landscape comparison")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, which="both")

        # (c) Visualise quadrature points for one observation (the largest)
        ax = axes[row, 2]
        j_max = int(I_obs.argmax())
        n_j = float((I_obs[j_max] + bg).clamp(min=0))
        sig_j = max(math.sqrt(max(n_j, 1)), 1)
        lam_c = max(n_j, bg + 0.1)
        lam_lo_j = max(bg + 0.01, lam_c - 4 * sig_j)
        lam_hi_j = lam_c + 4 * sig_j
        lam_pts = torch.linspace(lam_lo_j, lam_hi_j, 31)

        # Poisson PMF at these λ values
        from scipy.stats import poisson as poisson_dist
        lam_np = lam_pts.numpy()
        pmf = poisson_dist.pmf(int(n_j), lam_np)

        # Also show GenGauss weight at these ε (at true I0)
        sig_pts = (lam_pts - bg).clamp(min=1e-8)
        eps_pts = torch.log(sig_pts / I0_true)
        gg_w = torch.exp(-(eps_pts.abs() / alpha).pow(beta))

        ax2 = ax.twinx()
        ax.plot(lam_np, pmf, "steelblue", lw=2, label=f"Poisson(n={int(n_j)}; λ)")
        ax.axvline(float(I_obs[j_max]) + bg, color="red", ls="--", lw=1,
                   label=f"Observed λ={n_j:.0f}")
        ax.scatter(lam_np, pmf, c="steelblue", s=20, zorder=5)
        ax2.plot(lam_np, gg_w.numpy(), "seagreen", lw=2, ls="--",
                 label="GenGauss weight")
        ax.set_xlabel("λ (predicted photons)")
        ax.set_ylabel("Poisson PMF", color="steelblue")
        ax2.set_ylabel("GenGauss weight", color="seagreen")
        ax.set_title(f"Largest obs: I_obs={float(I_obs[j_max]):.0f}, "
                     f"n={n_j:.0f} photons\n31 quad points in λ-space")
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Fixed-grid vs Obs-centered (λ-space) quadrature  "
        f"(α={alpha}, β={beta}, bg={bg}, n={n_obs})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/obs_centered_quad.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
