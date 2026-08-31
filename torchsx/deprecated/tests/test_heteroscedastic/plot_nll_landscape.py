#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Visualise the Poisson × GenGauss NLL landscape for individual HKLs.

Two HKLs with identical ground truth (I0=50, alpha=1.2, beta=1.3),
each with 100 observations. Plot:
  - The sampled observations (histogram)
  - The NLL(I0) curve from the overlap integral
  - The true I0, median, and MLE locations
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


def compute_nll_curve(I_obs, alpha, beta, bg, I0_grid):
    """Compute Poisson × GenGauss marginal NLL for a single HKL over an I0 grid.

    Returns per-obs log-likelihoods too for diagnostic.
    """
    n_obs = len(I_obs)
    n_grid = len(I0_grid)

    # Quadrature
    R = _quad_range(beta)
    K = max(31, round(31 * R / 3.0))
    eps = torch.linspace(-R, R, K)
    d_eps = float(eps[1] - eps[0])
    log_w = -(eps.abs() ** beta) + math.log(beta) - math.log(2) \
            - float(torch.lgamma(torch.tensor(1.0 / beta))) + math.log(d_eps)
    s = torch.exp(alpha * eps)  # (K,)

    n_ph = (I_obs + bg).clamp(min=0)    # (J,)
    log_nf = torch.lgamma(n_ph + 1)     # (J,)

    nll = torch.zeros(n_grid)
    per_obs_logp = torch.zeros(n_grid, n_obs)  # for diagnostics

    for g in range(n_grid):
        lam = (s * I0_grid[g] + bg).clamp(min=1e-6)     # (K,)
        # log Poisson(n_j; lam_k) for all (k, j)
        log_poi = n_ph[None, :] * torch.log(lam[:, None]) \
                  - lam[:, None] - log_nf[None, :]        # (K, J)
        log_joint = log_poi + log_w[:, None]               # (K, J)
        log_p = torch.logsumexp(log_joint, dim=0)          # (J,)
        per_obs_logp[g] = log_p
        nll[g] = -log_p.sum()

    return nll, per_obs_logp


def main():
    alpha = 1.2
    beta = 1.3
    bg = 10.0
    n_obs = 100

    # Multiple HKLs at different I0 and seeds to show good + bad cases
    cases = [
        (10.0, 42,  "I₀=10, seed A"),
        (10.0, 777, "I₀=10, seed B"),
        (3.0,  42,  "I₀=3, seed A"),
        (3.0,  99,  "I₀=3, seed B"),
    ]

    I0_grid = torch.logspace(-0.5, 3.5, 500)
    I0_np = I0_grid.numpy()

    print(f"alpha={alpha}, beta={beta}, bg={bg}, n_obs={n_obs}")

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    for row, (I0_true, seed, label) in enumerate(cases):
        gen = torch.Generator().manual_seed(seed)
        eps = sample_gengauss(alpha, beta, n_obs, gen=gen)
        signal = (I0_true * torch.exp(eps)).clamp(min=0)
        I_obs = torch.poisson(signal + bg, generator=gen) - bg

        nll, pobs = compute_nll_curve(I_obs, alpha, beta, bg, I0_grid)
        mle = float(I0_grid[nll.argmin()])
        med = float(I_obs.median())
        n_neg = int((I_obs <= 0).sum())
        obs_np = I_obs.numpy()

        print(f"\n{label}: true={I0_true}, median={med:.1f}, MLE={mle:.1f}, "
              f"neg={n_neg}/{n_obs}, range=[{I_obs.min():.0f}, {I_obs.max():.0f}]")

        # (a) Histogram
        ax = axes[row, 0]
        hi_clip = min(obs_np.max(), max(500, I0_true * 20))
        bins = np.linspace(min(-bg, obs_np.min()), hi_clip, 60)
        ax.hist(obs_np[obs_np <= hi_clip], bins=bins, color="steelblue",
                alpha=0.7, edgecolor="white")
        n_clipped = int((obs_np > hi_clip).sum())
        ax.axvline(I0_true, color="red", ls="--", lw=2, label=f"True={I0_true}")
        ax.axvline(med, color="seagreen", ls="-", lw=2, label=f"Med={med:.1f}")
        ax.axvline(mle, color="darkorchid", ls="-", lw=2, label=f"MLE={mle:.1f}")
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("I_obs")
        ax.set_ylabel("Count")
        title = f"{label}: {n_neg} neg"
        if n_clipped:
            title += f", {n_clipped} clipped >{hi_clip:.0f}"
        ax.set_title(title)
        ax.legend(fontsize=7)

        # (b) NLL landscape
        ax = axes[row, 1]
        nll_np = nll.numpy()
        nll_rel = nll_np - nll_np.min()
        ax.plot(I0_np, nll_rel, "k-", lw=1.5)
        ax.axvline(I0_true, color="red", ls="--", lw=2, label=f"True={I0_true}")
        ax.axvline(med, color="seagreen", ls="-", lw=1.5, label=f"Med={med:.1f}")
        ax.axvline(mle, color="darkorchid", ls="-", lw=1.5, label=f"MLE={mle:.1f}")
        ax.set_xscale("log")
        ax.set_xlabel("I₀")
        ax.set_ylabel("NLL − NLL_min")
        ax.set_ylim(0, min(float(nll_rel.max()), 200))
        ax.set_title(f"NLL landscape (MLE/true = {mle/I0_true:.2f})")
        ax.legend(fontsize=7)

        # (c) Per-obs log-likelihood: MLE vs true
        ax = axes[row, 2]
        logp_true = pobs[torch.argmin((I0_grid - I0_true).abs())].numpy()
        logp_mle = pobs[nll.argmin()].numpy()
        delta = logp_mle - logp_true
        order = np.argsort(obs_np)
        colors = np.where(obs_np[order] > 0, "steelblue", "tomato")
        ax.bar(range(n_obs), delta[order], color=colors, alpha=0.7, width=1.0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Observation (sorted by I_obs)")
        ax.set_ylabel("Δlog p  (MLE − true)")
        ax.set_title(f"Per-obs Δlog p (sum={delta.sum():.1f})")
        top5 = np.argsort(np.abs(delta))[-5:]
        for idx in top5:
            rank = np.where(order == idx)[0][0]
            ax.annotate(f"{obs_np[idx]:.0f}", (rank, delta[idx]),
                       fontsize=6, ha="center",
                       va="bottom" if delta[idx] > 0 else "top")
    fig.suptitle(
        f"Poisson × GenGauss NLL landscape  "
        f"(α={alpha}, β={beta}, bg={bg}, n={n_obs})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/nll_landscape.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
