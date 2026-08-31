#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Test estimators in the weak-reflection regime (I0 = 0.1 to 3).

Checks whether signal_min=1.0 breaks for very weak reflections.
"""

from __future__ import annotations
import math, os, sys, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_scatter_bias import (
    sample_gengauss, estimate_median, estimate_irls, _log_gg_norm,
)
from plot_midpoint_comparison import estimate_gg_lbfgs
from plot_loglinear_tune import estimate_poisson_loglinear, _softplus_inv


def generate_weak(*, n_hkl, beta, alpha=1.2, I0_lo=0.1, I0_hi=3.0,
                  n_obs_per=100, bg=10., seed=42, device="cpu"):
    """Generate data with uniformly log-spaced I0 in [I0_lo, I0_hi]."""
    gen = torch.Generator().manual_seed(seed)
    log_I0 = torch.linspace(math.log(I0_lo), math.log(I0_hi), n_hkl)
    I0_true = torch.exp(log_I0)
    alpha_true = torch.full((n_hkl,), alpha)
    obs_per_hkl = torch.full((n_hkl,), n_obs_per, dtype=torch.long)
    N = n_hkl * n_obs_per

    hkl_id = torch.arange(n_hkl).repeat_interleave(n_obs_per)
    I0_po = I0_true.repeat_interleave(n_obs_per)

    eps = sample_gengauss(alpha, beta, N, generator=gen)
    I_signal = (I0_po * torch.exp(eps)).clamp(min=0)
    I_obs = torch.poisson(I_signal + bg, generator=gen) - bg

    return dict(
        I0_true=I0_true.to(device), alpha_true=alpha_true.to(device),
        obs_per_hkl=obs_per_hkl.to(device), I_obs=I_obs.to(device),
        hkl_id=hkl_id.to(device), bg=bg, N=N, n_hkl=n_hkl,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    beta = 1.3
    n_hkl = 2000

    data = generate_weak(n_hkl=n_hkl, beta=beta, I0_lo=0.1, I0_hi=3.0,
                         n_obs_per=100, device=device)
    true = data["I0_true"].cpu().numpy()
    lt = np.log(true)

    def _r(est):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    print(f"I0 range: [{true.min():.2f}, {true.max():.2f}]")

    # Baselines
    med = estimate_median(data).cpu().numpy()
    irls = estimate_irls(data, beta).cpu().numpy()
    gg = estimate_gg_lbfgs(data, beta, max_iter=200).cpu().numpy()
    print(f"  Median:          r={_r(med):.4f}")
    print(f"  IRLS:            r={_r(irls):.4f}")
    print(f"  GG LBFGS:        r={_r(gg):.4f}")

    # Loglinear with different signal_min
    print("\n--- signal_min sweep (N=51) ---")
    smin_vals = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    r_vals = []
    est_dict = {}
    for smin in smin_vals:
        est = estimate_poisson_loglinear(data, beta, N=51, signal_min=smin).cpu().numpy()
        r = _r(est)
        r_vals.append(r)
        est_dict[smin] = est
        print(f"  signal_min={smin:<6}  r={r:.4f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    lo, hi = 0.05, 5.0

    # (0,0) Scatter: GG LBFGS vs truth
    ax = axes[0, 0]
    ax.scatter(true, gg, s=4, alpha=0.3, c="steelblue", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ est")
    ax.set_title(f"GG LBFGS (r={_r(gg):.4f})")
    ax.grid(True, alpha=0.3, which="both")

    # (0,1) Best loglinear vs truth
    best_smin = smin_vals[np.argmax(r_vals)]
    best_est = est_dict[best_smin]
    ax = axes[0, 1]
    ax.scatter(true, best_est, s=4, alpha=0.3, c="darkorchid", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ est")
    ax.set_title(f"Loglinear smin={best_smin} (r={_r(best_est):.4f})")
    ax.grid(True, alpha=0.3, which="both")

    # (0,2) signal_min=1.0 (the one that was great for strong regime)
    ax = axes[0, 2]
    est_1 = est_dict[1.0]
    ax.scatter(true, est_1, s=4, alpha=0.3, c="tomato", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ est")
    ax.set_title(f"Loglinear smin=1.0 (r={_r(est_1):.4f})")
    ax.grid(True, alpha=0.3, which="both")

    # (1,0) signal_min sweep
    ax = axes[1, 0]
    ax.plot(smin_vals, r_vals, "o-", color="darkorchid", lw=2)
    ax.axhline(_r(gg), color="steelblue", ls="--", lw=1.5, label=f"GG LBFGS ({_r(gg):.4f})")
    ax.axhline(_r(irls), color="seagreen", ls=":", lw=1.5, label=f"IRLS ({_r(irls):.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("signal_min")
    ax.set_ylabel("r")
    ax.set_title("signal_min sweep — weak regime")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Ratio est/true vs true for different signal_min
    ax = axes[1, 1]
    for smin, col in [(0.1, "seagreen"), (0.5, "darkorchid"), (1.0, "tomato"), (2.0, "navy")]:
        ratio = est_dict[smin] / true
        order = np.argsort(true)
        # Binned median ratio
        n_bins = 20
        bins = np.logspace(np.log10(0.1), np.log10(3.0), n_bins + 1)
        bin_idx = np.digitize(true, bins)
        med_ratio = [np.median(ratio[bin_idx == i]) if (bin_idx == i).sum() > 0 else np.nan
                     for i in range(1, n_bins + 1)]
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        ax.plot(bin_centers, med_ratio, "o-", color=col, lw=1.5, ms=4,
                label=f"smin={smin}")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("I₀ true")
    ax.set_ylabel("median(I₀_est / I₀_true)")
    ax.set_title("Bias vs I₀ for different signal_min")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.0)

    # (1,2) Fraction of negative obs vs I0
    ax = axes[1, 2]
    I_obs = data["I_obs"].cpu().numpy()
    hkl_id = data["hkl_id"].cpu().numpy()
    frac_neg = []
    for h in range(n_hkl):
        mask = hkl_id == h
        frac_neg.append((I_obs[mask] <= 0).mean())
    ax.scatter(true, frac_neg, s=4, alpha=0.5, c="navy")
    ax.set_xscale("log")
    ax.set_xlabel("I₀ true")
    ax.set_ylabel("Fraction I_obs ≤ 0")
    ax.set_title("Negative observation fraction")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Weak regime: I₀ ∈ [0.1, 3.0], α=1.2, β={beta}, bg=10, n_obs=100",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "weak_regime.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
