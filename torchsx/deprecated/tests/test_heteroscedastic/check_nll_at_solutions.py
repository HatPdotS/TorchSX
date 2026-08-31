#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Evaluate the obs-centered Poisson NLL at each estimator's I0 solution.

For each HKL, compute NLL(I0) at:
  - I0_true (ground truth)
  - I0_median
  - I0_IRLS
  - I0_MLE (obs-centered Poisson)

If MLE has lowest NLL but worst correlation → it's overfitting.
If IRLS has lower NLL → optimization is broken.
"""

from __future__ import annotations

import math
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse everything from the scatter script
from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls,
    estimate_poisson_obs_centered, _log_gg_norm,
    sample_gengauss,
)


def eval_nll_per_hkl(data, I0_est, beta, n_per_obs=31):
    """Evaluate obs-centered Poisson NLL per HKL at given I0 values."""
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    I_obs = data["I_obs"]
    log_norm = _log_gg_norm(beta)

    alpha_po = data["alpha_true"].repeat_interleave(obs_per_hkl)
    n_ph = (I_obs + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)

    sigma = n_ph.clamp(min=1).sqrt()
    lam_center = torch.maximum(n_ph, torch.full_like(n_ph, bg + 0.1))
    lam_lo = (lam_center - 4 * sigma).clamp(min=bg + 0.1)
    lam_hi = lam_center + 4 * sigma

    t = torch.linspace(0, 1, n_per_obs, device=dev)
    lam = lam_lo[:, None] + (lam_hi - lam_lo)[:, None] * t[None, :]
    d_lam = (lam_hi - lam_lo) / (n_per_obs - 1)
    signal = lam - bg

    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_jac = -torch.log(signal)
    log_d_lam = torch.log(d_lam)
    log_poi_jac = log_poi + log_jac + log_d_lam[:, None]

    I0_po = I0_est.repeat_interleave(obs_per_hkl)
    eps = torch.log(signal / I0_po[:, None])
    log_gg = -(eps.abs() / alpha_po[:, None]).pow(beta) + log_norm
    log_integrand = log_gg + log_poi_jac
    log_p = torch.logsumexp(log_integrand, dim=1)  # (N,)

    # Sum per HKL
    nll_per_hkl = torch.zeros(n_hkl, device=dev)
    nll_per_hkl.scatter_add_(0, hkl_id, -log_p)
    return nll_per_hkl


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)

    beta = 1.3
    n_hkl = 5000

    data = generate_realistic(real_params=real_params, n_hkl=n_hkl,
                              beta=beta, device=device)
    true = data["I0_true"]

    print("Computing estimators ...", flush=True)
    med = estimate_median(data)
    irls = estimate_irls(data, beta)
    mle_oc = estimate_poisson_obs_centered(data, beta)

    print("Evaluating NLL at each solution ...", flush=True)
    nll_true = eval_nll_per_hkl(data, true.to(device), beta).cpu()
    nll_med = eval_nll_per_hkl(data, med.to(device), beta).cpu()
    nll_irls = eval_nll_per_hkl(data, irls.to(device), beta).cpu()
    nll_mle = eval_nll_per_hkl(data, mle_oc.to(device), beta).cpu()

    # Per-HKL: which solution has lowest NLL?
    all_nll = torch.stack([nll_true, nll_med, nll_irls, nll_mle], dim=1)
    best = all_nll.argmin(dim=1)
    names = ["True", "Median", "IRLS", "MLE"]
    for i, name in enumerate(names):
        print(f"  {name} is best NLL for {(best == i).sum()}/{n_hkl} HKLs")

    # Summary stats
    print(f"\n  Total NLL:")
    print(f"    True:   {nll_true.sum():.0f}")
    print(f"    Median: {nll_med.sum():.0f}  (Δ={nll_med.sum() - nll_true.sum():+.0f})")
    print(f"    IRLS:   {nll_irls.sum():.0f}  (Δ={nll_irls.sum() - nll_true.sum():+.0f})")
    print(f"    MLE:    {nll_mle.sum():.0f}  (Δ={nll_mle.sum() - nll_true.sum():+.0f})")

    # Correlation with truth
    true_np = true.cpu().numpy()
    lt = np.log(true_np)

    def _r(est):
        return np.corrcoef(lt, np.log(np.clip(est.cpu().numpy(), 1e-6, None)))[0, 1]

    print(f"\n  Correlation r(log I0_true, log I0_est):")
    print(f"    Median: {_r(med):.4f}")
    print(f"    IRLS:   {_r(irls):.4f}")
    print(f"    MLE:    {_r(mle_oc):.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (0,0) NLL at MLE vs NLL at true — per HKL
    ax = axes[0, 0]
    delta_mle = (nll_mle - nll_true).numpy()
    ax.hist(delta_mle, bins=100, color="darkorchid", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.set_xlabel("NLL(MLE) − NLL(true)")
    ax.set_ylabel("Count")
    frac_lower = (delta_mle < 0).mean()
    ax.set_title(f"MLE vs True: {frac_lower:.1%} of HKLs have MLE < True NLL")

    # (0,1) NLL at IRLS vs NLL at true
    ax = axes[0, 1]
    delta_irls = (nll_irls - nll_true).numpy()
    ax.hist(delta_irls, bins=100, color="seagreen", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.set_xlabel("NLL(IRLS) − NLL(true)")
    ax.set_ylabel("Count")
    frac_lower_irls = (delta_irls < 0).mean()
    ax.set_title(f"IRLS vs True: {frac_lower_irls:.1%} have IRLS < True NLL")

    # (0,2) NLL at MLE vs NLL at IRLS — the key question
    ax = axes[0, 2]
    delta_mle_irls = (nll_mle - nll_irls).numpy()
    ax.hist(delta_mle_irls, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.set_xlabel("NLL(MLE) − NLL(IRLS)")
    ax.set_ylabel("Count")
    frac_mle_wins = (delta_mle_irls < 0).mean()
    ax.set_title(f"MLE vs IRLS: MLE has lower NLL for {frac_mle_wins:.1%} of HKLs")

    # (1,0) Scatter: |log(est/true)| for MLE vs IRLS — who's closer to truth?
    ax = axes[1, 0]
    err_mle = np.abs(np.log(np.clip(mle_oc.cpu().numpy(), 1e-6, None)) - lt)
    err_irls = np.abs(np.log(np.clip(irls.cpu().numpy(), 1e-6, None)) - lt)
    ax.scatter(err_irls, err_mle, s=2, alpha=0.2, c="steelblue", edgecolors="none")
    mx = max(err_irls.max(), err_mle.max()) * 0.8
    ax.plot([0, mx], [0, mx], "k-", lw=1)
    ax.set_xlabel("|log(IRLS/true)|")
    ax.set_ylabel("|log(MLE/true)|")
    ax.set_title(f"Per-HKL error: MLE closer for "
                 f"{(err_mle < err_irls).mean():.1%}")
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_aspect("equal")

    # (1,1) NLL(MLE) - NLL(IRLS) vs I0_true — is there an I0 dependence?
    ax = axes[1, 1]
    ax.scatter(true_np, delta_mle_irls, s=2, alpha=0.2, c="steelblue", edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("I₀ true")
    ax.set_ylabel("NLL(MLE) − NLL(IRLS)")
    ax.set_title("NLL advantage vs I₀")
    ax.set_ylim(-50, 50)

    # (1,2) Error vs I0_true for both methods
    ax = axes[1, 2]
    ax.scatter(true_np, err_irls, s=2, alpha=0.15, c="seagreen", edgecolors="none", label="IRLS")
    ax.scatter(true_np, err_mle, s=2, alpha=0.15, c="darkorchid", edgecolors="none", label="MLE")
    ax.set_xscale("log")
    ax.set_xlabel("I₀ true")
    ax.set_ylabel("|log(est/true)|")
    ax.set_title("Per-HKL error vs I₀")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 3)

    fig.suptitle("NLL evaluation at each estimator's solution (obs-centered quadrature)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "nll_at_solutions.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
