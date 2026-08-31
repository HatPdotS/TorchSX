#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Posterior mean of log(I0) from the overlap integral.

Instead of MLE (mode of likelihood → overfits), compute:
  E[log I0 | data] = ∫ log(I0) · L(data|I0) · prior(I0) dI0 / Z

using the obs-centered quadrature for L(data|I0).
No optimization — just grid evaluation + weighted average.
"""

from __future__ import annotations
import math, os, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls,
    _log_gg_norm,
)


def estimate_posterior_mean(data, beta, n_per_obs=31, n_I0_grid=200, device="cpu"):
    """Posterior mean of log(I0) per HKL via grid evaluation.

    For each HKL:
      1. Evaluate obs-centered log-likelihood on a fine I0 grid
      2. Add flat prior on log(I0)
      3. Compute posterior mean: E[log I0 | data]
    """
    dev = device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    obs_per_hkl = data["obs_per_hkl"]
    I_obs = data["I_obs"]
    hkl_id = data["hkl_id"]
    log_norm = _log_gg_norm(beta)
    alpha_true = data["alpha_true"].to(dev)
    alpha_po = alpha_true.repeat_interleave(obs_per_hkl)

    n_ph = (I_obs + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)

    # Per-obs λ grid (precompute, I0-independent)
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
    log_pj = log_poi + log_jac + torch.log(d_lam)[:, None]  # (N, K)

    # I0 grid per HKL — centered on median, span ±2 decades
    I0_med = estimate_median(data).to(dev)
    log_I0_center = torch.log(I0_med.clamp(min=0.1))

    # Global I0 grid in log-space
    log_I0_offsets = torch.linspace(-3.0, 3.0, n_I0_grid, device=dev)  # offsets from median

    # Sort obs by hkl for grouped processing
    order = hkl_id.argsort()
    offsets = torch.zeros(n_hkl + 1, dtype=torch.long, device=dev)
    offsets[1:] = obs_per_hkl.cumsum(0)

    result = torch.zeros(n_hkl, device=dev)

    for h in range(n_hkl):
        s, e = offsets[h].item(), offsets[h + 1].item()
        idx = order[s:e]  # observation indices for this HKL
        n_obs_h = e - s

        # I0 grid for this HKL
        log_I0_grid = log_I0_center[h] + log_I0_offsets  # (G,)

        # Per-obs quadrature data for this HKL
        sig_h = signal[idx]       # (n_obs, K)
        lpj_h = log_pj[idx]       # (n_obs, K)
        alpha_h = alpha_po[idx]   # (n_obs,)

        # Evaluate log-likelihood at each I0 grid point
        log_lik = torch.zeros(n_I0_grid, device=dev)
        for g in range(n_I0_grid):
            I0 = torch.exp(log_I0_grid[g])
            eps = torch.log(sig_h / I0)  # (n_obs, K)
            log_gg = -(eps.abs() / alpha_h[:, None]).pow(beta) + log_norm
            log_integrand = log_gg + lpj_h
            log_p = torch.logsumexp(log_integrand, dim=1)  # (n_obs,)
            log_lik[g] = log_p.sum()

        # Posterior (flat prior on log I0 → uniform prior weights)
        # log posterior ∝ log_lik (flat prior adds nothing)
        log_post = log_lik - log_lik.max()  # normalize for stability
        post = torch.exp(log_post)
        post = post / post.sum()

        # Posterior mean of log(I0)
        result[h] = (post * log_I0_grid).sum()

    return torch.exp(result)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)

    beta = 1.3
    n_hkl = 5000

    data = generate_realistic(real_params=real_params, n_hkl=n_hkl,
                              beta=beta, device=device)
    true = data["I0_true"].cpu().numpy()
    lt = np.log(true)

    def _r(est):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    print("  Median ...", flush=True)
    med = estimate_median(data).cpu().numpy()
    print("  IRLS ...", flush=True)
    irls = estimate_irls(data, beta).cpu().numpy()
    print("  Posterior mean ...", flush=True)
    post_mean = estimate_posterior_mean(data, beta, device=device).cpu().numpy()

    r_med = _r(med)
    r_irls = _r(irls)
    r_post = _r(post_mean)

    print(f"\n  Median:         r={r_med:.4f}")
    print(f"  IRLS:           r={r_irls:.4f}")
    print(f"  Posterior mean: r={r_post:.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lo, hi = 0.3, 2e3

    # (0) Scatter: all three
    ax = axes[0]
    ax.scatter(true, med, s=2, alpha=0.12, c="tomato", edgecolors="none", label="Median")
    ax.scatter(true, irls, s=2, alpha=0.12, c="seagreen", edgecolors="none", label="IRLS")
    ax.scatter(true, post_mean, s=2, alpha=0.12, c="darkorchid", edgecolors="none",
               label="Posterior mean")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
    ax.set_title("All estimators")
    ax.text(0.05, 0.95,
            f"Post:  r={r_post:.4f}\nIRLS:  r={r_irls:.4f}\nMed:   r={r_med:.4f}",
            transform=ax.transAxes, fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3, which="both")

    # (1) IRLS vs posterior mean head-to-head
    ax = axes[1]
    ax.scatter(true, irls, s=2, alpha=0.15, c="seagreen", edgecolors="none", label="IRLS")
    ax.scatter(true, post_mean, s=2, alpha=0.15, c="darkorchid", edgecolors="none",
               label="Posterior mean")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
    ax.set_title(f"IRLS (r={r_irls:.4f}) vs Posterior mean (r={r_post:.4f})")
    ax.legend(loc="lower right", fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3, which="both")

    # (2) Where does posterior mean help?
    ax = axes[2]
    err_irls = np.abs(np.log(np.clip(irls, 1e-6, None)) - lt)
    err_post = np.abs(np.log(np.clip(post_mean, 1e-6, None)) - lt)
    improvement = err_irls - err_post

    log_bins = np.linspace(np.log(true.min()), np.log(true.max()), 15)
    bin_idx = np.digitize(lt, log_bins) - 1
    centers, fracs = [], []
    for b in range(len(log_bins) - 1):
        mask = bin_idx == b
        if mask.sum() < 10: continue
        centers.append(np.exp(0.5 * (log_bins[b] + log_bins[b+1])))
        fracs.append((improvement[mask] > 0).mean())

    ax.bar(range(len(centers)), fracs, color="darkorchid", alpha=0.7)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels([f"{c:.0f}" for c in centers], fontsize=7, rotation=45)
    ax.set_xlabel("I₀ true (bin center)")
    ax.set_ylabel("Frac. posterior mean beats IRLS")
    ax.set_title("Where does posterior mean help?")
    ax.set_ylim(0, 1)

    fig.suptitle(f"Posterior mean E[log I₀ | data] vs IRLS — real distributions, β={beta}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "scatter_posterior.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
