#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Check: does minimizing the GenGauss log-space NLL via LBFGS
reproduce the IRLS, or does it also overfit?

GenGauss NLL (positive obs only):
  NLL(μ) = Σ_j |log(I_obs_j) - μ|^β / α^β

IRLS solves this analytically. LBFGS should find the same minimum.
If it does → overfitting is specific to the Poisson likelihood.
If it doesn't → something else is going on.
"""

from __future__ import annotations
import math, os, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls,
    estimate_poisson_obs_centered, _log_gg_norm,
)


def estimate_gg_lbfgs(data, beta, max_iter=50):
    """Minimize GenGauss NLL in log-space via LBFGS (positive obs only)."""
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    I_obs = data["I_obs"]
    alpha_true = data["alpha_true"].to(dev)

    # Positive observations only
    pos = I_obs > 0
    log_I_pos = torch.log(I_obs[pos])
    hkl_pos = hkl_id[pos]
    alpha_po_pos = alpha_true.repeat_interleave(obs_per_hkl)[pos]

    # Init from median
    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)

    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        mu_po = log_I0[hkl_pos]  # (M,) — per positive obs
        residual = (log_I_pos - mu_po).abs().clamp(min=1e-8)
        # GenGauss NLL: |r/α|^β
        nll_per_obs = (residual / alpha_po_pos).pow(beta)
        loss = nll_per_obs.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


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
    print("  IRLS (closed form) ...", flush=True)
    irls = estimate_irls(data, beta).cpu().numpy()
    print("  GenGauss LBFGS (log-space, pos only) ...", flush=True)
    gg_lbfgs = estimate_gg_lbfgs(data, beta).cpu().numpy()
    print("  Poisson MLE (obs-centered) ...", flush=True)
    poi_oc = estimate_poisson_obs_centered(data, beta).cpu().numpy()

    r_med = _r(med)
    r_irls = _r(irls)
    r_gg = _r(gg_lbfgs)
    r_poi = _r(poi_oc)

    print(f"\n  Median:           r={r_med:.4f}")
    print(f"  IRLS (closed):    r={r_irls:.4f}")
    print(f"  GenGauss LBFGS:   r={r_gg:.4f}")
    print(f"  Poisson obs-ctr:  r={r_poi:.4f}")

    # How close is LBFGS to IRLS?
    ratio = gg_lbfgs / np.clip(irls, 1e-6, None)
    print(f"\n  GenGauss LBFGS vs IRLS:")
    print(f"    median(ratio): {np.median(ratio):.4f}")
    print(f"    std(log ratio): {np.std(np.log(ratio)):.4f}")
    print(f"    max |log ratio|: {np.max(np.abs(np.log(ratio))):.4f}")
    print(f"    r(LBFGS, IRLS): {np.corrcoef(np.log(irls), np.log(gg_lbfgs))[0,1]:.6f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lo, hi = 0.3, 2e3

    # (0) IRLS vs GenGauss LBFGS — do they match?
    ax = axes[0]
    ax.scatter(irls, gg_lbfgs, s=2, alpha=0.3, c="steelblue", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ IRLS (closed form)")
    ax.set_ylabel("I₀ GenGauss LBFGS")
    r_match = np.corrcoef(np.log(irls), np.log(gg_lbfgs))[0, 1]
    ax.set_title(f"IRLS vs GenGauss LBFGS (r={r_match:.6f})")
    ax.grid(True, alpha=0.3, which="both")

    # (1) All methods vs true
    ax = axes[1]
    ax.scatter(true, med, s=2, alpha=0.10, c="tomato", edgecolors="none", label="Median")
    ax.scatter(true, irls, s=2, alpha=0.10, c="seagreen", edgecolors="none", label="IRLS")
    ax.scatter(true, gg_lbfgs, s=2, alpha=0.10, c="steelblue", edgecolors="none", label="GG LBFGS")
    ax.scatter(true, poi_oc, s=2, alpha=0.10, c="darkorchid", edgecolors="none", label="Poisson OC")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
    ax.set_title("All methods vs true")
    ax.text(0.05, 0.95,
            f"Poi OC: r={r_poi:.4f}\nGG LB:  r={r_gg:.4f}\n"
            f"IRLS:   r={r_irls:.4f}\nMed:    r={r_med:.4f}",
            transform=ax.transAxes, fontsize=8, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=7, markerscale=3)
    ax.grid(True, alpha=0.3, which="both")

    # (2) Bar chart
    ax = axes[2]
    methods = ["Median", "IRLS (closed)", "GenGauss LBFGS", "Poisson obs-ctr"]
    rs = [r_med, r_irls, r_gg, r_poi]
    colors = ["tomato", "seagreen", "steelblue", "darkorchid"]
    bars = ax.barh(range(4), rs, color=colors, alpha=0.7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("r(log I₀_true, log I₀_est)")
    ax.set_title("Correlation comparison")
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.01)
    for bar, r in zip(bars, rs):
        ax.text(r + 0.002, bar.get_y() + bar.get_height()/2, f"{r:.4f}",
                va="center", fontsize=9)

    fig.suptitle(f"Does GenGauss NLL overfit like Poisson NLL? (β={beta}, real distributions)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "irls_vs_lbfgs.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
