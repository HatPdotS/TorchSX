#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Test adaptive regularization: obs-centered Poisson NLL + per-HKL IRLS prior.

Prior: log(I0) ~ N(log(I0_IRLS), σ²_h)
  where σ²_h = α_h² · Γ(3/β) / (Γ(1/β) · n_obs_h)
  — the variance of the log-space mean estimator under GenGauss.

This gives each HKL a regularization strength proportional to 1/n_obs.
"""

from __future__ import annotations

import math
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls,
    estimate_poisson_obs_centered, _log_gg_norm, _quad_range,
)


def estimate_poisson_adaptive_map(data, beta, strength=1.0, n_per_obs=31, max_iter=50):
    """Obs-centered Poisson NLL + adaptive per-HKL IRLS prior.

    loss = -Σ_j log p(n_j | I0, α, β)
           + Σ_h 0.5 * (log I0_h - log I0_IRLS_h)² / σ²_h

    σ²_h = strength × α_h² × Γ(3/β) / (Γ(1/β) × n_obs_h)
    """
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    I_obs = data["I_obs"]
    N = data["N"]
    log_norm = _log_gg_norm(beta)

    alpha_true = data["alpha_true"]
    alpha_po = alpha_true.repeat_interleave(obs_per_hkl)
    n_ph = (I_obs + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)

    # Per-obs λ grid
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
    log_pj = log_poi + log_jac + torch.log(d_lam)[:, None]

    # IRLS prior
    I0_irls = estimate_irls(data, beta)
    log_I0_prior = torch.log(I0_irls.clamp(min=0.1)).to(dev)

    # Per-HKL prior variance: σ²_h = strength × α² × Γ(3/β) / (Γ(1/β) × n_obs)
    g3b = float(torch.exp(torch.lgamma(torch.tensor(3.0 / beta))))
    g1b = float(torch.exp(torch.lgamma(torch.tensor(1.0 / beta))))
    var_factor = g3b / g1b  # ≈ σ²_LN / α² for GenGauss
    sigma2_prior = (strength * alpha_true.to(dev) ** 2 * var_factor
                    / obs_per_hkl.float().clamp(min=1))
    inv_sigma2 = 1.0 / sigma2_prior.clamp(min=1e-6)

    log_I0 = log_I0_prior.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        I0_po = torch.exp(log_I0).repeat_interleave(obs_per_hkl)
        eps = torch.log(signal / I0_po[:, None])
        log_gg = -(eps.abs() / alpha_po[:, None]).pow(beta) + log_norm
        log_integrand = log_gg + log_pj
        log_p = torch.logsumexp(log_integrand, dim=1)
        nll = torch.zeros(n_hkl, device=dev)
        nll.scatter_add_(0, hkl_id, -log_p)
        # Adaptive prior
        reg = 0.5 * (inv_sigma2 * (log_I0 - log_I0_prior) ** 2).sum()
        loss = nll.sum() + reg
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)
    print(f"Real data: {len(real_params['I0'])} HKLs, "
          f"I0 med={real_params['I0'].median():.1f}, "
          f"alpha med={real_params['alpha'].median():.3f}")

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
    print("  Poisson obs-centered (no reg) ...", flush=True)
    mle_oc = estimate_poisson_obs_centered(data, beta).cpu().numpy()

    # Scan regularization strengths
    strengths = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    map_results = {}
    for s in strengths:
        print(f"  Adaptive MAP (strength={s}) ...", flush=True)
        map_results[s] = estimate_poisson_adaptive_map(
            data, beta, strength=s).cpu().numpy()

    r_med = _r(med)
    r_irls = _r(irls)
    r_oc = _r(mle_oc)

    print(f"\n  Results:")
    print(f"    Median:         r={r_med:.4f}")
    print(f"    IRLS:           r={r_irls:.4f}")
    print(f"    Obs-ctr (unreg):r={r_oc:.4f}")
    for s in strengths:
        r = _r(map_results[s])
        print(f"    Adaptive s={s:<4}: r={r:.4f}")

    best_s = max(strengths, key=lambda s: _r(map_results[s]))
    best = map_results[best_s]
    r_best = _r(best)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lo, hi = 0.3, 2e3

    # (0) IRLS vs best adaptive MAP
    ax = axes[0]
    ax.scatter(true, irls, s=2, alpha=0.15, c="seagreen", edgecolors="none", label="IRLS")
    ax.scatter(true, best, s=2, alpha=0.15, c="darkorchid", edgecolors="none",
               label=f"Adaptive MAP s={best_s}")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
    ax.set_title(f"IRLS (r={r_irls:.4f}) vs Adaptive MAP (r={r_best:.4f})")
    ax.text(0.05, 0.95, f"MAP:  r={r_best:.4f}\nIRLS: r={r_irls:.4f}",
            transform=ax.transAxes, fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3, which="both")

    # (1) Bar chart — all methods
    ax = axes[1]
    methods = ["Median", "IRLS", "Obs-ctr unreg"] + [f"MAP s={s}" for s in strengths]
    rs = [r_med, r_irls, r_oc] + [_r(map_results[s]) for s in strengths]
    colors = (["tomato", "seagreen", "steelblue"]
              + ["darkorchid" if s == best_s else "plum" for s in strengths])
    bars = ax.barh(range(len(methods)), rs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xlabel("r(log I₀_true, log I₀_est)")
    ax.set_title("Regularization strength scan")
    ax.set_xlim(min(rs) - 0.01, max(rs) + 0.01)
    for bar, r in zip(bars, rs):
        ax.text(r + 0.001, bar.get_y() + bar.get_height()/2, f"{r:.4f}",
                va="center", fontsize=7)

    # (2) Improvement over IRLS vs I0
    ax = axes[2]
    err_irls = np.abs(np.log(np.clip(irls, 1e-6, None)) - lt)
    err_best = np.abs(np.log(np.clip(best, 1e-6, None)) - lt)
    delta = err_irls - err_best  # positive = MAP is better
    ax.scatter(true, delta, s=2, alpha=0.2, c="darkorchid", edgecolors="none")
    ax.axhline(0, color="k", ls="-", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("I₀ true")
    ax.set_ylabel("IRLS error − MAP error (>0 = MAP better)")
    pct_better = (delta > 0).mean()
    ax.set_title(f"MAP better for {pct_better:.1%} of HKLs")
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Adaptive Poisson MAP (obs-centered quad) — real distributions, β={beta}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "scatter_regularized.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
