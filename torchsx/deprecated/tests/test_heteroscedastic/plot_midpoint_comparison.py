#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare I0 estimators including midpoint-rule Poisson integration.

Methods:
  1. Median — robust baseline
  2. IRLS — log-space iterative reweighted LS (pos only)
  3. GenGauss LBFGS — log-space NLL optimized by LBFGS (pos only)
  4. Poisson midpoint — proper midpoint-rule integration in λ-space (all obs)
"""

from __future__ import annotations
import math, os, sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls, _log_gg_norm,
)


def estimate_gg_lbfgs(data, beta, max_iter=50):
    """GenGauss NLL in log-space, positive obs only."""
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    alpha_true = data["alpha_true"].to(dev)

    pos = data["I_obs"] > 0
    log_I_pos = torch.log(data["I_obs"][pos])
    hkl_pos = hkl_id[pos]
    alpha_po_pos = alpha_true.repeat_interleave(obs_per_hkl)[pos]

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        mu = log_I0[hkl_pos]
        r = (log_I_pos - mu).abs().clamp(min=1e-8)
        loss = (r / alpha_po_pos).pow(beta).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


def estimate_poisson_midpoint(data, beta, N=31, n_sigma=5, max_iter=50):
    """Poisson MLE with midpoint-rule integration in λ-space.

    For each obs with n_ph photons:
      Place N points uniformly at λ_k = n_ph + k·σ, k ∈ [-n_sigma, +n_sigma]
      Clamp λ_k ≥ bg + 0.1
      Cell widths via midpoint rule (half-distance to neighbors)
      Integrand = GG(ε_k) · Poi(n; λ_k) · |dε/dλ|_k · w_k
    """
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    alpha_true = data["alpha_true"].to(dev)
    log_gg_norm = _log_gg_norm(beta)

    alpha_po = alpha_true.repeat_interleave(obs_per_hkl)
    n_ph = (data["I_obs"] + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)
    N_obs = len(n_ph)

    # Build per-obs λ grid: ±n_sigma·σ around Poisson peak
    sigma = n_ph.clamp(min=1).sqrt()
    k = torch.linspace(-n_sigma, n_sigma, N, device=dev)  # (K,)
    lam = n_ph[:, None] + sigma[:, None] * k[None, :]     # (N_obs, K)
    lam = lam.clamp(min=bg + 0.1)

    # Cell widths via midpoint rule: w_k = (λ_{k+1} - λ_{k-1}) / 2
    # For uniform spacing: interior = Δλ, edges = Δλ/2
    # But after clamping, left-side points may collapse → use actual differences
    widths = torch.zeros_like(lam)                         # (N_obs, K)
    widths[:, 1:-1] = (lam[:, 2:] - lam[:, :-2]) / 2     # interior
    widths[:, 0] = (lam[:, 1] - lam[:, 0]) / 2            # left edge
    widths[:, -1] = (lam[:, -1] - lam[:, -2]) / 2         # right edge

    signal = lam - bg  # (N_obs, K) — always > 0.1 after clamp

    # Precompute I0-independent parts
    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]  # (N_obs, K)
    log_jac = -torch.log(signal)                                       # (N_obs, K)
    log_w = torch.log(widths.clamp(min=1e-30))                        # (N_obs, K)
    log_poi_jac_w = log_poi + log_jac + log_w                         # (N_obs, K)

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        I0_po = torch.exp(log_I0).repeat_interleave(obs_per_hkl)
        eps = torch.log(signal / I0_po[:, None])
        log_gg = -(eps.abs() / alpha_po[:, None]).pow(beta) + log_gg_norm
        log_integrand = log_gg + log_poi_jac_w
        log_p = torch.logsumexp(log_integrand, dim=1)
        nll = torch.zeros(n_hkl, device=dev)
        nll.scatter_add_(0, hkl_id, -log_p)
        loss = nll.sum()
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
    print("  GenGauss LBFGS (pos only) ...", flush=True)
    gg = estimate_gg_lbfgs(data, beta).cpu().numpy()
    print("  Poisson midpoint (all obs) ...", flush=True)
    poi = estimate_poisson_midpoint(data, beta).cpu().numpy()

    r_med, r_irls, r_gg, r_poi = _r(med), _r(irls), _r(gg), _r(poi)

    print(f"\n  Results:")
    print(f"    Median:           r={r_med:.4f}")
    print(f"    IRLS:             r={r_irls:.4f}")
    print(f"    GenGauss LBFGS:   r={r_gg:.4f}")
    print(f"    Poisson midpoint: r={r_poi:.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lo, hi = 0.3, 2e3

    # (0) Scatter: all vs true
    ax = axes[0]
    ax.scatter(true, irls, s=2, alpha=0.12, c="seagreen", edgecolors="none",
               label=f"IRLS ({r_irls:.4f})")
    ax.scatter(true, gg, s=2, alpha=0.12, c="steelblue", edgecolors="none",
               label=f"GG LBFGS ({r_gg:.4f})")
    ax.scatter(true, poi, s=2, alpha=0.12, c="darkorchid", edgecolors="none",
               label=f"Poi midpt ({r_poi:.4f})")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
    ax.set_title("All methods vs truth")
    ax.legend(loc="lower right", fontsize=7, markerscale=3)
    ax.grid(True, alpha=0.3, which="both")

    # (1) GG LBFGS vs Poisson midpoint head-to-head
    ax = axes[1]
    ax.scatter(gg, poi, s=2, alpha=0.3, c="darkorchid", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("GenGauss LBFGS"); ax.set_ylabel("Poisson midpoint")
    r_match = np.corrcoef(np.log(np.clip(gg, 1e-6, None)),
                          np.log(np.clip(poi, 1e-6, None)))[0, 1]
    ax.set_title(f"GG LBFGS vs Poi midpoint (r={r_match:.4f})")
    ax.grid(True, alpha=0.3, which="both")

    # (2) Bar chart
    ax = axes[2]
    methods = ["Median", "IRLS", "GenGauss LBFGS", "Poisson midpoint"]
    rs = [r_med, r_irls, r_gg, r_poi]
    colors = ["tomato", "seagreen", "steelblue", "darkorchid"]
    bars = ax.barh(range(4), rs, color=colors, alpha=0.7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("r(log I₀_true, log I₀_est)")
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.01)
    for bar, r in zip(bars, rs):
        ax.text(r + 0.002, bar.get_y() + bar.get_height()/2, f"{r:.4f}",
                va="center", fontsize=9)

    fig.suptitle(f"Midpoint-rule Poisson vs GenGauss LBFGS vs IRLS  (β={beta}, {n_hkl} HKLs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "midpoint_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
