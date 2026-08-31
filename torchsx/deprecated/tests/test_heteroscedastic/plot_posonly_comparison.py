#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare GenGauss LBFGS vs Poisson LBFGS — both on positive observations only.

If the per-obs NLLs are similar for positive obs, these should give similar results.
Any remaining difference isolates the likelihood shape vs negative-obs effect.
"""

from __future__ import annotations
import math, os, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def estimate_poisson_posonly(data, beta, n_per_obs=31, max_iter=50):
    """Poisson obs-centered NLL, positive observations only."""
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    alpha_true = data["alpha_true"].to(dev)
    log_norm = _log_gg_norm(beta)

    # Filter to positive only
    pos = data["I_obs"] > 0
    I_pos = data["I_obs"][pos]
    hkl_pos = hkl_id[pos]
    alpha_po_pos = alpha_true.repeat_interleave(obs_per_hkl)[pos]
    N_pos = int(pos.sum())

    n_ph = (I_pos + bg).clamp(min=0)
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

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        I0_po = torch.exp(log_I0)[hkl_pos]
        eps = torch.log(signal / I0_po[:, None])
        log_gg = -(eps.abs() / alpha_po_pos[:, None]).pow(beta) + log_norm
        log_integrand = log_gg + log_pj
        log_p = torch.logsumexp(log_integrand, dim=1)
        nll = torch.zeros(n_hkl, device=dev)
        nll.scatter_add_(0, hkl_pos, -log_p)
        loss = nll.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


def estimate_poisson_allobs(data, beta, n_per_obs=31, max_iter=50):
    """Poisson obs-centered NLL, all observations (including negative)."""
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    alpha_true = data["alpha_true"].to(dev)
    log_norm = _log_gg_norm(beta)

    alpha_po = alpha_true.repeat_interleave(obs_per_hkl)
    n_ph = (data["I_obs"] + bg).clamp(min=0)
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
    log_pj = log_poi + log_jac + torch.log(d_lam)[:, None]

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
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
    print("  Poisson LBFGS (pos only) ...", flush=True)
    poi_pos = estimate_poisson_posonly(data, beta).cpu().numpy()
    print("  Poisson LBFGS (all obs) ...", flush=True)
    poi_all = estimate_poisson_allobs(data, beta).cpu().numpy()

    r_med = _r(med)
    r_irls = _r(irls)
    r_gg = _r(gg)
    r_poi_pos = _r(poi_pos)
    r_poi_all = _r(poi_all)

    print(f"\n  Results:")
    print(f"    Median:              r={r_med:.4f}")
    print(f"    IRLS:                r={r_irls:.4f}")
    print(f"    GenGauss LBFGS:      r={r_gg:.4f}  (pos only)")
    print(f"    Poisson LBFGS pos:   r={r_poi_pos:.4f}  (pos only)")
    print(f"    Poisson LBFGS all:   r={r_poi_all:.4f}  (all obs)")

    # How close are GenGauss and Poisson (pos only)?
    ratio = poi_pos / np.clip(gg, 1e-6, None)
    print(f"\n  Poisson pos-only vs GenGauss LBFGS:")
    print(f"    r between them: {np.corrcoef(np.log(gg), np.log(poi_pos))[0,1]:.6f}")
    print(f"    median ratio: {np.median(ratio):.4f}")
    print(f"    std(log ratio): {np.std(np.log(ratio)):.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lo, hi = 0.3, 2e3

    # (0) Bar chart
    ax = axes[0]
    methods = ["Median", "IRLS", "GG LBFGS\n(pos only)", "Poisson LBFGS\n(pos only)",
               "Poisson LBFGS\n(all obs)"]
    rs = [r_med, r_irls, r_gg, r_poi_pos, r_poi_all]
    colors = ["tomato", "seagreen", "steelblue", "darkorchid", "gold"]
    bars = ax.barh(range(5), rs, color=colors, alpha=0.7)
    ax.set_yticks(range(5))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("r(log I₀_true, log I₀_est)")
    ax.set_title("Correlation comparison")
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.01)
    for bar, r in zip(bars, rs):
        ax.text(r + 0.002, bar.get_y() + bar.get_height()/2, f"{r:.4f}",
                va="center", fontsize=9)

    # (1) GenGauss vs Poisson pos-only scatter
    ax = axes[1]
    ax.scatter(gg, poi_pos, s=2, alpha=0.3, c="steelblue", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("GenGauss LBFGS (pos only)")
    ax.set_ylabel("Poisson LBFGS (pos only)")
    r_match = np.corrcoef(np.log(gg), np.log(poi_pos))[0, 1]
    ax.set_title(f"GG vs Poisson, both pos-only (r={r_match:.4f})")
    ax.grid(True, alpha=0.3, which="both")

    # (2) All three vs true
    ax = axes[2]
    ax.scatter(true, gg, s=2, alpha=0.12, c="steelblue", edgecolors="none",
               label=f"GG pos-only ({r_gg:.4f})")
    ax.scatter(true, poi_pos, s=2, alpha=0.12, c="darkorchid", edgecolors="none",
               label=f"Poi pos-only ({r_poi_pos:.4f})")
    ax.scatter(true, poi_all, s=2, alpha=0.12, c="gold", edgecolors="none",
               label=f"Poi all obs ({r_poi_all:.4f})")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
    ax.set_title("All vs true")
    ax.legend(loc="lower right", fontsize=7, markerscale=3)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Isolating the effect: GenGauss vs Poisson, pos-only vs all obs  (β={beta})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "posonly_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
