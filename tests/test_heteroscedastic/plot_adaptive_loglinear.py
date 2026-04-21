#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Adaptive log-linear quadrature: per-obs bounds from Poisson range.

signal_lo = max(ε, n_ph - bg - 5σ)  — per obs, auto-adapts to I0 regime
signal_hi = n_ph - bg + 5σ           — per obs

For strong obs: signal_lo ≈ 50 → entire grid is linear (like clamp)
For weak obs:   signal_lo ≈ 0.001 → grid is log near bg, linear above
No tuning parameters needed.
"""

from __future__ import annotations
import math, os, sys, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_scatter_bias import (
    generate_realistic, sample_gengauss, estimate_median, estimate_irls, _log_gg_norm,
)
from plot_midpoint_comparison import estimate_gg_lbfgs
from plot_convergence_sweep import estimate_poisson_midpoint
from verify_quadrature import _log_gg_density, _log_poi, mc_log_marginal


def _softplus_inv(y):
    """Numerically stable inverse softplus."""
    if isinstance(y, torch.Tensor):
        return torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(min=1e-8))))
    return y if y > 20 else math.log(math.expm1(max(y, 1e-8)))


def estimate_poisson_adaptive(data, beta, N=51, n_sigma=5, max_iter=50):
    """Poisson MLE with adaptive per-obs log-linear quadrature."""
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

    sigma = n_ph.clamp(min=1).sqrt()

    # Per-obs signal bounds
    signal_hi = (n_ph - bg + n_sigma * sigma).clamp(min=1.0)
    # Lower bound: 1% of signal_hi, floored at 0.001
    # This gives 2 decades of dynamic range for all obs.
    # For weak obs (signal_hi=6): signal_lo = 0.06 → covers sub-photon regime
    # For strong obs (signal_hi=150): signal_lo = 1.5 → stays above bg noise
    signal_lo = (signal_hi * 0.01).clamp(min=0.001)

    # Map to t-space via softplus_inv, uniform grid, map back
    t_lo = _softplus_inv(signal_lo)
    t_hi = _softplus_inv(signal_hi)

    frac = torch.linspace(0, 1, N, device=dev)
    t = t_lo[:, None] + (t_hi - t_lo)[:, None] * frac[None, :]
    signal = F.softplus(t)
    lam = signal + bg

    # Midpoint widths
    widths = torch.zeros_like(lam)
    widths[:, 1:-1] = (lam[:, 2:] - lam[:, :-2]) / 2
    widths[:, 0] = (lam[:, 1] - lam[:, 0]) / 2
    widths[:, -1] = (lam[:, -1] - lam[:, -2]) / 2

    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_jac = -torch.log(signal)
    log_w = torch.log(widths.clamp(min=1e-30))
    log_poi_jac_w = log_poi + log_jac + log_w

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


def generate_weak(*, n_hkl, beta, alpha=1.2, I0_lo=0.1, I0_hi=3.0,
                  n_obs_per=100, bg=10., seed=42, device="cpu"):
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

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)
    beta = 1.3

    def _r(est, lt):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    # ── Test 1: Realistic (mixed I0) ──
    print("=== Realistic regime (I0 med=9.2) ===")
    data_real = generate_realistic(real_params=real_params, n_hkl=5000,
                                    beta=beta, device=device)
    true_r = data_real["I0_true"].cpu().numpy()
    lt_r = np.log(true_r)

    r_gg = _r(estimate_gg_lbfgs(data_real, beta, max_iter=200).cpu().numpy(), lt_r)
    r_irls = _r(estimate_irls(data_real, beta).cpu().numpy(), lt_r)
    print(f"  GG LBFGS:     r={r_gg:.4f}")
    print(f"  IRLS:         r={r_irls:.4f}")

    for N in [31, 51, 101]:
        est_cl, _ = estimate_poisson_midpoint(data_real, beta, N=N, max_iter=50)
        r_cl = _r(est_cl.cpu().numpy(), lt_r)
        est_ad = estimate_poisson_adaptive(data_real, beta, N=N, max_iter=50)
        r_ad = _r(est_ad.cpu().numpy(), lt_r)
        print(f"  N={N:>3}  clamp: r={r_cl:.4f}   adaptive: r={r_ad:.4f}   Δ={r_ad-r_cl:+.4f}")

    # ── Test 2: Weak regime ──
    print("\n=== Weak regime (I0 ∈ [0.1, 3.0]) ===")
    data_weak = generate_weak(n_hkl=2000, beta=beta, device=device)
    true_w = data_weak["I0_true"].cpu().numpy()
    lt_w = np.log(true_w)

    r_gg_w = _r(estimate_gg_lbfgs(data_weak, beta, max_iter=200).cpu().numpy(), lt_w)
    r_irls_w = _r(estimate_irls(data_weak, beta).cpu().numpy(), lt_w)
    print(f"  GG LBFGS:     r={r_gg_w:.4f}")
    print(f"  IRLS:         r={r_irls_w:.4f}")

    for N in [31, 51, 101]:
        est_cl, _ = estimate_poisson_midpoint(data_weak, beta, N=N, max_iter=50)
        r_cl = _r(est_cl.cpu().numpy(), lt_w)
        est_ad = estimate_poisson_adaptive(data_weak, beta, N=N, max_iter=50)
        r_ad = _r(est_ad.cpu().numpy(), lt_w)
        print(f"  N={N:>3}  clamp: r={r_cl:.4f}   adaptive: r={r_ad:.4f}   Δ={r_ad-r_cl:+.4f}")

    # ── Test 3: Strong regime ──
    print("\n=== Strong regime (I0 ∈ [10, 1000]) ===")
    data_strong = generate_weak(n_hkl=2000, beta=beta, I0_lo=10., I0_hi=1000.,
                                device=device)
    true_s = data_strong["I0_true"].cpu().numpy()
    lt_s = np.log(true_s)

    r_gg_s = _r(estimate_gg_lbfgs(data_strong, beta, max_iter=200).cpu().numpy(), lt_s)
    print(f"  GG LBFGS:     r={r_gg_s:.4f}")

    for N in [31, 51]:
        est_cl, _ = estimate_poisson_midpoint(data_strong, beta, N=N, max_iter=50)
        r_cl = _r(est_cl.cpu().numpy(), lt_s)
        est_ad = estimate_poisson_adaptive(data_strong, beta, N=N, max_iter=50)
        r_ad = _r(est_ad.cpu().numpy(), lt_s)
        print(f"  N={N:>3}  clamp: r={r_cl:.4f}   adaptive: r={r_ad:.4f}   Δ={r_ad-r_cl:+.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (title, data, true, lt) in zip(axes, [
        ("Realistic (I₀ med=9.2)", data_real, true_r, lt_r),
        ("Weak (I₀ ∈ [0.1, 3])", data_weak, true_w, lt_w),
        ("Strong (I₀ ∈ [10, 1000])", data_strong, true_s, lt_s),
    ]):
        est_gg = estimate_gg_lbfgs(data, beta, max_iter=200).cpu().numpy()
        est_ad = estimate_poisson_adaptive(data, beta, N=51, max_iter=50).cpu().numpy()
        r_g = _r(est_gg, lt)
        r_a = _r(est_ad, lt)

        lo = true.min() * 0.3
        hi = true.max() * 3
        ax.scatter(true, est_gg, s=3, alpha=0.15, c="steelblue", edgecolors="none",
                   label=f"GG LBFGS ({r_g:.4f})")
        ax.scatter(true, est_ad, s=3, alpha=0.15, c="darkorchid", edgecolors="none",
                   label=f"Adaptive N=51 ({r_a:.4f})")
        ax.plot([lo, hi], [lo, hi], "k-", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8, markerscale=3)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Adaptive log-linear quadrature: works across all regimes (N=51)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "adaptive_loglinear.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
