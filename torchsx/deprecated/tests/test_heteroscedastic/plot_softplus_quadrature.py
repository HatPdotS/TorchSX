#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare clamp vs softplus quadrature grids.

Instead of:  lam = (n_ph + k·σ).clamp(min=bg+0.1)
Use:         signal = softplus(n_ph + k·σ - bg); lam = signal + bg

This smoothly compresses sampling near bg, giving better ε-resolution
for weak observations without affecting strong ones.
"""

from __future__ import annotations
import math, os, sys, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls, _log_gg_norm,
)
from plot_midpoint_comparison import estimate_gg_lbfgs
from verify_quadrature import _log_gg_density, _log_poi, mc_log_marginal


# ── Single-obs diagnostics ──

def _build_grid_clamp(n_ph, bg, N, n_sigma=5):
    sigma = max(math.sqrt(max(n_ph, 1)), 1.0)
    k = torch.linspace(-n_sigma, n_sigma, N)
    lam = (n_ph + sigma * k).clamp(min=bg + 0.1)
    return lam

def _build_grid_softplus(n_ph, bg, N, n_sigma=5):
    sigma = max(math.sqrt(max(n_ph, 1)), 1.0)
    k = torch.linspace(-n_sigma, n_sigma, N)
    raw_signal = n_ph + sigma * k - bg
    signal = F.softplus(raw_signal)
    return signal + bg

def _midpoint_widths(lam):
    N = len(lam)
    w = torch.zeros(N)
    w[1:-1] = (lam[2:] - lam[:-2]) / 2
    w[0] = (lam[1] - lam[0]) / 2
    w[-1] = (lam[-1] - lam[-2]) / 2
    return w

def _log_marginal(n_ph, I0, alpha, beta, bg, lam):
    w = _midpoint_widths(lam)
    signal = (lam - bg).clamp(min=1e-8)
    eps = torch.log(signal / I0)
    log_gg = _log_gg_density(eps, alpha, beta)
    log_poi = _log_poi(n_ph, lam)
    log_jac = -torch.log(signal)
    log_w = torch.log(w.clamp(min=1e-30))
    return float(torch.logsumexp(log_gg + log_poi + log_jac + log_w, dim=0))


# ── Full estimator with softplus grid ──

def estimate_poisson_softplus(data, beta, N=31, n_sigma=5, max_iter=50):
    """Poisson MLE with softplus-compressed quadrature grid."""
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
    k = torch.linspace(-n_sigma, n_sigma, N, device=dev)
    raw_signal = n_ph[:, None] + sigma[:, None] * k[None, :] - bg
    signal = F.softplus(raw_signal)  # smooth, always positive
    lam = signal + bg

    # Midpoint widths from actual (non-uniform) λ positions
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Part 1: Single-obs integral accuracy ──
    I0 = 3.0; alpha = 1.2; beta = 1.3; bg = 10.0

    print("=== Integral accuracy: clamp vs softplus ===")
    print(f"{'I_obs':>6} {'n_ph':>5} | {'MC':>8} | {'clamp31':>8} {'sp31':>8} {'clamp101':>8} {'sp101':>8}")
    print("-" * 70)

    for I_obs in [-8, -5, -2, 0, 1, 3, 5, 10, 50, 100]:
        n_ph = max(I_obs + bg, 0.0)
        mc = mc_log_marginal(n_ph, I0, alpha, beta, bg, n_mc=2_000_000)
        c31 = _log_marginal(n_ph, I0, alpha, beta, bg, _build_grid_clamp(n_ph, bg, 31))
        s31 = _log_marginal(n_ph, I0, alpha, beta, bg, _build_grid_softplus(n_ph, bg, 31))
        c101 = _log_marginal(n_ph, I0, alpha, beta, bg, _build_grid_clamp(n_ph, bg, 101))
        s101 = _log_marginal(n_ph, I0, alpha, beta, bg, _build_grid_softplus(n_ph, bg, 101))
        print(f"{I_obs:>6} {n_ph:>5.0f} | {mc:>8.3f} | {c31:>8.3f} {s31:>8.3f} {c101:>8.3f} {s101:>8.3f}")

    # ── Part 2: Gradient stability comparison ──
    print("\n=== Gradient stability: clamp vs softplus (at I0_true) ===")
    delta = 1e-4

    I_obs_test = list(range(-5, 20)) + list(range(20, 110, 10))
    grad_ref = []  # N=201 clamp as reference

    for I_obs in I_obs_test:
        n_ph = max(I_obs + bg, 0.0)
        lam201 = _build_grid_clamp(n_ph, bg, 201)
        lp = _log_marginal(n_ph, I0 * math.exp(delta), alpha, beta, bg, lam201)
        lm = _log_marginal(n_ph, I0 * math.exp(-delta), alpha, beta, bg, lam201)
        grad_ref.append((lp - lm) / (2 * delta))

    grad_ref = np.array(grad_ref)

    configs = [
        ("clamp N=31", lambda n, bg: _build_grid_clamp(n, bg, 31)),
        ("softplus N=31", lambda n, bg: _build_grid_softplus(n, bg, 31)),
        ("clamp N=101", lambda n, bg: _build_grid_clamp(n, bg, 101)),
        ("softplus N=101", lambda n, bg: _build_grid_softplus(n, bg, 101)),
    ]

    grad_errors = {name: [] for name, _ in configs}
    for I_obs in I_obs_test:
        n_ph = max(I_obs + bg, 0.0)
        for name, builder in configs:
            lam = builder(n_ph, bg)
            lp = _log_marginal(n_ph, I0 * math.exp(delta), alpha, beta, bg, lam)
            lm = _log_marginal(n_ph, I0 * math.exp(-delta), alpha, beta, bg, lam)
            g = (lp - lm) / (2 * delta)
            grad_errors[name].append(g)

    # ── Part 3: Full estimation comparison ──
    print("\n=== Full estimation comparison ===")
    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)

    n_hkl = 5000
    data = generate_realistic(real_params=real_params, n_hkl=n_hkl,
                              beta=beta, device=device)
    true = data["I0_true"].cpu().numpy()
    lt = np.log(true)
    def _r(est):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    r_gg = _r(estimate_gg_lbfgs(data, beta, max_iter=200).cpu().numpy())
    print(f"  GG LBFGS:        r={r_gg:.4f}")

    for N in [31, 51, 101]:
        from plot_convergence_sweep import estimate_poisson_midpoint
        est_clamp, _ = estimate_poisson_midpoint(data, beta, N=N, max_iter=50)
        r_clamp = _r(est_clamp.cpu().numpy())
        est_sp = estimate_poisson_softplus(data, beta, N=N, max_iter=50)
        r_sp = _r(est_sp.cpu().numpy())
        print(f"  N={N:>3}  clamp: r={r_clamp:.4f}   softplus: r={r_sp:.4f}   Δ={r_sp-r_clamp:+.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Grid visualization for weak obs
    ax = axes[0, 0]
    n_ph_weak = 11.0  # I_obs = 1
    lam_c = _build_grid_clamp(n_ph_weak, bg, 31)
    lam_s = _build_grid_softplus(n_ph_weak, bg, 31)
    sig_c = (lam_c - bg).numpy()
    sig_s = (lam_s - bg).numpy()
    eps_c = np.log(np.clip(sig_c, 1e-8, None) / I0)
    eps_s = np.log(np.clip(sig_s, 1e-8, None) / I0)
    ax.plot(eps_c, np.ones_like(eps_c) * 1.1, "|", color="tomato", ms=15, mew=2, label="clamp")
    ax.plot(eps_s, np.ones_like(eps_s) * 0.9, "|", color="steelblue", ms=15, mew=2, label="softplus")
    # GenGauss overlay
    eps_dense = torch.linspace(-6, 4, 500)
    gg_dense = torch.exp(_log_gg_density(eps_dense, alpha, beta))
    ax.plot(eps_dense.numpy(), gg_dense.numpy() / gg_dense.max().item() * 0.7 + 0.3,
            "seagreen", lw=1.5, label="GenGauss")
    ax.set_xlabel("ε = log(signal / I₀)")
    ax.set_title(f"Grid points in ε-space (I_obs=1, n={n_ph_weak:.0f})")
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 4)
    ax.grid(True, alpha=0.2)

    # (0,1) Grid visualization for strong obs
    ax = axes[0, 1]
    n_ph_strong = 60.0  # I_obs = 50
    lam_c = _build_grid_clamp(n_ph_strong, bg, 31)
    lam_s = _build_grid_softplus(n_ph_strong, bg, 31)
    sig_c = (lam_c - bg).numpy()
    sig_s = (lam_s - bg).numpy()
    eps_c = np.log(np.clip(sig_c, 1e-8, None) / I0)
    eps_s = np.log(np.clip(sig_s, 1e-8, None) / I0)
    ax.plot(eps_c, np.ones_like(eps_c) * 1.1, "|", color="tomato", ms=15, mew=2, label="clamp")
    ax.plot(eps_s, np.ones_like(eps_s) * 0.9, "|", color="steelblue", ms=15, mew=2, label="softplus")
    ax.plot(eps_dense.numpy(), gg_dense.numpy() / gg_dense.max().item() * 0.7 + 0.3,
            "seagreen", lw=1.5, label="GenGauss")
    ax.set_xlabel("ε = log(signal / I₀)")
    ax.set_title(f"Grid points in ε-space (I_obs=50, n={n_ph_strong:.0f})")
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 4)
    ax.grid(True, alpha=0.2)

    # (1,0) Gradient error comparison
    ax = axes[1, 0]
    colors_cfg = ["tomato", "steelblue", "lightsalmon", "lightsteelblue"]
    for (name, _), col in zip(configs, colors_cfg):
        ge = np.abs(np.array(grad_errors[name]) - grad_ref)
        ax.plot(I_obs_test, ge, lw=1.5, label=name, color=col)
    ax.set_xlabel("I_obs")
    ax.set_ylabel("|grad − ref|")
    ax.set_yscale("log")
    ax.set_title("Gradient error vs N=201 reference")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (1,1) ε-spacing at Poisson peak
    ax = axes[1, 1]
    I_obs_range = np.arange(-5, 200)
    builders = [("clamp", _build_grid_clamp, "tomato"),
                ("softplus", _build_grid_softplus, "steelblue")]
    for N, ls in [(31, "-"), (101, "--")]:
        for bname, builder, col in builders:
            spacings = []
            for Io in I_obs_range:
                n_ph_v = max(Io + bg, 0.0)
                lam = builder(n_ph_v, bg, N)
                sig = (lam - bg).clamp(min=1e-8)
                eps_arr = torch.log(sig / I0)
                idx_near = (eps_arr.abs()).argmin()
                idx = max(1, min(idx_near.item(), len(eps_arr) - 2))
                d_eps = float(eps_arr[idx + 1] - eps_arr[idx - 1]) / 2
                spacings.append(abs(d_eps))
            ax.plot(I_obs_range, spacings, color=col, ls=ls, lw=1.5,
                    label=f"{bname} N={N}")

    ax.axhline(alpha, color="black", ls="--", lw=1, label=f"α={alpha}")
    ax.axhline(alpha / 5, color="gray", ls=":", lw=1, label=f"α/5")
    ax.set_xlabel("I_obs")
    ax.set_ylabel("Δε near ε=0")
    ax.set_yscale("log")
    ax.set_title("ε-spacing near GenGauss peak")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Softplus vs clamp quadrature grid",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "softplus_quadrature.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
