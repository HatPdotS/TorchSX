#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Tune the log-linear quadrature: sweep signal_min and softplus scale."""

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
from plot_convergence_sweep import estimate_poisson_midpoint


def _softplus_inv(y):
    if isinstance(y, torch.Tensor):
        return torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(min=1e-8))))
    return y if y > 20 else math.log(math.expm1(max(y, 1e-8)))


def estimate_poisson_loglinear(data, beta, N=51, n_sigma=5, max_iter=50,
                                signal_min=0.01, scale=1.0):
    """Poisson MLE with tunable log-linear quadrature."""
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
    signal_max = (n_ph - bg + n_sigma * sigma).clamp(min=1.0)

    t_hi = _softplus_inv(signal_max / scale)
    t_lo = _softplus_inv(torch.tensor(signal_min / scale, device=dev))

    frac = torch.linspace(0, 1, N, device=dev)
    t = t_lo + (t_hi[:, None] - t_lo) * frac[None, :]
    signal = F.softplus(t) * scale
    lam = signal + bg

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

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)
    beta = 1.3; n_hkl = 5000

    data = generate_realistic(real_params=real_params, n_hkl=n_hkl,
                              beta=beta, device=device)
    true = data["I0_true"].cpu().numpy()
    lt = np.log(true)
    def _r(est):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    r_gg = _r(estimate_gg_lbfgs(data, beta, max_iter=200).cpu().numpy())
    print(f"  GG LBFGS:  r={r_gg:.4f}")

    # Sweep signal_min at N=51
    print("\n--- Sweep signal_min (N=51, scale=1.0) ---")
    smin_vals = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    r_smin = []
    for smin in smin_vals:
        est = estimate_poisson_loglinear(data, beta, N=51, signal_min=smin, scale=1.0)
        r = _r(est.cpu().numpy())
        r_smin.append(r)
        print(f"  signal_min={smin:<6}  r={r:.4f}")

    # Sweep scale at N=51, best signal_min
    best_smin = smin_vals[np.argmax(r_smin)]
    print(f"\n--- Sweep scale (N=51, signal_min={best_smin}) ---")
    scale_vals = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    r_scale = []
    for sc in scale_vals:
        est = estimate_poisson_loglinear(data, beta, N=51, signal_min=best_smin, scale=sc)
        r = _r(est.cpu().numpy())
        r_scale.append(r)
        print(f"  scale={sc:<6}  r={r:.4f}")

    # Best combo at different N
    best_scale = scale_vals[np.argmax(r_scale)]
    print(f"\n--- Best combo: signal_min={best_smin}, scale={best_scale} ---")
    for N in [31, 41, 51, 61, 81, 101]:
        est_ll = estimate_poisson_loglinear(data, beta, N=N,
                                             signal_min=best_smin, scale=best_scale)
        r_ll = _r(est_ll.cpu().numpy())
        est_cl, _ = estimate_poisson_midpoint(data, beta, N=N, max_iter=50)
        r_cl = _r(est_cl.cpu().numpy())
        print(f"  N={N:>3}  loglinear: r={r_ll:.4f}   clamp: r={r_cl:.4f}   Δ={r_ll-r_cl:+.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(smin_vals, r_smin, "o-", color="darkorchid", lw=2)
    ax.axhline(r_gg, color="steelblue", ls="--", lw=1.5, label=f"GG LBFGS ({r_gg:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("signal_min")
    ax.set_ylabel("r")
    ax.set_title("signal_min sweep (N=51, scale=1.0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(scale_vals, r_scale, "o-", color="darkorchid", lw=2)
    ax.axhline(r_gg, color="steelblue", ls="--", lw=1.5, label=f"GG LBFGS ({r_gg:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("scale")
    ax.set_ylabel("r")
    ax.set_title(f"scale sweep (N=51, signal_min={best_smin})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.text(0.05, 0.95, f"Best: signal_min={best_smin}, scale={best_scale}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="wheat"))
    ax.text(0.05, 0.80, f"GG LBFGS: r={r_gg:.4f}",
            transform=ax.transAxes, fontsize=10, va="top")
    ax.set_title("Summary")
    ax.axis("off")

    fig.suptitle("Log-linear quadrature tuning", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "loglinear_tune.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
