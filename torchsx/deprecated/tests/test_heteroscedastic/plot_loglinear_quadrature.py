#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Log-linear quadrature via softplus mapping.

Uniform grid in t-space → softplus → signal space:
  - t ≪ 0: signal ≈ exp(t) → log-spaced (good ε-resolution near bg)
  - t ≫ 0: signal ≈ t       → linear-spaced (good λ-resolution at peak)

This gives a single smooth grid that handles both weak and strong obs.
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
from plot_convergence_sweep import estimate_poisson_midpoint
from verify_quadrature import _log_gg_density, _log_poi, mc_log_marginal


def _softplus_inv(y):
    """Numerically stable inverse of softplus: log(exp(y) - 1).
    For y > 20, returns y (since exp(y) >> 1).
    """
    if isinstance(y, torch.Tensor):
        out = torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(min=1e-8))))
        return out
    else:
        if y > 20:
            return y
        return math.log(math.expm1(max(y, 1e-8)))


def build_loglinear_grid(n_ph, bg, N, n_sigma=5, device=None):
    """Build log-linear quadrature grid for a batch of observations.

    Args:
        n_ph: (N_obs,) photon counts
        bg: background level (scalar)
        N: number of quadrature points
        n_sigma: Poisson coverage in sigma units

    Returns:
        lam: (N_obs, N) quadrature points
        widths: (N_obs, N) midpoint-rule cell widths
    """
    if device is None:
        device = n_ph.device

    sigma = n_ph.clamp(min=1).sqrt()
    signal_max = (n_ph - bg + n_sigma * sigma).clamp(min=1.0)
    signal_min = 0.01

    t_hi = _softplus_inv(signal_max)
    t_lo = _softplus_inv(torch.tensor(signal_min, device=device))

    frac = torch.linspace(0, 1, N, device=device)
    t = t_lo + (t_hi[:, None] - t_lo) * frac[None, :]

    signal = F.softplus(t)
    lam = signal + bg

    widths = torch.zeros_like(lam)
    widths[:, 1:-1] = (lam[:, 2:] - lam[:, :-2]) / 2
    widths[:, 0] = (lam[:, 1] - lam[:, 0]) / 2
    widths[:, -1] = (lam[:, -1] - lam[:, -2]) / 2

    return lam, widths, signal


def estimate_poisson_loglinear(data, beta, N=51, n_sigma=5, max_iter=50):
    """Poisson MLE with log-linear softplus quadrature grid."""
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

    lam, widths, signal = build_loglinear_grid(n_ph, bg, N, n_sigma, device=dev)

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

    I0 = 3.0; alpha = 1.2; beta = 1.3; bg = 10.0

    # ── Part 1: Integral accuracy ──
    print("=== Integral accuracy ===")
    print(f"{'I_obs':>6} {'n_ph':>5} | {'MC':>8} | {'ll31':>8} {'ll51':>8} {'cl31':>8} {'cl101':>8}")
    print("-" * 65)

    for I_obs in [-8, -5, 0, 1, 3, 10, 50, 100]:
        n_ph_v = max(I_obs + bg, 0.0)
        mc = mc_log_marginal(n_ph_v, I0, alpha, beta, bg, n_mc=2_000_000)

        # Log-linear
        n_ph_t = torch.tensor([n_ph_v])
        for N, label in [(31, "ll31"), (51, "ll51")]:
            lam, widths, signal = build_loglinear_grid(n_ph_t, bg, N)
            sig = signal[0]
            eps = torch.log(sig / I0)
            log_gg = _log_gg_density(eps, alpha, beta)
            log_poi = _log_poi(n_ph_v, lam[0])
            log_jac = -torch.log(sig)
            log_w = torch.log(widths[0].clamp(min=1e-30))
            val = float(torch.logsumexp(log_gg + log_poi + log_jac + log_w, dim=0))
            if label == "ll31":
                ll31 = val
            else:
                ll51 = val

        # Clamp baselines
        from plot_integration_boxes import midpoint_log_marginal
        cl31 = midpoint_log_marginal(n_ph_v, I0, alpha, beta, bg, N=31)[0]
        cl101 = midpoint_log_marginal(n_ph_v, I0, alpha, beta, bg, N=101)[0]

        print(f"{I_obs:>6} {n_ph_v:>5.0f} | {mc:>8.3f} | {ll31:>8.3f} {ll51:>8.3f} {cl31:>8.3f} {cl101:>8.3f}")

    # ── Part 2: Estimation comparison ──
    print("\n=== Estimation comparison ===")
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
    r_irls = _r(estimate_irls(data, beta).cpu().numpy())
    print(f"  GG LBFGS:       r={r_gg:.4f}")
    print(f"  IRLS:           r={r_irls:.4f}")

    for N in [31, 41, 51, 61, 81, 101]:
        est_cl, _ = estimate_poisson_midpoint(data, beta, N=N, max_iter=50)
        r_cl = _r(est_cl.cpu().numpy())
        est_ll = estimate_poisson_loglinear(data, beta, N=N, max_iter=50)
        r_ll = _r(est_ll.cpu().numpy())
        print(f"  N={N:>3}  clamp: r={r_cl:.4f}   loglinear: r={r_ll:.4f}   Δ={r_ll-r_cl:+.4f}")

    # ── Part 3: Visualization ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    test_vis = [
        ("Weak: I_obs=0, n=10", 10.0),
        ("Medium: I_obs=3, n=13", 13.0),
        ("Strong: I_obs=50, n=60", 60.0),
    ]

    eps_dense = torch.linspace(-8, 5, 500)
    gg_dense = torch.exp(_log_gg_density(eps_dense, alpha, beta))
    gg_norm = gg_dense / gg_dense.max()

    for col, (label, n_ph_v) in enumerate(test_vis):
        n_ph_t = torch.tensor([n_ph_v])

        # Top row: grid points in ε-space
        ax = axes[0, col]
        ax.plot(eps_dense.numpy(), gg_norm.numpy(), "seagreen", lw=1.5, label="GenGauss")

        # Clamp grid
        sigma_v = max(math.sqrt(max(n_ph_v, 1)), 1.0)
        k = torch.linspace(-5, 5, 31)
        lam_cl = (n_ph_v + sigma_v * k).clamp(min=bg + 0.1)
        sig_cl = (lam_cl - bg).clamp(min=1e-8)
        eps_cl = torch.log(sig_cl / I0)
        ax.plot(eps_cl.numpy(), np.ones(31) * 1.1, "|", color="tomato",
                ms=12, mew=2, label="clamp N=31")

        # Log-linear grid
        lam_ll, _, sig_ll = build_loglinear_grid(n_ph_t, bg, 31)
        eps_ll = torch.log(sig_ll[0] / I0)
        ax.plot(eps_ll.numpy(), np.ones(31) * 0.9, "|", color="steelblue",
                ms=12, mew=2, label="loglinear N=31")

        # Log-linear N=51
        lam_ll51, _, sig_ll51 = build_loglinear_grid(n_ph_t, bg, 51)
        eps_ll51 = torch.log(sig_ll51[0] / I0)
        ax.plot(eps_ll51.numpy(), np.ones(51) * 0.75, "|", color="darkorchid",
                ms=10, mew=1.5, label="loglinear N=51")

        ax.set_xlabel("ε = log(signal / I₀)")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.set_xlim(-8, 5)
        ax.set_ylim(0.5, 1.2)
        ax.grid(True, alpha=0.2)

        # Bottom row: ε-spacing vs position
        ax = axes[1, col]
        # Clamp spacing
        d_eps_cl = torch.diff(eps_cl)
        ax.plot(eps_cl[:-1].numpy(), d_eps_cl.numpy(), "o-", color="tomato",
                ms=3, lw=1, label="clamp N=31")
        # Log-linear spacing
        d_eps_ll = torch.diff(eps_ll)
        ax.plot(eps_ll[:-1].numpy(), d_eps_ll.numpy(), "o-", color="steelblue",
                ms=3, lw=1, label="loglinear N=31")
        d_eps_ll51 = torch.diff(eps_ll51)
        ax.plot(eps_ll51[:-1].numpy(), d_eps_ll51.numpy(), "o-", color="darkorchid",
                ms=3, lw=1, label="loglinear N=51")

        ax.axhline(alpha, color="black", ls="--", lw=1, label=f"α={alpha}")
        ax.axhline(alpha / 5, color="gray", ls=":", lw=1, label="α/5")
        ax.set_xlabel("ε")
        ax.set_ylabel("Δε (spacing)")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.set_xlim(-8, 5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Log-linear (softplus) quadrature grid  (I₀={I0}, α={alpha}, β={beta}, bg={bg})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "loglinear_quadrature.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
