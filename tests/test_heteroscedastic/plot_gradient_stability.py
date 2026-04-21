#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Measure gradient stability vs I_obs by finite-difference comparison.

For each I_obs, compute the analytical gradient (via autograd through the
midpoint quadrature) and a finite-difference gradient. The discrepancy
reveals where the quadrature discretization corrupts the gradient.

We evaluate at I0 = I0_true = 3 with a small perturbation δ = 1e-4.
"""

from __future__ import annotations
import math, os, sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from verify_quadrature import _log_gg_density, _log_poi


def compute_log_marginal(n_ph, log_I0, alpha, beta, bg, N, n_sigma=5):
    """Midpoint-rule log-marginal for a single observation. Differentiable."""
    I0 = torch.exp(log_I0)
    sigma = max(math.sqrt(max(n_ph, 1)), 1.0)
    k = torch.linspace(-n_sigma, n_sigma, N)
    lam = n_ph + sigma * k
    lam = lam.clamp(min=bg + 0.1)

    widths = torch.zeros(N)
    widths[1:-1] = (lam[2:] - lam[:-2]) / 2
    widths[0] = (lam[1] - lam[0]) / 2
    widths[-1] = (lam[-1] - lam[-2]) / 2

    signal = lam - bg
    eps = torch.log(signal / I0)
    log_gg = _log_gg_density(eps, alpha, beta)
    log_poi = _log_poi(n_ph, lam)
    log_jac = -torch.log(signal)
    log_w = torch.log(widths.clamp(min=1e-30))
    log_integrand = log_gg + log_poi + log_jac + log_w
    return torch.logsumexp(log_integrand, dim=0)


def main():
    alpha = 1.2
    beta = 1.3
    bg = 10.0
    I0_true = 3.0
    delta = 1e-4

    I_obs_vals = np.concatenate([
        np.arange(-9, 0, 1),
        np.arange(0, 20, 1),
        np.arange(20, 110, 5),
        np.arange(110, 510, 20),
    ])
    N_vals = [11, 31, 51, 101, 201]
    colors = ["tomato", "darkorchid", "orange", "steelblue", "seagreen"]

    results = {N: [] for N in N_vals}

    for I_obs in I_obs_vals:
        n_ph = max(I_obs + bg, 0.0)
        for N in N_vals:
            # Autograd gradient
            log_I0 = torch.tensor(math.log(I0_true), requires_grad=True)
            log_p = compute_log_marginal(n_ph, log_I0, alpha, beta, bg, N)
            log_p.backward()
            grad_auto = float(log_I0.grad)

            # Finite difference gradient
            with torch.no_grad():
                log_I0_plus = torch.tensor(math.log(I0_true) + delta)
                log_I0_minus = torch.tensor(math.log(I0_true) - delta)
                lp_plus = float(compute_log_marginal(n_ph, log_I0_plus, alpha, beta, bg, N))
                lp_minus = float(compute_log_marginal(n_ph, log_I0_minus, alpha, beta, bg, N))
            grad_fd = (lp_plus - lp_minus) / (2 * delta)

            results[N].append({
                "I_obs": I_obs,
                "n_ph": n_ph,
                "grad_auto": grad_auto,
                "grad_fd": grad_fd,
                "grad_err": abs(grad_auto - grad_fd),
            })

    # Use N=201 finite-difference as "truth" reference
    grad_ref = np.array([r["grad_fd"] for r in results[201]])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Gradient vs I_obs for each N
    ax = axes[0, 0]
    for N, color in zip(N_vals, colors):
        grads = np.array([r["grad_fd"] for r in results[N]])
        ax.plot(I_obs_vals, grads, color=color, lw=1.2, label=f"N={N}", alpha=0.8)
    ax.set_xlabel("I_obs")
    ax.set_ylabel("∂ log p / ∂ log I₀")
    ax.set_title("Gradient (finite diff) vs I_obs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color="gray", ls=":", lw=0.5)

    # (0,1) Gradient error vs reference (N=201)
    ax = axes[0, 1]
    for N, color in zip(N_vals[:-1], colors[:-1]):
        grads = np.array([r["grad_fd"] for r in results[N]])
        err = np.abs(grads - grad_ref)
        ax.plot(I_obs_vals, err, color=color, lw=1.2, label=f"N={N}", alpha=0.8)
    ax.set_xlabel("I_obs")
    ax.set_ylabel("|grad(N) − grad(N=201)|")
    ax.set_title("Gradient error vs N=201 reference")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (1,0) Autograd vs finite-diff discrepancy (tests autograd correctness)
    ax = axes[1, 0]
    for N, color in zip(N_vals, colors):
        errs = np.array([r["grad_err"] for r in results[N]])
        ax.plot(I_obs_vals, errs, color=color, lw=1.2, label=f"N={N}", alpha=0.8)
    ax.set_xlabel("I_obs")
    ax.set_ylabel("|autograd − finite diff|")
    ax.set_title("Autograd vs finite-diff (sanity check)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (1,1) Effective ε-spacing at Poisson peak vs I_obs
    ax = axes[1, 1]
    for N, color in zip(N_vals, colors):
        eps_spacing = []
        for I_obs in I_obs_vals:
            n_ph = max(I_obs + bg, 0.0)
            sigma = max(math.sqrt(max(n_ph, 1)), 1.0)
            d_lam = 2 * 5 * sigma / (N - 1)
            signal_at_peak = max(n_ph - bg, 0.1)
            d_eps_peak = d_lam / max(signal_at_peak, 0.1)
            eps_spacing.append(d_eps_peak)
        ax.plot(I_obs_vals, eps_spacing, color=color, lw=1.5, label=f"N={N}")
    ax.axhline(alpha, color="black", ls="--", lw=1, label=f"α={alpha}")
    ax.axhline(alpha / 5, color="gray", ls=":", lw=1, label=f"α/5={alpha/5:.2f}")
    ax.set_xlabel("I_obs")
    ax.set_ylabel("Δε at Poisson peak")
    ax.set_title("ε-spacing at peak (need ≪ α for smooth grad)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Gradient stability vs I_obs  (I₀={I0_true}, α={alpha}, β={beta}, bg={bg})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "gradient_stability.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
