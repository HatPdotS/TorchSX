#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Visualize gradient noise from coarse quadrature.

For individual observations, plot:
  1. The NLL as a function of log(I0) at different N_quad
  2. The gradient d(NLL)/d(log I0) — should be jagged for small N
  3. The softmax weights across quadrature points
"""

from __future__ import annotations
import math, os, sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from verify_quadrature import _log_gg_density, _log_poi


def single_obs_nll_and_grad(n_ph, I0_grid, alpha, beta, bg, N, n_sigma=5):
    """Compute NLL and its gradient for a single observation over an I0 grid.

    Returns (nll, grad, softmax_at_true) arrays of shape (len(I0_grid),).
    """
    sigma = max(math.sqrt(max(n_ph, 1)), 1.0)
    k = torch.linspace(-n_sigma, n_sigma, N)
    lam = n_ph + sigma * k
    lam = lam.clamp(min=bg + 0.1)

    # Cell widths
    widths = torch.zeros(N)
    widths[1:-1] = (lam[2:] - lam[:-2]) / 2
    widths[0] = (lam[1] - lam[0]) / 2
    widths[-1] = (lam[-1] - lam[-2]) / 2

    signal = lam - bg  # (K,)
    log_poi = n_ph * torch.log(lam) - lam - float(torch.lgamma(torch.tensor(n_ph + 1.0)))
    log_jac = -torch.log(signal)
    log_w = torch.log(widths.clamp(min=1e-30))
    log_poi_jac_w = log_poi + log_jac + log_w  # (K,) — I0-independent

    nll_out = []
    grad_out = []
    softmax_out = []
    n_dominant_out = []

    for I0 in I0_grid:
        I0_t = torch.tensor(float(I0), requires_grad=True)
        eps = torch.log(signal / I0_t)  # (K,)
        log_gg = _log_gg_density(eps, alpha, beta)  # (K,)
        log_integrand = log_gg + log_poi_jac_w  # (K,)
        log_p = torch.logsumexp(log_integrand, dim=0)
        nll = -log_p
        nll.backward()

        # Softmax weights
        sm = torch.softmax(log_integrand, dim=0)

        nll_out.append(float(nll))
        grad_out.append(float(I0_t.grad))
        softmax_out.append(sm.detach().numpy().copy())
        n_dominant_out.append(int((sm > 0.01).sum()))

    return (np.array(nll_out), np.array(grad_out),
            np.array(softmax_out), np.array(n_dominant_out))


def main():
    alpha = 1.2
    beta = 1.3
    bg = 10.0
    I0_true = 3.0

    I0_grid = np.logspace(-0.3, 1.5, 300)

    # Test a weak and a strong observation
    test_cases = [
        ("Weak: I_obs=1, n=11", 11),
        ("Medium: I_obs=10, n=20", 20),
        ("Strong: I_obs=50, n=60", 60),
        ("Very strong: I_obs=200, n=210", 210),
    ]
    N_vals = [11, 31, 101, 201]
    colors = ["tomato", "darkorchid", "steelblue", "seagreen"]

    fig, axes = plt.subplots(4, 3, figsize=(20, 22))

    for row, (label, n_ph) in enumerate(test_cases):
        # Col 0: NLL landscape
        ax_nll = axes[row, 0]
        # Col 1: Gradient
        ax_grad = axes[row, 1]
        # Col 2: Number of dominant points and softmax entropy
        ax_dom = axes[row, 2]

        for N, color in zip(N_vals, colors):
            nll, grad, sm, n_dom = single_obs_nll_and_grad(
                n_ph, I0_grid, alpha, beta, bg, N)

            # Normalize NLL for overlay
            nll_plot = nll - nll.min()
            ax_nll.plot(I0_grid, nll_plot, color=color, lw=1.5,
                       label=f"N={N}", alpha=0.8)

            ax_grad.plot(I0_grid, grad, color=color, lw=1.0,
                        label=f"N={N}", alpha=0.8)

            # Softmax entropy = -Σ p log p (higher = more spread out)
            entropy = np.array([-np.sum(s * np.log(s + 1e-30)) for s in sm])
            ax_dom.plot(I0_grid, n_dom, color=color, lw=1.5,
                       label=f"N={N}", alpha=0.8)

        for ax in [ax_nll, ax_grad, ax_dom]:
            ax.axvline(I0_true, color="red", ls="--", lw=1, alpha=0.5)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.2)

        ax_nll.set_ylabel("NLL − min")
        ax_nll.set_ylim(0, 5)
        ax_nll.set_title(f"{label}")
        ax_nll.legend(fontsize=7)

        ax_grad.set_ylabel("∂NLL/∂log(I₀)")
        ax_grad.axhline(0, color="gray", ls=":", lw=0.5)
        ax_grad.set_title(f"Gradient")

        ax_dom.set_ylabel("# dominant pts (sm>1%)")
        ax_dom.set_title(f"Effective quadrature points")

        if row == 3:
            for ax in [ax_nll, ax_grad, ax_dom]:
                ax.set_xlabel("I₀")

    fig.suptitle(f"Gradient noise from coarse quadrature  (α={alpha}, β={beta}, bg={bg}, I₀_true={I0_true})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "gradient_noise.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
