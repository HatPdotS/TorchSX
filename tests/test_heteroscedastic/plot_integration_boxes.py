#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Visualize the overlap integral with proper midpoint-rule integration boxes.

For each observation, place N quadrature points at ±5σ_Poisson in λ-space,
compute cell widths via midpoint rule, evaluate GG × Poi × Jacobian at each,
and draw integration rectangles overlaid on the continuous curves.
"""

from __future__ import annotations
import math, os, sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Reuse existing helpers
sys.path.insert(0, os.path.dirname(__file__))
from verify_quadrature import _log_gg_density, _log_poi, mc_log_marginal


def midpoint_log_marginal(n_ph, I0, alpha, beta, bg, N=31, n_sigma=5):
    """Midpoint-rule integration in λ-space with proper cell widths.

    Returns (log_p, lam_pts, widths, log_integrand_density) for visualization.
    """
    sigma = max(math.sqrt(max(n_ph, 1)), 1.0)
    lam_pts = torch.linspace(n_ph - n_sigma * sigma, n_ph + n_sigma * sigma, N)
    lam_pts = lam_pts.clamp(min=bg + 0.1)

    # Cell widths via midpoint rule
    widths = torch.zeros(N)
    for k in range(N):
        if k == 0:
            widths[k] = (lam_pts[1] - lam_pts[0]) / 2
        elif k == N - 1:
            widths[k] = (lam_pts[-1] - lam_pts[-2]) / 2
        else:
            widths[k] = (lam_pts[k + 1] - lam_pts[k - 1]) / 2

    signal = (lam_pts - bg).clamp(min=1e-8)
    eps = torch.log(signal / I0)

    log_gg = _log_gg_density(eps, alpha, beta)
    log_poi = _log_poi(n_ph, lam_pts)
    log_jac = -torch.log(signal)  # dε/dλ = 1/(λ - bg)

    # Integrand density (per unit λ) = GG(ε) · Poi(n; λ) · |dε/dλ|
    log_density = log_gg + log_poi + log_jac

    # Full integrand with cell width
    log_integrand = log_density + torch.log(widths.clamp(min=1e-30))
    log_p = float(torch.logsumexp(log_integrand, dim=0))

    return log_p, lam_pts, widths, log_density


def main():
    I0 = 3.0
    alpha = 1.2
    beta = 1.3
    bg = 10.0
    N = 31

    test_obs = [-5, 0, 3, 10, 50, 100]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    print(f"Parameters: I0={I0}, α={alpha}, β={beta}, bg={bg}, N={N}")
    print(f"\n{'I_obs':>6} {'n_ph':>5} | {'MC':>10} {'Boxes':>10} | {'diff':>7}")
    print("-" * 55)

    for i, I_obs in enumerate(test_obs):
        n_ph = max(I_obs + bg, 0)

        # Our midpoint-rule integration
        log_p_box, lam_pts, widths, log_density = midpoint_log_marginal(
            n_ph, I0, alpha, beta, bg, N=N)

        # MC ground truth
        log_p_mc = mc_log_marginal(n_ph, I0, alpha, beta, bg, n_mc=2_000_000)

        diff = log_p_box - log_p_mc
        print(f"{I_obs:>6} {n_ph:>5.0f} | {log_p_mc:>10.3f} {log_p_box:>10.3f} | {diff:>+7.3f}")

        # ── Dense curves for continuous overlay ──
        sigma = max(math.sqrt(max(n_ph, 1)), 1.0)
        lam_dense = torch.linspace(max(bg + 0.01, n_ph - 6 * sigma),
                                   n_ph + 6 * sigma, 1000)
        signal_dense = (lam_dense - bg).clamp(min=1e-8)
        eps_dense = torch.log(signal_dense / I0)

        log_gg_dense = _log_gg_density(eps_dense, alpha, beta)
        log_poi_dense = _log_poi(n_ph, lam_dense)
        log_jac_dense = -torch.log(signal_dense)
        log_overlap_dense = log_gg_dense + log_poi_dense + log_jac_dense

        # Normalize each for visibility
        gg_jac_dense = torch.exp(log_gg_dense + log_jac_dense)
        gg_jac_norm = gg_jac_dense / gg_jac_dense.max().clamp(min=1e-30)

        poi_dense = torch.exp(log_poi_dense)
        poi_norm = poi_dense / poi_dense.max().clamp(min=1e-30)

        overlap_dense = torch.exp(log_overlap_dense)
        overlap_norm = overlap_dense / overlap_dense.max().clamp(min=1e-30)

        # Quadrature point densities (for box heights)
        box_density = torch.exp(log_density)
        box_norm = box_density / overlap_dense.max().clamp(min=1e-30)

        ax = axes[i]

        # Continuous curves
        ax.plot(lam_dense.numpy(), poi_norm.numpy(), "steelblue", lw=2,
                ls="--", label="Poisson", zorder=3)
        ax.plot(lam_dense.numpy(), gg_jac_norm.numpy(), "seagreen", lw=2,
                ls=":", label="GG×Jac", zorder=3)
        ax.fill_between(lam_dense.numpy(), overlap_norm.numpy(), alpha=0.15,
                        color="darkorchid", label="Overlap", zorder=2)
        ax.plot(lam_dense.numpy(), overlap_norm.numpy(), "darkorchid", lw=1.5,
                zorder=3)

        # Integration boxes
        for k in range(N):
            x_left = float(lam_pts[k]) - float(widths[k]) / 2
            h = float(box_norm[k])
            w = float(widths[k])
            if h > 1e-6:
                rect = patches.Rectangle(
                    (x_left, 0), w, h,
                    linewidth=0.8, edgecolor="red", facecolor="red",
                    alpha=0.25, zorder=4)
                ax.add_patch(rect)
                # Mark center
                ax.plot(float(lam_pts[k]), h, "r.", ms=3, zorder=5)

        # Reference lines
        ax.axvline(n_ph, color="navy", ls="--", lw=1, alpha=0.6,
                   label=f"n={n_ph:.0f}")
        ax.axvline(bg, color="gray", ls=":", lw=1, alpha=0.5,
                   label=f"bg={bg:.0f}")

        ax.set_xlabel("λ (photon count)")
        ax.set_ylabel("(normalized)")
        ax.set_title(f"I_obs={I_obs} (n={n_ph:.0f})\n"
                     f"box={log_p_box:.3f}  MC={log_p_mc:.3f}  "
                     f"Δ={diff:+.3f}")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Midpoint-rule integration boxes  (I₀={I0}, α={alpha}, β={beta}, bg={bg}, N={N})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "integration_boxes.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
