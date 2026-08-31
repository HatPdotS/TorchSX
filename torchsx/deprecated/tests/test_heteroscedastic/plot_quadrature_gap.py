#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Diagnose the quadrature coverage gap for outlier observations.

For a given (I0, alpha, beta, bg), show:
  1. The quadrature grid in ε-space and the predicted I at each point
  2. For various observed I_obs values, where the "explaining" ε must be
  3. The Poisson width at each prediction vs the gap between quadrature points
"""

import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _quad_range(beta, coverage=0.999):
    a = torch.tensor(1.0 / beta)
    lo, hi = 1.0, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(torch.special.gammainc(a, torch.tensor(mid**beta))) < coverage:
            lo = mid
        else:
            hi = mid
    return min(hi, 15.0)


def main():
    I0 = 10.0
    alpha = 1.2
    beta = 1.3
    bg = 10.0
    n_quad = 35  # typical effective count

    R = _quad_range(beta)
    print(f"Quadrature range R = {R:.3f} (standardised)")
    print(f"ε range = [-{alpha*R:.2f}, {alpha*R:.2f}]")

    eps_tilde = np.linspace(-R, R, n_quad)
    eps = alpha * eps_tilde
    d_eps_tilde = eps_tilde[1] - eps_tilde[0]

    # Predicted intensity at each quadrature point
    I_pred = I0 * np.exp(eps) + bg
    poisson_sigma = np.sqrt(np.maximum(I_pred, 1))

    # Gap between adjacent predictions (in intensity space)
    I_gap = np.diff(I_pred)

    print(f"\nQuadrature grid ({n_quad} points, Δε̃={d_eps_tilde:.3f}):")
    print(f"  ε̃ range: [{eps_tilde[0]:.2f}, {eps_tilde[-1]:.2f}]")
    print(f"  I_pred range: [{I_pred[0]:.1f}, {I_pred[-1]:.1f}]")
    print(f"  Max I_pred gap: {I_gap.max():.0f}")
    print(f"  Poisson σ at max I_pred: {poisson_sigma[-1]:.1f}")

    # For various outlier observations, find the required ε
    outlier_I = [5, 10, 30, 50, 100, 200, 500, 1000, 2000]
    print(f"\nOutlier analysis:")
    print(f"  {'I_obs':>7} {'ε_need':>7} {'ε̃_need':>7} {'in range?':>10} "
          f"{'nearest ε̃':>10} {'I_nearest':>10} {'Poi σ':>7} {'#σ away':>8}")
    print(f"  {'-'*75}")

    for I_obs in outlier_I:
        # Required ε to predict this observation
        I_signal = max(I_obs - bg, 0.1)
        eps_need = math.log(I_signal / I0) if I_signal > 0 else -10
        eps_tilde_need = eps_need / alpha

        in_range = abs(eps_tilde_need) <= R
        # Nearest quadrature point
        idx_near = np.argmin(np.abs(eps_tilde - eps_tilde_need))
        I_nearest = I_pred[idx_near]
        sigma_nearest = math.sqrt(max(I_nearest, 1))
        n_sigma = abs(I_obs - I_nearest) / sigma_nearest if sigma_nearest > 0 else 999

        print(f"  {I_obs:>7} {eps_need:>7.2f} {eps_tilde_need:>7.2f} "
              f"{'YES' if in_range else 'NO':>10} "
              f"{eps_tilde[idx_near]:>10.2f} {I_nearest:>10.1f} "
              f"{sigma_nearest:>7.1f} {n_sigma:>8.1f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Quadrature grid: predicted I vs ε̃
    ax = axes[0, 0]
    ax.plot(eps_tilde, I_pred, "k.-", markersize=4, lw=1)
    ax.fill_between(eps_tilde, I_pred - 2*poisson_sigma, I_pred + 2*poisson_sigma,
                    alpha=0.2, color="steelblue", label="±2σ Poisson")
    for I_obs, color in [(100, "orange"), (500, "red"), (1000, "darkred")]:
        I_sig = max(I_obs - bg, 0.1)
        et = math.log(I_sig / I0) / alpha
        ax.axhline(I_obs, color=color, ls="--", lw=1, alpha=0.7, label=f"I_obs={I_obs}")
        ax.axvline(et, color=color, ls=":", lw=1, alpha=0.7)
    ax.axvline(R, color="gray", ls="-", lw=2, alpha=0.5, label=f"R={R:.2f}")
    ax.axvline(-R, color="gray", ls="-", lw=2, alpha=0.5)
    ax.set_xlabel("ε̃ (standardised)")
    ax.set_ylabel("Predicted I = I₀·exp(αε̃) + bg")
    ax.set_yscale("log")
    ax.set_title(f"Quadrature grid: I₀={I0}, α={alpha}, n_quad={n_quad}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Gap between adjacent predictions vs Poisson sigma
    ax = axes[0, 1]
    mid_eps = 0.5 * (eps_tilde[:-1] + eps_tilde[1:])
    mid_I = 0.5 * (I_pred[:-1] + I_pred[1:])
    mid_sigma = np.sqrt(np.maximum(mid_I, 1))
    gap_in_sigma = I_gap / mid_sigma

    ax.semilogy(mid_eps, gap_in_sigma, "k.-", markersize=4)
    ax.axhline(1, color="orange", ls="--", lw=1, label="1σ gap")
    ax.axhline(3, color="red", ls="--", lw=1, label="3σ gap")
    ax.set_xlabel("ε̃")
    ax.set_ylabel("Gap between adjacent predictions (in Poisson σ)")
    ax.set_title("Quadrature spacing vs Poisson width")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Same as (b) but with more quad points
    ax = axes[1, 0]
    for nq, color, ls in [(10, "red", "-"), (35, "orange", "-"),
                           (100, "seagreen", "-"), (1000, "steelblue", "-")]:
        et = np.linspace(-R, R, nq)
        ep = alpha * et
        Ip = I0 * np.exp(ep) + bg
        Ig = np.diff(Ip)
        mI = 0.5 * (Ip[:-1] + Ip[1:])
        ms = np.sqrt(np.maximum(mI, 1))
        me = 0.5 * (et[:-1] + et[1:])
        ax.semilogy(me, Ig / ms, color=color, ls=ls, lw=1.5, label=f"n_quad={nq}")
    ax.axhline(1, color="gray", ls="--", lw=1, alpha=0.5, label="1σ")
    ax.axhline(3, color="gray", ls=":", lw=1, alpha=0.5, label="3σ")
    ax.set_xlabel("ε̃")
    ax.set_ylabel("Gap (in Poisson σ)")
    ax.set_title("Gap vs n_quad")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Per-observation log p contribution: well-covered vs outlier
    ax = axes[1, 1]
    # Generate one sample and show per-obs log p
    gen = torch.Generator().manual_seed(42)
    from plot_nll_landscape import sample_gengauss
    eps_sample = sample_gengauss(alpha, beta, 100, gen=gen)
    signal = (I0 * torch.exp(eps_sample)).clamp(min=0)
    I_obs_sample = torch.poisson(signal + bg, generator=gen) - bg

    # Compute log p at true I0 for each obs
    K = n_quad
    eps_q = torch.linspace(-R, R, K)
    d_e = float(eps_q[1] - eps_q[0])
    log_w = -(eps_q.abs() ** beta) + math.log(beta) - math.log(2) \
            - float(torch.lgamma(torch.tensor(1.0 / beta))) + math.log(d_e)
    s_q = torch.exp(alpha * eps_q)
    n_ph = (I_obs_sample + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)
    lam = (s_q * I0 + bg).clamp(min=1e-6)
    log_poi = n_ph[None, :] * torch.log(lam[:, None]) - lam[:, None] - log_nf[None, :]
    log_joint = log_poi + log_w[:, None]
    log_p = torch.logsumexp(log_joint, dim=0).numpy()

    obs_np = I_obs_sample.numpy()
    order = np.argsort(obs_np)
    colors = np.where(obs_np[order] > 0, "steelblue", "tomato")

    # Required ε̃ per obs
    eps_needed = np.log(np.maximum(obs_np - bg, 0.01) / I0) / alpha

    ax.scatter(obs_np[order], log_p[order], c=colors, s=15, alpha=0.7)
    ax.set_xlabel("I_obs (sorted)")
    ax.set_ylabel("log p(I_obs | I₀=true)")
    ax.set_title("Per-obs log-likelihood at true I₀\n(red=negative obs)")
    ax.grid(True, alpha=0.3)
    # Annotate extreme observations
    for idx in np.argsort(log_p)[:5]:
        ax.annotate(f"I={obs_np[idx]:.0f}\nε̃={eps_needed[idx]:.1f}",
                   (obs_np[idx], log_p[idx]),
                   fontsize=6, ha="center", va="top")

    fig.suptitle(f"Quadrature coverage analysis  (I₀={I0}, α={alpha}, β={beta}, bg={bg})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/quadrature_gap.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
