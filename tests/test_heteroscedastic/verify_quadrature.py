#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Verify the obs-centered quadrature against Monte Carlo and dense fixed-grid.

For a single observation, compute p(n | I0) three ways:
  1. Monte Carlo (1M ε samples — ground truth)
  2. Dense fixed-grid quadrature in ε-space (10000 points)
  3. Obs-centered quadrature in λ-space (31 points)

If all three agree, the quadrature is correct.
Then sweep I0 and check the NLL landscape.
"""

from __future__ import annotations
import math, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sample_gengauss(alpha, beta, size, gen=None):
    g = torch._standard_gamma(torch.full((size,), 1.0 / beta), generator=gen)
    signs = 2.0 * torch.bernoulli(torch.full((size,), 0.5), generator=gen) - 1.0
    return signs * alpha * g.pow(1.0 / beta)


def _log_gg_density(eps, alpha, beta):
    """Full GenGauss log-density including all normalization terms."""
    return (math.log(beta) - math.log(2) - math.log(alpha)
            - float(torch.lgamma(torch.tensor(1.0 / beta)))
            - (eps.abs() / alpha).pow(beta))


def _log_poi(n, lam):
    """Log Poisson PMF."""
    lam = lam.clamp(min=1e-30)
    return n * torch.log(lam) - lam - torch.lgamma(torch.tensor(n + 1.0))


def mc_log_marginal(n_obs, I0, alpha, beta, bg, n_mc=2_000_000):
    """Monte Carlo estimate of log p(n | I0)."""
    gen = torch.Generator().manual_seed(0)
    eps = sample_gengauss(alpha, beta, n_mc, gen=gen)
    lam = (I0 * torch.exp(eps) + bg).clamp(min=1e-30)
    log_poi = _log_poi(n_obs, lam)
    # p(n|I0) = E_eps[Poi(n; I0*exp(eps)+bg)] ≈ mean of Poi values
    # log p ≈ logsumexp(log_poi) - log(n_mc)
    return float(torch.logsumexp(log_poi, dim=0) - math.log(n_mc))


def fixed_grid_log_marginal(n_obs, I0, alpha, beta, bg, n_quad=10000):
    """Dense fixed-grid quadrature in ε-space."""
    from plot_scatter_bias import _quad_range
    R = _quad_range(beta)
    eps = torch.linspace(-R, R, n_quad)
    d_eps = float(eps[1] - eps[0])
    log_gg = _log_gg_density(eps, alpha, beta)
    lam = (I0 * torch.exp(eps) + bg).clamp(min=1e-30)
    log_poi = _log_poi(n_obs, lam)
    log_integrand = log_gg + log_poi + math.log(d_eps)
    return float(torch.logsumexp(log_integrand, dim=0))


def obs_centered_log_marginal(n_obs, I0, alpha, beta, bg, n_per_obs=31):
    """Obs-centered quadrature in λ-space."""
    n_ph = max(n_obs + bg, 0)
    sigma = max(math.sqrt(max(n_ph, 1)), 1)
    lam_lo = max(bg + 0.1, n_ph - 4 * sigma)
    lam_hi = max(lam_lo + 1.0, n_ph + 4 * sigma)
    lam = torch.linspace(lam_lo, lam_hi, n_per_obs)
    d_lam = float(lam[1] - lam[0])
    signal = (lam - bg).clamp(min=1e-8)
    eps = torch.log(signal / I0)
    log_gg = _log_gg_density(eps, alpha, beta)
    log_poi = _log_poi(n_ph, lam)
    log_jac = -torch.log(signal)  # dε/dλ = 1/(λ - bg)
    log_integrand = log_gg + log_poi + log_jac + math.log(d_lam)
    return float(torch.logsumexp(log_integrand, dim=0))


def gg_log_likelihood(I_obs, I0, alpha, beta):
    """GenGauss log-likelihood in log-space (positive I_obs only)."""
    if I_obs <= 0:
        return float('nan')
    eps = math.log(I_obs / I0)
    # Full density including Jacobian d(log I_obs)/dI_obs = 1/I_obs
    return _log_gg_density(torch.tensor(eps), alpha, beta).item() - math.log(I_obs)


def main():
    I0 = 3.0
    alpha = 1.2
    beta = 1.3
    bg = 10.0

    print(f"Parameters: I0={I0}, α={alpha}, β={beta}, bg={bg}")
    print(f"\nGenerative model:")
    print(f"  ε ~ GenGauss(0, {alpha}, {beta})")
    print(f"  n ~ Poisson(I0·exp(ε) + bg) = Poisson({I0}·exp(ε) + {bg})")
    print(f"  I_obs = n - bg")
    print(f"  n = I_obs + bg")

    # ── Verify for specific observations ──
    test_obs = [-8, -5, -3, -1, 0, 1, 2, 3, 5, 10, 20, 50, 100]

    print(f"\n{'I_obs':>6} {'n_ph':>5} | {'MC':>10} {'Fixed10k':>10} {'ObsCtr31':>10} {'GenGauss':>10} | {'Fix-MC':>7} {'OC-MC':>7}")
    print("-" * 85)

    for I_obs in test_obs:
        n_ph = I_obs + bg
        mc = mc_log_marginal(n_ph, I0, alpha, beta, bg)
        fixed = fixed_grid_log_marginal(n_ph, I0, alpha, beta, bg)
        oc = obs_centered_log_marginal(I_obs, I0, alpha, beta, bg)
        gg = gg_log_likelihood(I_obs, I0, alpha, beta)

        d_fix = fixed - mc
        d_oc = oc - mc
        gg_str = f"{gg:>10.3f}" if not math.isnan(gg) else f"{'NaN':>10}"
        print(f"{I_obs:>6} {n_ph:>5.0f} | {mc:>10.3f} {fixed:>10.3f} {oc:>10.3f} {gg_str} | {d_fix:>+7.3f} {d_oc:>+7.3f}")

    # ── NLL landscape: all methods over I0 grid ──
    I0_grid = torch.logspace(-0.5, 2.5, 200)
    test_n = [5, 13, 20, 110]  # n_photons (= I_obs + bg for I_obs = -5, 3, 10, 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, n_ph in enumerate(test_n):
        I_obs_val = n_ph - bg
        ax = axes[idx // 2, idx % 2]

        nll_mc = []
        nll_fix = []
        nll_oc = []
        nll_gg = []

        for I0_candidate in I0_grid.tolist():
            nll_mc.append(-mc_log_marginal(n_ph, I0_candidate, alpha, beta, bg, n_mc=500_000))
            nll_fix.append(-fixed_grid_log_marginal(n_ph, I0_candidate, alpha, beta, bg))
            nll_oc.append(-obs_centered_log_marginal(I_obs_val, I0_candidate, alpha, beta, bg))
            nll_gg.append(-gg_log_likelihood(I_obs_val, I0_candidate, alpha, beta))

        nll_mc = np.array(nll_mc)
        nll_fix = np.array(nll_fix)
        nll_oc = np.array(nll_oc)
        nll_gg = np.array(nll_gg)
        I0_np = I0_grid.numpy()

        # Normalize to min=0
        ref = np.nanmin(nll_mc)
        ax.plot(I0_np, nll_mc - ref, "k-", lw=2, label="MC (500k)")
        ax.plot(I0_np, nll_fix - ref, "b--", lw=1.5, label="Fixed 10k")
        ax.plot(I0_np, nll_oc - ref, "r:", lw=2, label="Obs-ctr 31")
        valid_gg = np.isfinite(nll_gg)
        if valid_gg.any():
            gg_ref = np.nanmin(nll_gg[valid_gg])
            ax.plot(I0_np[valid_gg], nll_gg[valid_gg] - gg_ref, "g-.", lw=1.5, label="GenGauss")
        ax.axvline(I0, color="red", ls="--", lw=1, alpha=0.5, label=f"True I₀={I0}")
        ax.set_xscale("log")
        ax.set_xlabel("I₀")
        ax.set_ylabel("NLL − min")
        ax.set_ylim(0, 10)
        ax.set_title(f"I_obs = {I_obs_val:.0f} (n = {n_ph:.0f} photons)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Quadrature verification: MC vs Fixed vs Obs-centered  "
                 f"(α={alpha}, β={beta}, bg={bg})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/verify_quadrature.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
