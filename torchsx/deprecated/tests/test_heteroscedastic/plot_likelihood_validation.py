#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Validate the per-obs likelihood by comparing to empirical histogram.

Generate 1M samples from the forward model, histogram them, and overlay
the theoretical p(I_obs | I0) from both GenGauss and Poisson models.
If the theory is right, it should match the histogram exactly.
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


def _log_gg_norm(beta):
    return math.log(beta) - math.log(2) - float(torch.lgamma(torch.tensor(1.0 / beta)))


def compute_theoretical(I_obs_vals, I0, alpha, beta, bg, n_per_obs=61):
    """Compute both theoretical likelihoods at given I_obs values."""
    N = len(I_obs_vals)
    log_gg_norm = _log_gg_norm(beta)

    # ── GenGauss (log-space, continuous) ──
    # p_GG(I_obs) = p_GG(ε=log(I_obs/I0)) · |dε/dI_obs| = p_GG(ε) / I_obs
    pos = I_obs_vals > 0
    log_p_gg = torch.full((N,), float('nan'))
    r = torch.log(I_obs_vals[pos] / I0)
    log_p_gg[pos] = -(r.abs() / alpha).pow(beta) + log_gg_norm - torch.log(I_obs_vals[pos])

    # ── Poisson marginal (obs-centered quadrature) ──
    n_ph = (I_obs_vals + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)

    sigma = n_ph.clamp(min=1).sqrt()
    lam_center = torch.maximum(n_ph, torch.full_like(n_ph, bg + 0.1))
    lam_lo = (lam_center - 4 * sigma).clamp(min=bg + 0.1)
    lam_hi = lam_center + 4 * sigma
    t = torch.linspace(0, 1, n_per_obs)
    lam = lam_lo[:, None] + (lam_hi - lam_lo)[:, None] * t[None, :]
    d_lam = (lam_hi - lam_lo) / (n_per_obs - 1)
    signal = lam - bg

    eps = torch.log(signal / I0)
    log_norm = _log_gg_norm(beta)
    log_gg_q = -(eps.abs() / alpha).pow(beta) + log_norm
    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_jac = -torch.log(signal)
    log_integrand = log_gg_q + log_poi + log_jac + torch.log(d_lam)[:, None]
    log_p_poi = torch.logsumexp(log_integrand, dim=1)

    return log_p_gg, log_p_poi


def main():
    n_samples = 1_000_000
    bg = 10.0
    alpha = 1.2
    beta = 1.3

    I0_vals = [3.0, 10.0, 30.0]

    fig, axes = plt.subplots(len(I0_vals), 2, figsize=(14, 4 * len(I0_vals)))

    for row, I0 in enumerate(I0_vals):
        gen = torch.Generator().manual_seed(42)

        # Forward model: ε → I_signal → Poisson → I_obs
        eps = sample_gengauss(alpha, beta, n_samples, gen=gen)
        I_signal = (I0 * torch.exp(eps)).clamp(min=0)
        I_obs = torch.poisson(I_signal + bg, generator=gen) - bg
        obs_np = I_obs.numpy()

        n_neg = (I_obs <= 0).sum()
        print(f"I0={I0}: range=[{I_obs.min():.0f}, {I_obs.max():.0f}], "
              f"neg={n_neg}/{n_samples} ({100*n_neg/n_samples:.1f}%)")

        # Theoretical curves
        # For integer I_obs values (Poisson is discrete)
        i_min = max(int(obs_np.min()) - 1, -int(bg))
        i_max = min(int(np.percentile(obs_np, 99.9)) + 1, 1000)
        I_eval = torch.arange(i_min, i_max + 1, dtype=torch.float32)
        log_p_gg, log_p_poi = compute_theoretical(I_eval, I0, alpha, beta, bg)

        # Convert to PMF (Poisson is discrete, bin width = 1)
        p_poi = torch.exp(log_p_poi).numpy()
        p_gg = torch.exp(log_p_gg).numpy()  # continuous density, multiply by bin width=1
        I_eval_np = I_eval.numpy()

        # Empirical histogram (integer bins since Poisson is discrete)
        bins = np.arange(i_min - 0.5, i_max + 1.5, 1)
        hist, _ = np.histogram(obs_np, bins=bins, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # ── Left: linear scale, near peak ──
        ax = axes[row, 0]
        ax.bar(bin_centers, hist, width=0.8, color="steelblue", alpha=0.5,
               label="Empirical (1M samples)")
        valid_gg = np.isfinite(p_gg)
        ax.plot(I_eval_np[valid_gg], p_gg[valid_gg], "seagreen", lw=2,
                label="GenGauss theory")
        ax.plot(I_eval_np, p_poi, "darkorchid", lw=2, ls="--",
                label="Poisson theory")
        ax.axvline(I0, color="red", ls="--", lw=1, label=f"I₀={I0}")
        ax.set_xlabel("I_obs")
        ax.set_ylabel("P(I_obs)")
        ax.set_xlim(i_min - 1, min(I0 * 15, i_max))
        ax.set_title(f"I₀={I0}: linear scale")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── Right: log scale, show tails ──
        ax = axes[row, 1]
        mask_h = hist > 0
        ax.scatter(bin_centers[mask_h], hist[mask_h], s=8, c="steelblue",
                   alpha=0.5, label="Empirical", zorder=5)
        ax.plot(I_eval_np[valid_gg], p_gg[valid_gg], "seagreen", lw=2,
                label="GenGauss theory")
        ax.plot(I_eval_np, p_poi, "darkorchid", lw=2, ls="--",
                label="Poisson theory")
        ax.axvline(I0, color="red", ls="--", lw=1, label=f"I₀={I0}")
        ax.set_xlabel("I_obs")
        ax.set_ylabel("P(I_obs) [log scale]")
        ax.set_yscale("log")
        ax.set_xlim(i_min - 1, i_max)
        ymin = max(1e-8, hist[mask_h].min() * 0.1) if mask_h.any() else 1e-8
        ax.set_ylim(ymin, ax.get_ylim()[1] * 2)
        ax.set_title(f"I₀={I0}: log scale (tails)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Likelihood validation: 1M samples vs theory  "
                 f"(α={alpha}, β={beta}, bg={bg})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/likelihood_validation.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
