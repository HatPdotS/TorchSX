#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare per-observation likelihood under GenGauss vs Poisson model.

For I0=10, alpha=1.2, beta=1.3, bg=10:
  - Generate observations
  - For each obs, compute:
    1. GenGauss log-likelihood: log p_GG = -|log(I_obs/I0)/α|^β + norm
    2. Poisson marginal log-likelihood: log p = log ∫ GG(ε) · Poi(n; I0·eᵋ+bg) dε
  - Plot both as function of I_obs
"""

from __future__ import annotations
import math, torch
import matplotlib; matplotlib.use("Agg")
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
    I0 = 3.0
    alpha = 1.2
    beta = 1.3
    bg = 10.0

    # GenGauss normalization
    log_gg_norm = math.log(beta) - math.log(2) - math.log(alpha) \
                  - float(torch.lgamma(torch.tensor(1.0 / beta)))

    # Range of I_obs to evaluate — focus on low signal
    I_obs_range = torch.linspace(-10, 200, 1000)

    # ── GenGauss log-likelihood (log-space, positive only) ──
    # p_GG(I_obs | I0) = (β/(2αΓ(1/β))) · exp(-|log(I_obs/I0)/α|^β) · (1/I_obs)
    # The 1/I_obs is the Jacobian for change of variable from ε to I_obs
    pos_mask = I_obs_range > 0
    log_gg = torch.full_like(I_obs_range, float('nan'))
    I_pos = I_obs_range[pos_mask]
    r = torch.log(I_pos / I0)
    log_gg[pos_mask] = -(r.abs() / alpha).pow(beta) + log_gg_norm - torch.log(I_pos)

    # ── Poisson marginal log-likelihood (obs-centered quadrature) ──
    n_per_obs = 31
    log_poi_norm = math.log(beta) - math.log(2) \
                   - float(torch.lgamma(torch.tensor(1.0 / beta)))

    n_ph = (I_obs_range + bg).clamp(min=0)
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
    log_gg_q = -(eps.abs() / alpha).pow(beta) + log_poi_norm
    log_poi_q = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_jac = -torch.log(signal)
    log_integrand = log_gg_q + log_poi_q + log_jac + torch.log(d_lam)[:, None]
    log_poisson_marginal = torch.logsumexp(log_integrand, dim=1)

    # ── Also generate actual samples to show where they fall ──
    gen = torch.Generator().manual_seed(42)
    from plot_scatter_bias import sample_gengauss
    eps_sample = sample_gengauss(alpha, beta, 100, gen=gen)
    sig = (I0 * torch.exp(eps_sample)).clamp(min=0)
    I_obs_samples = torch.poisson(sig + bg, generator=gen) - bg

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = I_obs_range.numpy()

    # (0,0) Both log-likelihoods vs I_obs
    ax = axes[0, 0]
    ax.plot(x, log_gg.numpy(), "seagreen", lw=2, label="GenGauss (log-space)")
    ax.plot(x, log_poisson_marginal.numpy(), "darkorchid", lw=2, label="Poisson marginal")
    ax.axvline(I0, color="red", ls="--", lw=1, label=f"I₀={I0}")
    ax.set_xlabel("I_obs")
    ax.set_ylabel("log p(I_obs | I₀)")
    ax.set_title("Per-observation log-likelihood")
    ax.legend(fontsize=8)
    ax.set_xlim(-12, 100)
    ax.grid(True, alpha=0.3)

    # (0,1) Same but wider range, log x-axis for positive
    ax = axes[0, 1]
    pos_x = x[x > 0.5]
    ax.plot(pos_x, log_gg.numpy()[x > 0.5], "seagreen", lw=2, label="GenGauss")
    ax.plot(pos_x, log_poisson_marginal.numpy()[x > 0.5], "darkorchid", lw=2, label="Poisson")
    ax.axvline(I0, color="red", ls="--", lw=1, label=f"I₀={I0}")
    ax.set_xscale("log")
    ax.set_xlabel("I_obs (log scale)")
    ax.set_ylabel("log p(I_obs | I₀)")
    ax.set_title("Log-likelihood (log x-axis)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) NLL contribution: -log p (what gets summed)
    ax = axes[1, 0]
    nll_gg = -log_gg.numpy()
    nll_poi = -log_poisson_marginal.numpy()
    ax.plot(x, nll_gg, "seagreen", lw=2, label="GenGauss NLL")
    ax.plot(x, nll_poi, "darkorchid", lw=2, label="Poisson NLL")
    ax.axvline(I0, color="red", ls="--", lw=1)
    # Show actual samples
    for s in I_obs_samples[:20]:
        ax.axvline(float(s), color="steelblue", alpha=0.15, lw=1)
    ax.set_xlabel("I_obs")
    ax.set_ylabel("-log p  (NLL contribution)")
    ax.set_title("NLL contribution per observation")
    ax.set_xlim(-12, 200)
    ax.set_ylim(0, 20)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Ratio: how much more does each model penalise outliers?
    ax = axes[1, 1]
    # Difference in NLL: Poisson penalty - GenGauss penalty
    diff = nll_poi - nll_gg
    valid = np.isfinite(diff) & (x > 0.5)
    ax.plot(x[valid], diff[valid], "k-", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axvline(I0, color="red", ls="--", lw=1, label=f"I₀={I0}")
    ax.set_xlabel("I_obs")
    ax.set_ylabel("Poisson NLL − GenGauss NLL")
    ax.set_title("Extra penalty from Poisson model\n(>0 = Poisson penalises more)")
    ax.set_xlim(-5, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Print some specific values
    print(f"\nPer-observation likelihoods at I0={I0}, α={alpha}, β={beta}, bg={bg}:")
    print(f"{'I_obs':>8} {'log p_GG':>10} {'log p_Poi':>10} {'NLL_GG':>8} {'NLL_Poi':>8} {'Poi-GG':>8}")
    print("-" * 60)
    for I_val in [-8, -5, -3, -1, 0, 1, 2, 3, 5, 10, 20, 50, 100, 200]:
        idx = torch.argmin((I_obs_range - I_val).abs())
        gg_val = float(log_gg[idx]) if not math.isnan(float(log_gg[idx])) else float('nan')
        poi_val = float(log_poisson_marginal[idx])
        ngg = -gg_val if not math.isnan(gg_val) else float('nan')
        npoi = -poi_val
        diff_val = npoi - ngg if not math.isnan(ngg) else float('nan')
        print(f"{I_val:>8} {gg_val:>10.2f} {poi_val:>10.2f} {ngg:>8.2f} {npoi:>8.2f} {diff_val:>8.2f}")

    fig.suptitle(f"Per-observation likelihood: GenGauss vs Poisson  "
                 f"(I₀={I0}, α={alpha}, β={beta}, bg={bg})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/per_obs_likelihood.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
