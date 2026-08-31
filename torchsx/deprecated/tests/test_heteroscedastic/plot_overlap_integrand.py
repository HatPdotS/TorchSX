#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Plot the overlap integrand for individual observations.

For I0=3, α=1.2, β=1.3, bg=10, plot:
  f(ε) = GenGauss(ε) × Poisson(n; I0·exp(ε) + bg)
as a function of ε, for various I_obs values.
"""

import math, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    I0 = 3.0
    alpha = 1.2
    beta = 1.3
    bg = 10.0

    # Dense ε grid
    eps = torch.linspace(-6, 8, 2000)

    # GenGauss density
    log_gg = (math.log(beta) - math.log(2) - math.log(alpha)
              - float(torch.lgamma(torch.tensor(1.0 / beta)))
              - (eps.abs() / alpha).pow(beta))

    test_obs = [-8, -5, -2, 0, 1, 3, 5, 10, 20, 50, 100, 500]

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()

    for i, I_obs in enumerate(test_obs):
        n_ph = max(I_obs + bg, 0)  # photon count

        # Poisson as function of ε: Poi(n; I0·exp(ε) + bg)
        lam = (I0 * torch.exp(eps) + bg).clamp(min=1e-30)
        log_poi = n_ph * torch.log(lam) - lam - float(torch.lgamma(torch.tensor(n_ph + 1.0)))

        # Integrand (unnormalized)
        log_integrand = log_gg + log_poi
        integrand = torch.exp(log_integrand - log_integrand.max())  # normalize for plotting

        # Individual components (normalized for overlay)
        gg_norm = torch.exp(log_gg - log_gg.max())
        poi_norm = torch.exp(log_poi - log_poi.max())

        # Where does I_obs fall in ε-space?
        if I_obs > 0:
            eps_obs = math.log(I_obs / I0)
        else:
            eps_obs = None

        # Where does the integrand peak?
        peak_idx = integrand.argmax()
        eps_peak = float(eps[peak_idx])
        lam_peak = float(I0 * math.exp(eps_peak) + bg)
        signal_peak = lam_peak - bg

        ax = axes[i]
        ax.fill_between(eps.numpy(), integrand.numpy(), alpha=0.3, color="darkorchid",
                        label="Integrand (overlap)")
        ax.plot(eps.numpy(), gg_norm.numpy(), "seagreen", lw=1.5, ls="--", label="GenGauss")
        ax.plot(eps.numpy(), poi_norm.numpy(), "steelblue", lw=1.5, ls=":", label="Poisson")

        if eps_obs is not None:
            ax.axvline(eps_obs, color="red", ls="--", lw=1.5,
                       label=f"ε(I_obs)={eps_obs:.2f}")
        ax.axvline(eps_peak, color="darkorchid", ls="-", lw=1,
                   label=f"peak ε={eps_peak:.2f}")
        ax.axvline(0, color="gray", ls=":", lw=0.5)

        ax.set_xlabel("ε")
        ax.set_ylabel("(normalized)")
        n_neg_frac = ""
        if I_obs <= 0:
            n_neg_frac = f" (n={n_ph:.0f} photons)"
        ax.set_title(f"I_obs={I_obs}{n_neg_frac}\n"
                     f"peak at ε={eps_peak:.2f}, signal={signal_peak:.1f}")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlim(-6, 8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Overlap integrand: GG(ε) × Poi(n; I₀·eᵋ+bg)\n"
                 f"I₀={I0}, α={alpha}, β={beta}, bg={bg}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/overlap_integrand.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
