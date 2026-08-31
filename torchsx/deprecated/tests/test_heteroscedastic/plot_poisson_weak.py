#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Visualise the Poisson distribution for weak reflections.

For various I_signal values (0.1, 1, 3, 10, 30) with bg=10,
show the Poisson(n; I_signal + bg) distribution and where bg sits.
"""

import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


def main():
    bg = 10.0
    signals = [0.1, 0.5, 1, 3, 10, 30]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, I_sig in enumerate(signals):
        lam = I_sig + bg
        ax = axes[i]

        # Poisson PMF
        n_max = int(lam + 5 * math.sqrt(lam)) + 5
        n_vals = np.arange(0, n_max + 1)
        pmf = poisson.pmf(n_vals, lam)

        # I_obs = n - bg
        I_obs = n_vals - bg

        # Color: red for I_obs <= 0, blue for positive
        colors = np.where(I_obs <= 0, "tomato", "steelblue")
        ax.bar(I_obs, pmf, color=colors, alpha=0.8, width=0.8, edgecolor="none")

        # Mark true I_signal
        ax.axvline(I_sig, color="red", ls="--", lw=2, label=f"I_signal = {I_sig}")
        ax.axvline(0, color="gray", ls=":", lw=1, label="I_obs = 0")

        # Fraction negative
        frac_neg = poisson.cdf(bg, lam)  # P(n <= bg) = P(I_obs <= 0)
        frac_neg_strict = poisson.cdf(bg - 1, lam)  # P(n < bg) = P(I_obs < 0)

        ax.set_xlabel("I_obs = n_photons − bg")
        ax.set_ylabel("P(n)")
        ax.set_title(f"I_signal = {I_sig},  λ = {lam:.1f}\n"
                     f"P(I_obs ≤ 0) = {frac_neg:.1%},  P(I_obs < 0) = {frac_neg_strict:.1%}")
        ax.legend(fontsize=8)
        ax.set_xlim(-bg - 2, max(I_obs[-1], 3 * I_sig + 5))

    fig.suptitle(f"Poisson distribution after background subtraction (bg = {bg})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/poisson_weak.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
