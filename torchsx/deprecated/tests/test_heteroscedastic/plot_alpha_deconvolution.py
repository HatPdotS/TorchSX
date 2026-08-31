#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Correct alpha for Poisson noise via the MLE of the convolved distribution.

The observed log-residual is r = ε + ε_poi where:
  ε ~ GenGauss(0, α, β)           — what we want
  ε_poi ~ approx N(0, σ_poi²)    — Poisson noise, σ_poi² ≈ 1/(I0+bg)

The MLE of α under the GenGauss model when the data is actually
GenGauss * Gaussian is biased. We can compute the expected bias
by evaluating:

  E[α_MLE | α_true, σ_poi] = α such that
  α^β = (β) · E_{r~GG*N}[|r|^β]

where the expectation is over the convolved distribution.
This can be computed numerically for a grid of (α, σ_poi) values,
giving us a lookup table to invert: given α_raw and σ_poi, find α_true.

Alternative simpler approach: fit α_true directly via MLE of the
convolved distribution, marginalizing over ε_poi. This is equivalent
to what fit_I0_poisson_log does for I0, but for α.

Let's try the lookup table approach first — it's simpler and more
transparent.
"""

from __future__ import annotations

import os, sys, time, math
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.test_heteroscedastic.generate import sample_gengauss
from torchsx.deprecated.merge._dstep import fit_I0_poisson_log, fit_alpha


def expected_alpha_mle(alpha_true, sigma_poi, beta, n_mc=500000):
    """MC estimate of what fit_alpha returns when data = GenGauss(α) + N(0, σ_poi).

    Returns E[α_MLE] where α_MLE^β = β · E[|r|^β] and r = ε + ε_poi.
    """
    # Sample ε ~ GenGauss
    eps = sample_gengauss(alpha_true, beta, n_mc).numpy()
    # Sample ε_poi ~ N(0, σ_poi)
    eps_poi = np.random.randn(n_mc) * sigma_poi
    r = eps + eps_poi

    alpha_mle_pow_b = beta * np.mean(np.abs(r)**beta)
    return alpha_mle_pow_b**(1.0/beta)


def invert_alpha(alpha_raw, sigma_poi, beta, n_mc=500000):
    """Given the measured α_raw and σ_poi, find α_true such that
    expected_alpha_mle(α_true, σ_poi, β) = α_raw.

    Uses bisection.
    """
    # If sigma_poi is very small, no correction needed
    if sigma_poi < 1e-4:
        return alpha_raw

    def residual(alpha_true):
        return expected_alpha_mle(alpha_true, sigma_poi, beta, n_mc) - alpha_raw

    # α_true must be smaller than α_raw (noise inflates)
    # But could be very small if most of the width is Poisson
    try:
        result = brentq(residual, 0.001, alpha_raw * 1.5, xtol=0.001)
        return max(result, 0.001)
    except ValueError:
        return alpha_raw  # can't correct, return raw


def build_correction_table(beta, sigma_poi_vals, alpha_true_vals, n_mc=200000):
    """Build a 2D table: alpha_measured[i,j] = f(alpha_true[i], sigma_poi[j]).

    This table allows fast inversion via interpolation.
    """
    n_alpha = len(alpha_true_vals)
    n_sigma = len(sigma_poi_vals)
    table = np.zeros((n_alpha, n_sigma))

    for j, sp in enumerate(sigma_poi_vals):
        for i, at in enumerate(alpha_true_vals):
            table[i, j] = expected_alpha_mle(at, sp, beta, n_mc)

    return table


def correct_alpha_table(alpha_raw_np, sigma_poi_np, beta,
                        alpha_grid, sigma_grid, table):
    """Use pre-computed table to correct alpha values via interpolation."""
    from scipy.interpolate import RegularGridInterpolator

    # For each (alpha_raw, sigma_poi), find alpha_true such that
    # table(alpha_true, sigma_poi) = alpha_raw
    # Do this per-HKL via bisection on the interpolator
    corrected = np.zeros_like(alpha_raw_np)

    interp = RegularGridInterpolator(
        (alpha_grid, sigma_grid), table,
        method="linear", bounds_error=False, fill_value=None)

    for h in range(len(alpha_raw_np)):
        ar = alpha_raw_np[h]
        sp = np.clip(sigma_poi_np[h], sigma_grid[0], sigma_grid[-1])

        def residual(at):
            at = np.clip(at, alpha_grid[0], alpha_grid[-1])
            return float(interp([[at, sp]])) - ar

        try:
            corrected[h] = brentq(residual, 0.01, max(ar * 1.2, alpha_grid[-1] * 0.99),
                                   xtol=0.005)
        except ValueError:
            corrected[h] = ar

    return corrected


def generate_one_trial(I0_true, alpha_true, beta_val, obs_per_hkl, background, seed):
    n_hkl = len(I0_true)
    N = n_hkl * obs_per_hkl
    gen = torch.Generator().manual_seed(seed)
    hkl_id = torch.arange(n_hkl).repeat_interleave(obs_per_hkl)
    I0_po = I0_true.repeat_interleave(obs_per_hkl)

    epsilon = torch.zeros(N)
    for h in range(n_hkl):
        lo, hi = h * obs_per_hkl, (h + 1) * obs_per_hkl
        epsilon[lo:hi] = sample_gengauss(
            float(alpha_true[h]), beta_val, obs_per_hkl, generator=gen)

    I_signal = (I0_po * torch.exp(epsilon)).clamp(min=0)
    total_photons = torch.poisson(I_signal + background, generator=gen)
    I_obs = total_photons - background
    obs_counts = torch.full((n_hkl,), obs_per_hkl, dtype=torch.long)
    return dict(I_obs=I_obs, hkl_id=hkl_id, obs_counts=obs_counts, n_hkl=n_hkl)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    beta_val = 1.3
    background = 10.0

    # ── Step 1: Build correction table ──────────────────────────
    print("Building correction table (this takes ~60s)...")
    alpha_grid = np.linspace(0.05, 2.0, 40)
    sigma_grid = np.linspace(0.0, 1.0, 25)  # σ_poi = 1/sqrt(I0+bg)
    t0 = time.time()
    table = build_correction_table(beta_val, sigma_grid, alpha_grid, n_mc=200000)
    print(f"  Table built in {time.time()-t0:.1f}s, shape={table.shape}")

    # Show the table: what alpha_mle do we get for alpha_true=0.4 at various σ_poi?
    print(f"\n  Expected α_MLE for α_true=0.4, β={beta_val}:")
    idx_04 = np.argmin(np.abs(alpha_grid - 0.4))
    print(f"    σ_poi:      ", "  ".join(f"{s:.2f}" for s in sigma_grid[::3]))
    print(f"    α_MLE:      ", "  ".join(f"{table[idx_04, j]:.3f}" for j in range(0, len(sigma_grid), 3)))

    print(f"\n  Expected α_MLE for α_true=1.2:")
    idx_12 = np.argmin(np.abs(alpha_grid - 1.2))
    print(f"    σ_poi:      ", "  ".join(f"{s:.2f}" for s in sigma_grid[::3]))
    print(f"    α_MLE:      ", "  ".join(f"{table[idx_12, j]:.3f}" for j in range(0, len(sigma_grid), 3)))

    # ── Step 2: Test on synthetic data ──────────────────────────
    n_hkl = 300
    obs_per_hkl = 50
    n_trials = 200

    gen_gt = torch.Generator().manual_seed(0)
    log_I0 = torch.empty(n_hkl).uniform_(np.log(1.0), np.log(10000.0), generator=gen_gt)
    I0_true = torch.exp(log_I0)

    bins = np.array([0, 3, 10, 30, 100, 300, 1000, 10000, 1e6])

    for alpha_val in [0.4, 1.2]:
        alpha_true = torch.full((n_hkl,), alpha_val)
        beta_true = torch.full((n_hkl,), beta_val)

        alpha_raw_all = np.zeros((n_trials, n_hkl))
        alpha_corr_all = np.zeros((n_trials, n_hkl))

        print(f"\n{'='*60}")
        print(f"ALPHA = {alpha_val}, n_trials={n_trials}")
        print(f"{'='*60}")
        t0 = time.time()

        for trial in range(n_trials):
            data = generate_one_trial(I0_true, alpha_true, beta_val, obs_per_hkl,
                                      background, seed=1000 + trial)
            I_obs = data["I_obs"].to(device)
            hkl_id = data["hkl_id"].to(device)
            obs_counts = data["obs_counts"].to(device)
            scale = torch.ones(len(I_obs), device=device)

            offsets = torch.zeros(n_hkl + 1, dtype=torch.long, device=device)
            offsets[1:] = obs_counts.cumsum(0)
            med = torch.zeros(n_hkl, device=device)
            for h in range(n_hkl):
                med[h] = I_obs[offsets[h]:offsets[h+1]].median()
            I0_init = med.clamp(min=0.1)

            I0_p, _ = fit_I0_poisson_log(
                I_obs, hkl_id, scale, alpha_true.to(device), beta_true.to(device),
                I0_init, obs_counts, n_hkl, bg=background, n_quad=51, max_iter=50)

            pos = I_obs > 0
            I_pos = I_obs[pos]
            hkl_pos = hkl_id[pos]
            obs_counts_pos = torch.zeros(n_hkl, dtype=torch.long, device=device)
            obs_counts_pos.scatter_add_(0, hkl_pos, torch.ones_like(hkl_pos))

            alpha_raw = fit_alpha(I_pos, hkl_pos, I0_p, beta_true.to(device),
                                  obs_counts_pos, n_hkl)

            alpha_raw_np = alpha_raw.cpu().numpy()
            # σ_poi per HKL: 1/sqrt(I0 + bg)
            sigma_poi = (1.0 / np.sqrt(np.clip(I0_p.cpu().numpy() + background, 1, None)))

            alpha_corr = correct_alpha_table(
                alpha_raw_np, sigma_poi, beta_val,
                alpha_grid, sigma_grid, table)

            alpha_raw_all[trial] = alpha_raw_np
            alpha_corr_all[trial] = alpha_corr

            if (trial + 1) % 50 == 0:
                print(f"  trial {trial+1}/{n_trials}  ({time.time()-t0:.1f}s)")

        print(f"Done in {time.time()-t0:.1f}s")

        print(f"\n  {'I0 range':>16s}  {'raw α':>10s}  {'corrected α':>12s}  {'true':>8s}")
        true_np = I0_true.numpy()
        for i in range(len(bins) - 1):
            mask = (true_np >= bins[i]) & (true_np < bins[i+1])
            if mask.sum() == 0:
                continue
            raw = np.mean(alpha_raw_all[:, mask])
            corr = np.mean(alpha_corr_all[:, mask])
            print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {raw:10.4f}  "
                  f"{corr:12.4f}  {alpha_val:8.4f}")

        raw_rmse = np.sqrt(np.mean((alpha_raw_all - alpha_val)**2))
        corr_rmse = np.sqrt(np.mean((alpha_corr_all - alpha_val)**2))
        print(f"\n  RMSE: raw={raw_rmse:.4f}, corrected={corr_rmse:.4f}  "
              f"(improvement: {(1-corr_rmse/raw_rmse)*100:.0f}%)")

    # ── Plot correction table ───────────────────────────────────
    outdir = os.path.dirname(__file__)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of the correction table
    ax = axes[0]
    im = ax.imshow(table, aspect="auto", origin="lower",
                   extent=[sigma_grid[0], sigma_grid[-1],
                           alpha_grid[0], alpha_grid[-1]])
    ax.set_xlabel("σ_poi = 1/√(I0+bg)")
    ax.set_ylabel("α_true")
    ax.set_title("α_MLE(α_true, σ_poi)")
    plt.colorbar(im, ax=ax)
    # Contour lines
    ax.contour(sigma_grid, alpha_grid, table,
               levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
               colors="white", linewidths=0.5)

    # Slices
    ax = axes[1]
    for at_val in [0.2, 0.4, 0.8, 1.2]:
        idx = np.argmin(np.abs(alpha_grid - at_val))
        ax.plot(sigma_grid, table[idx, :], "-", lw=2,
                label=f"α_true={at_val:.1f}")
        ax.axhline(at_val, ls=":", alpha=0.3)
    ax.set_xlabel("σ_poi = 1/√(I0+bg)")
    ax.set_ylabel("α_MLE (what fit_alpha returns)")
    ax.set_title("Inflation of α_MLE by Poisson noise")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(outdir, "mc_alpha_deconvolution.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
