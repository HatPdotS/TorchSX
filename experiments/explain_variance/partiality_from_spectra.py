#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Partiality simulation using measured per-shot spectra + B-factor relp model.

The relp width is resolution-dependent (disorder broadening):
    σ_relp(d) = √(σ₀² + B_eff × s²)   where s = 1/(2d)

For each measured spectrum S(E), partiality at Ewald offset δ:
    p(δ) = ∫ relp(δ - ε; σ_relp) × shell(ε) dε

Vectorised: all spectra × all offsets in one matmul per (σ_relp, d).
"""

import numpy as np
import h5py
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
import os, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

OBSERVED_SIGMA_LN = 1.56
BS_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"


def load_spectra(max_spectra=5000):
    """Load spectra. Returns (N_spec, N_channels) arrays."""
    files = sorted(glob.glob(f"{BS_DIR}/acq*.BSDATA.h5"))
    all_x, all_y = [], []
    for f_path in files:
        try:
            f = h5py.File(f_path, "r")
            sx = f["SARFE10-PSSS059:SPECTRUM_X/data"][:]
            sy = f["SARFE10-PSSS059:SPECTRUM_Y/data"][:]
            f.close()
        except Exception:
            continue
        good = np.clip(sy, 0, None).sum(axis=1) > 100
        all_x.append(sx[good])
        all_y.append(sy[good])
        if sum(len(x) for x in all_x) >= max_spectra:
            break
    X = np.concatenate(all_x)[:max_spectra]
    Y = np.concatenate(all_y)[:max_spectra]
    return X, Y


def precompute_shells(spec_x, spec_y, d_spacing, E0):
    """
    Convert all spectra to Ewald shell profiles at a given d-spacing.

    Returns (shell_offsets, shell_weights) where:
      shell_offsets: (N_channels,) — shared offset grid (all spectra have same x)
      shell_weights: (N_spec, N_channels) — normalised weights per spectrum
    """
    s = 1.0 / d_spacing
    # All spectra share the same energy grid (verified from data)
    offsets = s * (spec_x[0] - E0) / E0  # (N_channels,)
    doff = np.abs(np.median(np.diff(offsets)))

    weights = np.clip(spec_y, 0, None)  # (N_spec, N_channels)
    row_sums = weights.sum(axis=1, keepdims=True) * doff
    row_sums = np.clip(row_sums, 1e-10, None)
    weights = weights / row_sums  # normalised

    return offsets, weights, doff


def simulate_fast(shell_offsets, shell_weights, doff, sigma_relp,
                  n_offsets=300, threshold=0.01):
    """
    Compute partiality for all spectra at uniform offsets. Fully vectorised.

    shell_offsets: (M,) reciprocal-space positions of shell channels
    shell_weights: (N_spec, M) normalised spectral weights
    sigma_relp: float — Gaussian relp width (Å⁻¹)

    Returns log(p) for all (spectrum, offset) pairs above threshold.
    """
    # Offset sampling range
    shell_extent = shell_offsets[-1] - shell_offsets[0]
    cutoff = 3 * sigma_relp + shell_extent / 2
    deltas = np.linspace(-cutoff, cutoff, n_offsets)  # (D,)

    # Relp evaluated at (delta - shell_offset) for all combinations
    # diff: (D, M)
    diff = deltas[:, None] - shell_offsets[None, :]
    relp = np.exp(-0.5 * diff**2 / sigma_relp**2)  # (D, M)

    # Partiality: matrix multiply (N_spec, M) × (M, D) → (N_spec, D)
    p = shell_weights @ (relp * doff).T  # (N_spec, D)

    # Flatten, threshold, return log
    p_flat = p.ravel()
    detected = p_flat > threshold
    if detected.sum() < 10:
        return np.array([])
    return np.log(p_flat[detected])


def main():
    t0 = time.time()
    print("Loading spectra...")
    spec_x, spec_y = load_spectra(max_spectra=5000)
    print(f"  {len(spec_x):,} spectra, {spec_x.shape[1]} channels")

    E0 = np.median(spec_x)
    centroids = np.array([np.sum(x * np.clip(y, 0, None)) / np.clip(y, 0, None).sum()
                          for x, y in zip(spec_x, spec_y)])
    print(f"  E0 = {E0:.1f} eV, σ(centroid) = {np.std(centroids):.2f} eV")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    d_spacings = [0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

    sigma0_list = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]       # Å⁻¹
    B_eff_list = [0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]  # controls resolution dependence

    print(f"\nSimulating {len(sigma0_list)}×{len(B_eff_list)}×{len(d_spacings)} = "
          f"{len(sigma0_list)*len(B_eff_list)*len(d_spacings)} parameter points...")

    results = {}  # (sigma0, B_eff, d) → σ(log p)

    for d in d_spacings:
        # Precompute shells once per d
        shell_off, shell_w, doff = precompute_shells(spec_x, spec_y, d, E0)

        for sigma0 in sigma0_list:
            for B_eff in B_eff_list:
                s = 1.0 / (2 * d)
                sigma_relp = np.sqrt(sigma0**2 + B_eff * s**2)

                log_p = simulate_fast(shell_off, shell_w, doff, sigma_relp)
                if len(log_p) < 100:
                    results[(sigma0, B_eff, d)] = np.nan
                    continue
                results[(sigma0, B_eff, d)] = np.std(log_p)

        print(f"  d = {d:.1f} Å done ({time.time()-t0:.0f}s)")

    # Print table for selected params
    print(f"\n{'σ₀':>8} {'B_eff':>8} | ", end="")
    for d in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        print(f"d={d:<4} ", end="")
    print()
    print("─" * 80)
    for sigma0 in sigma0_list:
        for B_eff in B_eff_list:
            print(f"{sigma0:8.0e} {B_eff:8.0e} | ", end="")
            for d in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
                v = results.get((sigma0, B_eff, d), np.nan)
                print(f"{v:5.3f}  " if np.isfinite(v) else "  N/A  ", end="")
            print()

    # ── Figures ──

    # Fig 1: σ(log p) vs resolution for each (σ₀, B_eff)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Partiality variance from measured SASE spectra\n"
        "σ_relp(d) = √(σ₀² + B_eff/(4d²))",
        fontsize=12, fontweight="bold",
    )

    # Observed curve
    d_fine = np.array(d_spacings)
    sigma_obs = np.clip(2.25 - 1.16 / d_fine**2, 0.3, None)

    for ax, sigma0 in zip(axes.flat[:len(sigma0_list)], sigma0_list):
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(B_eff_list)))
        for B_eff, col in zip(B_eff_list, colors):
            sigs = [results.get((sigma0, B_eff, d), np.nan) for d in d_spacings]
            label = f"B={B_eff:.0e}" if B_eff > 0 else "B=0"
            ax.plot(d_spacings, sigs, "o-", color=col, lw=1.5, ms=3, label=label)
        ax.plot(d_fine, sigma_obs, "r--", lw=2, alpha=0.7, label="observed")
        ax.axhline(OBSERVED_SIGMA_LN, color="red", ls=":", lw=1, alpha=0.4)
        ax.set_xlabel("d-spacing (Å)")
        ax.set_ylabel("σ(log p)")
        ax.set_title(f"σ₀ = {sigma0:.0e} Å⁻¹")
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylim(0, 2.5)

    for ax in axes.flat[len(sigma0_list):]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUTDIR / "fig_partiality_from_spectra.png", dpi=150)
    print(f"\n  → {OUTDIR / 'fig_partiality_from_spectra.png'}")
    plt.close(fig)

    # Fig 2: Best-fit to observed curve
    print("\nFinding best-fit parameters...")
    best_cost = np.inf
    best_params = None
    for sigma0 in sigma0_list:
        for B_eff in B_eff_list:
            sigs = []
            obs = []
            for d in d_spacings:
                v = results.get((sigma0, B_eff, d), np.nan)
                if np.isfinite(v):
                    sigs.append(v)
                    obs.append(max(2.25 - 1.16 / d**2, 0.3))
            if len(sigs) < 5:
                continue
            cost = np.mean((np.array(sigs) - np.array(obs))**2)
            if cost < best_cost:
                best_cost = cost
                best_params = (sigma0, B_eff)

    if best_params:
        s0, B = best_params
        print(f"  Best fit: σ₀ = {s0:.1e} Å⁻¹, B_eff = {B:.1e}")
        print(f"  RMS error: {np.sqrt(best_cost):.4f}")

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sigs = [results.get((s0, B, d), np.nan) for d in d_spacings]
        ax.plot(d_spacings, sigs, "o-", color="#534AB7", lw=2, ms=6,
                label=f"Simulated: σ₀={s0:.0e}, B={B:.0e}")
        d_f = np.linspace(0.9, 10, 100)
        ax.plot(d_f, np.clip(2.25 - 1.16 / d_f**2, 0.3, None), "r--", lw=2,
                label="Observed: 2.25 − 1.16/d²")
        ax.set_xlabel("d-spacing (Å)")
        ax.set_ylabel("σ(log p)")
        ax.set_title(f"Best-fit partiality model vs observed σ_LN(d)\n"
                     f"σ₀={s0:.0e}, B_eff={B:.0e}, RMSE={np.sqrt(best_cost):.3f}")
        ax.legend()
        ax.set_ylim(0, 2.5)
        fig.tight_layout()
        fig.savefig(OUTDIR / "fig_bestfit_vs_observed.png", dpi=150)
        print(f"  → {OUTDIR / 'fig_bestfit_vs_observed.png'}")
        plt.close(fig)

    # Fig 3: Spectra
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in np.random.choice(len(spec_x), 30, replace=False):
        ax.plot(spec_x[i], np.clip(spec_y[i], 0, None), alpha=0.15, color="#534AB7", lw=0.5)
    mean_y = np.mean(spec_y[:2000], axis=0)
    ax.plot(spec_x[0], np.clip(mean_y, 0, None), color="red", lw=2, label="mean")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity")
    ax.set_title("Shot-to-shot SASE spectra")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_spectral_variation.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_spectral_variation.png'}")
    plt.close(fig)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Done. Figures in {OUTDIR}/")


if __name__ == "__main__":
    np.random.seed(42)
    main()
