#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Does the SASE spectral structure matter for partiality variance?

Compare σ(log p) from:
  1. Real measured SASE spectra (spiky, variable)
  2. Per-shot Gaussian with same FWHM as each real spectrum
  3. Single mean Gaussian (average FWHM, no shot-to-shot variation)
  4. Top-hat with same FWHM

If (1) ≈ (2): SASE spikiness doesn't matter, only the bandwidth
If (2) > (3): shot-to-shot BW variation adds variance
If (1) > (2): the multi-mode SASE structure itself adds variance
"""

import numpy as np
import h5py
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

BS_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"
OBSERVED_SIGMA_LN = 1.56


def load_spectra(max_spectra=5000):
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
    return np.concatenate(all_x)[:max_spectra], np.concatenate(all_y)[:max_spectra]


def measure_fwhm(x, y):
    """FWHM of each spectrum. Returns (N,) array in eV."""
    yc = np.clip(y, 0, None)
    fwhms = np.zeros(len(x))
    for i in range(len(x)):
        hm = yc[i].max() / 2
        above = yc[i] > hm
        if above.any():
            fwhms[i] = x[i][above][-1] - x[i][above][0]
    return fwhms


def make_gaussian_spectra(x_grid, centroids, fwhms):
    """Generate Gaussian spectra with given centroids and FWHMs."""
    sigmas = fwhms / 2.355
    # (N, M) where M = len(x_grid[0])
    diff = x_grid - centroids[:, None]  # (N, M)
    y = np.exp(-0.5 * diff**2 / sigmas[:, None]**2)
    return y


def make_tophat_spectra(x_grid, centroids, fwhms):
    """Generate top-hat spectra with given centroids and FWHMs."""
    hw = fwhms / 2
    y = ((x_grid - centroids[:, None]).abs() if hasattr(x_grid, 'abs')
         else np.abs(x_grid - centroids[:, None])) <= hw[:, None]
    return y.astype(float)


def precompute_shells(spec_x, spec_y, d_spacing, E0):
    s = 1.0 / d_spacing
    offsets = s * (spec_x[0] - E0) / E0
    doff = np.abs(np.median(np.diff(offsets)))
    weights = np.clip(spec_y, 0, None)
    row_sums = weights.sum(axis=1, keepdims=True) * doff
    row_sums = np.clip(row_sums, 1e-10, None)
    weights = weights / row_sums
    return offsets, weights, doff


def simulate_fast(shell_offsets, shell_weights, doff, sigma_relp,
                  n_offsets=300, threshold=0.01):
    shell_extent = shell_offsets[-1] - shell_offsets[0]
    cutoff = 3 * sigma_relp + shell_extent / 2
    deltas = np.linspace(-cutoff, cutoff, n_offsets)
    diff = deltas[:, None] - shell_offsets[None, :]
    relp = np.exp(-0.5 * diff**2 / sigma_relp**2)
    p = shell_weights @ (relp * doff).T
    p_flat = p.ravel()
    detected = p_flat > threshold
    if detected.sum() < 10:
        return np.nan
    return np.std(np.log(p_flat[detected]))


def main():
    t0 = time.time()
    print("Loading spectra...")
    spec_x, spec_y = load_spectra(5000)
    N = len(spec_x)
    print(f"  {N:,} spectra")

    E0 = np.median(spec_x)
    centroids = np.array([
        np.sum(x * np.clip(y, 0, None)) / np.clip(y, 0, None).sum()
        for x, y in zip(spec_x, spec_y)
    ])
    fwhms = measure_fwhm(spec_x, spec_y)
    mean_fwhm = np.mean(fwhms)
    print(f"  FWHM: mean={mean_fwhm:.2f} eV, std={np.std(fwhms):.2f} eV")
    print(f"  BW: mean={mean_fwhm/E0*100:.3f}%")

    # Generate synthetic spectra
    x_grid = spec_x  # same energy grid

    # 1. Per-shot Gaussian (same centroid + FWHM as each real spectrum)
    gauss_per_shot = make_gaussian_spectra(x_grid, centroids, fwhms)

    # 2. Single mean Gaussian (same for all shots)
    mean_centroid = np.mean(centroids)
    gauss_mean = make_gaussian_spectra(
        x_grid,
        np.full(N, mean_centroid),
        np.full(N, mean_fwhm),
    )

    # 3. Top-hat per shot
    tophat_per_shot = (np.abs(x_grid - centroids[:, None]) <= (fwhms / 2)[:, None]).astype(np.float32)

    # 4. Single mean top-hat
    tophat_one = (np.abs(x_grid[0] - mean_centroid) <= (mean_fwhm / 2)).astype(np.float32)
    tophat_mean = np.broadcast_to(tophat_one[None, :], (N, x_grid.shape[1])).copy()

    models = [
        ("Real SASE", spec_x, spec_y),
        ("Gaussian (per-shot BW)", spec_x, gauss_per_shot),
        ("Gaussian (mean BW)", spec_x, gauss_mean),
        ("Top-hat (per-shot BW)", spec_x, tophat_per_shot),
        ("Top-hat (mean BW)", spec_x, tophat_mean),
    ]

    # Scan d-spacings and a few σ_relp values
    d_spacings = [0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    sigma_relps = [1e-4, 2e-4, 5e-4, 1e-3]

    print(f"\nComparing spectral models...")
    all_results = {}

    for name, sx, sy in models:
        print(f"\n  {name}:")
        for sig_r in sigma_relps:
            sigs = []
            for d in d_spacings:
                shell_off, shell_w, doff = precompute_shells(sx, sy, d, E0)
                s = simulate_fast(shell_off, shell_w, doff, sig_r)
                sigs.append(s)
            all_results[(name, sig_r)] = sigs
            if sig_r in [2e-4, 5e-4]:
                vals = " ".join(f"{s:.3f}" if np.isfinite(s) else "N/A" for s in sigs)
                print(f"    σ_relp={sig_r:.0e}: {vals}")

    # ── Figure: comparison across models ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Does SASE spectral structure matter for partiality variance?\n"
        "Comparing real spectra vs smooth approximations",
        fontsize=12, fontweight="bold",
    )

    colors = ["#534AB7", "#D85A30", "#1D9E75", "#B5338A", "#666666"]
    markers = ["o", "s", "^", "D", "v"]

    for ax, sig_r in zip(axes.flat, sigma_relps):
        for (name, _, _), col, mk in zip(models, colors, markers):
            sigs = all_results[(name, sig_r)]
            ax.plot(d_spacings, sigs, f"{mk}-", color=col, lw=2, ms=5, label=name)
        ax.axhline(OBSERVED_SIGMA_LN, color="red", ls="--", lw=1.5, alpha=0.5)
        ax.set_xlabel("d-spacing (Å)")
        ax.set_ylabel("σ(log p)")
        ax.set_title(f"σ_relp = {sig_r:.0e} Å⁻¹")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 2.0)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUTDIR / "fig_spectral_shape_comparison.png", dpi=150)
    print(f"\n  → {OUTDIR / 'fig_spectral_shape_comparison.png'}")
    plt.close(fig)

    # ── Summary table at d=2 Å ──
    print("\n" + "=" * 75)
    print("SPECTRAL SHAPE COMPARISON — σ(log p)")
    print("=" * 75)
    d_ref = 2.0
    d_idx = d_spacings.index(d_ref)
    print(f"\n  At d = {d_ref} Å:")
    print(f"  {'Model':>30} | ", end="")
    for sig_r in sigma_relps:
        print(f"σ_r={sig_r:.0e} ", end="")
    print()
    print("  " + "─" * 70)
    for name, _, _ in models:
        print(f"  {name:>30} | ", end="")
        for sig_r in sigma_relps:
            v = all_results[(name, sig_r)][d_idx]
            print(f"  {v:.4f}  " if np.isfinite(v) else "   N/A   ", end="")
        print()

    # Compute relative differences
    print(f"\n  Relative to 'Real SASE' (% difference):")
    for name, _, _ in models[1:]:
        print(f"  {name:>30} | ", end="")
        for sig_r in sigma_relps:
            real = all_results[("Real SASE", sig_r)][d_idx]
            other = all_results[(name, sig_r)][d_idx]
            if np.isfinite(real) and np.isfinite(other) and real > 0:
                pct = (other - real) / real * 100
                print(f"  {pct:+5.1f}%  ", end="")
            else:
                print(f"   N/A   ", end="")
        print()
    print("=" * 75)

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    np.random.seed(42)
    main()
