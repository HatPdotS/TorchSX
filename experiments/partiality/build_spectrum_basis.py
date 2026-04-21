#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -p hour
#SBATCH -t 1:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/build_spectrum_basis.out

"""Build the SASE spectrum basis from BS data files.

Loads ~12k SASE spectra, normalises to a common energy axis, decomposes via SVD
to find the principal modes.  Also computes convolved spectra at multiple relp
widths γ — these decompose better than raw spectra (low-pass filtering).

Output: basis tensor (K, n_energy) and convolved bases at γ ∈ {γ_1, ..., γ_M}.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BS_DIR = Path("/das/work/p17/p17490/Peter/torchsx/experiments/sfx_scan/bs_data_files")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/results")
OUTDIR.mkdir(exist_ok=True)

E0 = 12960.0  # central beam energy (eV)
N_ENERGY_BINS = 256  # resampled common energy axis
DELTA_E_RANGE = 80.0  # ±40 eV around E0

# Reference reciprocal vector magnitude for the convolution.
# For each observation, the actual q rescales the spectrum in |r| space.
Q_REF = 0.5  # Å⁻¹ (corresponds to ~2 Å resolution)

# Relp widths for convolved-basis decomposition
GAMMA_VALUES = np.array([1e-3, 2e-3, 4e-3, 6e-3, 8e-3])  # Å⁻¹

N_BASIS = 10  # number of principal components to keep

# ── Load all spectra ──────────────────────────────────────────
print("Loading spectra...")
files = sorted(BS_DIR.glob("acq*.BSDATA.h5"))
print(f"Found {len(files)} files")

all_spectra = []
all_pulse_ids = []
all_e_axes = []

for f in files:
    try:
        with h5py.File(f, "r") as h:
            x = h["SARFE10-PSSS059:SPECTRUM_X/data"][:]  # (N, 2560) eV
            y = h["SARFE10-PSSS059:SPECTRUM_Y/data"][:]  # (N, 2560) intensity
            pid = h["SARFE10-PSSS059:SPECTRUM_X/pulse_id"][:]
            # Filter pulses with valid signal
            valid = (y.sum(axis=1) > 0)
            all_spectra.append(y[valid])
            all_pulse_ids.append(pid[valid])
            all_e_axes.append(x[valid])
    except Exception as e:
        print(f"  skip {f.name}: {e}")

spectra = np.concatenate(all_spectra, axis=0)
pulse_ids = np.concatenate(all_pulse_ids)
e_axes = np.concatenate(all_e_axes, axis=0)

print(f"Loaded {len(spectra)} spectra, axis shape per pulse: {e_axes.shape}")

# ── Resample to common energy axis ────────────────────────────
print(f"\nResampling to common axis: {N_ENERGY_BINS} bins, ±{DELTA_E_RANGE/2:.1f} eV around {E0} eV")

e_common = np.linspace(E0 - DELTA_E_RANGE/2, E0 + DELTA_E_RANGE/2, N_ENERGY_BINS)
spec_resampled = np.zeros((len(spectra), N_ENERGY_BINS), dtype=np.float32)

for i in range(len(spectra)):
    spec_resampled[i] = np.interp(e_common, e_axes[i], spectra[i], left=0, right=0)

# Normalise each spectrum by its sum
sums = spec_resampled.sum(axis=1, keepdims=True).clip(min=1e-10)
spec_normed = spec_resampled / sums

print(f"  median total intensity: {np.median(sums):.0f}")
print(f"  spectra with all zeros: {(sums.flatten() < 1).sum()}")

# Filter out empty spectra
keep = sums.flatten() > 1
spec_normed = spec_normed[keep]
print(f"  kept: {len(spec_normed)} spectra")

# ── Raw spectrum decomposition ────────────────────────────────
print("\n── Raw spectrum SVD ──")
mean_spec = spec_normed.mean(axis=0)
spec_centered = spec_normed - mean_spec

U, S, Vt = np.linalg.svd(spec_centered, full_matrices=False)
print(f"  Singular values: {S[:10]}")
print(f"  Variance explained by top 10: {(S[:10]**2 / (S**2).sum() * 100).cumsum()}")

# ── Convolved spectrum decomposition ──────────────────────────
print("\n── Convolved spectrum SVD ──")

# Convert energy axis to wavelength deviation (Δλ/λ = -ΔE/E)
# Then to |r| coordinate via |r| = q_ref × Δλ/λ = -q_ref × ΔE/E
delta_E = e_common - E0
delta_lambda_over_lambda = -delta_E / E0  # signed
r_coord = Q_REF * delta_lambda_over_lambda  # signed |r| coord (Å⁻¹)
print(f"  r coord range: [{r_coord.min():.5f}, {r_coord.max():.5f}] Å⁻¹")

# For each γ, convolve each spectrum with a Gaussian relp profile of width γ
# in |r| space.  The relp profile is in |r|, so we convolve the spectrum
# (now expressed as a function of r_coord) with G(r; 0, γ).

dr = r_coord[1] - r_coord[0]
print(f"  r coord spacing: {dr:.6f} Å⁻¹")

convolved_per_gamma = {}
basis_per_gamma = {}
mean_per_gamma = {}

for gamma in GAMMA_VALUES:
    # Gaussian kernel in r space
    sigma_in_bins = gamma / abs(dr)
    if sigma_in_bins < 0.5:
        # Spectrum already finer than γ — use raw
        conv = spec_normed.copy()
    else:
        # FFT-based Gaussian convolution along the energy axis
        from scipy.ndimage import gaussian_filter1d
        conv = gaussian_filter1d(spec_normed, sigma=sigma_in_bins, axis=1, mode="constant")
    # Re-normalise after convolution
    conv = conv / conv.sum(axis=1, keepdims=True).clip(min=1e-10)

    mean_c = conv.mean(axis=0)
    centered_c = conv - mean_c
    Uc, Sc, Vct = np.linalg.svd(centered_c, full_matrices=False)
    var_explained = (Sc[:N_BASIS] ** 2 / (Sc ** 2).sum() * 100).cumsum()
    print(f"  γ={gamma*1e3:.1f}e-3 Å⁻¹: top {N_BASIS} components explain "
          f"{var_explained[-1]:.1f}% of variance")
    print(f"    cumulative: {var_explained}")

    convolved_per_gamma[gamma] = conv
    basis_per_gamma[gamma] = Vct[:N_BASIS]  # (N_BASIS, N_ENERGY_BINS)
    mean_per_gamma[gamma] = mean_c

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# Top row: example raw spectra
ax = axes[0, 0]
for i in range(min(20, len(spec_normed))):
    ax.plot(e_common, spec_normed[i], lw=0.5, alpha=0.6)
ax.plot(e_common, mean_spec, "k-", lw=2, label="mean")
ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Intensity (norm)")
ax.set_title(f"Example raw spectra (n={len(spec_normed)})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Cumulative variance for raw
ax = axes[0, 1]
var_cum = (S ** 2 / (S ** 2).sum() * 100).cumsum()
ax.plot(np.arange(1, len(var_cum) + 1), var_cum, "C0-o", ms=4)
ax.set_xlabel("Component"); ax.set_ylabel("Cumulative variance (%)")
ax.set_title(f"Raw spectrum variance ({var_cum[N_BASIS-1]:.1f}% with top {N_BASIS})")
ax.set_xlim(0, 30); ax.grid(True, alpha=0.3)
ax.axhline(95, color="gray", ls="--", lw=1)

# Top components for raw
ax = axes[0, 2]
for k in range(min(5, Vt.shape[0])):
    ax.plot(e_common, Vt[k], lw=1.2, label=f"PC{k+1}")
ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Component amplitude")
ax.set_title("Raw spectrum: top 5 components")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Middle/bottom rows: convolved at different γ
sorted_gammas = sorted(GAMMA_VALUES)
for i, gamma in enumerate(sorted_gammas[:3]):
    ax = axes[1, i]
    conv = convolved_per_gamma[gamma]
    for j in range(min(20, len(conv))):
        ax.plot(e_common, conv[j], lw=0.5, alpha=0.6)
    ax.plot(e_common, mean_per_gamma[gamma], "k-", lw=2)
    ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Intensity (norm)")
    ax.set_title(f"Convolved γ={gamma*1e3:.1f}e-3 Å⁻¹")
    ax.grid(True, alpha=0.3)

for i, gamma in enumerate(sorted_gammas[:3]):
    ax = axes[2, i]
    basis = basis_per_gamma[gamma]
    for k in range(min(5, basis.shape[0])):
        ax.plot(e_common, basis[k], lw=1.2, label=f"PC{k+1}")
    ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Component")
    ax.set_title(f"γ={gamma*1e3:.1f}e-3: top components")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTDIR / "spectrum_basis.png", dpi=150)
print(f"\nSaved spectrum_basis.png")
plt.close(fig)

# ── Save basis ────────────────────────────────────────────────
torch.save({
    "e_axis_eV": torch.from_numpy(e_common).float(),
    "E0": E0,
    "Q_REF": Q_REF,
    "delta_lambda_over_lambda": torch.from_numpy(delta_lambda_over_lambda).float(),
    "r_coord": torch.from_numpy(r_coord).float(),
    "gamma_values": torch.from_numpy(GAMMA_VALUES).float(),
    "raw_basis": torch.from_numpy(Vt[:N_BASIS]).float(),
    "raw_mean": torch.from_numpy(mean_spec).float(),
    "raw_variance_explained": torch.from_numpy((S[:N_BASIS]**2/(S**2).sum()).cumsum()).float(),
    "convolved_basis_per_gamma": {
        float(g): torch.from_numpy(basis_per_gamma[g]).float() for g in GAMMA_VALUES
    },
    "convolved_mean_per_gamma": {
        float(g): torch.from_numpy(mean_per_gamma[g]).float() for g in GAMMA_VALUES
    },
    "n_spectra": len(spec_normed),
}, OUTDIR / "spectrum_basis.pt")

print(f"Saved spectrum_basis.pt: {len(spec_normed)} spectra, {N_BASIS} basis components")
print("Done.")
