#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -p hour
#SBATCH -t 1:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/study_spectrum_compression.out

"""Study the compressibility of SASE spectra under different representations.

Compares:
  (a) PCA on raw spectra (real space)
  (b) PCA on FFT magnitudes (translation-invariant)
  (c) PCA on low-frequency-truncated FFT (kept modes 0..K)
  (d) PCA on real-space-smoothed (Gaussian convolved) spectra

For each, compute:
  - Cumulative variance vs number of components
  - Reconstruction error vs information retained
  - Effective intrinsic dimension

Also: variance per Fourier mode → noise floor estimate.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BS_DIR = Path("/das/work/p17/p17490/Peter/torchsx/experiments/sfx_scan/bs_data_files")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/results")

E0 = 12960.0
N_BINS = 256
DELTA_E = 80.0  # ±40 eV around E0

# ── Load spectra (same as build_spectrum_basis.py) ────────────
print("Loading spectra...")
files = sorted(BS_DIR.glob("acq*.BSDATA.h5"))
all_y, all_x = [], []
for f in files:
    try:
        with h5py.File(f, "r") as h:
            x = h["SARFE10-PSSS059:SPECTRUM_X/data"][:]
            y = h["SARFE10-PSSS059:SPECTRUM_Y/data"][:]
            valid = (y.sum(axis=1) > 0)
            all_y.append(y[valid])
            all_x.append(x[valid])
    except Exception:
        pass
y_raw = np.concatenate(all_y, axis=0)
x_raw = np.concatenate(all_x, axis=0)
print(f"Loaded {len(y_raw)} spectra")

# Resample to common axis
e_common = np.linspace(E0 - DELTA_E/2, E0 + DELTA_E/2, N_BINS)
spec = np.zeros((len(y_raw), N_BINS), dtype=np.float32)
for i in range(len(y_raw)):
    spec[i] = np.interp(e_common, x_raw[i], y_raw[i], left=0, right=0)

# Filter and normalise
sums = spec.sum(axis=1)
keep = sums > 1
spec = spec[keep]
spec = spec / spec.sum(axis=1, keepdims=True)
print(f"Kept {len(spec)} non-empty spectra")

# Center the data
mean_spec = spec.mean(axis=0)
spec_centered = spec - mean_spec

# ══════════════════════════════════════════════════════════════
# (a) PCA on raw spectra (baseline)
# ══════════════════════════════════════════════════════════════
print("\n── (a) Raw real-space PCA ──")
U, S, Vt = np.linalg.svd(spec_centered, full_matrices=False)
var_a = S ** 2
var_a_cum = (var_a / var_a.sum() * 100).cumsum()
print(f"  Top 5 cumulative variance: {var_a_cum[:5]}")
print(f"  Components for 95%: {(var_a_cum >= 95).argmax() + 1}")
print(f"  Components for 99%: {(var_a_cum >= 99).argmax() + 1}")

# ══════════════════════════════════════════════════════════════
# (b) FFT analysis
# ══════════════════════════════════════════════════════════════
print("\n── FFT analysis ──")

# Compute FFT of each centered spectrum
spec_fft = np.fft.rfft(spec_centered, axis=1)  # (N, N_freq)
n_freq = spec_fft.shape[1]
print(f"  Number of (real) frequency bins: {n_freq}")

# Variance per frequency bin (across spectra)
var_per_freq = np.var(np.abs(spec_fft), axis=0)
power_per_freq = (np.abs(spec_fft) ** 2).mean(axis=0)
freqs = np.arange(n_freq) / DELTA_E  # cycles per eV
freqs_lambda = freqs * E0  # cycles per (δλ/λ × E0)

print(f"  Power at f=0: {power_per_freq[0]:.2e}")
print(f"  Power at f=N/4 (mid): {power_per_freq[n_freq//4]:.2e}")
print(f"  Power at f=N/2 (max): {power_per_freq[-1]:.2e}")
print(f"  Ratio (low/mid/high): "
      f"{power_per_freq[0]/power_per_freq[-1]:.2e} / "
      f"{power_per_freq[n_freq//4]/power_per_freq[-1]:.2e} / 1")

# ══════════════════════════════════════════════════════════════
# (c) FFT-truncated PCA at various cutoffs K
# ══════════════════════════════════════════════════════════════
print("\n── (c) FFT-truncated PCA ──")

K_values = [3, 5, 10, 15, 20, 30, 50]
truncated_results = {}

for K in K_values:
    # Truncate to first K Fourier modes (in addition to DC)
    spec_fft_trunc = spec_fft.copy()
    spec_fft_trunc[:, K:] = 0
    # Inverse FFT to get smoothed real-space spectra
    spec_lp = np.fft.irfft(spec_fft_trunc, n=N_BINS, axis=1)

    # PCA on the low-pass-filtered data (real space)
    spec_lp_centered = spec_lp - spec_lp.mean(axis=0)
    U_t, S_t, Vt_t = np.linalg.svd(spec_lp_centered, full_matrices=False)
    var_t = S_t ** 2
    var_t_cum = (var_t / var_t.sum() * 100).cumsum() if var_t.sum() > 0 else np.zeros(10)

    # Information retained = sum of var of original spectra in low-pass region
    info_retained = (var_t.sum() / var_a.sum()) * 100

    # Components needed for 95% / 99% of the LOW-PASS variance
    n_95 = (var_t_cum >= 95).argmax() + 1 if (var_t_cum >= 95).any() else len(var_t_cum)
    n_99 = (var_t_cum >= 99).argmax() + 1 if (var_t_cum >= 99).any() else len(var_t_cum)

    truncated_results[K] = {
        "spec_lp": spec_lp,
        "var_cum": var_t_cum,
        "info_retained": info_retained,
        "n_95": n_95,
        "n_99": n_99,
    }
    print(f"  K={K:>3}: info_retained={info_retained:5.1f}%  "
          f"n_components for 95%={n_95}  for 99%={n_99}")

# ══════════════════════════════════════════════════════════════
# (d) Real-space convolved spectra (for comparison)
# ══════════════════════════════════════════════════════════════
print("\n── (d) Real-space convolved spectra ──")

from scipy.ndimage import gaussian_filter1d

# Map γ in r-coord (Å⁻¹) to bins via Q_REF (assumes same scaling as build_spectrum_basis.py)
Q_REF = 0.5
gamma_values_real = [1e-3, 2e-3, 4e-3, 6e-3, 8e-3]
dr = (DELTA_E / E0) * Q_REF / N_BINS  # spacing in r-coord per bin

convolved_results = {}
for gamma in gamma_values_real:
    sigma_bins = gamma / abs(dr)
    if sigma_bins < 0.5:
        spec_conv = spec.copy()
    else:
        spec_conv = gaussian_filter1d(spec, sigma=sigma_bins, axis=1, mode="constant")
    spec_conv = spec_conv / spec_conv.sum(axis=1, keepdims=True).clip(1e-10)
    spec_conv_centered = spec_conv - spec_conv.mean(axis=0)
    U_c, S_c, Vt_c = np.linalg.svd(spec_conv_centered, full_matrices=False)
    var_c = S_c ** 2
    var_c_cum = (var_c / var_c.sum() * 100).cumsum() if var_c.sum() > 0 else np.zeros(10)
    info_retained = (var_c.sum() / var_a.sum()) * 100
    n_95 = (var_c_cum >= 95).argmax() + 1 if (var_c_cum >= 95).any() else len(var_c_cum)

    convolved_results[gamma] = {
        "var_cum": var_c_cum,
        "info_retained": info_retained,
        "sigma_bins": sigma_bins,
        "n_95": n_95,
    }
    print(f"  γ={gamma*1e3:.1f}e-3 ({sigma_bins:.1f} bins): "
          f"info={info_retained:5.1f}%  n95={n_95}")

# ══════════════════════════════════════════════════════════════
# (e) Effective dimension via "noise floor" estimate
# ══════════════════════════════════════════════════════════════
print("\n── (e) Noise floor analysis ──")

# Look at the eigenvalue spectrum of raw PCA
# The "noise floor" is where eigenvalues plateau
log_var = np.log10(var_a + 1e-30)
print(f"  log10 eigenvalues (top 20): {log_var[:20]}")
# Largest gap in log-eigenvalues
diffs = np.diff(log_var[:50])
gap_idx = diffs.argmin()  # most negative diff = biggest drop
print(f"  Largest drop at component {gap_idx + 1} → {gap_idx + 2}")

# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Cumulative variance for raw vs FFT-truncated
ax = axes[0, 0]
ax.plot(np.arange(1, 31), var_a_cum[:30], "k-o", lw=2, ms=4, label="Raw")
for K in [5, 10, 20]:
    var_cum = truncated_results[K]["var_cum"][:30]
    info = truncated_results[K]["info_retained"]
    ax.plot(np.arange(1, len(var_cum) + 1), var_cum,
            "-", lw=1.5, label=f"FFT trunc K={K} ({info:.0f}% info)")
ax.set_xlabel("Component"); ax.set_ylabel("Cumulative variance (%)")
ax.set_title("Variance retention vs PCA component")
ax.set_xlim(0, 30); ax.set_ylim(0, 105); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.axhline(95, color="gray", ls=":", lw=1)

# Plot 2: Power spectrum (variance per Fourier mode)
ax = axes[0, 1]
ax.semilogy(freqs_lambda, power_per_freq, "C0-")
ax.semilogy(freqs_lambda, var_per_freq, "C3--", label="Variance |c_k|")
ax.set_xlabel("Frequency (cycles per δλ/λ × E₀)")
ax.set_ylabel("Power")
ax.set_title("FFT power spectrum")
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 3: Information retained vs K
ax = axes[0, 2]
K_arr = np.array(K_values)
info_arr = np.array([truncated_results[K]["info_retained"] for K in K_values])
ax.plot(K_arr, info_arr, "C0-o", lw=2, ms=6)
ax.set_xlabel("FFT cutoff K (modes kept)")
ax.set_ylabel("Information retained (%)")
ax.set_title("FFT truncation: information loss")
ax.grid(True, alpha=0.3)
ax.axhline(95, color="gray", ls=":", lw=1, label="95%")
ax.legend()

# Plot 4: Example spectra at different truncations
ax = axes[1, 0]
n_show = 5
indices = np.linspace(0, len(spec) - 1, n_show).astype(int)
for j, idx in enumerate(indices):
    ax.plot(e_common, spec[idx] + j * 0.005, "k-", lw=0.5, alpha=0.5)
    ax.plot(e_common, truncated_results[5]["spec_lp"][idx] + mean_spec + j * 0.005,
            "C3-", lw=1.5, alpha=0.8)
ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Intensity (offset)")
ax.set_title(f"Black=raw, Red=FFT trunc K=5")
ax.grid(True, alpha=0.3)

# Plot 5: Variance per Fourier mode (focused on low-frequency)
ax = axes[1, 1]
ax.semilogy(np.arange(n_freq), var_per_freq, "C0-")
ax.set_xlabel("Fourier mode index")
ax.set_ylabel("Variance across spectra")
ax.set_title("Per-mode variance (where the info lives)")
ax.set_xlim(0, 50)
ax.grid(True, alpha=0.3)
# Mark the "knee"
diff = -np.diff(np.log(var_per_freq[:50] + 1e-30))
knee = diff.argmax() if len(diff) > 0 else 0
ax.axvline(knee, color="red", ls="--", lw=1, label=f"knee at k={knee}")
ax.legend()

# Plot 6: Comparison of all approaches
ax = axes[1, 2]
methods = [("Raw", var_a, 100.0)]
for K in [5, 10, 20]:
    methods.append((
        f"FFT K={K}",
        truncated_results[K]["var_cum"],
        truncated_results[K]["info_retained"],
    ))
for gamma in [2e-3, 4e-3]:
    methods.append((
        f"Conv γ={gamma*1e3:.0f}e-3",
        convolved_results[gamma]["var_cum"],
        convolved_results[gamma]["info_retained"],
    ))

# For raw, convert to cumulative percentage
raw_cum = (var_a / var_a.sum() * 100).cumsum()
ax.plot(np.arange(1, 21), raw_cum[:20], "k-o", lw=2, ms=4, label="Raw (100%)")
for K in [5, 10, 20]:
    vc = truncated_results[K]["var_cum"][:20]
    info = truncated_results[K]["info_retained"]
    ax.plot(np.arange(1, len(vc) + 1), vc, "--", lw=1.5,
            label=f"FFT K={K} ({info:.0f}%)")
for gamma in [2e-3, 4e-3]:
    vc = convolved_results[gamma]["var_cum"][:20]
    info = convolved_results[gamma]["info_retained"]
    ax.plot(np.arange(1, len(vc) + 1), vc, ":", lw=2,
            label=f"Conv γ={gamma*1e3:.0f}e-3 ({info:.0f}%)")
ax.set_xlabel("Component"); ax.set_ylabel("Cumulative variance of method (%)")
ax.set_title("Compressibility comparison")
ax.set_xlim(0, 20); ax.set_ylim(0, 105); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.axhline(95, color="gray", ls=":", lw=1)

fig.tight_layout()
fig.savefig(OUTDIR / "spectrum_compression_study.png", dpi=150)
print(f"\nSaved spectrum_compression_study.png")
plt.close(fig)

# Save analysis
torch.save({
    "n_spectra": len(spec),
    "raw_var_cum": torch.from_numpy(var_a_cum).float(),
    "raw_eigenvalues": torch.from_numpy(var_a).float(),
    "fft_per_freq_variance": torch.from_numpy(var_per_freq).float(),
    "fft_per_freq_power": torch.from_numpy(power_per_freq).float(),
    "truncation_results": {
        K: {
            "info_retained": truncated_results[K]["info_retained"],
            "var_cum": torch.from_numpy(truncated_results[K]["var_cum"]).float(),
            "n_95": truncated_results[K]["n_95"],
            "n_99": truncated_results[K]["n_99"],
        } for K in K_values
    },
    "convolved_results": {
        gamma: {
            "info_retained": convolved_results[gamma]["info_retained"],
            "var_cum": torch.from_numpy(convolved_results[gamma]["var_cum"]).float(),
            "n_95": convolved_results[gamma]["n_95"],
        } for gamma in gamma_values_real
    },
}, OUTDIR / "spectrum_compression_study.pt")

print("Done.")
