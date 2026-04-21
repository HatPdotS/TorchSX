"""Visualize random sample ROIs at various signal percentiles.

Writes PNG images showing observed, predicted, and residual patches
for ROIs sampled from different integrated intensity percentile bins.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from torchsx import SFXStream
from torchsx.scaling import (
    CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs,
)
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import (
    RefinableGeometry, RelpProfile, PCASpectrum,
    PixelForwardModel, PixelLikelihood, precompute_spectral_overlap,
)
from torchsx.forward.batch import ROIBatchLoader
from torchsx.io._roi_store import ROIStore

OUT_DIR = "roi_plots"
N_SAMPLES = 5  # samples per percentile bin
SEED = 123

# ── Load ─────────────────────────────────────────────────────────────────
print("Loading...")
stream = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
stream = stream.remap_filenames(
    r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
    "/das/work/p17/p17490/Peter/torchsx/h5files" + r"/\1/",
)
core = stream._core
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)

roi_store = ROIStore("delay_-45_rois_11x11.safetensors")
grids = precompute_spectral_overlap(
    np.zeros(1), np.zeros((1, 1)),
    cache_path="delay_-45_pca_grids_v2.npz",
)
crystal_to_shot = np.arange(core.n_crystals) % 12194

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile(n_unique=hkl_idx.n_unique, sigma_r_init=0.001, sigma_t_init=0.0007)
pca_spectrum = PCASpectrum(grids, core.n_crystals, crystal_to_shot)

model = PixelForwardModel(
    structure_factors=sf, orientation=orientation,
    crystal_scale=crystal_scale, crystal_bfactor=crystal_bfactor,
    relp_profile=relp, spectrum=pca_spectrum,
    geometry=geometry, wavelength=inp.wavelength,
).to(device)

loader = ROIBatchLoader(
    roi_store=roi_store, core=core, hkl_index=hkl_idx,
    geometry=geometry, n_crystals_per_batch=500, device=device,
)

# ── Collect all ROIs ─────────────────────────────────────────────────────
print("Running forward on all batches...")

all_obs = []
all_pred = []
all_signal = []
all_valid = []
all_integral_center = []
all_bg = []
all_F_sq = []
all_scale = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        result = model.forward_debug(batch)
        c = batch.patches.shape[1] // 2

        all_obs.append(batch.patches.float().cpu().numpy())
        all_pred.append(result.n_pred.cpu().numpy())
        all_signal.append(result.n_signal.cpu().numpy())
        all_valid.append((batch.valid > 0).cpu().numpy())
        all_integral_center.append(result.integral[:, c, c].cpu().numpy())
        all_bg.append(batch.background.cpu().numpy())
        all_F_sq.append(result.F_sq.cpu().numpy())
        all_scale.append(result.scale.cpu().numpy())

obs = np.concatenate(all_obs)
pred = np.concatenate(all_pred)
signal = np.concatenate(all_signal)
valid = np.concatenate(all_valid)
integral = np.concatenate(all_integral_center)
bg = np.concatenate(all_bg)
F_sq = np.concatenate(all_F_sq)
scale = np.concatenate(all_scale)

N = len(obs)
print(f"Total ROIs: {N:,}")

# Compute per-ROI integrated signal (sum of signal over valid pixels)
integrated_signal = np.array([signal[i][valid[i]].sum() for i in range(N)])

# ── Sample ROIs from percentile bins ─────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.RandomState(SEED)

# Sort by integrated signal
order = np.argsort(integrated_signal)[::-1]  # descending

bins = [
    ("top_1pct",  0.00, 0.01),
    ("top_5pct",  0.01, 0.05),
    ("top_10pct", 0.05, 0.10),
    ("top_20pct", 0.10, 0.20),
    ("top_50pct", 0.20, 0.50),
    ("top_90pct", 0.50, 0.90),
]

for bin_name, frac_lo, frac_hi in bins:
    idx_lo = int(N * frac_lo)
    idx_hi = int(N * frac_hi)
    pool = order[idx_lo:idx_hi]
    if len(pool) == 0:
        continue
    samples = rng.choice(pool, size=min(N_SAMPLES, len(pool)), replace=False)

    fig, axes = plt.subplots(N_SAMPLES, 4, figsize=(16, 4 * N_SAMPLES))
    if N_SAMPLES == 1:
        axes = axes[None, :]

    fig.suptitle(f"ROIs: {bin_name} (integrated signal rank {frac_lo*100:.0f}%-{frac_hi*100:.0f}%)",
                 fontsize=14, y=1.01)

    for row, idx in enumerate(samples):
        o = obs[idx]
        p = pred[idx]
        s = signal[idx]
        v = valid[idx]
        resid = o - p
        # Mask invalid pixels
        o_masked = np.where(v, o, np.nan)
        p_masked = np.where(v, p, np.nan)
        resid_masked = np.where(v, resid, np.nan)
        s_masked = np.where(v, s, np.nan)

        # Color scale: symmetric for residuals, shared for obs/pred
        vmax_obs = max(np.nanmax(np.abs(o_masked)), np.nanmax(np.abs(p_masked)), 1)
        vmax_resid = max(np.nanmax(np.abs(resid_masked)), 1)

        nll_pix = np.nanmean(
            0.5 * resid_masked**2 / np.clip(p_masked, 1, None)
            + 0.5 * np.log(np.clip(p_masked, 1, None))
        )

        # Observed
        ax = axes[row, 0]
        im = ax.imshow(o_masked, cmap="viridis", vmin=0, vmax=vmax_obs,
                        interpolation="nearest", origin="lower")
        ax.set_title(f"Observed (max={np.nanmax(o_masked):.0f})")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Predicted
        ax = axes[row, 1]
        im = ax.imshow(p_masked, cmap="viridis", vmin=0, vmax=vmax_obs,
                        interpolation="nearest", origin="lower")
        ax.set_title(f"Predicted (max={np.nanmax(p_masked):.1f})")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Signal only (no background)
        ax = axes[row, 2]
        im = ax.imshow(s_masked, cmap="hot", vmin=0,
                        interpolation="nearest", origin="lower")
        ax.set_title(f"Signal (int={integral[idx]:.4f})")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Residual
        ax = axes[row, 3]
        im = ax.imshow(resid_masked, cmap="RdBu_r", vmin=-vmax_resid, vmax=vmax_resid,
                        interpolation="nearest", origin="lower")
        ax.set_title(f"Residual (NLL/pix={nll_pix:.1f})")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Annotation
        axes[row, 0].set_ylabel(
            f"F²={F_sq[idx]:.1f}  scale={scale[idx]:.3f}\n"
            f"bg={bg[idx]:.1f}  sig_tot={integrated_signal[idx]:.1f}",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{bin_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

# ── Summary plot: NLL vs integrated signal ────────────────────────────────
print("Computing per-ROI NLL...")
roi_nll = np.zeros(N)
roi_npix = np.zeros(N)
for i in range(N):
    m = valid[i]
    o = obs[i][m]
    p = pred[i][m]
    roi_nll[i] = (0.5 * (o - p)**2 / np.clip(p, 1, None) + 0.5 * np.log(np.clip(p, 1, None))).sum()
    roi_npix[i] = m.sum()

roi_nll_pp = roi_nll / np.clip(roi_npix, 1, None)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# NLL/pix vs integrated signal
ax = axes[0]
ax.scatter(integrated_signal, roi_nll_pp, s=0.5, alpha=0.3)
ax.set_xlabel("Integrated signal (photons)")
ax.set_ylabel("NLL / pixel")
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_title("NLL vs integrated signal")
ax.axhline(y=np.median(roi_nll_pp), color="r", ls="--", alpha=0.5,
           label=f"median={np.median(roi_nll_pp):.1f}")
ax.legend()

# Histogram of integrated signal
ax = axes[1]
ax.hist(np.log10(np.clip(integrated_signal, 1e-3, None)), bins=100, alpha=0.7)
ax.set_xlabel("log10(integrated signal)")
ax.set_ylabel("count")
ax.set_title("Distribution of integrated signal")

# Histogram of center-pixel integral
ax = axes[2]
ax.hist(np.log10(np.clip(integral, 1e-8, None)), bins=100, alpha=0.7)
ax.set_xlabel("log10(center-pixel integral)")
ax.set_ylabel("count")
ax.set_title("Distribution of spectral integral")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "summary.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out_path}")

print("\nDone.")
