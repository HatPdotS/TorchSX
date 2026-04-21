"""Predict spot shapes for a single crystal and plot alongside real ROI data."""

import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torchsx import SFXStream
from torchsx.scaling import CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import RefinableGeometry, RelpProfile, GaussianSpectrum, PixelForwardModel
from torchsx.forward.batch import _build_pixel_grids, _compute_pixel_geometry, ROIBatch
from torchsx.io._excise import excise_patches

# ── Load stream ──────────────────────────────────────────────────────────
stream = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
core = stream._core
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
wavelength = inp.wavelength

print(f"Loaded: {core.n_crystals:,} crystals, {core.n_observations:,} obs")
print(f"λ = {wavelength:.4f} Å ({12398.4/wavelength:.0f} eV)")

# ── Pick crystal with most reflections ───────────────────────────────────
cid = core.n_refl.argmax().item()
obs_mask = core.crystal_id == cid
obs_idx = obs_mask.nonzero(as_tuple=True)[0]
n_spots = obs_idx.shape[0]
print(f"\nCrystal {cid}: {n_spots} reflections")

# ── Extract real ROI patches from HDF5 ───────────────────────────────────
PATCH_SIZE = 21
print(f"\nExcising {n_spots} patches ({PATCH_SIZE}x{PATCH_SIZE}) from HDF5...")
excised = excise_patches(core, patch_size=PATCH_SIZE, obs_mask=obs_mask)
# excised.patches: (n_spots, PATCH_SIZE, PATCH_SIZE)
# excised.obs_index maps back to original obs indices
real_patches = excised.patches
real_valid = excised.valid
print(f"  Done. Patches shape: {real_patches.shape}")
print(f"  Value range: [{real_patches.min():.0f}, {real_patches.max():.0f}]")

# ── Build model ──────────────────────────────────────────────────────────
geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
# σ_r tight (intrinsic crystal quality) — spectral integration creates radial elongation
# σ_t broader (mosaicity + divergence)
relp = RelpProfile(n_unique=hkl_idx.n_unique, sigma_r_init=0.0005, sigma_t_init=0.002)
spectrum = GaussianSpectrum(
    n_crystals=core.n_crystals,
    mean_energy_eV=core.beam_energy.median().item(),
    sigma_eV=30.0, n_bins=20, n_sigma=3.0,
)
model = PixelForwardModel(
    structure_factors=sf, orientation=orientation,
    crystal_scale=crystal_scale, crystal_bfactor=crystal_bfactor,
    relp_profile=relp, spectrum=spectrum,
    geometry=geometry, wavelength=wavelength,
)

# ── Per-crystal observation tensors ───────────────────────────────────────
crystal_id = core.crystal_id[obs_idx]
hkl = core.hkl[obs_idx]
hkl_id = hkl_idx.hkl_id[obs_idx]
panel_id = core.panel_id[obs_idx]
fs_center = core.fs[obs_idx].float()
ss_center = core.ss[obs_idx].float()
I_obs = core.I[obs_idx]

# ── Calibrate scale using the real patches for this crystal ──────────────
# Compare observed pixel sums to unit-scale predicted sums
with torch.no_grad():
    fs_g, ss_g, pid_g = _build_pixel_grids(
        fs_center, ss_center, panel_id, PATCH_SIZE, PATCH_SIZE,
    )
    sh, tt, ph, po, so = _compute_pixel_geometry(geometry, fs_g, ss_g, pid_g)

    # Estimate background from real patch borders
    from torchsx.forward.corrections import estimate_background
    bg_est = estimate_background(real_patches, real_valid, border_width=3)

    cal_batch = ROIBatch(
        patches=real_patches,
        valid=real_valid,
        crystal_id=crystal_id,
        hkl_id=hkl_id,
        hkl=hkl,
        panel_id=panel_id,
        fs_center=fs_center,
        ss_center=ss_center,
        background=bg_est,
        s_hat_out=sh, two_theta=tt, phi=ph, polarization=po, solid_angle=so,
    )
    pred_unit = model(cal_batch)
    # Signal = predicted - background
    sig_unit = (pred_unit - bg_est[:, None, None]).reshape(n_spots, -1).sum(dim=1)
    # Observed signal = sum of (patch - background)
    obs_signal = (real_patches - bg_est[:, None, None]).reshape(n_spots, -1).sum(dim=1)

    valid_cal = (sig_unit > 1e-4) & (obs_signal > 10)
    ratio = obs_signal[valid_cal] / sig_unit[valid_cal]
    log_scale = ratio.median().log().item()
    crystal_scale.log_G.data.fill_(log_scale)
    print(f"Scale calibrated from real patches: log(S) = {log_scale:.2f} (S = {math.exp(log_scale):.0f})")
    print(f"  Using {valid_cal.sum()} / {n_spots} spots for calibration")
    print(f"  Obs signal: median={obs_signal[valid_cal].median():.0f}, "
          f"pred signal: median={sig_unit[valid_cal].median():.4f}")

# ── Re-predict with calibrated scale ─────────────────────────────────────
with torch.no_grad():
    pred = model(cal_batch)  # (n_spots, PATCH_SIZE, PATCH_SIZE) — predicted counts (bg + signal)

# ── Per-ROI Gaussian NLL ──────────────────────────────────────────────────
with torch.no_grad():
    mask = real_valid > 0  # (n_spots, PS, PS)
    pred_c = pred.clamp(min=1e-3)
    # Per-pixel NLL: 0.5 * (obs - pred)^2 / pred + 0.5 * log(pred)
    per_pixel_nll = 0.5 * (real_patches - pred_c) ** 2 / pred_c + 0.5 * torch.log(pred_c)
    per_pixel_nll[~mask] = 0.0
    n_valid = mask.sum(dim=(1, 2)).float()
    per_roi_loss = per_pixel_nll.sum(dim=(1, 2)) / n_valid.clamp(min=1)  # mean NLL per pixel

print(f"Per-ROI loss (mean NLL/pixel): median={per_roi_loss.median():.1f}, "
      f"range=[{per_roi_loss.min():.1f}, {per_roi_loss.max():.1f}]")

# ── Sort by observed intensity (brightest first) ─────────────────────────
order = I_obs.abs().argsort(descending=True)
peak_signal = (pred - bg_est[:, None, None]).reshape(n_spots, -1).max(dim=1).values

n_visible = (peak_signal > 0.1).sum().item()
print(f"\nPredicted spots: {n_visible}/{n_spots} with peak signal > 0.1 counts")
print(f"Peak signal: max={peak_signal.max():.1f}, median={peak_signal.median():.3f}")
print(f"Background: median={bg_est.median():.0f}, range=[{bg_est.min():.0f}, {bg_est.max():.0f}]")

# ── Plot: observed vs predicted, side by side ─────────────────────────────
N_SHOW = min(50, n_spots)
n_cols = 10  # 5 pairs of (obs, pred)
n_rows = math.ceil(N_SHOW / 5)  # 5 spots per row (each spot = 2 columns)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8),
                         squeeze=False)

for i in range(N_SHOW):
    row = i // 5
    col_obs = (i % 5) * 2
    col_pred = col_obs + 1
    idx = order[i].item()

    obs_img = real_patches[idx].numpy()
    pred_img = pred[idx].numpy()
    h, k, l = hkl[idx].tolist()
    I_val = I_obs[idx].item()
    bg = bg_est[idx].item()

    # Shared color scale: use observed data range
    vmin = obs_img.min()
    vmax = max(obs_img.max(), pred_img.max(), vmin + 1)

    loss = per_roi_loss[idx].item()

    # Observed
    ax_obs = axes[row][col_obs]
    ax_obs.imshow(obs_img, cmap="hot", interpolation="none", vmin=vmin, vmax=vmax)
    ax_obs.set_title(f"({h},{k},{l}) I={I_val:.0f}\nL={loss:.1f}", fontsize=5, pad=1)
    ax_obs.set_xticks([])
    ax_obs.set_yticks([])
    if i % 5 == 0:
        ax_obs.set_ylabel(f"row {row}", fontsize=5)

    # Predicted
    ax_pred = axes[row][col_pred]
    ax_pred.imshow(pred_img, cmap="hot", interpolation="none", vmin=vmin, vmax=vmax)
    ax_pred.set_title(f"pred bg={bg:.0f}\npk={peak_signal[idx].item():.1f}", fontsize=5, pad=1)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])

fig.suptitle(f"Crystal {cid}: observed (left) vs predicted (right) — {N_SHOW} brightest\n"
             f"Patch {PATCH_SIZE}x{PATCH_SIZE}, scale={math.exp(log_scale):.0f}",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])

outpath = "spot_comparison.png"
fig.savefig(outpath, dpi=150)
print(f"\nSaved to {outpath}")
plt.close(fig)

# ── Detail plot: top 12 spots with cross-sections ────────────────────────
N_DETAIL = 12
fig2, axes2 = plt.subplots(3, N_DETAIL, figsize=(N_DETAIL * 2.5, 7.5))

for i in range(N_DETAIL):
    idx = order[i].item()
    obs_img = real_patches[idx].numpy()
    pred_img = pred[idx].numpy()
    h, k, l = hkl[idx].tolist()
    I_val = I_obs[idx].item()
    bg = bg_est[idx].item()
    pk = peak_signal[idx].item()
    d_star = orientation.predict_h_lab(
        hkl[idx:idx+1].float(), crystal_id[idx:idx+1]
    ).norm().item()
    d_spacing = 1.0 / d_star if d_star > 0 else 0

    vmin = obs_img.min()
    vmax = max(obs_img.max(), pred_img.max(), vmin + 1)

    loss = per_roi_loss[idx].item()

    # Row 0: observed
    ax = axes2[0, i]
    ax.imshow(obs_img, cmap="hot", interpolation="none", vmin=vmin, vmax=vmax)
    ax.set_title(f"({h},{k},{l}) d={d_spacing:.2f}A\nI={I_val:.0f} L={loss:.1f}", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel("observed", fontsize=7)

    # Row 1: predicted
    ax = axes2[1, i]
    ax.imshow(pred_img, cmap="hot", interpolation="none", vmin=vmin, vmax=vmax)
    ax.set_title(f"pred pk={pk:.1f} bg={bg:.0f}", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel("predicted", fontsize=7)

    # Row 2: cross-sections (observed + predicted overlay)
    ax2 = axes2[2, i]
    center = PATCH_SIZE // 2
    ax2.plot(obs_img[center, :], 'k-o', lw=1, ms=1.5, label='obs row', alpha=0.7)
    ax2.plot(pred_img[center, :], 'r-', lw=1.5, label='pred row')
    ax2.plot(obs_img[:, center], 'k--o', lw=1, ms=1.5, label='obs col', alpha=0.7)
    ax2.plot(pred_img[:, center], 'b-', lw=1.5, label='pred col')
    ax2.set_xlim(0, PATCH_SIZE - 1)
    ax2.tick_params(labelsize=5)
    if i == 0:
        ax2.legend(fontsize=4, loc='upper right')
        ax2.set_ylabel("cross-section", fontsize=7)

fig2.suptitle(f"Crystal {cid}: top {N_DETAIL} spots — observed vs predicted + cross-sections",
              fontsize=10)
fig2.tight_layout(rect=[0, 0, 1, 0.94])

outpath2 = "spot_detail.png"
fig2.savefig(outpath2, dpi=150)
print(f"Saved to {outpath2}")
plt.close(fig2)
