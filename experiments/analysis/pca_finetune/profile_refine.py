#!/usr/bin/env python
"""ROI-based profile-fitting refinement.

For each 11×11 patch around a predicted spot:
  1. Compute predicted spot SHAPE (profile) from geometry/PCA — no F², no scale
  2. Fit amplitude per reflection analytically (closed-form, detached)
  3. Poisson NLL on full patch → gradients only through profile shape

This eliminates F²-scale degeneracy: geometry/PCA learn spot shape, not intensity.
Fitted amplitudes give structure factors as a byproduct.

Usage:
    sbatch analysis/pca_finetune/submit_profile.sh
    sbatch analysis/pca_finetune/submit_profile_A.sh  # half A
"""

import os
import time
from pathlib import Path

import numpy as np
import h5py
import bitshuffle.h5  # noqa: F401
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchsx import SFXStream
from torchsx.scaling import (
    CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs,
)
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import (
    RefinableGeometry,
    RelpProfile,
    PCASpectrum,
    PixelForwardModel,
    precompute_spectral_overlap,
)
from torchsx.forward.batch import ROIBatchLoader
from torchsx.io._roi_store import ROIStore

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--half", type=str, default=None, choices=["A", "B"],
                     help="Run on half-dataset: A=(delay_-45,-136), B=(delay_-91,-182)")
_parser.add_argument("--tag", type=str, default=None, help="Output tag suffix")
_args = _parser.parse_args()

ALL_STREAMS = ["delay_-45", "delay_-91", "delay_-136", "delay_-182"]
HALF_A = ["delay_-45", "delay_-136"]
HALF_B = ["delay_-91", "delay_-182"]

if _args.half == "A":
    STREAM_NAMES = HALF_A
elif _args.half == "B":
    STREAM_NAMES = HALF_B
else:
    STREAM_NAMES = ALL_STREAMS

RUN_TAG = _args.tag or (_args.half or "full")
STREAM_DIR = "sfx_scan/streams"
H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
SPECTRA_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"
PCA_CACHE_PATH = "delay_-45_pca_grids_v2.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMAP_RE = r"/sf/alvra/data/p21957/raw/([^/]+)/data/"
REMAP_TO = H5_ROOT + r"/\1/"

N_PCA_MODES = 5
SIGMA_R_INIT = 0.0002       # near delta function (σ_relp ≈ 0)
SIGMA_T_INIT = 0.001
GRADIENT_CLIP = 5.0
N_CRYSTALS_PER_BATCH = 200

N_EPOCHS = 50
LR_PCA = 5e-4
LR_ORIENTATION = 1e-4

# Regularization
REG_PCA = 1e-5
REG_ORIENT = 1e-4

# ROI file pattern
ROI_PATTERN = "{name}_rois_11x11.safetensors"

OUT_DIR = Path(__file__).resolve().parent
print(f"Run tag: {RUN_TAG}, streams: {STREAM_NAMES}")

# ══════════════════════════════════════════════════════════════════════════
# 1. Load streams
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Step 1: Loading streams")
print("=" * 70)

t0 = time.time()
streams = []
for name in STREAM_NAMES:
    sfx_path = os.path.join(STREAM_DIR, f"{name}.sfx")
    s = SFXStream.load(sfx_path)
    s = s.remap_filenames(REMAP_RE, REMAP_TO)
    c = s._core
    print(f"  {name}: {c.n_crystals:,} crystals, {c.n_observations:,} obs")
    streams.append(s)

stream = SFXStream.concat(streams)
core = stream._core
print(f"  Combined: {core.n_crystals:,} crystals, {core.n_observations:,} obs")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 2. Load ROI patches
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 2: Loading ROI patches")
print("=" * 70)

t0 = time.time()
if len(STREAM_NAMES) == len(ALL_STREAMS):
    # Full dataset — use combined store
    roi_path = "combined_4stream_rois_11x11.safetensors"
    print(f"  Loading combined ROI store: {roi_path}")
    roi_store = ROIStore(roi_path)
else:
    # Half dataset — concatenate per-stream stores
    roi_paths = [ROI_PATTERN.format(name=name) for name in STREAM_NAMES]
    print(f"  Concatenating ROI stores: {roi_paths}")
    stores = [ROIStore(p) for p in roi_paths]
    half_roi_path = OUT_DIR / f"half_{RUN_TAG}_rois_11x11.safetensors"
    roi_store = ROIStore.concat(stores, half_roi_path)
    print(f"  Saved concatenated store: {half_roi_path}")

n_rois = roi_store["patches"].shape[0]
ph, pw = roi_store["patches"].shape[1], roi_store["patches"].shape[2]
print(f"  ROIs: {n_rois:,}, patch size: {ph}×{pw}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 3. PCA spectral overlap
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 3: PCA spectral overlap")
print("=" * 70)

t0 = time.time()
all_y = []
for i in range(1, 200):
    fn = os.path.join(SPECTRA_DIR, f"acq{i:04d}.BSDATA.h5")
    if not os.path.exists(fn):
        break
    f = h5py.File(fn, "r")
    all_y.append(f["SARFE10-PSSS059:SPECTRUM_Y/data"][:])
    f.close()
spectra = np.concatenate(all_y, axis=0).astype(np.float32)
n_spectra = spectra.shape[0]
del all_y

f = h5py.File(os.path.join(SPECTRA_DIR, "acq0001.BSDATA.h5"), "r")
energy_eV = f["SARFE10-PSSS059:SPECTRUM_X/data"][0]
f.close()

grids = precompute_spectral_overlap(
    energy_eV, spectra, n_modes=N_PCA_MODES,
    n_ds=200, n_vr=50, n_sr=20,
    ds_r_range=(-0.01, 0.01), v_r_range=(0.01, 0.8),
    sigma_r_range=(0.0001, 0.01),
    max_shots_pca=2000, cache_path=PCA_CACHE_PATH,
)
print(f"  {n_spectra:,} spectra, PCA variance: {grids.explained_variance.cumsum()[-1]:.4f}")
print(f"  Done in {time.time()-t0:.1f}s")

device = torch.device(DEVICE)

# ══════════════════════════════════════════════════════════════════════════
# 4. Build model
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 4: Building model")
print("=" * 70)

inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
wavelength = inp.wavelength
n_unique = hkl_idx.n_unique

geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile(n_unique=n_unique,
                   sigma_r_init=SIGMA_R_INIT, sigma_t_init=SIGMA_T_INIT,
                   per_reflection=False)
crystal_to_shot = np.arange(core.n_crystals) % n_spectra
pca_spectrum = PCASpectrum(grids, core.n_crystals, crystal_to_shot)

model = PixelForwardModel(
    structure_factors=sf, orientation=orientation,
    crystal_scale=crystal_scale, crystal_bfactor=crystal_bfactor,
    relp_profile=relp, spectrum=pca_spectrum,
    geometry=geometry, wavelength=wavelength,
).to(device)

loader = ROIBatchLoader(
    roi_store=roi_store, core=core, hkl_index=hkl_idx,
    geometry=geometry, n_crystals_per_batch=N_CRYSTALS_PER_BATCH,
    device=device,
)
print(f"  Unique reflections: {n_unique:,}")
print(f"  Batches/epoch: {len(loader)}")
print(f"  σ_r = {SIGMA_R_INIT} Å⁻¹ (near delta function)")

# ══════════════════════════════════════════════════════════════════════════
# 5. Setup optimizer (orientation + PCA only)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 5: Setup optimizer")
print("=" * 70)

# Freeze everything except orientation and PCA
for p in model.parameters():
    p.requires_grad_(False)

orientation.delta_omega.requires_grad_(True)
pca_spectrum.coeffs.requires_grad_(True)

param_groups = [
    {"params": [orientation.delta_omega], "lr": LR_ORIENTATION},
    {"params": [pca_spectrum.coeffs], "lr": LR_PCA},
]
optimizer = torch.optim.Adam(param_groups)

n_trainable = sum(p.numel() for g in param_groups for p in g["params"])
print(f"  Trainable: {n_trainable:,} params")
print(f"    Orientation delta_omega: {orientation.delta_omega.numel():,}")
print(f"    PCA coeffs: {pca_spectrum.coeffs.numel():,}")
print(f"  Profile fitting: amplitude per reflection fitted analytically (no F², no scale)")

# ══════════════════════════════════════════════════════════════════════════
# 6. Profile-fitting forward pass helper
# ══════════════════════════════════════════════════════════════════════════

def profile_forward(model, batch):
    """Compute profile-fitting loss for a batch of ROIs.

    Returns (loss, n_valid, mean_A, frac_positive).
    """
    # Ensure geometry (computes s_hat_out, polarization, solid_angle)
    batch = model._ensure_geometry(batch)
    ph_half, pw_half = batch.patches.shape[1] // 2, batch.patches.shape[2] // 2

    # Step 1: orientation
    s_hat_center = batch.s_hat_out[:, ph_half, pw_half, :]
    h_lab, h_hat, d_star, s_sq, friedel_sign = model.compute_h_lab(
        batch.hkl, batch.crystal_id, s_hat_center,
    )

    # Step 2: spectral integral
    integral = model.compute_spectral_integral_pca(
        batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
    )

    # Step 3: profile shape = P × Ω × integral (no F², no scale)
    profile = batch.polarization * batch.solid_angle * integral  # (B, ph, pw)

    # Step 4: gain and background
    gain = model.geometry.gain(batch.panel_id)[:, None, None]  # (B, 1, 1)
    bg = batch.background[:, None, None]                        # (B, 1, 1)

    # Step 5: closed-form amplitude (detached)
    with torch.no_grad():
        obs = batch.patches
        valid_f = batch.valid.float()
        excess = (obs / gain.clamp(min=1e-6) - bg) * valid_f
        p_valid = profile.detach() * valid_f
        A = (excess * p_valid).sum(dim=(1, 2)) / p_valid.pow(2).sum(dim=(1, 2)).clamp(min=1e-10)
        A = A.clamp(min=0)

    # Step 6: predicted counts (gradient through profile only)
    n_pred = (gain * (bg + A[:, None, None] * profile)).clamp(min=1e-3)

    # Step 7: Poisson NLL on valid pixels
    valid_mask = batch.valid > 0
    poisson_nll = n_pred - obs * torch.log(n_pred)
    loss = (poisson_nll * valid_mask).sum() / valid_mask.sum().clamp(min=1)

    # Diagnostics
    n_valid = valid_mask.sum().item()
    mean_A = A.mean().item()
    frac_pos = (A > 0).float().mean().item()

    return loss, n_valid, mean_A, frac_pos, A

# ══════════════════════════════════════════════════════════════════════════
# 7. Initial evaluation
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 7: Initial evaluation")
print("=" * 70)

t0 = time.time()
init_losses, init_As = [], []

with torch.no_grad():
    for batch in loader:
        loss, n_valid, mean_A, frac_pos, A = profile_forward(model, batch)
        init_losses.append(loss.item())
        init_As.append(mean_A)
        del batch, loss, A

init_loss = np.mean(init_losses)
init_A = np.mean(init_As)
print(f"  Initial Poisson NLL: {init_loss:.4f}")
print(f"  Initial mean amplitude: {init_A:.2f}")
print(f"  Evaluated in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 8. Profile refinement loop
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 8: Profile refinement")
print("=" * 70)

epoch_losses = []
epoch_orient_rms = []
epoch_mean_As = []
epoch_frac_pos = []

t_start = time.time()

for epoch in range(N_EPOCHS):
    epoch_t0 = time.time()

    sum_loss = 0.0
    sum_A = 0.0
    sum_frac = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        loss, n_valid, mean_A, frac_pos, A = profile_forward(model, batch)

        # Regularization
        reg = (REG_PCA * pca_spectrum.coeffs.pow(2).sum()
               + REG_ORIENT * orientation.delta_omega.pow(2).sum())

        total_loss = loss + reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in param_groups for p in g["params"]], GRADIENT_CLIP
        )
        optimizer.step()

        sum_loss += loss.item()
        sum_A += mean_A
        sum_frac += frac_pos
        n_batches += 1
        del batch, loss, total_loss, A

    avg_loss = sum_loss / max(n_batches, 1)
    avg_A = sum_A / max(n_batches, 1)
    avg_frac = sum_frac / max(n_batches, 1)
    orient_rms = orientation.delta_omega.detach().pow(2).mean().sqrt().item()
    dt = time.time() - epoch_t0

    epoch_losses.append(avg_loss)
    epoch_orient_rms.append(orient_rms)
    epoch_mean_As.append(avg_A)
    epoch_frac_pos.append(avg_frac)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: "
              f"NLL={avg_loss:.4f} "
              f"A_mean={avg_A:.2f} "
              f"A_pos={avg_frac:.3f} "
              f"Δω_rms={orient_rms:.5f} "
              f"{dt:.1f}s")

total_time = time.time() - t_start
print(f"\n  Total training time: {total_time:.0f}s")

# ══════════════════════════════════════════════════════════════════════════
# 9. Extract F² from fitted amplitudes
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 9: Extract F² from fitted amplitudes")
print("=" * 70)

t0 = time.time()
A_sum = torch.zeros(n_unique)
A_count = torch.zeros(n_unique, dtype=torch.long)

with torch.no_grad():
    for batch in loader:
        loss, n_valid, mean_A, frac_pos, A = profile_forward(model, batch)
        hkl_id_cpu = batch.hkl_id.cpu()
        A_cpu = A.cpu()
        A_sum.scatter_add_(0, hkl_id_cpu, A_cpu)
        A_count.scatter_add_(0, hkl_id_cpu, torch.ones_like(hkl_id_cpu, dtype=torch.long))
        del batch, loss, A

has_data = A_count > 0
F_sq_merged = torch.zeros(n_unique)
F_sq_merged[has_data] = (A_sum[has_data] / A_count[has_data].float()).clamp(min=0.01)

print(f"  F² from {has_data.sum():,} / {n_unique:,} unique hkl")
print(f"  F² median: {F_sq_merged[has_data].median():.2f}")
print(f"  F² mean:   {F_sq_merged[has_data].mean():.2f}")
print(f"  Done in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 10. Final evaluation
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 10: Final evaluation")
print("=" * 70)

final_losses = []
with torch.no_grad():
    for batch in loader:
        loss, n_valid, mean_A, frac_pos, A = profile_forward(model, batch)
        final_losses.append(loss.item())
        del batch, loss, A

final_loss = np.mean(final_losses)
print(f"  Final NLL:  {final_loss:.4f}  (was {init_loss:.4f}, Δ={final_loss-init_loss:+.4f})")
print(f"  Final Δω rms: {orientation.delta_omega.detach().pow(2).mean().sqrt().item():.5f} rad")

# ══════════════════════════════════════════════════════════════════════════
# 11. Visualization
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 11: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
epochs_arr = np.arange(1, N_EPOCHS + 1)

# Plot 1: Training curve
ax = axes[0, 0]
ax.plot(epochs_arr, epoch_losses, "b-", lw=1.5)
ax.axhline(init_loss, color="b", ls="--", lw=0.8, alpha=0.5, label=f"Init={init_loss:.4f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Poisson NLL")
ax.set_title("Training curve")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 2: Orientation correction
ax = axes[0, 1]
ax.plot(epochs_arr, epoch_orient_rms, "c-", lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("RMS Δω (rad)")
ax.set_title("Orientation correction evolution")
ax.grid(True, alpha=0.3)

# Plot 3: Mean amplitude
ax = axes[0, 2]
ax.plot(epochs_arr, epoch_mean_As, "m-", lw=1.5)
ax.axhline(init_A, color="m", ls="--", lw=0.8, alpha=0.5, label=f"Init={init_A:.2f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean fitted amplitude")
ax.set_title("Amplitude evolution")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 4: Orientation correction distribution
ax = axes[1, 0]
dw = orientation.delta_omega.detach().cpu().numpy()
dw_norm = np.linalg.norm(dw, axis=1)
ax.hist(np.degrees(dw_norm), bins=100, alpha=0.7, color="coral")
ax.set_xlabel("|Δω| (degrees)")
ax.set_ylabel("Crystals")
ax.set_title(f"Orientation shift (median={np.degrees(np.median(dw_norm)):.3f}°)")
ax.grid(True, alpha=0.3)

# Plot 5: F² distribution
ax = axes[1, 1]
log_F = torch.log10(F_sq_merged[has_data].clamp(min=1e-3))
ax.hist(log_F.numpy(), bins=100, alpha=0.7, color="steelblue")
ax.set_xlabel("log₁₀(F²)")
ax.set_ylabel("Reflections")
ax.set_title(f"F² distribution (median={F_sq_merged[has_data].median():.2f})")
ax.grid(True, alpha=0.3)

# Plot 6: Fraction positive amplitude
ax = axes[1, 2]
ax.plot(epochs_arr, epoch_frac_pos, "g-", lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Fraction A > 0")
ax.set_title("Positive amplitude fraction")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Profile refinement: {core.n_crystals:,} crystals, {n_unique:,} reflections, "
    f"σ_r={SIGMA_R_INIT}\n"
    f"NLL: {init_loss:.4f} → {final_loss:.4f}, "
    f"Δω_rms={orientation.delta_omega.detach().pow(2).mean().sqrt().item():.5f} rad",
    fontsize=11, fontweight="bold",
)
fig.tight_layout()
out_path = OUT_DIR / f"profile_refine_{RUN_TAG}.png"
fig.savefig(out_path, dpi=150)
print(f"  Saved {out_path}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# 12. Save refined parameters
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 12: Save refined parameters")
print("=" * 70)

unique_hkl = hkl_idx.unique_hkl
state = {
    "F_sq_merged": F_sq_merged,
    "delta_omega": orientation.delta_omega.detach().cpu(),
    "pca_coeffs": pca_spectrum.coeffs.detach().cpu(),
    "unique_hkl": unique_hkl.cpu(),
    "A_sum": A_sum,
    "A_count": A_count,
}
ckpt_path = OUT_DIR / f"profile_refine_state_{RUN_TAG}.pt"
torch.save(state, ckpt_path)
print(f"  Saved checkpoint: {ckpt_path}")

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Summary")
print("=" * 70)
print(f"  Crystals:       {core.n_crystals:,}")
print(f"  Unique hkl:     {n_unique:,}")
print(f"  ROIs:           {n_rois:,}")
print(f"  Epochs:         {N_EPOCHS}")
print(f"  σ_r:            {SIGMA_R_INIT} Å⁻¹")
print()
print(f"  NLL:            {init_loss:.4f} → {final_loss:.4f}  ({100*(final_loss-init_loss)/abs(init_loss):+.2f}%)")
print(f"  Δω rms:         0.00000 → {orientation.delta_omega.detach().pow(2).mean().sqrt().item():.5f} rad")
print(f"  F² median:      {F_sq_merged[has_data].median():.2f}")
print(f"  Training time:  {total_time:.0f}s")
print(f"\nDone.")
