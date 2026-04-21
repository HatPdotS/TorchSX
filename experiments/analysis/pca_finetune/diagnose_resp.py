#!/usr/bin/env python
"""Quick diagnostic: why is responsibility so low?

Inspects the PCA spectral overlap grid values at small σ_r to understand
whether the overlap magnitude drops as σ_r → 0.
"""

import os
import numpy as np
import h5py
import torch

from torchsx.forward import precompute_spectral_overlap, PCASpectrum

SPECTRA_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"
PCA_CACHE_PATH = "delay_-45_pca_grids_v2.npz"
N_PCA_MODES = 5

# Load PCA grids (cached, fast)
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
print(f"PCA grids loaded: mean_grid shape={grids.mean_grid.shape}")
print(f"  ds_r_range: {grids.ds_r_range}")
print(f"  v_r_range:  {grids.v_r_range}")
print(f"  log_sr_range: {grids.log_sr_range}")
print(f"  σ_r range: {np.exp(grids.log_sr_range[0]):.6f} — {np.exp(grids.log_sr_range[1]):.6f} Å⁻¹")

# The mean_grid has shape (n_ds, n_vr, n_sr)
# Let's look at the overlap values along the σ_r axis
mean = grids.mean_grid  # (200, 50, 20)

print(f"\n--- Mean grid statistics ---")
print(f"  Overall: min={mean.min():.6f}, max={mean.max():.6f}, mean={mean.mean():.6f}")

# At ds_r=0 (center), different v_r and σ_r
ds_center = mean.shape[0] // 2  # ds_r = 0
print(f"\n--- At ds_r=0 (center of relp): overlap vs (v_r, σ_r) ---")
print(f"  {'σ_r':>10s}", end="")
sr_indices = [0, 4, 9, 14, 19]  # sample across σ_r axis
log_sr = np.linspace(grids.log_sr_range[0], grids.log_sr_range[1], 20)
for si in sr_indices:
    print(f"  σ_r={np.exp(log_sr[si]):.5f}", end="")
print()

vr_vals = np.linspace(grids.v_r_range[0], grids.v_r_range[1], 50)
for vi in [0, 10, 24, 40, 49]:
    print(f"  v_r={vr_vals[vi]:.3f}  ", end="")
    for si in sr_indices:
        print(f"  {mean[ds_center, vi, si]:12.6f}", end="")
    print()

# Check: does the grid at small σ_r have reasonable values?
print(f"\n--- σ_r dependence at ds_r=0, v_r=median ---")
vi_mid = 24  # middle of v_r range
for si in range(20):
    sr = np.exp(log_sr[si])
    val = mean[ds_center, vi_mid, si]
    print(f"  σ_r={sr:.6f} Å⁻¹  overlap={val:.6f}")

# Now test PCASpectrum.overlap() directly
print(f"\n--- Direct overlap query at different σ_r ---")
n_test = 1
crystal_to_shot = np.array([0])  # just one crystal
pca = PCASpectrum(grids, n_test, crystal_to_shot)
print(f"  PCA coeffs for crystal 0: {pca.coeffs.data[0].tolist()}")

for sr_val in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]:
    # Query at ds_r=0 (on the relp), typical v_r, given σ_r
    ds_r = torch.zeros(1, 1, 1)
    v_r = torch.full((1, 1, 1), 0.3)  # typical v_r for medium resolution
    log_sr = torch.full((1,), np.log(sr_val))

    with torch.no_grad():
        overlap = pca.overlap(ds_r, v_r, log_sr, torch.tensor([0]))
    print(f"  σ_r={sr_val:.4f}  overlap at ds_r=0, v_r=0.3: {overlap.item():.6f}")

# Also check at various ds_r offsets
print(f"\n--- Overlap vs ds_r offset at σ_r=0.0002 and σ_r=0.005 ---")
for sr_val in [0.0002, 0.005]:
    log_sr = torch.full((1,), np.log(sr_val))
    print(f"  σ_r = {sr_val}:")
    for ds_r_val in [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]:
        ds_r = torch.full((1, 1, 1), ds_r_val)
        v_r = torch.full((1, 1, 1), 0.3)
        with torch.no_grad():
            overlap = pca.overlap(ds_r, v_r, log_sr, torch.tensor([0]))
        print(f"    ds_r={ds_r_val:.4f}  overlap={overlap.item():.6f}")

print("\nDone.")
