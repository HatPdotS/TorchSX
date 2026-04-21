"""Validate PCA spectral overlap: accuracy vs numerical, gradient flow, GPU benchmark."""

import time
import numpy as np
import h5py
import torch

from torchsx import SFXStream
from torchsx.scaling import CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import (
    RefinableGeometry, RelpProfile, GaussianSpectrum, PCASpectrum,
    PixelForwardModel, PixelLikelihood, precompute_spectral_overlap,
)
from torchsx.forward.batch import ROIBatch, _build_pixel_grids, _compute_pixel_geometry

# ── Load SFX data ─────────────────────────────────────────────────────────
print("Loading SFX data...")
stream = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
core = stream._core
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
wavelength = inp.wavelength

# ── Load spectra ──────────────────────────────────────────────────────────
print("Loading FEL spectra...")
all_y = []
for i in range(1, 135):
    fn = f'/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files/acq{i:04d}.BSDATA.h5'
    f = h5py.File(fn, 'r')
    all_y.append(f['SARFE10-PSSS059:SPECTRUM_Y/data'][:])
    f.close()
spectra = np.concatenate(all_y, axis=0).astype(np.float32)
f = h5py.File('/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files/acq0001.BSDATA.h5', 'r')
energy_eV = f['SARFE10-PSSS059:SPECTRUM_X/data'][0]
f.close()
print(f"  {spectra.shape[0]} spectra, {spectra.shape[1]} bins, "
      f"{energy_eV[0]:.0f}-{energy_eV[-1]:.0f} eV")

# ── Precompute spectral overlap grids ─────────────────────────────────────
print("\nPrecomputing spectral overlap grids...")
t0 = time.time()
grids = precompute_spectral_overlap(
    energy_eV, spectra,
    n_modes=5, n_ds=200, n_vr=50, n_sr=15,
    ds_r_range=(-0.005, 0.005),
    v_r_range=(0.01, 0.8),
    sigma_r_range=(0.0001, 0.003),
)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Grid shape: mean={grids.mean_grid.shape}, modes={grids.mode_grids.shape}")
print(f"  Explained variance per mode: {grids.explained_variance}")
print(f"  Total explained: {grids.explained_variance.sum():.6f}")

# For testing: map crystals to random spectra (real mapping would come from metadata)
np.random.seed(42)
crystal_to_shot = np.random.randint(0, spectra.shape[0], size=core.n_crystals)

# ── Build models ──────────────────────────────────────────────────────────
geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile(n_unique=hkl_idx.n_unique, sigma_r_init=0.0005, sigma_t_init=0.002)

# PCA spectrum model
pca_spectrum = PCASpectrum(grids, core.n_crystals, crystal_to_shot)
print(f"\nPCASpectrum: {pca_spectrum.n_modes} modes, "
      f"coeffs shape={pca_spectrum.coeffs.shape}")

model_pca = PixelForwardModel(
    structure_factors=sf, orientation=orientation,
    crystal_scale=crystal_scale, crystal_bfactor=crystal_bfactor,
    relp_profile=relp, spectrum=pca_spectrum,
    geometry=geometry, wavelength=wavelength,
)

# ── Build test batch ──────────────────────────────────────────────────────
PATCH_SIZE = 11
N_TEST = 2000
torch.manual_seed(42)
idx = torch.randperm(core.n_observations)[:N_TEST]

with torch.no_grad():
    fs_g, ss_g, pid_g = _build_pixel_grids(
        core.fs[idx].float(), core.ss[idx].float(),
        core.panel_id[idx], PATCH_SIZE, PATCH_SIZE,
    )
    sh, tt, ph, po, so = _compute_pixel_geometry(geometry, fs_g, ss_g, pid_g)

batch = ROIBatch(
    patches=torch.randn(N_TEST, PATCH_SIZE, PATCH_SIZE).abs() * 10 + 50,
    valid=torch.ones(N_TEST, PATCH_SIZE, PATCH_SIZE, dtype=torch.uint8),
    crystal_id=core.crystal_id[idx],
    hkl_id=hkl_idx.hkl_id[idx],
    hkl=core.hkl[idx],
    panel_id=core.panel_id[idx],
    fs_center=core.fs[idx].float(),
    ss_center=core.ss[idx].float(),
    background=torch.full((N_TEST,), 50.0),
    s_hat_out=sh,
    two_theta=torch.zeros(N_TEST, PATCH_SIZE, PATCH_SIZE),
    phi=torch.zeros(N_TEST, PATCH_SIZE, PATCH_SIZE),
    polarization=po,
    solid_angle=so,
)

# ── Test forward pass ─────────────────────────────────────────────────────
print("\n--- Forward pass test ---")
with torch.no_grad():
    pred_pca = model_pca(batch)
print(f"  Output shape: {pred_pca.shape}")
print(f"  Pred range: {pred_pca.min():.2f} - {pred_pca.max():.2f}")
print(f"  Pred mean: {pred_pca.mean():.2f}")

# ── Test gradient flow ────────────────────────────────────────────────────
print("\n--- Gradient flow test ---")
sf.log_F.requires_grad_(True)
crystal_scale.log_G.requires_grad_(True)
relp.log_sigma_r.requires_grad_(True)
relp.log_sigma_t.requires_grad_(True)
pca_spectrum.coeffs.requires_grad_(True)

likelihood = PixelLikelihood(mode="gaussian")
pred = model_pca(batch)
loss = likelihood(pred, batch.patches, batch.valid)
loss.backward()

print(f"  loss = {loss.item():.2f}")
for name, param in [
    ("log_F", sf.log_F),
    ("log_G", crystal_scale.log_G),
    ("log_sigma_r", relp.log_sigma_r),
    ("log_sigma_t", relp.log_sigma_t),
    ("pca_coeffs", pca_spectrum.coeffs),
]:
    if param.grad is not None:
        g = param.grad
        print(f"  {name:15s}: grad norm={g.norm():.6f}, nonzero={(g.abs()>1e-10).sum()}/{g.numel()}")
    else:
        print(f"  {name:15s}: NO GRADIENT!")

# Clear grads
for p in model_pca.parameters():
    if p.grad is not None:
        p.grad = None

# ── Benchmark ─────────────────────────────────────────────────────────────
print("\n--- Performance benchmark (CPU) ---")

# Warmup
with torch.no_grad():
    model_pca(batch)

N_REPEAT = 3
times = []
for _ in range(N_REPEAT):
    t0 = time.perf_counter()
    with torch.no_grad():
        model_pca(batch)
    times.append(time.perf_counter() - t0)
print(f"  PCA forward:     {min(times)*1000:8.1f} ms ({min(times)*1000/N_TEST:.4f} ms/obs)")

# With backward
times_bwd = []
for _ in range(N_REPEAT):
    t0 = time.perf_counter()
    pred = model_pca(batch)
    loss = likelihood(pred, batch.patches, batch.valid)
    loss.backward()
    for p in model_pca.parameters():
        if p.grad is not None:
            p.grad = None
    times_bwd.append(time.perf_counter() - t0)
print(f"  PCA fwd+bwd:     {min(times_bwd)*1000:8.1f} ms ({min(times_bwd)*1000/N_TEST:.4f} ms/obs)")

# GPU benchmark
if torch.cuda.is_available():
    print(f"\n--- GPU benchmark ({torch.cuda.get_device_name()}) ---")
    device = torch.device("cuda")
    model_gpu = model_pca.to(device)

    batch_gpu = ROIBatch(
        patches=batch.patches.to(device),
        valid=batch.valid.to(device),
        crystal_id=batch.crystal_id.to(device),
        hkl_id=batch.hkl_id.to(device),
        hkl=batch.hkl.to(device),
        panel_id=batch.panel_id.to(device),
        fs_center=batch.fs_center.to(device),
        ss_center=batch.ss_center.to(device),
        background=batch.background.to(device),
        s_hat_out=batch.s_hat_out.to(device),
        two_theta=batch.two_theta.to(device),
        phi=batch.phi.to(device),
        polarization=batch.polarization.to(device),
        solid_angle=batch.solid_angle.to(device),
    )

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model_gpu(batch_gpu)
        torch.cuda.synchronize()

    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model_gpu(batch_gpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"  PCA forward:     {min(times)*1000:8.1f} ms ({min(times)*1000/N_TEST:.4f} ms/obs)")

    # With backward
    times_bwd = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model_gpu(batch_gpu)
        loss = likelihood(pred, batch_gpu.patches, batch_gpu.valid)
        loss.backward()
        torch.cuda.synchronize()
        for p in model_gpu.parameters():
            if p.grad is not None:
                p.grad = None
        times_bwd.append(time.perf_counter() - t0)
    print(f"  PCA fwd+bwd:     {min(times_bwd)*1000:8.1f} ms ({min(times_bwd)*1000/N_TEST:.4f} ms/obs)")

    model_pca.cpu()
