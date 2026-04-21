"""Validate analytical spectral integral against numerical, and benchmark."""

import time
import torch

from torchsx import SFXStream
from torchsx.scaling import CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import RefinableGeometry, RelpProfile, GaussianSpectrum, PixelForwardModel, PixelLikelihood
from torchsx.forward.batch import ROIBatch, _build_pixel_grids, _compute_pixel_geometry

# ── Load data ─────────────────────────────────────────────────────────────
stream = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
core = stream._core
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
wavelength = inp.wavelength
N_OBS = core.n_observations

print(f"Loaded: {core.n_crystals:,} crystals, {N_OBS:,} obs")

# ── Build model ───────────────────────────────────────────────────────────
geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
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

PATCH_SIZE = 11
N_TEST = 2000

# ── Build a test batch ────────────────────────────────────────────────────
torch.manual_seed(42)
idx = torch.randperm(N_OBS)[:N_TEST]

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

# ── Compare numerical vs analytical ──────────────────────────────────────
print("\n--- Accuracy comparison (2000 obs) ---")

# Step 1: compute h_lab (shared)
phw = PATCH_SIZE
s_hat_center = batch.s_hat_out[:, phw // 2, phw // 2, :]
with torch.no_grad():
    h_lab, h_hat, d_star, s_sq, friedel_sign = model.compute_h_lab(
        batch.hkl, batch.crystal_id, s_hat_center,
    )

    # Numerical
    integral_num = model.compute_spectral_integral(
        batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
    )

    # Analytical
    integral_ana = model.compute_spectral_integral_analytical(
        batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
    )

# Compare where signal is significant
mask = integral_num > integral_num.max() * 1e-4
rel_err = ((integral_ana - integral_num) / integral_num.clamp(min=1e-30))[mask]
print(f"  Pixels with signal: {mask.sum().item():,} / {integral_num.numel():,}")
print(f"  Relative error:  median={rel_err.abs().median():.6f}, "
      f"max={rel_err.abs().max():.6f}, mean={rel_err.abs().mean():.6f}")
print(f"  Integral sum:  numerical={integral_num.sum():.4f}, "
      f"analytical={integral_ana.sum():.4f}")

# Full forward pass comparison
with torch.no_grad():
    model.analytical_integral = False
    pred_num = model(batch)
    model.analytical_integral = True
    pred_ana = model(batch)

signal_mask = (pred_num - 50.0).abs() > 0.1  # non-background pixels
rel_err_pred = ((pred_ana - pred_num) / pred_num)[signal_mask]
print(f"\n  Full forward prediction:")
print(f"  Relative error on signal pixels: median={rel_err_pred.abs().median():.6f}, "
      f"max={rel_err_pred.abs().max():.6f}")

# ── Benchmark ────────────────────────────────────────────────────────────
print("\n--- Performance benchmark (2000 obs, 3 repeats) ---")
N_REPEAT = 3

for name, use_analytical in [("Numerical (20 bins)", False), ("Analytical", True)]:
    model.analytical_integral = use_analytical

    # Warmup
    with torch.no_grad():
        model(batch)

    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(batch)
        times.append(time.perf_counter() - t0)

    best = min(times) * 1000
    print(f"  {name:25s}: {best:8.1f} ms  ({best/N_TEST:.3f} ms/obs)")

# Benchmark with gradients
print("\n--- With backward pass ---")
sf.log_F.requires_grad_(True)
crystal_scale.log_G.requires_grad_(True)
likelihood = PixelLikelihood(mode="gaussian")

for name, use_analytical in [("Numerical (20 bins)", False), ("Analytical", True)]:
    model.analytical_integral = use_analytical

    # Warmup
    pred = model(batch)
    loss = likelihood(pred, batch.patches, batch.valid)
    loss.backward()
    sf.log_F.grad = None
    crystal_scale.log_G.grad = None

    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        pred = model(batch)
        loss = likelihood(pred, batch.patches, batch.valid)
        loss.backward()
        sf.log_F.grad = None
        crystal_scale.log_G.grad = None
        times.append(time.perf_counter() - t0)

    best = min(times) * 1000
    print(f"  {name:25s}: {best:8.1f} ms  ({best/N_TEST:.3f} ms/obs)")
