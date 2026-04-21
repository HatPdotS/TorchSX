"""Benchmark geometry pipeline: frozen vs unfrozen, precomputed vs on-the-fly."""

import time
import torch
import numpy as np

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
from torchsx.forward.batch import ROIBatchLoader, _build_pixel_grids, _compute_pixel_geometry
from torchsx.io._roi_store import ROIStore
from torchsx.forward.refine import _set_requires_grad

# ── Load data ────────────────────────────────────────────────────────────
print("Loading data...")
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
    np.zeros(1), np.zeros((1, 1)),  # dummy
    cache_path="delay_-45_pca_grids.npz",
)
crystal_to_shot = np.arange(core.n_crystals) % 12194

# ── Build model ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile(n_unique=hkl_idx.n_unique, sigma_r_init=0.0005, sigma_t_init=0.002)
pca_spectrum = PCASpectrum(grids, core.n_crystals, crystal_to_shot)

model = PixelForwardModel(
    structure_factors=sf, orientation=orientation,
    crystal_scale=crystal_scale, crystal_bfactor=crystal_bfactor,
    relp_profile=relp, spectrum=pca_spectrum,
    geometry=geometry, wavelength=inp.wavelength,
).to(device)

likelihood = PixelLikelihood(mode="gaussian")

print(f"Device: {device}")
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

# ── Build loader (new style: no precomputed geometry) ────────────────────
loader = ROIBatchLoader(
    roi_store=roi_store, core=core, hkl_index=hkl_idx,
    geometry=geometry, n_crystals_per_batch=500, device=device,
)

# ── Benchmark: pixel_to_s_hat cost ──────────────────────────────────────
print("\n" + "="*60)
print("PIXEL GEOMETRY COST")
print("="*60)

batch = next(iter(loader)).to(device)
B, ph, pw = batch.patches.shape
n_pixels = B * ph * pw
print(f"Batch: {B} obs × {ph}×{pw} = {n_pixels:,} pixels")

# no_grad
with torch.no_grad():
    times = []
    for _ in range(10):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        s_hat_out, tt, phi, pol, sol = _compute_pixel_geometry(
            geometry, batch.fs_grid, batch.ss_grid, batch.pid_grid)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"  _compute_pixel_geometry (no_grad): {min(times)*1000:.1f} ms")

# with grad + backward
times = []
for _ in range(5):
    geometry.zero_grad()
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    s_hat_out, tt, phi, pol, sol = _compute_pixel_geometry(
        geometry, batch.fs_grid, batch.ss_grid, batch.pid_grid)
    (s_hat_out.sum() + pol.sum() + sol.sum()).backward()
    if device.type == "cuda": torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
print(f"  _compute_pixel_geometry + backward: {min(times)*1000:.1f} ms")

# ── Warmup model ────────────────────────────────────────────────────────
print("\nWarming up model...")
for _ in range(3):
    with torch.no_grad():
        model(batch)
if device.type == "cuda":
    torch.cuda.synchronize()

# ── Full epoch: geometry frozen ──────────────────────────────────────────
print("\n" + "="*60)
print("FULL EPOCH: geometry FROZEN")
print("="*60)
_set_requires_grad(model, frozen=["geometry"])
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
opt = torch.optim.Adam(trainable, lr=2e-4)

if device.type == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()
n_batches = 0
for batch in loader:
    batch = batch.to(device)
    opt.zero_grad()
    pred = model(batch)
    loss = likelihood(pred, batch.patches.float(), batch.valid)
    loss.backward()
    opt.step()
    n_batches += 1
if device.type == "cuda": torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"  {n_batches} batches in {elapsed:.2f}s ({elapsed/n_batches*1000:.0f} ms/batch)")

# Check geometry grad
for name, p in geometry.named_parameters():
    has_grad = p.grad is not None and p.grad.abs().max() > 0
    print(f"  geometry.{name}: grad={'YES' if has_grad else 'no'}")

# ── Full epoch: geometry UNFROZEN ────────────────────────────────────────
print("\n" + "="*60)
print("FULL EPOCH: geometry UNFROZEN")
print("="*60)
_set_requires_grad(model, frozen=[])
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
opt = torch.optim.Adam(trainable, lr=1e-4)

if device.type == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()
n_batches = 0
for batch in loader:
    batch = batch.to(device)
    opt.zero_grad()
    pred = model(batch)
    loss = likelihood(pred, batch.patches.float(), batch.valid)
    loss.backward()
    opt.step()
    n_batches += 1
if device.type == "cuda": torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"  {n_batches} batches in {elapsed:.2f}s ({elapsed/n_batches*1000:.0f} ms/batch)")

# Check geometry grad
for name, p in geometry.named_parameters():
    has_grad = p.grad is not None and p.grad.abs().max() > 0
    print(f"  geometry.{name}: grad={'YES' if has_grad else 'no'}, "
          f"max|g|={p.grad.abs().max():.6f}" if has_grad else "")

# Check other grads
for label, param in [("log_F", sf.log_F), ("log_G", crystal_scale.log_G)]:
    if param.grad is not None:
        print(f"  {label}: |grad|={param.grad.norm():.4f}")
    else:
        print(f"  {label}: NO GRAD")

# ── Per-batch timing breakdown ───────────────────────────────────────────
print("\n" + "="*60)
print("PER-BATCH BREAKDOWN (geometry unfrozen, single batch)")
print("="*60)

batch = next(iter(loader)).to(device)
B = batch.patches.shape[0]
print(f"  Batch size: {B} obs")

# Data loading (approx — just the loader overhead)
times_load = []
for _ in range(3):
    loader_iter = iter(loader)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    b = next(loader_iter).to(device)
    if device.type == "cuda": torch.cuda.synchronize()
    times_load.append(time.perf_counter() - t0)
print(f"  Data load + .to(device):   {min(times_load)*1000:.1f} ms")

# Forward only
opt.zero_grad()
if device.type == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()
pred = model(batch)
if device.type == "cuda": torch.cuda.synchronize()
t_fwd = time.perf_counter() - t0
print(f"  Forward:                   {t_fwd*1000:.1f} ms")

# Likelihood
t0 = time.perf_counter()
loss = likelihood(pred, batch.patches.float(), batch.valid)
if device.type == "cuda": torch.cuda.synchronize()
t_lik = time.perf_counter() - t0
print(f"  Likelihood:                {t_lik*1000:.1f} ms")

# Backward
t0 = time.perf_counter()
loss.backward()
if device.type == "cuda": torch.cuda.synchronize()
t_bwd = time.perf_counter() - t0
print(f"  Backward:                  {t_bwd*1000:.1f} ms")

# Optimizer step
t0 = time.perf_counter()
opt.step()
if device.type == "cuda": torch.cuda.synchronize()
t_step = time.perf_counter() - t0
print(f"  Optimizer step:            {t_step*1000:.1f} ms")

print(f"  Total:                     {(t_fwd+t_lik+t_bwd+t_step)*1000:.1f} ms")

print("\nDone.")
