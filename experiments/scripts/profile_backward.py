"""Profile where backward pass time is spent."""

import time
import torch

from torchsx import SFXStream
from torchsx.scaling import CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import RefinableGeometry, RelpProfile, GaussianSpectrum, PixelForwardModel, PixelLikelihood
from torchsx.forward.batch import ROIBatch, _build_pixel_grids, _compute_pixel_geometry

# ── Load & build ──────────────────────────────────────────────────────────
stream = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
core = stream._core
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
wavelength = inp.wavelength

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
likelihood = PixelLikelihood(mode="gaussian")

sf.log_F.requires_grad_(True)
crystal_scale.log_G.requires_grad_(True)

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
    crystal_id=core.crystal_id[idx], hkl_id=hkl_idx.hkl_id[idx],
    hkl=core.hkl[idx], panel_id=core.panel_id[idx],
    fs_center=core.fs[idx].float(), ss_center=core.ss[idx].float(),
    background=torch.full((N_TEST,), 50.0),
    s_hat_out=sh, two_theta=torch.zeros(N_TEST, PATCH_SIZE, PATCH_SIZE),
    phi=torch.zeros(N_TEST, PATCH_SIZE, PATCH_SIZE),
    polarization=po, solid_angle=so,
)

# ── Profile individual steps ─────────────────────────────────────────────
print(f"Profiling {N_TEST} obs, patch {PATCH_SIZE}x{PATCH_SIZE}, analytical integral")
print(f"{'='*70}")

N = 5

# Step 1: compute_h_lab
s_hat_center = batch.s_hat_out[:, PATCH_SIZE // 2, PATCH_SIZE // 2, :]
times = []
for _ in range(N):
    t0 = time.perf_counter()
    h_lab, h_hat, d_star, s_sq, friedel_sign = model.compute_h_lab(
        batch.hkl, batch.crystal_id, s_hat_center,
    )
    times.append(time.perf_counter() - t0)
print(f"Step 1 (h_lab) forward:        {min(times)*1000:8.2f} ms")

# h_lab backward
h_lab, h_hat, d_star, s_sq, friedel_sign = model.compute_h_lab(
    batch.hkl, batch.crystal_id, s_hat_center,
)
dummy = h_lab.sum()
times = []
for _ in range(N):
    dummy = h_lab.sum()
    t0 = time.perf_counter()
    dummy.backward(retain_graph=True)
    times.append(time.perf_counter() - t0)
    orientation.delta_omega.grad = None
print(f"Step 1 (h_lab) backward:       {min(times)*1000:8.2f} ms")

# Step 2: spectral integral (analytical)
times = []
for _ in range(N):
    h_lab_d, h_hat_d = h_lab.detach(), h_hat.detach()
    t0 = time.perf_counter()
    integral = model.compute_spectral_integral_analytical(
        batch.s_hat_out, h_lab_d, h_hat_d, batch.hkl_id, batch.crystal_id,
    )
    times.append(time.perf_counter() - t0)
print(f"\nStep 2 (analytical integral) forward:  {min(times)*1000:8.2f} ms")

# backward through integral
h_lab_d, h_hat_d = h_lab.detach(), h_hat.detach()
integral = model.compute_spectral_integral_analytical(
    batch.s_hat_out, h_lab_d, h_hat_d, batch.hkl_id, batch.crystal_id,
)
dummy = integral.sum()
times = []
for _ in range(N):
    integral = model.compute_spectral_integral_analytical(
        batch.s_hat_out, h_lab_d, h_hat_d, batch.hkl_id, batch.crystal_id,
    )
    dummy = integral.sum()
    t0 = time.perf_counter()
    dummy.backward()
    times.append(time.perf_counter() - t0)
    relp.log_sigma_r.grad = None
    relp.log_sigma_t.grad = None
    spectrum.log_energy.grad = None
    spectrum.log_sigma_eV.grad = None
print(f"Step 2 (analytical integral) backward: {min(times)*1000:8.2f} ms")

# Step 2 numerical for comparison
times = []
for _ in range(N):
    h_lab_d, h_hat_d = h_lab.detach(), h_hat.detach()
    t0 = time.perf_counter()
    integral_n = model.compute_spectral_integral(
        batch.s_hat_out, h_lab_d, h_hat_d, batch.hkl_id, batch.crystal_id,
    )
    times.append(time.perf_counter() - t0)
print(f"\nStep 2 (numerical integral) forward:   {min(times)*1000:8.2f} ms")

integral_n = model.compute_spectral_integral(
    batch.s_hat_out, h_lab_d, h_hat_d, batch.hkl_id, batch.crystal_id,
)
dummy = integral_n.sum()
times = []
for _ in range(N):
    integral_n = model.compute_spectral_integral(
        batch.s_hat_out, h_lab_d, h_hat_d, batch.hkl_id, batch.crystal_id,
    )
    dummy = integral_n.sum()
    t0 = time.perf_counter()
    dummy.backward()
    times.append(time.perf_counter() - t0)
    relp.log_sigma_r.grad = None
    relp.log_sigma_t.grad = None
    spectrum.log_energy.grad = None
    spectrum.log_sigma_eV.grad = None
print(f"Step 2 (numerical integral) backward:  {min(times)*1000:8.2f} ms")

# Step 3: compute_signal
print()
times = []
for _ in range(N):
    t0 = time.perf_counter()
    n_signal, F_sq, scale = model.compute_signal(batch, integral.detach(), s_sq.detach())
    times.append(time.perf_counter() - t0)
print(f"Step 3 (signal) forward:       {min(times)*1000:8.2f} ms")

n_signal, F_sq, scale = model.compute_signal(batch, integral.detach(), s_sq.detach())
dummy = n_signal.sum()
times = []
for _ in range(N):
    n_signal, F_sq, scale = model.compute_signal(batch, integral.detach(), s_sq.detach())
    dummy = n_signal.sum()
    t0 = time.perf_counter()
    dummy.backward()
    times.append(time.perf_counter() - t0)
    sf.log_F.grad = None
    crystal_scale.log_G.grad = None
print(f"Step 3 (signal) backward:      {min(times)*1000:8.2f} ms")

# Likelihood
print()
pred = torch.randn(N_TEST, PATCH_SIZE, PATCH_SIZE).abs() * 100 + 50
pred.requires_grad_(True)
times = []
for _ in range(N):
    t0 = time.perf_counter()
    loss = likelihood(pred, batch.patches, batch.valid)
    times.append(time.perf_counter() - t0)
print(f"Likelihood forward:            {min(times)*1000:8.2f} ms")

loss = likelihood(pred, batch.patches, batch.valid)
times = []
for _ in range(N):
    loss = likelihood(pred, batch.patches, batch.valid)
    t0 = time.perf_counter()
    loss.backward(retain_graph=True)
    times.append(time.perf_counter() - t0)
    pred.grad = None
print(f"Likelihood backward:           {min(times)*1000:8.2f} ms")

# Full end-to-end
print(f"\n{'='*70}")
print("Full model end-to-end:")

# Count autograd graph nodes
model.analytical_integral = True
pred = model(batch)
loss = likelihood(pred, batch.patches, batch.valid)

# Count saved tensors in the graph
n_nodes = 0
seen = set()
queue = [loss.grad_fn]
while queue:
    node = queue.pop()
    if node is None or id(node) in seen:
        continue
    seen.add(id(node))
    n_nodes += 1
    for child, _ in node.next_functions:
        queue.append(child)
print(f"Autograd graph nodes (analytical): {n_nodes}")

model.analytical_integral = False
pred = model(batch)
loss = likelihood(pred, batch.patches, batch.valid)
n_nodes = 0
seen = set()
queue = [loss.grad_fn]
while queue:
    node = queue.pop()
    if node is None or id(node) in seen:
        continue
    seen.add(id(node))
    n_nodes += 1
    for child, _ in node.next_functions:
        queue.append(child)
print(f"Autograd graph nodes (numerical):  {n_nodes}")

# Memory: count saved tensors
print()
model.analytical_integral = True
pred = model(batch)
loss = likelihood(pred, batch.patches, batch.valid)

saved = []
seen = set()
queue = [loss.grad_fn]
while queue:
    node = queue.pop()
    if node is None or id(node) in seen:
        continue
    seen.add(id(node))
    if hasattr(node, 'saved_tensors'):
        try:
            for t in node.saved_tensors:
                saved.append(t.nelement() * t.element_size())
        except RuntimeError:
            pass
    for child, _ in node.next_functions:
        queue.append(child)
total_saved_mb = sum(saved) / 1e6
print(f"Saved tensors (analytical): {len(saved)}, total {total_saved_mb:.1f} MB")
