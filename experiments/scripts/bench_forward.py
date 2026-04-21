"""Benchmark forward + backward pass for varying batch sizes."""

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
print(f"λ = {wavelength:.4f} Å")

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
likelihood = PixelLikelihood(mode="gaussian")

# Enable gradients for F and scale
for p in model.parameters():
    p.requires_grad_(False)
sf.log_F.requires_grad_(True)
crystal_scale.log_G.requires_grad_(True)

PATCH_SIZE = 11
N_WARMUP = 1
N_REPEAT = 3

# ── Precompute geometry for all obs ───────────────────────────────────────
print(f"\nPrecomputing geometry for {N_OBS:,} obs, patch {PATCH_SIZE}x{PATCH_SIZE}...")
t0 = time.time()

GEOM_CHUNK = 200_000
all_s_hat, all_pol, all_sol = [], [], []
for start in range(0, N_OBS, GEOM_CHUNK):
    end = min(start + GEOM_CHUNK, N_OBS)
    with torch.no_grad():
        fs_g, ss_g, pid_g = _build_pixel_grids(
            core.fs[start:end].float(), core.ss[start:end].float(),
            core.panel_id[start:end], PATCH_SIZE, PATCH_SIZE,
        )
        sh, tt, ph, po, so = _compute_pixel_geometry(geometry, fs_g, ss_g, pid_g)
        all_s_hat.append(sh)
        all_pol.append(po)
        all_sol.append(so)

s_hat_all = torch.cat(all_s_hat, dim=0)
pol_all = torch.cat(all_pol, dim=0)
sol_all = torch.cat(all_sol, dim=0)
del all_s_hat, all_pol, all_sol
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Memory: s_hat {s_hat_all.nelement()*4/1e9:.2f} GB, "
      f"corrections {pol_all.nelement()*4/1e9:.2f} GB")

# ── Benchmark function ────────────────────────────────────────────────────
def bench_batch(n_obs, device="cpu"):
    """Benchmark forward + backward for n_obs observations."""
    idx = torch.randperm(N_OBS)[:n_obs]

    batch = ROIBatch(
        patches=torch.randn(n_obs, PATCH_SIZE, PATCH_SIZE, device=device).abs() * 10 + 50,
        valid=torch.ones(n_obs, PATCH_SIZE, PATCH_SIZE, dtype=torch.uint8, device=device),
        crystal_id=core.crystal_id[idx].to(device),
        hkl_id=hkl_idx.hkl_id[idx].to(device),
        hkl=core.hkl[idx].to(device),
        panel_id=core.panel_id[idx].to(device),
        fs_center=core.fs[idx].float().to(device),
        ss_center=core.ss[idx].float().to(device),
        background=torch.full((n_obs,), 50.0, device=device),
        s_hat_out=s_hat_all[idx].to(device),
        two_theta=torch.zeros(n_obs, PATCH_SIZE, PATCH_SIZE, device=device),
        phi=torch.zeros(n_obs, PATCH_SIZE, PATCH_SIZE, device=device),
        polarization=pol_all[idx].to(device),
        solid_angle=sol_all[idx].to(device),
    )

    # Warmup
    for _ in range(N_WARMUP):
        pred = model(batch)
        loss = likelihood(pred, batch.patches, batch.valid)
        loss.backward()
        sf.log_F.grad = None
        crystal_scale.log_G.grad = None

    # Timed runs
    fwd_times = []
    bwd_times = []
    total_times = []

    for _ in range(N_REPEAT):
        if device != "cpu":
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        pred = model(batch)
        loss = likelihood(pred, batch.patches, batch.valid)
        if device != "cpu":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        loss.backward()
        if device != "cpu":
            torch.cuda.synchronize()
        t_bwd = time.perf_counter()

        fwd_times.append(t_fwd - t_start)
        bwd_times.append(t_bwd - t_fwd)
        total_times.append(t_bwd - t_start)

        sf.log_F.grad = None
        crystal_scale.log_G.grad = None

    # Check gradients are nonzero
    pred = model(batch)
    loss = likelihood(pred, batch.patches, batch.valid)
    loss.backward()
    grad_F_norm = sf.log_F.grad.norm().item()
    grad_G_norm = crystal_scale.log_G.grad.norm().item()
    sf.log_F.grad = None
    crystal_scale.log_G.grad = None

    return {
        "n_obs": n_obs,
        "fwd_ms": min(fwd_times) * 1000,
        "bwd_ms": min(bwd_times) * 1000,
        "total_ms": min(total_times) * 1000,
        "loss": loss.item(),
        "grad_F_norm": grad_F_norm,
        "grad_G_norm": grad_G_norm,
        "n_pixels": n_obs * PATCH_SIZE * PATCH_SIZE,
    }


# ── Run benchmarks ────────────────────────────────────────────────────────
batch_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

print(f"\n{'='*90}")
print(f"BENCHMARK: forward + backward, patch {PATCH_SIZE}x{PATCH_SIZE}, "
      f"n_bins={spectrum.n_bins}, CPU")
print(f"{'='*90}")
print(f"{'n_obs':>8} {'n_pixels':>10} {'fwd(ms)':>10} {'bwd(ms)':>10} "
      f"{'total(ms)':>10} {'ms/obs':>8} {'loss':>12} {'|∇F|':>10} {'|∇G|':>10}")
print("-" * 90)

for n in batch_sizes:
    if n > N_OBS:
        break
    torch.manual_seed(42)
    r = bench_batch(n)
    print(f"{r['n_obs']:>8,} {r['n_pixels']:>10,} {r['fwd_ms']:>10.1f} {r['bwd_ms']:>10.1f} "
          f"{r['total_ms']:>10.1f} {r['total_ms']/r['n_obs']:>8.2f} "
          f"{r['loss']:>12.1f} {r['grad_F_norm']:>10.2f} {r['grad_G_norm']:>10.2f}")

# ── GPU benchmark if available ────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"\n{'='*90}")
    print(f"BENCHMARK: GPU ({torch.cuda.get_device_name()})")
    print(f"{'='*90}")
    model_gpu = model.to(DEVICE)
    print(f"{'n_obs':>8} {'n_pixels':>10} {'fwd(ms)':>10} {'bwd(ms)':>10} "
          f"{'total(ms)':>10} {'ms/obs':>8} {'loss':>12} {'|∇F|':>10} {'|∇G|':>10}")
    print("-" * 90)

    for n in batch_sizes:
        if n > N_OBS:
            break
        torch.manual_seed(42)
        r = bench_batch(n, device=DEVICE)
        print(f"{r['n_obs']:>8,} {r['n_pixels']:>10,} {r['fwd_ms']:>10.1f} {r['bwd_ms']:>10.1f} "
              f"{r['total_ms']:>10.1f} {r['total_ms']/r['n_obs']:>8.2f} "
              f"{r['loss']:>12.1f} {r['grad_F_norm']:>10.2f} {r['grad_G_norm']:>10.2f}")

    model.cpu()
