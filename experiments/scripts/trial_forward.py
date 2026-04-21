"""Trial: pixel-level forward model on delay_-45 — all 1.6M obs, GPU mini-batch SGD."""

import time
import torch

from torchsx import SFXStream
from torchsx.scaling import (
    CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs,
)
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import (
    RefinableGeometry, RelpProfile, GaussianSpectrum,
    PixelForwardModel, PixelLikelihood, ROIBatch,
)
from torchsx.forward.batch import _build_pixel_grids, _compute_pixel_geometry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  {torch.cuda.get_device_name()}")

PATCH_SIZE = 11
MINI_BATCH = 20_000   # obs per mini-batch (GPU can handle much more)
N_EPOCHS = 30

# ── 1. Load ──────────────────────────────────────────────────────────────
t0 = time.time()
stream = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
core = stream._core
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
wavelength = inp.wavelength
N_OBS = core.n_observations
print(f"\nLoaded in {time.time()-t0:.1f}s")
print(f"  {core.n_crystals:,} crystals, {N_OBS:,} obs, {core.n_images:,} images")
print(f"  {hkl_idx.n_unique:,} unique HKL, λ={wavelength:.4f} Å")

# ── 2. Build model ───────────────────────────────────────────────────────
geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile.from_partiality(
    n_unique=hkl_idx.n_unique, sigma_0=0.05, B_relp=0.5,
    s_sq=inp.s_sq, hkl_id=hkl_idx.hkl_id,
)
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
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters")

# ── 3. Precompute per-pixel geometry (on CPU, then move model to GPU) ────
print(f"\nPrecomputing geometry for {N_OBS:,} obs, patch {PATCH_SIZE}x{PATCH_SIZE}...")
t1 = time.time()

GEOM_CHUNK = 100_000
all_s_hat = []
all_pol = []
all_sol = []

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
print(f"  Done in {time.time()-t1:.1f}s, "
      f"{s_hat_all.nelement()*4/1e9:.2f} GB (s_hat) + "
      f"{pol_all.nelement()*4/1e9:.2f} GB (corrections)")

# ── 4. Observation-level tensors (keep on CPU, move per batch) ───────────
obs_crystal_id = core.crystal_id
obs_hkl_id = hkl_idx.hkl_id
obs_hkl = core.hkl
obs_panel_id = core.panel_id
obs_fs = core.fs.float()
obs_ss = core.ss.float()

# ── 5. Calibrate crystal scale (on CPU, model not yet on GPU) ────────────
print("\nCalibrating scale...")
CAL_SIZE = 10_000
torch.manual_seed(42)
cal_idx = torch.randperm(N_OBS)[:CAL_SIZE]

with torch.no_grad():
    cal_batch = ROIBatch(
        patches=torch.zeros(CAL_SIZE, PATCH_SIZE, PATCH_SIZE),
        valid=torch.ones(CAL_SIZE, PATCH_SIZE, PATCH_SIZE, dtype=torch.uint8),
        crystal_id=obs_crystal_id[cal_idx],
        hkl_id=obs_hkl_id[cal_idx],
        hkl=obs_hkl[cal_idx],
        panel_id=obs_panel_id[cal_idx],
        fs_center=obs_fs[cal_idx],
        ss_center=obs_ss[cal_idx],
        background=torch.full((CAL_SIZE,), 50.0),
        s_hat_out=s_hat_all[cal_idx],
        two_theta=torch.zeros(CAL_SIZE, PATCH_SIZE, PATCH_SIZE),
        phi=torch.zeros(CAL_SIZE, PATCH_SIZE, PATCH_SIZE),
        polarization=pol_all[cal_idx],
        solid_angle=sol_all[cal_idx],
    )
    pred_unit = model(cal_batch)
    sig_unit = (pred_unit - 50.0).reshape(CAL_SIZE, -1).sum(dim=1)
    I_cal = core.I[cal_idx]
    valid_cal = (sig_unit > 1e-4) & (I_cal > 1.0)
    ratio = I_cal[valid_cal] / sig_unit[valid_cal]
    log_scale = ratio.median().log().item()
    crystal_scale.log_G.data.fill_(log_scale)
    print(f"  log(S) = {log_scale:.2f} (S = {ratio.median():.0f})")

    # Signal check
    pred_check = model(cal_batch)
    peak = (pred_check - 50.0).reshape(CAL_SIZE, -1).max(dim=1).values
    print(f"  Peak signal: median={peak.median():.1f}, max={peak.max():.1f}, "
          f">10: {(peak > 10).sum()}, >100: {(peak > 100).sum()}")

# ── 6. Move model to GPU, generate synthetic ground truth ────────────────
model = model.to(DEVICE)
likelihood = PixelLikelihood(mode="gaussian").to(DEVICE)
print(f"\nModel moved to {DEVICE}")
print("Generating ground truth for all obs...")
t2 = time.time()
gt_all = torch.zeros(N_OBS, PATCH_SIZE, PATCH_SIZE)
GT_CHUNK = 50_000

with torch.no_grad():
    for start in range(0, N_OBS, GT_CHUNK):
        end = min(start + GT_CHUNK, N_OBS)
        bs = end - start
        b = ROIBatch(
            patches=torch.zeros(bs, PATCH_SIZE, PATCH_SIZE, device=DEVICE),
            valid=torch.ones(bs, PATCH_SIZE, PATCH_SIZE, dtype=torch.uint8, device=DEVICE),
            crystal_id=obs_crystal_id[start:end].to(DEVICE),
            hkl_id=obs_hkl_id[start:end].to(DEVICE),
            hkl=obs_hkl[start:end].to(DEVICE),
            panel_id=obs_panel_id[start:end].to(DEVICE),
            fs_center=obs_fs[start:end].to(DEVICE),
            ss_center=obs_ss[start:end].to(DEVICE),
            background=torch.full((bs,), 50.0, device=DEVICE),
            s_hat_out=s_hat_all[start:end].to(DEVICE),
            two_theta=torch.zeros(bs, PATCH_SIZE, PATCH_SIZE, device=DEVICE),
            phi=torch.zeros(bs, PATCH_SIZE, PATCH_SIZE, device=DEVICE),
            polarization=pol_all[start:end].to(DEVICE),
            solid_angle=sol_all[start:end].to(DEVICE),
        )
        gt_all[start:end] = model(b).cpu()

noisy_all = torch.poisson(gt_all.clamp(min=0.1))
signal = gt_all - 50.0
peak_per_obs = signal.reshape(N_OBS, -1).max(dim=1).values
print(f"  Done in {time.time()-t2:.1f}s")
print(f"  Signal: median peak={peak_per_obs.median():.2f}, max={peak_per_obs.max():.1f}")
print(f"  Obs with peak>1: {(peak_per_obs > 1).sum():,}, "
      f">10: {(peak_per_obs > 10).sum():,}, "
      f">100: {(peak_per_obs > 100).sum():,}")

# ── 7. Perturb parameters ───────────────────────────────────────────────
true_log_F = sf.log_F.data.clone()
true_log_G = crystal_scale.log_G.data.clone()

sf.log_F.data += 0.1 * torch.randn_like(sf.log_F.data)
crystal_scale.log_G.data += 0.2 * torch.randn_like(crystal_scale.log_G.data)

with torch.no_grad():
    f_err_0 = (sf.log_F.data - true_log_F).abs().mean().item()
    g_err_0 = (crystal_scale.log_G.data - true_log_G).abs().mean().item()
    F_true = torch.exp(true_log_F)
    F_pert = torch.exp(sf.log_F.data)
    cc_0 = torch.corrcoef(torch.stack([F_true.double(), F_pert.double()]))[0, 1].item()
print(f"\nInitial perturbation: |ΔlogF|={f_err_0:.4f}, |ΔlogG|={g_err_0:.4f}, CC(F)={cc_0:.4f}")

# ── 8. Mini-batch refinement ────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"REFINEMENT: F + scale, {N_OBS:,} obs, mini-batch={MINI_BATCH:,}, {N_EPOCHS} epochs")
print(f"{'='*70}")

for p in model.parameters():
    p.requires_grad_(False)
sf.log_F.requires_grad_(True)
crystal_scale.log_G.requires_grad_(True)

optimizer = torch.optim.Adam([
    {"params": sf.log_F, "lr": 2e-3},
    {"params": crystal_scale.log_G, "lr": 5e-3},
])

t3 = time.time()
for epoch in range(N_EPOCHS):
    perm = torch.randperm(N_OBS)
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, N_OBS, MINI_BATCH):
        end = min(start + MINI_BATCH, N_OBS)
        idx = perm[start:end]
        bs = idx.shape[0]

        mini = ROIBatch(
            patches=noisy_all[idx].to(DEVICE),
            valid=torch.ones(bs, PATCH_SIZE, PATCH_SIZE, dtype=torch.uint8, device=DEVICE),
            crystal_id=obs_crystal_id[idx].to(DEVICE),
            hkl_id=obs_hkl_id[idx].to(DEVICE),
            hkl=obs_hkl[idx].to(DEVICE),
            panel_id=obs_panel_id[idx].to(DEVICE),
            fs_center=obs_fs[idx].to(DEVICE),
            ss_center=obs_ss[idx].to(DEVICE),
            background=torch.full((bs,), 50.0, device=DEVICE),
            s_hat_out=s_hat_all[idx].to(DEVICE),
            two_theta=torch.zeros(bs, PATCH_SIZE, PATCH_SIZE, device=DEVICE),
            phi=torch.zeros(bs, PATCH_SIZE, PATCH_SIZE, device=DEVICE),
            polarization=pol_all[idx].to(DEVICE),
            solid_angle=sol_all[idx].to(DEVICE),
        )

        optimizer.zero_grad()
        n_pred = model(mini)
        loss = likelihood(n_pred, mini.patches, mini.valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([sf.log_F, crystal_scale.log_G], 5.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    # Metrics
    with torch.no_grad():
        f_err = (sf.log_F.data - true_log_F).abs().mean().item()
        g_err = (crystal_scale.log_G.data - true_log_G).abs().mean().item()
        F_ref = torch.exp(sf.log_F.data)
        cc = torch.corrcoef(torch.stack([F_true.double(), F_ref.double()]))[0, 1].item()

    elapsed = time.time() - t3
    avg_loss = epoch_loss / n_batches
    print(f"Epoch {epoch+1:2d}/{N_EPOCHS} ({elapsed:5.0f}s): "
          f"loss={avg_loss:12.1f}  |ΔlogF|={f_err:.4f}  |ΔlogG|={g_err:.4f}  CC(F)={cc:.4f}")

# ── 9. Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Data: {core.n_crystals:,} crystals, {N_OBS:,} obs")
print(f"Parameters: {hkl_idx.n_unique:,} F + {core.n_crystals:,} G = {hkl_idx.n_unique + core.n_crystals:,}")
print(f"|ΔlogF|: {f_err_0:.4f} → {f_err:.4f} ({'↓' if f_err < f_err_0 else '↑'})")
print(f"|ΔlogG|: {g_err_0:.4f} → {g_err:.4f} ({'↓' if g_err < g_err_0 else '↑'})")
print(f"CC(F):   {cc_0:.4f} → {cc:.4f} ({'↑' if cc > cc_0 else '↓'})")
print(f"Total time: {time.time()-t0:.0f}s")
