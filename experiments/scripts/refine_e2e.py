"""End-to-end pixel-level refinement on multiple SFX streams (~32K crystals).

Usage:
    # On a GPU node (e.g. via SLURM):
    .venv/bin/python refine_e2e.py

Stages:
    1. Load 4 streams, remap filenames, concat, excise ROI patches (11×11)
    2. Load FEL spectra, precompute PCA spectral overlap grids (shared)
    3. Build forward model with PCA spectrum
    4. Staged refinement: bootstrap → spectral → joint → full
    5. Report final merged intensities
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import torch

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
    PixelLikelihood,
    precompute_spectral_overlap,
    RefinementStage,
    predict_reflections,
    update_observations,
)
from torchsx.forward.batch import ROIBatchLoader
from torchsx.io._roi_store import ROIStore

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════
STREAM_NAMES = [
    "delay_-45",
    "delay_-91",
    "delay_-136",
    "delay_-182",
]
STREAM_DIR = "sfx_scan/streams"
H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
SPECTRA_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"
COMBINED_ROI_PATH = "combined_4stream_rois_11x11.safetensors"
PCA_CACHE_PATH = "delay_-45_pca_grids_v2.npz"  # shared PCA for all streams
PATCH_SIZE = 11
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PCA spectrum settings
N_PCA_MODES = 5
SIGMA_R_INIT = 0.001   # Å⁻¹ (radial: mosaicity + divergence)
SIGMA_T_INIT = 0.0007  # Å⁻¹ (tangential: ~1 pixel FWHM at 75µm/94mm)

# Refinement settings
N_CRYSTALS_PER_BATCH = 500  # ~5K obs per batch
GRADIENT_CLIP = 5.0

# Regularization weights
REG_SCALE = 0.01     # L2 on log_G
REG_BFACTOR = 0.01   # L2 on B values
REG_PCA = 0.001      # L2 on PCA coeffs
REG_RELP_R = 0.1     # L2 on σ_r deviation from resolution-bin mean
REG_RELP_T = 0.1     # L2 on σ_t deviation from global mean
WEIGHT_FLOOR = 0.05   # minimum loss weight for any ROI

# Spot re-prediction settings
REPREDICT_EVERY = 10  # re-predict spots every N epochs (0 = disabled)
REPREDICT_SIGMA = 0.005  # Å⁻¹, effective relp width for Ewald cutoff

# ══════════════════════════════════════════════════════════════════════════
# 1. Load and concatenate streams
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"Step 1: Loading {len(STREAM_NAMES)} streams")
print("=" * 70)

t0 = time.time()
streams = []
for name in STREAM_NAMES:
    sfx_path = os.path.join(STREAM_DIR, f"{name}.sfx")
    s = SFXStream.load(sfx_path)
    s = s.remap_filenames(
        r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
        H5_ROOT + r"/\1/",
    )
    c = s._core
    print(f"  {name}: {c.n_crystals:,} crystals, {c.n_observations:,} obs")
    streams.append(s)

stream = SFXStream.concat(streams)
core = stream._core
print(f"  Combined: {core.n_crystals:,} crystals, {core.n_observations:,} obs, "
      f"{core.n_images:,} images")
print(f"  Loaded in {time.time()-t0:.1f}s")

# Verify a file exists
fnames = stream.filenames
if fnames:
    test_fn = fnames[0]
    assert Path(test_fn).exists(), f"Data file not found: {test_fn}"
    print(f"  Data files accessible: ✓")

# Prepare scaler inputs (UB matrices, wavelength, etc.)
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
wavelength = inp.wavelength
print(f"  Wavelength: {wavelength:.4f} Å")
print(f"  Unique reflections: {hkl_idx.n_unique:,}")

# Pre-compute per-unique-reflection resolution (s_sq = (1/(2d))²)
# Used for relp width regularization
s_sq_per_obs = inp.s_sq
s_sq_unique = torch.zeros(hkl_idx.n_unique)
s_sq_count = torch.zeros(hkl_idx.n_unique)
s_sq_unique.scatter_add_(0, hkl_idx.hkl_id, s_sq_per_obs)
s_sq_count.scatter_add_(0, hkl_idx.hkl_id, torch.ones_like(s_sq_per_obs))
s_sq_unique = s_sq_unique / s_sq_count.clamp(min=1)
# Assign resolution bins (20 bins)
N_RES_BINS = 20
_, bin_edges = torch.histogram(s_sq_unique.float(), bins=N_RES_BINS)
res_bin = torch.bucketize(s_sq_unique, bin_edges[1:-1])  # (n_unique,) int
print(f"  Resolution bins for relp restraints: {N_RES_BINS}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Excise ROI patches (per-stream, then concat)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"Step 2: ROI patches ({PATCH_SIZE}×{PATCH_SIZE})")
print("=" * 70)

if Path(COMBINED_ROI_PATH).exists():
    print(f"  Loading existing combined ROI store: {COMBINED_ROI_PATH}")
    roi_store = ROIStore(COMBINED_ROI_PATH)
else:
    # Excise per-stream (cached), then concatenate
    per_stream_stores = []
    for name, s in zip(STREAM_NAMES, streams):
        roi_path = f"{name}_rois_{PATCH_SIZE}x{PATCH_SIZE}.safetensors"
        if Path(roi_path).exists():
            print(f"  Loading cached: {roi_path}")
            per_stream_stores.append(ROIStore(roi_path))
        else:
            print(f"  Excising {name}...")
            t1 = time.time()
            store = s.excise_rois(roi_path, patch_size=PATCH_SIZE, n_workers=1)
            print(f"    Done in {time.time()-t1:.1f}s")
            per_stream_stores.append(store)

    print(f"  Concatenating {len(per_stream_stores)} ROI stores...")
    t1 = time.time()
    roi_store = ROIStore.concat(per_stream_stores, COMBINED_ROI_PATH)
    print(f"    Done in {time.time()-t1:.1f}s")
    # Close per-stream handles
    for s in per_stream_stores:
        s.close()

patches_shape = roi_store["patches"].shape
print(f"  Patches shape: {patches_shape}")
print(f"  Storage: {Path(COMBINED_ROI_PATH).stat().st_size / 1e9:.2f} GB")

# ══════════════════════════════════════════════════════════════════════════
# 3. Load FEL spectra and precompute PCA overlap (shared across streams)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 3: PCA spectral overlap (shared)")
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
print(f"  Loaded {n_spectra:,} spectra in {time.time()-t0:.1f}s")

print("  Computing PCA spectral overlap grids...")
t0 = time.time()
grids = precompute_spectral_overlap(
    energy_eV, spectra,
    n_modes=N_PCA_MODES,
    n_ds=200, n_vr=50, n_sr=20,
    ds_r_range=(-0.01, 0.01),
    v_r_range=(0.01, 0.8),
    sigma_r_range=(0.0001, 0.01),
    max_shots_pca=2000,
    cache_path=PCA_CACHE_PATH,
)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Explained variance: {grids.explained_variance.cumsum()[-1]:.4f}")

# Map ALL crystals to spectra (sequential wrap — same beamtime, shared PCA)
crystal_to_shot = np.arange(core.n_crystals) % n_spectra
print(f"  Crystal→spectrum mapping: sequential wrap over {n_spectra} spectra")

# ══════════════════════════════════════════════════════════════════════════
# 4. Build forward model
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 4: Building forward model")
print("=" * 70)

geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile(
    n_unique=hkl_idx.n_unique,
    sigma_r_init=SIGMA_R_INIT,
    sigma_t_init=SIGMA_T_INIT,
)
pca_spectrum = PCASpectrum(grids, core.n_crystals, crystal_to_shot)

model = PixelForwardModel(
    structure_factors=sf,
    orientation=orientation,
    crystal_scale=crystal_scale,
    crystal_bfactor=crystal_bfactor,
    relp_profile=relp,
    spectrum=pca_spectrum,
    geometry=geometry,
    wavelength=wavelength,
)

device = torch.device(DEVICE)
model = model.to(device)
res_bin = res_bin.to(device)
print(f"  Device: {device}")

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {n_params:,}")
print(f"  Trainable: {n_trainable:,}")

# ══════════════════════════════════════════════════════════════════════════
# 5. Build data loaders (90/10 train/val split)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 5: Data loader (90/10 train/val split)")
print("=" * 70)

torch.manual_seed(42)
n_obs = core.n_observations
perm = torch.randperm(n_obs)
n_val = n_obs // 10
val_indices = perm[:n_val].sort().values
train_indices = perm[n_val:].sort().values

train_mask = torch.zeros(n_obs, dtype=torch.bool)
train_mask[train_indices] = True
val_mask = torch.zeros(n_obs, dtype=torch.bool)
val_mask[val_indices] = True

print(f"  Train: {train_mask.sum().item():,} obs ({100*train_mask.sum().item()/n_obs:.1f}%)")
print(f"  Val:   {val_mask.sum().item():,} obs ({100*val_mask.sum().item()/n_obs:.1f}%)")

train_loader = ROIBatchLoader(
    roi_store=roi_store, core=core, hkl_index=hkl_idx,
    geometry=geometry, n_crystals_per_batch=N_CRYSTALS_PER_BATCH,
    device=device, obs_mask=train_mask,
)
val_loader = ROIBatchLoader(
    roi_store=roi_store, core=core, hkl_index=hkl_idx,
    geometry=geometry, n_crystals_per_batch=N_CRYSTALS_PER_BATCH,
    device=device, obs_mask=val_mask,
)
print(f"  Train batches/epoch: {len(train_loader)}")
print(f"  Val batches/epoch: {len(val_loader)}")

# ══════════════════════════════════════════════════════════════════════════
# 6. Refinement stages
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 6: Staged refinement")
print("=" * 70)

likelihood = PixelLikelihood(mode="gaussian")

stages = [
    RefinementStage(
        name="bootstrap",
        n_epochs=50,
        lr=1e-3,
        frozen=["orientation", "spectrum", "geometry"],
    ),
    RefinementStage(
        name="spectral",
        n_epochs=50,
        lr=5e-4,
        frozen=["orientation", "geometry"],
    ),
    RefinementStage(
        name="joint",
        n_epochs=50,
        lr=2e-4,
        frozen=["geometry"],
    ),
    RefinementStage(
        name="full",
        n_epochs=50,
        lr=1e-4,
        frozen=[],
    ),
]

from torchsx.forward.refine import _set_requires_grad, _module_for_group


def relp_regularization(relp_module, res_bin, n_bins):
    """Regularize relp widths toward smooth resolution-dependent model.

    σ_r: each reflection is pulled toward the mean of its resolution bin.
    σ_t: each reflection is pulled toward the global mean.
    This allows smooth variation with resolution without a parametric form.
    """
    log_sr = relp_module.log_sigma_r
    log_st = relp_module.log_sigma_t

    # σ_r: regularize toward resolution-bin mean
    bin_sum_r = torch.zeros(n_bins, device=log_sr.device)
    bin_count = torch.zeros(n_bins, device=log_sr.device)
    bin_sum_r.scatter_add_(0, res_bin, log_sr)
    bin_count.scatter_add_(0, res_bin, torch.ones_like(log_sr))
    bin_mean_r = bin_sum_r / bin_count.clamp(min=1)
    target_r = bin_mean_r[res_bin]  # (n_unique,)
    reg_r = REG_RELP_R * (log_sr - target_r.detach()).pow(2).sum()

    # σ_t: regularize toward global mean
    target_t = log_st.mean().detach()
    reg_t = REG_RELP_T * (log_st - target_t).pow(2).sum()

    return reg_r + reg_t


# ── Auto-calibrate global F scale ─────────────────────────────────────
print("\nCalibrating global F scale...")
with torch.no_grad():
    cal_batch = next(iter(train_loader)).to(device)
    cal_result = model.forward_debug(cal_batch)

    integral_center = cal_result.integral[:, cal_batch.patches.shape[1]//2,
                                            cal_batch.patches.shape[2]//2]
    good = integral_center > integral_center.quantile(0.75)
    if good.sum() > 10:
        obs_peak = cal_batch.patches[good].float().max(dim=-1).values.max(dim=-1).values
        pred_signal_peak = cal_result.n_signal[good].max(dim=-1).values.max(dim=-1).values
        ratio = (obs_peak / pred_signal_peak.clamp(min=1e-6)).median()
        log_gs_init = torch.log(ratio.clamp(min=1.0))
        sf.log_global_scale.data.fill_(log_gs_init.item())
        print(f"  Estimated global scale: {ratio.item():.1f} (logGS={log_gs_init.item():.2f})")
    else:
        print("  Warning: too few ROIs with meaningful integral, keeping logGS=0")

total_t0 = time.time()
obs_weight = None  # computed once in first stage

for stage in stages:
    print(f"\n--- Stage: {stage.name} ({stage.n_epochs} epochs, lr={stage.lr}) ---")
    if stage.frozen:
        print(f"    Frozen: {stage.frozen}")

    _set_requires_grad(model, stage.frozen)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"    Trainable params: {n_train:,}")

    if not trainable:
        print("    No trainable parameters, skipping.")
        continue

    optimizer = torch.optim.Adam(trainable, lr=stage.lr)

    # Precompute data-driven weights (once, first stage only)
    if obs_weight is None:
        print("    Computing data-driven loss weights (signal significance)...")
        from torchsx.forward.corrections import signal_significance
        obs_weight = torch.zeros(core.n_observations, device="cpu")
        weight_loader = ROIBatchLoader(
            roi_store=roi_store, core=core, hkl_index=hkl_idx,
            geometry=geometry, n_crystals_per_batch=N_CRYSTALS_PER_BATCH,
            device=device,
        )
        for batch in weight_loader:
            batch = batch.to(device)
            snr = signal_significance(batch.patches.float(), batch.valid,
                                      batch.background, center_radius=2)
            obs_weight[batch.obs_idx.cpu()] = snr.cpu()
        del weight_loader
        obs_weight = WEIGHT_FLOOR + (1.0 - WEIGHT_FLOOR) * torch.sigmoid((obs_weight - 3.0) / 1.0)
        n_sig = (obs_weight > 0.5).sum().item()
        print(f"    Weight stats: median={obs_weight.median():.4f}  "
              f"mean={obs_weight.mean():.4f}  "
              f"ROIs with >3σ signal: {n_sig:,} ({100*n_sig/n_obs:.1f}%)")

    for epoch in range(stage.n_epochs):
        epoch_t0 = time.time()
        total_loss = 0.0
        total_pixels = 0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            n_pred = model(batch)

            roi_weight = obs_weight[batch.obs_idx.cpu()].to(device)
            loss = likelihood(n_pred, batch.patches.float(), batch.valid,
                              weights=roi_weight)

            # Regularization
            reg = torch.tensor(0.0, device=device)
            if crystal_scale.log_G.requires_grad:
                reg = reg + REG_SCALE * (crystal_scale.log_G ** 2).sum()
            if crystal_bfactor.B.requires_grad:
                reg = reg + REG_BFACTOR * (crystal_bfactor.B ** 2).sum()
            if pca_spectrum.coeffs.requires_grad:
                reg = reg + REG_PCA * (pca_spectrum.coeffs ** 2).sum()
            if relp.log_sigma_r.requires_grad:
                reg = reg + relp_regularization(relp, res_bin, N_RES_BINS)

            (loss + reg).backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_pixels += batch.valid.sum().item()
            n_batches += 1

        train_lpp = total_loss / max(total_pixels, 1)

        # Validation loss
        val_loss = 0.0
        val_pixels = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                n_pred = model(batch)
                w_val = obs_weight[batch.obs_idx.cpu()].to(device)
                loss = likelihood(n_pred, batch.patches.float(),
                                  batch.valid, weights=w_val)
                val_loss += loss.item()
                val_pixels += batch.valid.sum().item()
        val_lpp = val_loss / max(val_pixels, 1)

        epoch_time = time.time() - epoch_t0

        # Parameter stats
        grad_F = sf.log_F.grad
        grad_F_norm = grad_F.norm().item() if grad_F is not None else 0.0
        grad_G = crystal_scale.log_G.grad
        grad_G_norm = grad_G.norm().item() if grad_G is not None else 0.0
        gs_val = sf.log_global_scale.item()
        sr_med = torch.exp(relp.log_sigma_r).median().item()
        st_med = torch.exp(relp.log_sigma_t).median().item()

        print(
            f"    Epoch {epoch+1}/{stage.n_epochs}: "
            f"train={train_lpp:.6f}  val={val_lpp:.6f}  "
            f"logGS={gs_val:.2f}  "
            f"σ_r={sr_med:.5f}  σ_t={st_med:.5f}  "
            f"|∇F|={grad_F_norm:.2f}  |∇G|={grad_G_norm:.2f}  "
            f"time={epoch_time:.1f}s"
        )

        # ── Periodic spot re-prediction ──────────────────────────────
        if (REPREDICT_EVERY > 0
                and "geometry" not in stage.frozen
                and (epoch + 1) % REPREDICT_EVERY == 0):
            print(f"    Re-predicting spots (epoch {epoch+1})...")
            t_rp = time.time()
            with torch.no_grad():
                cell = core.cell  # (n_crystals, 6) — a,b,c in Å, α,β,γ in deg
                d_min = 1.5  # Å
                predicted = predict_reflections(
                    orientation=orientation,
                    geometry=geometry,
                    cell=cell.to(device),
                    d_min=d_min,
                    wavelength=wavelength,
                    sigma_cutoff=3.0,
                    relp_sigma=REPREDICT_SIGMA,
                )
                n_pred_refl = predicted.hkl.shape[0]
                print(f"      Predicted {n_pred_refl:,} reflections")

                # Update observation positions + re-excise moved ROIs
                core, roi_store = update_observations(
                    core, predicted, roi_store,
                    threshold=0.5, n_workers=1,
                )

            # Rebuild loaders with updated core/ROI store
            train_loader = ROIBatchLoader(
                roi_store=roi_store, core=core, hkl_index=hkl_idx,
                geometry=geometry, n_crystals_per_batch=N_CRYSTALS_PER_BATCH,
                device=device, obs_mask=train_mask,
            )
            val_loader = ROIBatchLoader(
                roi_store=roi_store, core=core, hkl_index=hkl_idx,
                geometry=geometry, n_crystals_per_batch=N_CRYSTALS_PER_BATCH,
                device=device, obs_mask=val_mask,
            )
            print(f"      Done in {time.time()-t_rp:.1f}s")

total_time = time.time() - total_t0
print(f"\n  Total refinement time: {total_time:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 7. Results summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 7: Results")
print("=" * 70)

with torch.no_grad():
    global_scale = torch.exp(sf.log_global_scale).item()
    F_sq = torch.exp(sf.log_F * 2).cpu()
    scale = torch.exp(crystal_scale.log_G).cpu()
    sigma_r = torch.exp(relp.log_sigma_r).cpu()
    sigma_t = torch.exp(relp.log_sigma_t).cpu()

print(f"  Global scale: {global_scale:.2f} (log={sf.log_global_scale.item():.2f})")
print(f"  |F|² range: {F_sq.min():.2f} - {F_sq.max():.2f}, median={F_sq.median():.2f}")
print(f"  Scale range: {scale.min():.4f} - {scale.max():.4f}, median={scale.median():.4f}")
print(f"  σ_r range: {sigma_r.min():.6f} - {sigma_r.max():.6f}, median={sigma_r.median():.6f}")
print(f"  σ_t range: {sigma_t.min():.6f} - {sigma_t.max():.6f}, median={sigma_t.median():.6f}")

if isinstance(pca_spectrum, PCASpectrum):
    coeffs = pca_spectrum.coeffs.detach().cpu()
    print(f"  PCA coeffs: {coeffs.shape}, norm range: "
          f"{coeffs.norm(dim=1).min():.4f} - {coeffs.norm(dim=1).max():.4f}")

print(f"\nDone. Total wall time: {total_time:.0f}s")
