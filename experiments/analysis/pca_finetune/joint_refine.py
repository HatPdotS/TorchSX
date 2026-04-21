#!/usr/bin/env python
"""Joint refinement with closed-form F² and gradient-based PCA/scale/geometry.

Block coordinate descent:
  Phase A (per epoch): Accumulate weighted-merge statistics → closed-form F² update
  Phase B (per batch):  Gradient step for PCA coefficients, crystal scale, B-factor

The closed-form F² update is:
  For each unique hkl h:
    A_pair = scale_c × pol × omega × integral  (partiality without F²)
    excess_p = k_p/gain - bg_p                  (observed signal per pixel)
    attributed_i = frac_i × excess_p            (EM attribution to pair i)
    F²_h = Σ_pairs attributed_i / Σ_pairs A_i  (weighted merge)

Usage:
    sbatch --partition=gpu-day --gres=gpu:1 ...
    .venv/bin/python -u analysis/pca_finetune/joint_refine.py
"""

import json
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
    SparsePixelLoss,
    PixelIndexMapper,
    SparsePixelBatchLoader,
)
from torchsx.forward._sparse_data import SparseFrameStore
from torchsx.forward.sparse_batch import build_frame_to_obs
from safetensors.torch import load_file

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
COMPRESSED_DIR = Path("/das/work/p17/p17490/Peter/torchsx/compressed_z1.5")
SPECTRA_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"
PCA_CACHE_PATH = "delay_-45_pca_grids_v2.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMAP_RE = r"/sf/alvra/data/p21957/raw/([^/]+)/data/"
REMAP_TO = H5_ROOT + r"/\1/"

N_PCA_MODES = 5
SIGMA_R_INIT = 0.0002       # near delta function (σ_relp ≈ 0)
SIGMA_T_INIT = 0.001
FRAMES_PER_BATCH = 50
SEARCH_RADIUS = 20.0
GRADIENT_CLIP = 5.0
VAL_FRAC = 0.1

N_EPOCHS = 50
LR_PCA = 5e-4
LR_SCALE = 1e-3
LR_BFACTOR = 1e-3

# Regularization
REG_SCALE = 1e-4       # L2 on log_G
REG_BFACTOR = 1e-4     # L2 on B
REG_PCA = 1e-5         # L2 on PCA coefficients

# F² update settings
F_SQ_FLOOR = 0.01      # minimum F² value
F_SQ_MIN_DEN = 1e-6    # minimum denominator for F² merge

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
# 2. Good mask & radial binning
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 2: Good mask & radial binning")
print("=" * 70)

any_h5 = Path(H5_ROOT) / "run0121-s2_inj55_scan_0p5uJ" / "acq0001.JF06T32V07.h5"
with h5py.File(any_h5, "r") as f:
    good_mask_np = np.asarray(f["data/JF06T32V07/meta/pixel_mask"][()], dtype=bool)
good_mask = torch.from_numpy(good_mask_np)
n_good = good_mask.sum().item()
print(f"  Good pixels: {n_good:,} / {good_mask.numel():,}")

from torchsx.geometry import Geometry
geom = Geometry.from_stream(Path("merged_368.stream"))
r_map = geom.radial_map()
N_RADIAL_BINS = 500
r_good = r_map[good_mask_np]
radial_bin_edges = np.linspace(0, r_good.max() + 1, N_RADIAL_BINS + 1).astype(np.float32)
bin_assign = np.clip(
    np.digitize(r_good, radial_bin_edges) - 1, 0, N_RADIAL_BINS - 1
).astype(np.int64)
bin_assign_t = torch.from_numpy(bin_assign)
pixel_mapper = PixelIndexMapper(good_mask, device=torch.device("cpu"))
print(f"  PixelIndexMapper: {pixel_mapper.n_good:,} good pixels")

# ══════════════════════════════════════════════════════════════════════════
# 3. Load sparse compressed data
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 3: Loading sparse compressed data")
print("=" * 70)
t0 = time.time()

meta = core.metadata
filenames = meta["image"]["filename"]
events = meta["image"]["event"]

h5_to_sparse: dict[str, Path] = {}
for fn in set(filenames):
    parent = os.path.basename(os.path.dirname(fn))
    h5_stem = os.path.splitext(os.path.basename(fn))[0]
    sparse_name = f"{parent}_{h5_stem}.sparse.safetensors"
    sparse_path = COMPRESSED_DIR / sparse_name
    if sparse_path.exists():
        h5_to_sparse[fn] = sparse_path

print(f"  Mapped {len(h5_to_sparse):,} H5 files to sparse files")

all_profiles, all_signal_idx, all_signal_frame, all_signal_val = [], [], [], []
h5_offsets: dict[str, tuple[int, int]] = {}
global_frame_offset = 0
z_threshold = None

for h5_path, sparse_path in sorted(h5_to_sparse.items()):
    tensors = load_file(str(sparse_path))
    json_path = sparse_path.with_name(
        sparse_path.name.replace(".sparse.safetensors", ".sparse.json")
    )
    with open(json_path) as f:
        smeta = json.load(f)
    n_frames = tensors["profiles"].shape[0]
    h5_offsets[h5_path] = (global_frame_offset, n_frames)
    all_profiles.append(tensors["profiles"])
    all_signal_idx.append(tensors["signal_idx"])
    all_signal_frame.append((tensors["signal_frame"].long() + global_frame_offset).int())
    all_signal_val.append(tensors["signal_val"])
    if z_threshold is None:
        z_threshold = smeta["z_threshold"]
    global_frame_offset += n_frames

combined_store = SparseFrameStore(
    profiles=torch.cat(all_profiles), signal_idx=torch.cat(all_signal_idx),
    signal_frame=torch.cat(all_signal_frame), signal_val=torch.cat(all_signal_val),
    bin_edges=torch.from_numpy(radial_bin_edges), z_threshold=z_threshold,
    n_good_pixels=n_good, detector_shape=tuple(good_mask.shape),
)
del all_profiles, all_signal_idx, all_signal_frame, all_signal_val
print(f"  Total frames: {global_frame_offset:,}")
print(f"  Signal pixels: {combined_store.signal_idx.shape[0]:,}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 4. PCA spectral overlap
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 4: PCA spectral overlap")
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
# 5. Build model
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 5: Building model")
print("=" * 70)

inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
wavelength = inp.wavelength
n_unique = hkl_idx.n_unique

frame_to_obs = build_frame_to_obs(core, filenames, events, h5_offsets)
n_mapped = sum(t.numel() for t in frame_to_obs.values())
print(f"  Mapped: {n_mapped:,} / {core.n_observations:,} obs")

geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile(n_unique=hkl_idx.n_unique,
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

loader = SparsePixelBatchLoader(
    sparse_store=combined_store, frame_to_obs=frame_to_obs, core=core,
    hkl_index=hkl_idx, pixel_mapper=pixel_mapper, bin_assign=bin_assign_t,
    search_radius=SEARCH_RADIUS, frames_per_batch=FRAMES_PER_BATCH,
    device=device, prefetch=2,
)
print(f"  Unique reflections: {n_unique:,}")
print(f"  Batches/epoch: {len(loader)}")
print(f"  σ_r = {SIGMA_R_INIT} Å⁻¹ (near delta function)")

# ══════════════════════════════════════════════════════════════════════════
# 6. Setup optimizer (gradient-based params only)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 6: Setup optimizer")
print("=" * 70)

# F² is updated via closed-form — freeze from gradient
sf.log_F.requires_grad_(False)
sf.log_global_scale.requires_grad_(False)

# Relp widths frozen (σ_relp ≈ 0, width is spectral)
relp.log_sigma_r.requires_grad_(False)
relp.log_sigma_t.requires_grad_(False)

# Orientation frozen (from CrystFEL indexing)
for p in orientation.parameters():
    p.requires_grad_(False)

# Geometry frozen for now (could enable later)
for p in geometry.parameters():
    p.requires_grad_(False)

# Trainable: PCA coefficients, crystal scale, B-factor
pca_spectrum.coeffs.requires_grad_(True)
crystal_scale.log_G.requires_grad_(True)
crystal_bfactor.B.requires_grad_(True)

param_groups = [
    {"params": [pca_spectrum.coeffs], "lr": LR_PCA},
    {"params": [crystal_scale.log_G], "lr": LR_SCALE},
    {"params": [crystal_bfactor.B], "lr": LR_BFACTOR},
]
optimizer = torch.optim.Adam(param_groups)

n_trainable = sum(p.numel() for g in param_groups for p in g["params"])
print(f"  Trainable (gradient): {n_trainable:,} params")
print(f"    PCA coeffs: {pca_spectrum.coeffs.numel():,}")
print(f"    Crystal scale: {crystal_scale.log_G.numel():,}")
print(f"    B-factor: {crystal_bfactor.B.numel():,}")
print(f"  F² updated via closed-form merge ({n_unique:,} unique hkl)")

loss_fn = SparsePixelLoss(val_frac=VAL_FRAC)

# ══════════════════════════════════════════════════════════════════════════
# 7. Initial evaluation
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 7: Initial evaluation")
print("=" * 70)

t0 = time.time()
init_losses, init_val_losses, init_resps = [], [], []

with torch.no_grad():
    for pairs in loader:
        result = model.forward_pairs(pairs)
        lr = loss_fn(result, pairs)
        init_losses.append(lr.loss.item())
        if lr.val_loss is not None:
            init_val_losses.append(lr.val_loss.item())
        init_resps.append(lr.mean_responsibility)
        del pairs, result, lr

init_train = np.mean(init_losses)
init_val = np.mean(init_val_losses) if init_val_losses else float("nan")
init_resp = np.mean(init_resps)

# Save initial F² for comparison
F_sq_init = torch.exp(sf.log_global_scale + 2.0 * sf.log_F).detach().cpu().clone()

print(f"  Initial train NLL: {init_train:.4f}")
print(f"  Initial val NLL:   {init_val:.4f}")
print(f"  Initial resp:      {init_resp:.4f}")
print(f"  Initial F² median: {F_sq_init.median():.2f}")
print(f"  Evaluated in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 8. Joint refinement loop
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 8: Joint refinement")
print("=" * 70)

epoch_train_losses = []
epoch_val_losses = []
epoch_resps = []
epoch_F_sq_medians = []
epoch_scale_stds = []

t_start = time.time()

for epoch in range(N_EPOCHS):
    epoch_t0 = time.time()

    # ── Phase A + B combined: forward pass, accumulate F² stats, gradient step ──
    # Accumulators for closed-form F² (on CPU to avoid GPU OOM)
    num_F = torch.zeros(n_unique)   # Σ attributed signal
    den_F = torch.zeros(n_unique)   # Σ partiality A_pair
    count_F = torch.zeros(n_unique, dtype=torch.long)  # for diagnostics

    sum_loss = 0.0
    sum_val = 0.0
    sum_resp = 0.0
    n_batches = 0

    for pairs in loader:
        optimizer.zero_grad()

        # Forward pass
        result = model.forward_pairs(pairs)
        lr = loss_fn(result, pairs)

        # ── Accumulate F² statistics (detached, no gradient) ──
        with torch.no_grad():
            # F² per reflection (current values)
            F_sq_refl = sf.predict_I(pairs.hkl_id)  # (N_refl,)

            # Partiality without F²: A = n_signal / F²
            F_sq_pair = F_sq_refl[pairs.refl_idx]  # (N_pairs,)
            A_pair = result.n_signal_pairs.detach() / F_sq_pair.clamp(min=1e-10)

            # Excess signal per pixel: (observed - background) / gain
            gain = model.geometry.gain(pairs.pixel_panel)  # (N_pixels,)
            excess_pixel = pairs.pixel_counts.float() / gain.clamp(min=1e-6) - pairs.pixel_bg

            # EM-style attribution: split excess among pairs hitting same pixel
            n_signal_pix = torch.zeros(pairs.n_pixels, device=device)
            n_signal_pix.scatter_add_(0, pairs.pixel_idx, result.n_signal_pairs.detach())
            # Add background to denominator for proper fractional attribution
            total_pix = n_signal_pix + pairs.pixel_bg
            signal_frac = result.n_signal_pairs.detach() / total_pix[pairs.pixel_idx].clamp(min=1e-10)

            # Attributed signal for each pair
            attributed = signal_frac * excess_pixel[pairs.pixel_idx]

            # Scatter into hkl_id buckets (move to CPU)
            hkl_pair = pairs.hkl_id[pairs.refl_idx]  # (N_pairs,) unique hkl index
            num_F.scatter_add_(0, hkl_pair.cpu(), attributed.cpu())
            den_F.scatter_add_(0, hkl_pair.cpu(), A_pair.cpu())
            count_F.scatter_add_(0, hkl_pair.cpu(), torch.ones_like(hkl_pair.cpu(), dtype=torch.long))

        # ── Gradient step for PCA, scale, B-factor ──
        # Add regularization
        reg = (REG_SCALE * crystal_scale.log_G.pow(2).sum()
               + REG_BFACTOR * crystal_bfactor.B.pow(2).sum()
               + REG_PCA * pca_spectrum.coeffs.pow(2).sum())

        total_loss = lr.loss + reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in param_groups for p in g["params"]], GRADIENT_CLIP
        )
        optimizer.step()

        sum_loss += lr.loss.item()
        if lr.val_loss is not None:
            sum_val += lr.val_loss.item()
        sum_resp += lr.mean_responsibility
        n_batches += 1
        del pairs, result, lr, total_loss

    # ── Phase A completion: closed-form F² update ──
    with torch.no_grad():
        has_data = den_F > F_SQ_MIN_DEN
        F_sq_new = torch.full((n_unique,), F_SQ_FLOOR)
        F_sq_new[has_data] = (num_F[has_data] / den_F[has_data]).clamp(min=F_SQ_FLOOR)

        # Update log_F (absorbing global scale)
        log_gs = sf.log_global_scale.item()
        sf.log_F.data.copy_(
            (0.5 * torch.log(F_sq_new) - 0.5 * log_gs).to(device)
        )

        F_sq_median = F_sq_new[has_data].median().item()
        n_updated = has_data.sum().item()

    # ── Epoch statistics ──
    avg_loss = sum_loss / max(n_batches, 1)
    avg_val = sum_val / max(n_batches, 1)
    avg_resp = sum_resp / max(n_batches, 1)
    scale_std = crystal_scale.log_G.detach().std().item()
    B_mean = crystal_bfactor.B.detach().mean().item()
    dt = time.time() - epoch_t0

    epoch_train_losses.append(avg_loss)
    epoch_val_losses.append(avg_val)
    epoch_resps.append(avg_resp)
    epoch_F_sq_medians.append(F_sq_median)
    epoch_scale_stds.append(scale_std)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: "
              f"train={avg_loss:.4f} val={avg_val:.4f} "
              f"resp={avg_resp:.4f} "
              f"F²_med={F_sq_median:.2f} "
              f"scale_σ={scale_std:.4f} B̄={B_mean:.2f} "
              f"F²_updated={n_updated}/{n_unique}  "
              f"{dt:.1f}s")

total_time = time.time() - t_start
print(f"\n  Total training time: {total_time:.0f}s")

# ══════════════════════════════════════════════════════════════════════════
# 9. Final evaluation
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 9: Final evaluation")
print("=" * 70)

final_losses, final_val_losses, final_resps = [], [], []
with torch.no_grad():
    for pairs in loader:
        result = model.forward_pairs(pairs)
        lr = loss_fn(result, pairs)
        final_losses.append(lr.loss.item())
        if lr.val_loss is not None:
            final_val_losses.append(lr.val_loss.item())
        final_resps.append(lr.mean_responsibility)
        del pairs, result, lr

final_train = np.mean(final_losses)
final_val = np.mean(final_val_losses) if final_val_losses else float("nan")
final_resp = np.mean(final_resps)

F_sq_final = torch.exp(sf.log_global_scale + 2.0 * sf.log_F).detach().cpu()

print(f"  Final train NLL:   {final_train:.4f}  (was {init_train:.4f}, Δ={final_train-init_train:+.4f})")
print(f"  Final val NLL:     {final_val:.4f}  (was {init_val:.4f}, Δ={final_val-init_val:+.4f})")
print(f"  Final resp:        {final_resp:.4f}  (was {init_resp:.4f})")
print(f"  Final F² median:   {F_sq_final.median():.2f}  (was {F_sq_init.median():.2f})")
print(f"  Final scale σ:     {crystal_scale.log_G.std().item():.4f}")
print(f"  Final B mean:      {crystal_bfactor.B.mean().item():.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 10. Visualization
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 10: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
epochs = np.arange(1, N_EPOCHS + 1)

# Plot 1: Training curve
ax = axes[0, 0]
ax.plot(epochs, epoch_train_losses, "b-", label="Train NLL", lw=1.5)
ax.plot(epochs, epoch_val_losses, "r-", label="Val NLL", lw=1.5)
ax.axhline(init_train, color="b", ls="--", lw=0.8, alpha=0.5, label=f"Init train={init_train:.4f}")
ax.axhline(init_val, color="r", ls="--", lw=0.8, alpha=0.5, label=f"Init val={init_val:.4f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Poisson NLL")
ax.set_title("Training curve")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 2: Responsibility
ax = axes[0, 1]
ax.plot(epochs, epoch_resps, "g-", lw=1.5)
ax.axhline(init_resp, color="g", ls="--", lw=0.8, alpha=0.5, label=f"Init={init_resp:.4f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean responsibility")
ax.set_title("Signal fraction at sparse pixels")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 3: F² evolution
ax = axes[0, 2]
ax.plot(epochs, epoch_F_sq_medians, "m-", lw=1.5)
ax.axhline(F_sq_init.median().item(), color="m", ls="--", lw=0.8, alpha=0.5,
           label=f"Init={F_sq_init.median():.2f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Median F²")
ax.set_title("Structure factor evolution")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 4: F² scatter (init vs final)
ax = axes[1, 0]
log_F_init = torch.log10(F_sq_init.clamp(min=1e-3))
log_F_final = torch.log10(F_sq_final.clamp(min=1e-3))
ax.scatter(log_F_init.numpy(), log_F_final.numpy(), s=1, alpha=0.3, rasterized=True)
lims = [min(log_F_init.min(), log_F_final.min()).item(),
        max(log_F_init.max(), log_F_final.max()).item()]
ax.plot(lims, lims, "r-", lw=0.8, label="y = x")
ax.set_xlabel("log₁₀(F²) initial (CrystFEL merge)")
ax.set_ylabel("log₁₀(F²) refined")
ax.set_title(f"F² refinement ({n_unique:,} reflections)")
ax.legend(fontsize=7)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Plot 5: Crystal scale distribution
ax = axes[1, 1]
log_G = crystal_scale.log_G.detach().cpu().numpy()
ax.hist(log_G, bins=100, alpha=0.7, color="steelblue")
ax.set_xlabel("log(G) — crystal scale")
ax.set_ylabel("Crystals")
ax.set_title(f"Scale distribution (σ={np.std(log_G):.4f})")
ax.axvline(0, color="r", ls="--", lw=0.8)
ax.grid(True, alpha=0.3)

# Plot 6: Scale std + resp evolution
ax = axes[1, 2]
ax2 = ax.twinx()
ax.plot(epochs, epoch_scale_stds, "b-", lw=1.5, label="scale σ")
ax2.plot(epochs, epoch_resps, "g-", lw=1.5, label="resp")
ax.set_xlabel("Epoch")
ax.set_ylabel("Scale σ(log G)", color="b")
ax2.set_ylabel("Responsibility", color="g")
ax.set_title("Scale spread & responsibility")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Joint refinement: {core.n_crystals:,} crystals, {n_unique:,} reflections, "
    f"σ_r={SIGMA_R_INIT}\n"
    f"NLL: {init_train:.4f} → {final_train:.4f} (train), "
    f"{init_val:.4f} → {final_val:.4f} (val), "
    f"resp: {init_resp:.4f} → {final_resp:.4f}",
    fontsize=11, fontweight="bold",
)
fig.tight_layout()
out_path = OUT_DIR / f"joint_refine_{RUN_TAG}.png"
fig.savefig(out_path, dpi=150)
print(f"  Saved {out_path}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# 11. Save refined parameters
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 11: Save refined parameters")
print("=" * 70)

state = {
    "log_F": sf.log_F.detach().cpu(),
    "log_global_scale": sf.log_global_scale.detach().cpu(),
    "log_G": crystal_scale.log_G.detach().cpu(),
    "B": crystal_bfactor.B.detach().cpu(),
    "pca_coeffs": pca_spectrum.coeffs.detach().cpu(),
    "unique_hkl": sf.unique_hkl.cpu(),
}
ckpt_path = OUT_DIR / f"joint_refine_state_{RUN_TAG}.pt"
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
print(f"  Epochs:         {N_EPOCHS}")
print(f"  σ_r:            {SIGMA_R_INIT} Å⁻¹")
print()
print(f"  Train NLL:      {init_train:.4f} → {final_train:.4f}  ({100*(final_train-init_train)/abs(init_train):+.2f}%)")
print(f"  Val NLL:        {init_val:.4f} → {final_val:.4f}  ({100*(final_val-init_val)/abs(init_val):+.2f}%)")
print(f"  Responsibility: {init_resp:.4f} → {final_resp:.4f}")
print(f"  F² median:      {F_sq_init.median():.2f} → {F_sq_final.median():.2f}")
print(f"  Scale σ:        0.0000 → {crystal_scale.log_G.std().item():.4f}")
print(f"  B mean:         0.00 → {crystal_bfactor.B.mean().item():.2f}")
print(f"  Training time:  {total_time:.0f}s")
print(f"\nDone.")
