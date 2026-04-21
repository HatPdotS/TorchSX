#!/usr/bin/env python
"""Fine-tune PCA spectral coefficients against sparse pixel data.

Since σ_relp ≈ 0 (large crystal domains), the observed partiality IS the
SASE spectral kernel.  The PCA coefficients (5 per crystal) encode each
crystal's SASE spectrum.  Currently crystal_to_shot uses a sequential
modulo wrap — essentially arbitrary.  Fine-tuning PCA coefficients against
observed pixel data should improve data–model agreement.

This script:
  1. Loads the full pixel-pair pipeline (SparsePixelBatchLoader)
  2. Evaluates initial Poisson NLL with arbitrary PCA assignment
  3. Fine-tunes ONLY pca_spectrum.coeffs (freezes F², scale, geometry, etc.)
  4. Compares before/after metrics and visualises the improvement

Usage:
    .venv/bin/python -u analysis/pca_finetune/finetune_pca.py
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
STREAM_NAMES = ["delay_-45", "delay_-91", "delay_-136", "delay_-182"]
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

N_EPOCHS = 100
LR_PCA = 5e-4

OUT_DIR = Path(__file__).resolve().parent

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
print(f"  Unique reflections: {hkl_idx.n_unique:,}")
print(f"  Batches/epoch: {len(loader)}")
print(f"  σ_r = {SIGMA_R_INIT} Å⁻¹ (near delta function)")

# ══════════════════════════════════════════════════════════════════════════
# 6. Initial evaluation (before PCA tuning)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 6: Initial evaluation (before PCA tuning)")
print("=" * 70)

loss_fn = SparsePixelLoss(val_frac=VAL_FRAC)

# Save initial PCA coefficients
coeffs_init = pca_spectrum.coeffs.detach().cpu().clone()

t0 = time.time()
init_losses = []
init_val_losses = []
init_resps = []
init_pairs = []

with torch.no_grad():
    for pairs in loader:
        result = model.forward_pairs(pairs)
        lr = loss_fn(result, pairs)
        init_losses.append(lr.loss.item())
        if lr.val_loss is not None:
            init_val_losses.append(lr.val_loss.item())
        init_resps.append(lr.mean_responsibility)
        init_pairs.append(lr.mean_pairs_per_refl)
        del pairs, result, lr

init_train = np.mean(init_losses)
init_val = np.mean(init_val_losses) if init_val_losses else float("nan")
init_resp = np.mean(init_resps)
init_ppair = np.mean(init_pairs)

print(f"  Initial train NLL: {init_train:.4f}")
print(f"  Initial val NLL:   {init_val:.4f}")
print(f"  Initial resp:      {init_resp:.4f}")
print(f"  Initial pairs/refl:{init_ppair:.1f}")
print(f"  Evaluated in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 7. Fine-tune PCA coefficients only
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 7: Fine-tuning PCA coefficients")
print("=" * 70)

# Freeze everything except PCA coefficients
sf.log_F.requires_grad_(False)
sf.log_global_scale.requires_grad_(False)
crystal_scale.log_G.requires_grad_(False)
crystal_bfactor.B.requires_grad_(False)
relp.log_sigma_r.requires_grad_(False)
relp.log_sigma_t.requires_grad_(False)
for p in geometry.parameters():
    p.requires_grad_(False)
for p in orientation.parameters():
    p.requires_grad_(False)

# Only PCA coefficients are trainable
pca_spectrum.coeffs.requires_grad_(True)
trainable = [pca_spectrum.coeffs]
n_trainable = sum(p.numel() for p in trainable)
print(f"  Trainable: {n_trainable:,} params ({core.n_crystals} crystals × {N_PCA_MODES} modes)")

optimizer = torch.optim.Adam(trainable, lr=LR_PCA)

epoch_train_losses = []
epoch_val_losses = []
epoch_resps = []

t0 = time.time()
for epoch in range(N_EPOCHS):
    epoch_t0 = time.time()
    sum_loss = 0.0
    sum_val = 0.0
    sum_resp = 0.0
    sum_pairs = 0.0
    n_batches = 0

    for pairs in loader:
        optimizer.zero_grad()
        result = model.forward_pairs(pairs)
        lr = loss_fn(result, pairs)
        lr.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
        optimizer.step()

        sum_loss += lr.loss.item()
        if lr.val_loss is not None:
            sum_val += lr.val_loss.item()
        sum_resp += lr.mean_responsibility
        sum_pairs += lr.mean_pairs_per_refl
        n_batches += 1
        del pairs, result, lr

    avg_loss = sum_loss / max(n_batches, 1)
    avg_val = sum_val / max(n_batches, 1)
    avg_resp = sum_resp / max(n_batches, 1)
    avg_pairs = sum_pairs / max(n_batches, 1)
    dt = time.time() - epoch_t0

    epoch_train_losses.append(avg_loss)
    epoch_val_losses.append(avg_val)
    epoch_resps.append(avg_resp)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        with torch.no_grad():
            pca_rms = pca_spectrum.coeffs.pow(2).mean().sqrt().item()
            delta_pca = (pca_spectrum.coeffs.cpu() - coeffs_init).pow(2).mean().sqrt().item()
        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: "
              f"train={avg_loss:.4f} val={avg_val:.4f} "
              f"resp={avg_resp:.4f} pairs/refl={avg_pairs:.1f} "
              f"pca_rms={pca_rms:.4f} Δpca={delta_pca:.4f}  "
              f"{dt:.1f}s")

total_time = time.time() - t0
print(f"\n  Total training time: {total_time:.0f}s")

# ══════════════════════════════════════════════════════════════════════════
# 8. Final evaluation (after PCA tuning)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 8: Final evaluation (after PCA tuning)")
print("=" * 70)

coeffs_final = pca_spectrum.coeffs.detach().cpu().clone()

final_losses = []
final_val_losses = []
final_resps = []

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

print(f"  Final train NLL:   {final_train:.4f}  (was {init_train:.4f}, Δ={final_train - init_train:+.4f})")
print(f"  Final val NLL:     {final_val:.4f}  (was {init_val:.4f}, Δ={final_val - init_val:+.4f})")
print(f"  Final resp:        {final_resp:.4f}  (was {init_resp:.4f}, Δ={final_resp - init_resp:+.4f})")

# ══════════════════════════════════════════════════════════════════════════
# 9. PCA coefficient analysis
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 9: PCA coefficient analysis")
print("=" * 70)

delta_coeffs = coeffs_final - coeffs_init  # (n_crystals, n_modes)
shift_norm = delta_coeffs.norm(dim=1)      # per-crystal L2 shift

print(f"\n  Per-crystal coefficient shift (L2 norm):")
print(f"    min:    {shift_norm.min():.6f}")
print(f"    median: {shift_norm.median():.6f}")
print(f"    mean:   {shift_norm.mean():.6f}")
print(f"    max:    {shift_norm.max():.6f}")

print(f"\n  Per-mode statistics:")
print(f"  {'Mode':>6s}  {'init_rms':>10s}  {'final_rms':>10s}  {'mean_shift':>10s}  {'std_shift':>10s}  {'expl_var':>10s}")
for m in range(N_PCA_MODES):
    init_rms = coeffs_init[:, m].pow(2).mean().sqrt().item()
    final_rms = coeffs_final[:, m].pow(2).mean().sqrt().item()
    shift_mean = delta_coeffs[:, m].mean().item()
    shift_std = delta_coeffs[:, m].std().item()
    expl_var = grids.explained_variance[m]
    print(f"  {m:6d}  {init_rms:10.4f}  {final_rms:10.4f}  {shift_mean:+10.4f}  {shift_std:10.4f}  {expl_var:10.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 10. Visualization
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 10: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training curve
ax = axes[0, 0]
epochs = np.arange(1, N_EPOCHS + 1)
ax.plot(epochs, epoch_train_losses, "b-", label="Train NLL", linewidth=1.5)
ax.plot(epochs, epoch_val_losses, "r-", label="Val NLL", linewidth=1.5)
ax.axhline(init_train, color="b", ls="--", lw=0.8, alpha=0.5, label=f"Init train={init_train:.4f}")
ax.axhline(init_val, color="r", ls="--", lw=0.8, alpha=0.5, label=f"Init val={init_val:.4f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Poisson NLL")
ax.set_title("Training curve (PCA coeffs only)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Responsibility over training
ax = axes[0, 1]
ax.plot(epochs, epoch_resps, "g-", linewidth=1.5)
ax.axhline(init_resp, color="g", ls="--", lw=0.8, alpha=0.5, label=f"Init={init_resp:.4f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean responsibility")
ax.set_title("Signal fraction at sparse pixels")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Per-mode coefficient shift distribution
ax = axes[1, 0]
for m in range(N_PCA_MODES):
    vals = delta_coeffs[:, m].numpy()
    ax.hist(vals, bins=50, alpha=0.5, label=f"Mode {m}")
ax.set_xlabel("Coefficient shift (final − init)")
ax.set_ylabel("Crystals")
ax.set_title("PCA coefficient changes per mode")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: Initial vs final coefficients for mode 0 (largest variance)
ax = axes[1, 1]
ax.scatter(coeffs_init[:, 0].numpy(), coeffs_final[:, 0].numpy(),
           s=1, alpha=0.3, rasterized=True)
lims = [min(coeffs_init[:, 0].min(), coeffs_final[:, 0].min()).item(),
        max(coeffs_init[:, 0].max(), coeffs_final[:, 0].max()).item()]
ax.plot(lims, lims, "r-", lw=0.8, label="y = x")
ax.set_xlabel("Initial coeff (mode 0)")
ax.set_ylabel("Final coeff (mode 0)")
ax.set_title(f"Mode 0 coefficient scatter (expl_var={grids.explained_variance[0]:.3f})")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"PCA fine-tuning: {core.n_crystals:,} crystals, {N_PCA_MODES} modes, "
    f"σ_r={SIGMA_R_INIT}\n"
    f"NLL: {init_train:.4f} → {final_train:.4f} (train), "
    f"{init_val:.4f} → {final_val:.4f} (val)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
out_path = OUT_DIR / "pca_finetune_results.png"
fig.savefig(out_path, dpi=150)
print(f"  Saved {out_path}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# 11. Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Summary")
print("=" * 70)
print(f"  Crystals:     {core.n_crystals:,}")
print(f"  PCA modes:    {N_PCA_MODES}")
print(f"  Epochs:       {N_EPOCHS}")
print(f"  σ_r:          {SIGMA_R_INIT} Å⁻¹ (near delta function)")
print(f"  LR:           {LR_PCA}")
print(f"")
print(f"  Train NLL:    {init_train:.4f} → {final_train:.4f}  ({100*(final_train-init_train)/abs(init_train):+.2f}%)")
print(f"  Val NLL:      {init_val:.4f} → {final_val:.4f}  ({100*(final_val-init_val)/abs(init_val):+.2f}%)")
print(f"  Resp:         {init_resp:.4f} → {final_resp:.4f}")
print(f"  Coeff shift:  median={shift_norm.median():.4f}, max={shift_norm.max():.4f}")
print(f"  Training time:{total_time:.0f}s")
print(f"\nDone.")
