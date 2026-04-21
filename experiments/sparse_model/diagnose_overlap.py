"""Diagnostic: visualize predicted vs observed sparse patterns.

Finds observations with many sparse pixels (likely strong reflections),
shows the observed sparse pattern alongside the model's predicted signal,
and tracks how predictions evolve over training epochs.

Usage:
    .venv/bin/python -u sparse_model/diagnose_overlap.py
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
from matplotlib.colors import LogNorm

from torchsx import SFXStream
from torchsx.scaling import (
    CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs,
)
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import (
    RefinableGeometry, RelpProfile, PCASpectrum, PixelForwardModel,
    precompute_spectral_overlap, AttractionLoss, PixelIndexMapper,
)
from torchsx.forward._sparse_data import SparseFrameStore
from torchsx.forward.sparse_batch import SparseBatchLoader, build_frame_to_obs
from safetensors.torch import load_file

# ══════════════════════════════════════════════════════════════════════════
# Configuration — same as refine_sparse.py
# ══════════════════════════════════════════════════════════════════════════
STREAM_NAMES = ["delay_-45", "delay_-91", "delay_-136", "delay_-182"]
STREAM_DIR = "sfx_scan/streams"
H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
COMPRESSED_DIR = Path("/das/work/p17/p17490/Peter/torchsx/compressed_z1.5")
SPECTRA_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"
PCA_CACHE_PATH = "delay_-45_pca_grids_v2.npz"
PATCH_SIZE = 11
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMAP_RE = r"/sf/alvra/data/p21957/raw/([^/]+)/data/"
REMAP_TO = H5_ROOT + r"/\1/"

N_PCA_MODES = 5
SIGMA_R_INIT = 0.005
SIGMA_T_INIT = 0.005
FRAMES_PER_BATCH = 50

# Diagnostic settings
N_TOP_OBS = 12           # number of strong observations to track
SNAPSHOT_EPOCHS = [0, 5, 10, 20, 50]  # epochs at which to snapshot
N_TRAIN_EPOCHS = 50      # total training epochs for the diagnostic
BLUR_SIGMA = 3.0         # attraction blur width (pixels)
BETA = 0.1               # Poisson NLL weight relative to attraction
LR = 1e-3

# ══════════════════════════════════════════════════════════════════════════
# 1. Load streams
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading streams")
print("=" * 70)

t0 = time.time()
streams = []
for name in STREAM_NAMES:
    sfx_path = os.path.join(STREAM_DIR, f"{name}.sfx")
    s = SFXStream.load(sfx_path)
    s = s.remap_filenames(REMAP_RE, REMAP_TO)
    streams.append(s)

stream = SFXStream.concat(streams)
core = stream._core
print(f"  {core.n_crystals:,} crystals, {core.n_observations:,} obs ({time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════
# 2. Good mask & radial binning
# ══════════════════════════════════════════════════════════════════════════
any_h5 = Path(H5_ROOT) / "run0121-s2_inj55_scan_0p5uJ" / "acq0001.JF06T32V07.h5"
with h5py.File(any_h5, "r") as f:
    good_mask_np = np.asarray(f["data/JF06T32V07/meta/pixel_mask"][()], dtype=bool)
good_mask = torch.from_numpy(good_mask_np)

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

# ══════════════════════════════════════════════════════════════════════════
# 3. Load sparse data
# ══════════════════════════════════════════════════════════════════════════
print("Loading sparse data...")
t0 = time.time()
meta = core.metadata
filenames = meta["image"]["filename"]
events = meta["image"]["event"]

h5_to_sparse = {}
for fn in set(filenames):
    parent = os.path.basename(os.path.dirname(fn))
    h5_stem = os.path.splitext(os.path.basename(fn))[0]
    sparse_name = f"{parent}_{h5_stem}.sparse.safetensors"
    sparse_path = COMPRESSED_DIR / sparse_name
    if sparse_path.exists():
        h5_to_sparse[fn] = sparse_path

all_profiles, all_signal_idx, all_signal_frame, all_signal_val = [], [], [], []
h5_offsets = {}
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
    n_good_pixels=good_mask.sum().item(), detector_shape=tuple(good_mask.shape),
)
del all_profiles, all_signal_idx, all_signal_frame, all_signal_val
print(f"  Loaded in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 4. PCA spectra
# ══════════════════════════════════════════════════════════════════════════
print("Loading spectra...")
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
print(f"  Done ({time.time()-t0:.1f}s)")

device = torch.device(DEVICE)

# ══════════════════════════════════════════════════════════════════════════
# 5. Build model (all crystals, no split)
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding model...")
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)
frame_to_obs = build_frame_to_obs(core, filenames, events, h5_offsets)

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
    geometry=geometry, wavelength=inp.wavelength,
).to(device)

loader = SparseBatchLoader(
    sparse_store=combined_store, frame_to_obs=frame_to_obs, core=core,
    hkl_index=hkl_idx, pixel_mapper=pixel_mapper, bin_assign=bin_assign_t,
    patch_size=PATCH_SIZE, frames_per_batch=FRAMES_PER_BATCH,
    device=device, z_threshold=combined_store.z_threshold, prefetch=2,
)
print(f"  {sum(p.numel() for p in model.parameters()):,} total params")
print(f"  {len(loader)} batches/epoch")

# ══════════════════════════════════════════════════════════════════════════
# 6. Find observations with most sparse pixels
# ══════════════════════════════════════════════════════════════════════════
print("\nScanning for strong observations...")

best_obs = []  # list of (n_sparse, batch_copy, obs_index_in_batch)

with torch.no_grad():
    for batch in loader:
        B = batch.is_sparse.shape[0]
        sparse_per_obs = batch.is_sparse.sum(dim=(1, 2))  # (B,)
        for i in range(B):
            n_sp = sparse_per_obs[i].item()
            if len(best_obs) < N_TOP_OBS:
                best_obs.append((n_sp, batch, i))
                best_obs.sort(key=lambda x: x[0], reverse=True)
            elif n_sp > best_obs[-1][0]:
                best_obs[-1] = (n_sp, batch, i)
                best_obs.sort(key=lambda x: x[0], reverse=True)

# Extract individual observations into a mini-batch
# We need: is_sparse, sparse_counts, background, valid, threshold,
# and the batch fields needed for forward_sparse
print(f"  Top {N_TOP_OBS} observations by sparse pixel count:")
for rank, (n_sp, batch, idx) in enumerate(best_obs):
    hkl = batch.hkl[idx].cpu().tolist()
    print(f"    #{rank+1}: {n_sp} sparse pixels, hkl=({hkl[0]:.0f},{hkl[1]:.0f},{hkl[2]:.0f})")


def extract_obs(best_obs, device):
    """Build a mini-batch from the selected observations."""
    from torchsx.forward.sparse_batch import SparseBatch

    indices = []
    batches = []
    for _, batch, idx in best_obs:
        indices.append(idx)
        batches.append(batch)

    # Gather fields from each source batch
    fields = {}
    for attr in [
        "crystal_id", "hkl_id", "hkl", "panel_id",
        "fs_center", "ss_center", "obs_idx",
        "fs_grid", "ss_grid", "pid_grid",
        "background", "is_sparse", "sparse_counts", "threshold", "valid",
    ]:
        tensors = []
        for batch, idx in zip(batches, indices):
            t = getattr(batch, attr)
            if t is not None:
                tensors.append(t[idx:idx+1].to(device))
        if tensors:
            fields[attr] = torch.cat(tensors, dim=0)
        else:
            fields[attr] = None

    return SparseBatch(**fields)


diag_batch = extract_obs(best_obs, device)
del best_obs  # free batch references

print(f"  Diagnostic batch: {diag_batch.is_sparse.shape[0]} observations")

# ══════════════════════════════════════════════════════════════════════════
# 7. Snapshot function
# ══════════════════════════════════════════════════════════════════════════
def snapshot_predictions(model, batch, epoch_label):
    """Run forward model and return (n_signal, lam) for the diagnostic batch."""
    with torch.no_grad():
        result = model.forward_sparse(batch)
        gain = model.geometry.gain(batch.panel_id)[:, None, None]
        n_signal = result.n_signal.cpu()
        lam = (gain * (batch.background + result.n_signal)).cpu()
        bg = batch.background.cpu()
    return {
        "epoch": epoch_label,
        "n_signal": n_signal,
        "lam": lam,
        "bg": bg,
    }


# Initial snapshot (epoch 0, before any training)
snapshots = [snapshot_predictions(model, diag_batch, "init")]
print(f"\n  Snapshot at init: mean n_signal at sparse = "
      f"{snapshots[0]['n_signal'][diag_batch.is_sparse.cpu()].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 8. Train with snapshots
# ══════════════════════════════════════════════════════════════════════════
print(f"\nTraining {N_TRAIN_EPOCHS} epochs (snapshots at {SNAPSHOT_EPOCHS})...")

loss_fn = AttractionLoss(blur_sigma=BLUR_SIGMA, beta=BETA, val_frac=0.0)

# Only train F² (same as latest refine_sparse.py)
sf.log_global_scale.requires_grad_(False)
relp.log_sigma_r.requires_grad_(False)
relp.log_sigma_t.requires_grad_(False)
crystal_scale.log_G.requires_grad_(False)
crystal_bfactor.B.requires_grad_(False)
for p in geometry.parameters():
    p.requires_grad_(False)
for p in orientation.parameters():
    p.requires_grad_(False)
for p in pca_spectrum.parameters():
    p.requires_grad_(False)

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=LR)
print(f"  Trainable: {sum(p.numel() for p in trainable):,} params")

for epoch in range(N_TRAIN_EPOCHS):
    sum_loss = 0.0
    sum_resp = 0.0
    sum_attract = 0.0
    sum_smooth = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()
        result = model.forward_sparse(batch)
        gain = model.geometry.gain(batch.panel_id)[:, None, None]
        lr = loss_fn(
            result.n_signal, batch.background, batch.is_sparse,
            batch.sparse_counts, batch.threshold, batch.valid, gain,
        )
        lr.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()

        sum_loss += lr.loss.item()
        sum_resp += lr.mean_responsibility
        sum_attract += lr.attraction.item()
        sum_smooth += lr.mean_smoothed_signal
        n_batches += 1
        del batch, result, lr, gain

    avg_loss = sum_loss / max(n_batches, 1)
    avg_resp = sum_resp / max(n_batches, 1)
    avg_attract = sum_attract / max(n_batches, 1)
    avg_smooth = sum_smooth / max(n_batches, 1)
    print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} attract={avg_attract:.4f} "
          f"smooth={avg_smooth:.3f} resp={avg_resp:.4f}")

    if (epoch + 1) in SNAPSHOT_EPOCHS:
        snap = snapshot_predictions(model, diag_batch, f"ep{epoch+1}")
        snapshots.append(snap)
        mean_sig = snap["n_signal"][diag_batch.is_sparse.cpu()].mean()
        print(f"    → Snapshot: mean n_signal at sparse = {mean_sig:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 9. Create visualization
# ══════════════════════════════════════════════════════════════════════════
print("\nCreating visualization...")

obs_sparse = diag_batch.sparse_counts.cpu().float()
obs_is_sparse = diag_batch.is_sparse.cpu()
obs_valid = diag_batch.valid.cpu()
N = diag_batch.is_sparse.shape[0]
n_snaps = len(snapshots)

# Figure layout: N_TOP_OBS rows × (1 + n_snaps) columns
# Col 0: observed sparse pattern
# Col 1..n_snaps: predicted n_signal at each snapshot
n_cols = 1 + n_snaps
fig, axes = plt.subplots(N, n_cols, figsize=(3 * n_cols, 3 * N),
                         squeeze=False)

for i in range(N):
    hkl = diag_batch.hkl[i].cpu().tolist()
    n_sp = obs_is_sparse[i].sum().item()

    # Column 0: observed sparse pixels
    ax = axes[i, 0]
    obs_img = torch.where(obs_is_sparse[i], obs_sparse[i], torch.tensor(float("nan")))
    valid_img = obs_valid[i].float()
    # Show valid region as light gray, sparse pixels as colored
    ax.imshow(valid_img.numpy(), cmap="Greys_r", vmin=0, vmax=2, alpha=0.3)
    im = ax.imshow(obs_img.numpy(), cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.7)
    if i == 0:
        ax.set_title("Observed\n(sparse pixels)")
    ax.set_ylabel(f"({hkl[0]:.0f},{hkl[1]:.0f},{hkl[2]:.0f})\n{n_sp} sparse",
                   fontsize=8)

    # Columns 1..n_snaps: predicted signal
    # Find global vmax for this observation across all snapshots
    vmax_sig = max(snap["n_signal"][i].max().item() for snap in snapshots)
    vmax_sig = max(vmax_sig, 0.1)  # avoid zero

    for j, snap in enumerate(snapshots):
        ax = axes[i, 1 + j]
        sig = snap["n_signal"][i].numpy()

        im = ax.imshow(sig, cmap="viridis", interpolation="nearest",
                       vmin=0, vmax=vmax_sig)
        # Overlay sparse pixel locations as red dots
        spy, spx = torch.where(obs_is_sparse[i])
        ax.scatter(spx.numpy(), spy.numpy(), c="red", s=8, marker="x",
                   linewidths=0.5, alpha=0.7)
        plt.colorbar(im, ax=ax, shrink=0.7)
        if i == 0:
            ax.set_title(f"Predicted\n{snap['epoch']}")

        # Annotate with signal at sparse pixels
        sig_at_sparse = snap["n_signal"][i][obs_is_sparse[i]].mean().item()
        resp_i = snap["n_signal"][i][obs_is_sparse[i]].sum().item() / (
            snap["lam"][i][obs_is_sparse[i]].sum().item() + 1e-10
        )
        ax.text(0.02, 0.98, f"sig@sp={sig_at_sparse:.2f}\nresp={resp_i:.3f}",
                transform=ax.transAxes, fontsize=6, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # Remove tick labels for cleanliness
    for j in range(n_cols):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

fig.suptitle(f"Predicted vs Observed — σ_r={SIGMA_R_INIT}, blur_σ={BLUR_SIGMA}, "
             f"β={BETA}, patch={PATCH_SIZE}×{PATCH_SIZE}", fontsize=12)
fig.tight_layout()

out_path = "sparse_model/diagnose_overlap.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"  Saved to {out_path}")

# ══════════════════════════════════════════════════════════════════════════
# 10. Summary statistics
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

for snap in snapshots:
    sig_at_sparse = snap["n_signal"][obs_is_sparse].mean().item()
    sig_max = snap["n_signal"].max().item()
    sig_mean = snap["n_signal"].mean().item()
    # Overlap: fraction of sparse pixels with n_signal > 0.1
    overlap = (snap["n_signal"][obs_is_sparse] > 0.1).float().mean().item()
    print(f"  {snap['epoch']:>8s}: sig@sparse={sig_at_sparse:.4f}  "
          f"sig_max={sig_max:.2f}  sig_mean={sig_mean:.4f}  "
          f"overlap(>0.1)={overlap:.3f}")

print("\nDone.")
