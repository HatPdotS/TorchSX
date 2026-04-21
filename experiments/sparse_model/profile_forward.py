"""Profile the sparse forward pass: batch construction + forward model + loss."""

import json
import os
import time
from pathlib import Path

import numpy as np
import h5py
import bitshuffle.h5  # noqa: F401
import torch

from torchsx import SFXStream
from torchsx.scaling import (
    CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs,
)
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import (
    RefinableGeometry, RelpProfile, PCASpectrum, PixelForwardModel,
    precompute_spectral_overlap, CensoredPoissonLoss, PixelIndexMapper,
)
from torchsx.forward._sparse_data import SparseFrameStore
from torchsx.forward.sparse_batch import SparseBatchLoader, build_frame_to_obs
from safetensors.torch import load_file

# ── Config (same as refine_sparse.py) ─────────────────────────────────
STREAM_DIR = "sfx_scan/streams"
H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
COMPRESSED_DIR = Path("/das/work/p17/p17490/Peter/torchsx/compressed")
SPECTRA_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"
PCA_CACHE_PATH = "delay_-45_pca_grids_v2.npz"
REMAP_RE = r"/sf/alvra/data/p21957/raw/([^/]+)/data/"
REMAP_TO = H5_ROOT + r"/\1/"
PATCH_SIZE = 11
FRAMES_PER_BATCH = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load data (reuse from refine_sparse.py) ───────────────────────────
print("Loading data...")
t0 = time.time()

streams = []
for name in ["delay_-45", "delay_-91", "delay_-136", "delay_-182"]:
    s = SFXStream.load(os.path.join(STREAM_DIR, f"{name}.sfx"))
    s = s.remap_filenames(REMAP_RE, REMAP_TO)
    streams.append(s)
stream = SFXStream.concat(streams)
core = stream._core
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)

# Good mask + radial bins
any_h5 = Path(H5_ROOT) / "run0121-s2_inj55_scan_0p5uJ" / "acq0001.JF06T32V07.h5"
with h5py.File(any_h5, "r") as f:
    good_mask_np = np.asarray(f["data/JF06T32V07/meta/pixel_mask"][()], dtype=bool)
good_mask = torch.from_numpy(good_mask_np)
n_good = good_mask.sum().item()

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

# Sparse data
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
    profiles=torch.cat(all_profiles),
    signal_idx=torch.cat(all_signal_idx),
    signal_frame=torch.cat(all_signal_frame),
    signal_val=torch.cat(all_signal_val),
    bin_edges=torch.from_numpy(radial_bin_edges),
    z_threshold=z_threshold,
    n_good_pixels=n_good,
    detector_shape=tuple(good_mask.shape),
)
del all_profiles, all_signal_idx, all_signal_frame, all_signal_val

frame_to_obs_t = build_frame_to_obs(core, filenames, events, h5_offsets)

# PCA spectra
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
grids = precompute_spectral_overlap(
    h5py.File(os.path.join(SPECTRA_DIR, "acq0001.BSDATA.h5"), "r")["SARFE10-PSSS059:SPECTRUM_X/data"][0],
    spectra, n_modes=5, n_ds=200, n_vr=50, n_sr=20,
    ds_r_range=(-0.01, 0.01), v_r_range=(0.01, 0.8), sigma_r_range=(0.0001, 0.01),
    max_shots_pca=2000, cache_path=PCA_CACHE_PATH,
)
crystal_to_shot = np.arange(core.n_crystals) % n_spectra

# Model
wavelength = inp.wavelength
geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile(n_unique=hkl_idx.n_unique, sigma_r_init=0.001, sigma_t_init=0.0007)
pca_spectrum = PCASpectrum(grids, core.n_crystals, crystal_to_shot)
model = PixelForwardModel(
    structure_factors=sf, orientation=orientation, crystal_scale=crystal_scale,
    crystal_bfactor=crystal_bfactor, relp_profile=relp, spectrum=pca_spectrum,
    geometry=geometry, wavelength=wavelength,
)
device = torch.device(DEVICE)
model = model.to(device)

print(f"\nMoving static data to GPU...")
t_gpu0 = time.time()
loader = SparseBatchLoader(
    sparse_store=combined_store, frame_to_obs=frame_to_obs_t, core=core,
    hkl_index=hkl_idx, pixel_mapper=pixel_mapper, bin_assign=bin_assign_t,
    patch_size=PATCH_SIZE, frames_per_batch=FRAMES_PER_BATCH,
    device=device, z_threshold=z_threshold, prefetch=0,  # no prefetch for profiling
)
if device.type == "cuda":
    torch.cuda.synchronize()
print(f"  GPU data transfer: {time.time()-t_gpu0:.1f}s")
if device.type == "cuda":
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_res = torch.cuda.memory_reserved() / 1e9
    print(f"  GPU memory: {mem_alloc:.2f} GB allocated, {mem_res:.2f} GB reserved")
loss_fn = CensoredPoissonLoss()

print(f"Setup done in {time.time()-t0:.1f}s")
print(f"Batches per epoch: {len(loader)}")
print(f"Device: {device}")

# ── Profile ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Profiling 10 batches (no prefetch, CUDA sync between steps)")
print("=" * 70)

N_PROFILE = 10

times = {
    "make_batch": [],
    "ensure_geometry": [],
    "compute_h_lab": [],
    "spectral_integral": [],
    "compute_signal": [],
    "bg_gain_pred": [],
    "loss_fn": [],
    "backward": [],
    "total": [],
}

# Pre-build batch list (bypass prefetch thread)
print("\nBuilding batches...")
import torch

n_frames = len(loader._frame_keys)
frame_order = torch.randperm(n_frames)
batch_groups = []
for start in range(0, n_frames, loader.frames_per_batch):
    end = min(start + loader.frames_per_batch, n_frames)
    batch_groups.append(frame_order[start:end])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

from collections import defaultdict
loader._profile_times = defaultdict(list)

for i in range(min(N_PROFILE, len(batch_groups))):
    group = batch_groups[i]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total = time.perf_counter()

    # 1. Make batch (GPU)
    t0 = time.perf_counter()
    batch = loader._make_batch(group)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_make = time.perf_counter() - t0

    if batch is None:
        print(f"  Batch {i}: empty, skipping")
        continue

    B = batch.crystal_id.shape[0]

    # 2. Ensure geometry
    t0 = time.perf_counter()
    batch = model._ensure_geometry(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_geom = time.perf_counter() - t0

    ph, pw = batch.background.shape[1], batch.background.shape[2]

    # 4. compute_h_lab
    t0 = time.perf_counter()
    s_hat_center = batch.s_hat_out[:, ph // 2, pw // 2, :]
    h_lab, h_hat, d_star, s_sq, friedel_sign = model.compute_h_lab(
        batch.hkl, batch.crystal_id, s_hat_center,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_hlab = time.perf_counter() - t0

    # 5. Spectral integral
    t0 = time.perf_counter()
    integral = model.compute_spectral_integral_pca(
        batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_spec = time.perf_counter() - t0

    # 6. Compute signal
    t0 = time.perf_counter()
    n_signal, F_sq, scale = model.compute_signal(batch, integral, s_sq)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_sig = time.perf_counter() - t0

    # 7. BG + gain + prediction
    t0 = time.perf_counter()
    bg = batch.background
    gain = model.geometry.gain(batch.panel_id)[:, None, None]
    n_pred = (gain * (bg + n_signal)).clamp(min=1e-3)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bg = time.perf_counter() - t0

    # 8. Loss
    t0 = time.perf_counter()
    loss_result = loss_fn(
        n_signal, batch.background, batch.is_sparse, batch.sparse_counts,
        batch.threshold, batch.valid, gain,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_loss = time.perf_counter() - t0

    # 9. Backward
    t0 = time.perf_counter()
    optimizer.zero_grad()
    loss_result.loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0

    t_total_end = time.perf_counter() - t_total

    times["make_batch"].append(t_make)
    times["ensure_geometry"].append(t_geom)
    times["compute_h_lab"].append(t_hlab)
    times["spectral_integral"].append(t_spec)
    times["compute_signal"].append(t_sig)
    times["bg_gain_pred"].append(t_bg)
    times["loss_fn"].append(t_loss)
    times["backward"].append(t_bwd)
    times["total"].append(t_total_end)

    print(f"  Batch {i}: B={B:,}  "
          f"make={t_make*1000:.0f}ms  "
          f"geom={t_geom*1000:.0f}ms  h_lab={t_hlab*1000:.0f}ms  "
          f"spec={t_spec*1000:.0f}ms  sig={t_sig*1000:.0f}ms  "
          f"bg={t_bg*1000:.0f}ms  loss={t_loss*1000:.0f}ms  "
          f"bwd={t_bwd*1000:.0f}ms  TOTAL={t_total_end*1000:.0f}ms")

# Summary
print("\n" + "=" * 70)
print("Summary (mean over batches, ms)")
print("=" * 70)
for key in times:
    vals = times[key]
    if vals:
        mean = sum(vals) / len(vals) * 1000
        mn = min(vals) * 1000
        mx = max(vals) * 1000
        pct = mean / (sum(times["total"]) / len(times["total"]) * 1000) * 100
        print(f"  {key:20s}: {mean:7.1f}ms  (min={mn:.0f}, max={mx:.0f})  {pct:5.1f}%")

# _make_batch internals
print("\n" + "=" * 70)
print("_make_batch internals (mean over batches, ms)")
print("=" * 70)
pt = loader._profile_times
total_internal = 0
for key in ["gather", "core_index", "pixel_grids", "flat_map", "background",
            "searchsorted", "threshold", "to_device"]:
    vals = pt.get(key, [])
    if vals:
        mean = sum(vals) / len(vals) * 1000
        total_internal += mean
        print(f"  {key:20s}: {mean:7.1f}ms")
print(f"  {'TOTAL':20s}: {total_internal:7.1f}ms")
