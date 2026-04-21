"""Quick diagnostic: why are we getting only ~1 pair per reflection?"""
import json, os, time
from pathlib import Path
import numpy as np
import h5py
import bitshuffle.h5  # noqa
import torch
from torchsx import SFXStream
from torchsx.forward import PixelIndexMapper
from torchsx.forward._sparse_data import SparseFrameStore
from torchsx.forward.sparse_batch import build_frame_to_obs
from torchsx.geometry import Geometry
from torchsx.forward._pixel_matching import match_pixels_to_reflections
from torchsx.merge._hkl import compute_hkl_index
from safetensors.torch import load_file

STREAM_DIR = "sfx_scan/streams"
H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
COMPRESSED_DIR = Path("/das/work/p17/p17490/Peter/torchsx/compressed_z1.5")
REMAP_RE = r"/sf/alvra/data/p21957/raw/([^/]+)/data/"
REMAP_TO = H5_ROOT + r"/\1/"

# Load one stream
s = SFXStream.load(os.path.join(STREAM_DIR, "delay_-45.sfx"))
s = s.remap_filenames(REMAP_RE, REMAP_TO)
core = s._core
meta = core.metadata
filenames = meta["image"]["filename"]
events = meta["image"]["event"]
print(f"Stream: {core.n_crystals} crystals, {core.n_observations} obs")

# Good mask
any_h5 = Path(H5_ROOT) / "run0121-s2_inj55_scan_0p5uJ" / "acq0001.JF06T32V07.h5"
with h5py.File(any_h5, "r") as f:
    good_mask_np = np.asarray(f["data/JF06T32V07/meta/pixel_mask"][()], dtype=bool)
good_mask = torch.from_numpy(good_mask_np)
pixel_mapper = PixelIndexMapper(good_mask)

# Radial binning
geom = Geometry.from_stream(Path("merged_368.stream"))
r_map = geom.radial_map()
N_RADIAL_BINS = 500
r_good = r_map[good_mask_np]
radial_bin_edges = np.linspace(0, r_good.max() + 1, N_RADIAL_BINS + 1).astype(np.float32)
bin_assign = np.clip(np.digitize(r_good, radial_bin_edges) - 1, 0, N_RADIAL_BINS - 1).astype(np.int64)
bin_assign_t = torch.from_numpy(bin_assign)

# Load one sparse file
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
    json_path = sparse_path.with_name(sparse_path.name.replace(".sparse.safetensors", ".sparse.json"))
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

store = SparseFrameStore(
    profiles=torch.cat(all_profiles), signal_idx=torch.cat(all_signal_idx),
    signal_frame=torch.cat(all_signal_frame), signal_val=torch.cat(all_signal_val),
    bin_edges=torch.from_numpy(radial_bin_edges), z_threshold=z_threshold,
    n_good_pixels=good_mask.sum().item(), detector_shape=tuple(good_mask.shape),
)

# Build frame_to_obs
frame_to_obs = build_frame_to_obs(core, filenames, events, h5_offsets)
print(f"Frames with obs: {len(frame_to_obs)}")

# Panel map
panel_map = core.geometry.build_panel_map()

# Look at a few frames
hkl_idx = compute_hkl_index(core.hkl)

# Build signal frame offset table
sig_frame = store.signal_frame.long()
order = sig_frame.argsort()
sig_frame_sorted = sig_frame[order]
sig_idx_sorted = store.signal_idx.long()[order]
sig_val_sorted = store.signal_val[order]
unique_sig_frames, sig_counts = sig_frame_sorted.unique_consecutive(return_counts=True)
sig_ends = sig_counts.cumsum(0)
sig_starts = sig_ends - sig_counts
frame_offsets = {f.item(): (s.item(), e.item()) for f, s, e in zip(unique_sig_frames, sig_starts, sig_ends)}

frame_keys = sorted(frame_to_obs.keys())
print(f"\nChecking first 10 frames with observations:")
for fk in frame_keys[:10]:
    obs = frame_to_obs[fk]
    n_obs = obs.numel()

    # Reflection positions
    r_ss = core.ss[obs].float()
    r_fs = core.fs[obs].float()
    r_panel = core.panel_id[obs]

    # Sparse pixels for this frame
    if fk in frame_offsets:
        s, e = frame_offsets[fk]
        flat_idx = sig_idx_sorted[s:e]
        pix_ss, pix_fs = pixel_mapper.flat_to_ss_fs(flat_idx)
        pix_panel = panel_map[pix_ss.long()]
        n_pix = flat_idx.shape[0]
    else:
        n_pix = 0
        pix_ss = torch.empty(0)
        pix_fs = torch.empty(0)
        pix_panel = torch.empty(0, dtype=torch.int16)

    print(f"\n  Frame {fk}: {n_obs} reflections, {n_pix} sparse pixels")

    if n_obs > 0 and n_pix > 0:
        # Check panel distribution
        refl_panels = r_panel.unique().tolist()
        pix_panels = pix_panel.unique().tolist()
        print(f"    Refl panels: {refl_panels[:10]}")
        print(f"    Pixel panels: {pix_panels[:10]}")

        # Check position ranges
        print(f"    Refl ss: [{r_ss.min():.0f}, {r_ss.max():.0f}]  fs: [{r_fs.min():.0f}, {r_fs.max():.0f}]")
        print(f"    Pixel ss: [{pix_ss.min():.0f}, {pix_ss.max():.0f}]  fs: [{pix_fs.min():.0f}, {pix_fs.max():.0f}]")

        # Try matching
        refl_frame_t = torch.full((n_obs,), fk, dtype=torch.long)
        pixel_frame_t = torch.full((n_pix,), fk, dtype=torch.long)

        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            r_ss, r_fs, r_panel, refl_frame_t,
            pix_ss.float(), pix_fs.float(), pix_panel, pixel_frame_t,
            sig_val_sorted[s:e].int(), torch.ones(n_pix),
            search_radius=20.0,
        )
        print(f"    Matched pairs: {refl_idx.shape[0]}  ({refl_idx.shape[0]/max(n_obs,1):.1f} pairs/refl)")

        if refl_idx.shape[0] > 0:
            # Show some distances
            matched_r_ss = r_ss[refl_idx[:5]]
            matched_r_fs = r_fs[refl_idx[:5]]
            matched_p_ss = pix_ss[pixel_idx[:5]].float()
            matched_p_fs = pix_fs[pixel_idx[:5]].float()
            dists = torch.sqrt((matched_r_ss - matched_p_ss)**2 + (matched_r_fs - matched_p_fs)**2)
            print(f"    First 5 distances: {dists.tolist()}")

        # Also try without panel constraint for comparison
        # Compute distances without panel check
        dss = r_ss[:, None] - pix_ss[None, :].float()
        dfs = r_fs[:, None] - pix_fs[None, :].float()
        dist = torch.sqrt(dss**2 + dfs**2)
        n_within = (dist <= 20.0).sum().item()
        n_same_panel = (r_panel[:, None] == pix_panel[None, :]).sum().item()
        n_both = ((dist <= 20.0) & (r_panel[:, None] == pix_panel[None, :])).sum().item()
        print(f"    Within 20px (any panel): {n_within}  same panel: {n_same_panel}  both: {n_both}")
