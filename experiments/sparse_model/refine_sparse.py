"""Split-half pixel-level refinement against sparse compressed data.

Splits crystals into two independent halves, refines each separately,
and compares the resulting structure factors (CC_1/2, R_split).

Usage:
    .venv/bin/python refine_sparse.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import h5py
import bitshuffle.h5  # noqa: F401 — registers HDF5 filter
import torch

from torchsx import SFXStream
from torchsx.scaling import (
    CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs,
)
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.merge._stats import match_halves, cc_half, r_split, shell_stats, format_shell_table
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
PATCH_SIZE = 11
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMAP_RE = r"/sf/alvra/data/p21957/raw/([^/]+)/data/"
REMAP_TO = H5_ROOT + r"/\1/"

N_PCA_MODES = 5
SIGMA_R_INIT = 0.005       # ~5 pixels — wide enough to ensure initial overlap
SIGMA_T_INIT = 0.005
FRAMES_PER_BATCH = 50
GRADIENT_CLIP = 5.0
VAL_FRAC = 0.1

REG_SCALE = 0.01
REG_BFACTOR = 0.01
REG_PCA = 0.0          # don't kill PCA — it encodes real spectral variation
SEARCH_RADIUS = 20.0   # pixels — search radius for matching
SIGMA_ERR_R = 0.01     # radial position error (A^-1) — start wide
SIGMA_ERR_T = 0.005    # tangential position error
N_EPOCHS = 100
LR = 1e-3


def refine_half(
    label: str,
    half_stream: SFXStream,
    combined_store: SparseFrameStore,
    h5_offsets: dict[str, tuple[int, int]],
    pixel_mapper: PixelIndexMapper,
    bin_assign_t: torch.Tensor,
    grids,
    n_spectra: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine one half and return (unique_hkl, F_sq) on CPU.

    Returns
    -------
    unique_hkl : (N_unique, 3) int16
    F_sq : (N_unique,) float32
    """
    print(f"\n{'='*70}")
    print(f"  Refining half: {label}")
    print(f"{'='*70}")

    core = half_stream._core
    meta = core.metadata
    filenames = meta["image"]["filename"]
    events = meta["image"]["event"]

    print(f"  {core.n_crystals:,} crystals, {core.n_observations:,} obs")

    inp = prepare_scaler_inputs(core, n_shells=20)
    hkl_idx = compute_hkl_index(core.hkl)
    wavelength = inp.wavelength
    print(f"  Unique reflections: {hkl_idx.n_unique:,}")

    # Frame-to-obs mapping
    frame_to_obs = build_frame_to_obs(core, filenames, events, h5_offsets)
    n_mapped = sum(t.numel() for t in frame_to_obs.values())
    print(f"  Mapped: {n_mapped:,} / {core.n_observations:,} obs")

    # Build model
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
    print(f"  Batches/epoch: {len(loader)}")

    loss_fn = SparsePixelLoss(val_frac=VAL_FRAC)

    # Freeze global scale — degenerate with per-crystal scales
    sf.log_global_scale.requires_grad_(False)

    # Freeze relp widths — prevent the model from shrinking to predict nothing
    relp.log_sigma_r.requires_grad_(False)
    relp.log_sigma_t.requires_grad_(False)

    # Save initial F² for comparison
    with torch.no_grad():
        Fsq_init = torch.exp(sf.log_F * 2).cpu().clone()
        logF_init = sf.log_F.detach().clone()

    # Single optimizer — per-observation normalization fixes gradient dilution
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR)
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        epoch_t0 = time.time()

        sum_loss = 0.0
        sum_val = 0.0
        sum_resp = 0.0
        sum_pairs = 0.0
        n_batches = 0

        # Optional: anneal sigma_err from wide to narrow
        progress = epoch / max(N_EPOCHS - 1, 1)
        sigma_err_r = SIGMA_ERR_R * (1.0 - 0.9 * progress)
        sigma_err_t = SIGMA_ERR_T * (1.0 - 0.9 * progress)

        for pairs in loader:
            optimizer.zero_grad()
            result = model.forward_pairs(
                pairs,
                sigma_err_r=sigma_err_r,
                sigma_err_t=sigma_err_t,
            )

            lr = loss_fn(result, pairs)

            reg = torch.tensor(0.0, device=device)
            reg = reg + REG_SCALE * (crystal_scale.log_G ** 2).sum()
            reg = reg + REG_BFACTOR * (crystal_bfactor.B ** 2).sum()
            reg = reg + REG_PCA * (pca_spectrum.coeffs ** 2).sum()

            (lr.loss + reg).backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            optimizer.step()

            sum_loss += lr.loss.item()
            if lr.val_loss is not None:
                sum_val += lr.val_loss.item()
            sum_resp += lr.mean_responsibility
            sum_pairs += lr.mean_pairs_per_refl
            n_batches += 1

            del pairs, result, lr, reg

        avg_loss = sum_loss / max(n_batches, 1)
        avg_val = sum_val / max(n_batches, 1)
        avg_resp = sum_resp / max(n_batches, 1)
        avg_pairs = sum_pairs / max(n_batches, 1)
        dt = time.time() - epoch_t0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                logG_rms = crystal_scale.log_G.pow(2).mean().sqrt().item()
                B_rms = crystal_bfactor.B.pow(2).mean().sqrt().item()
                pca_rms = pca_spectrum.coeffs.pow(2).mean().sqrt().item()
                logF_delta = (sf.log_F - logF_init).abs().mean().item()
                sr = torch.exp(relp.log_sigma_r[0]).item()
                st = torch.exp(relp.log_sigma_t[0]).item()
            print(f"    Epoch {epoch+1:3d}/{N_EPOCHS}: "
                  f"train={avg_loss:.4f} val={avg_val:.4f}  "
                  f"pairs/refl={avg_pairs:.1f} "
                  f"logG={logG_rms:.3f} B={B_rms:.3f} pca={pca_rms:.3f} "
                  f"dlogF={logF_delta:.4f} "
                  f"sig_err_r={sigma_err_r:.4f} "
                  f"resp={avg_resp:.4f}  {dt:.1f}s")

    print(f"  Refinement time: {time.time()-t0:.0f}s")

    # Extract refined F²
    with torch.no_grad():
        F_sq = torch.exp(sf.log_F * 2).cpu()
    unique_hkl = hkl_idx.unique_hkl

    print(f"  |F|² range: {F_sq.min():.2f} - {F_sq.max():.2f}, median={F_sq.median():.2f}")

    # Compare initial vs refined F²
    ratio = F_sq / Fsq_init.clamp(min=1e-6)
    delta_log = (torch.log(F_sq.clamp(min=1e-6)) - torch.log(Fsq_init.clamp(min=1e-6))).abs()
    cc_init = torch.corrcoef(torch.stack([Fsq_init, F_sq]))[0, 1].item()
    changed_2x = ((ratio > 2.0) | (ratio < 0.5)).sum().item()
    print(f"  F² init vs refined:")
    print(f"    CC(init, refined) = {cc_init:.4f}")
    print(f"    median |log ratio| = {delta_log.median():.3f}")
    print(f"    changed >2x: {changed_2x:,} / {F_sq.numel():,} ({100*changed_2x/F_sq.numel():.1f}%)")

    # Clean up GPU memory
    del model, loader, optimizer
    torch.cuda.empty_cache()

    return unique_hkl, F_sq


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
    s = s.remap_filenames(REMAP_RE, REMAP_TO)
    c = s._core
    print(f"  {name}: {c.n_crystals:,} crystals, {c.n_observations:,} obs")
    streams.append(s)

stream = SFXStream.concat(streams)
core = stream._core
print(f"  Combined: {core.n_crystals:,} crystals, {core.n_observations:,} obs")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# 2. Load good mask & build radial bin assignment
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
# 3. Load sparse compressed files
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
# 4. Load FEL spectra and precompute PCA overlap
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
# 5. Split crystals into two halves and refine independently
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 5: Split-half refinement")
print("=" * 70)

half_a, half_b = stream.split_random(n_parts=2, seed=42)
print(f"  Half A: {half_a.n_crystals:,} crystals")
print(f"  Half B: {half_b.n_crystals:,} crystals")

hkl_a, Fsq_a = refine_half(
    "A", half_a, combined_store, h5_offsets,
    pixel_mapper, bin_assign_t, grids, n_spectra, device,
)

hkl_b, Fsq_b = refine_half(
    "B", half_b, combined_store, h5_offsets,
    pixel_mapper, bin_assign_t, grids, n_spectra, device,
)

# ══════════════════════════════════════════════════════════════════════════
# 6. Compare structure factors
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 6: Split-half comparison")
print("=" * 70)

# Use F = sqrt(F²) for CC and R_split (amplitude, not intensity)
F_a = torch.sqrt(Fsq_a.clamp(min=1e-6))
F_b = torch.sqrt(Fsq_b.clamp(min=1e-6))

cell = core.cell.median(dim=0).values
matched = match_halves(hkl_a, F_a, hkl_b, F_b, cell)

print(f"  Matched reflections: {matched.F1.shape[0]:,}")
if matched.F1.numel() > 0:
    cc = cc_half(matched)
    r = r_split(matched)
    print(f"  CC_1/2:  {cc:.4f}")
    print(f"  R_split: {r:.4f}")

    shells = shell_stats(matched, n_shells=20)
    print(f"\n{format_shell_table(shells, overall_cc=cc, overall_r=r)}")

    # ── Scatter plot of half A vs half B structure factors ────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1 = matched.F1.numpy()
    f2 = matched.F2.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: log-log scatter of F (amplitudes)
    ax = axes[0]
    ax.scatter(f1, f2, s=1, alpha=0.3, rasterized=True)
    vmin = min(f1[f1 > 0].min(), f2[f2 > 0].min())
    vmax = max(f1.max(), f2.max())
    ax.plot([vmin, vmax], [vmin, vmax], "r-", lw=0.8, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|F| half A")
    ax.set_ylabel("|F| half B")
    ax.set_title(f"|F| scatter (CC₁/₂ = {cc:.3f}, R_split = {r:.3f})")
    ax.set_aspect("equal")
    ax.legend()

    # Right: log-log scatter of F² (intensities)
    ax = axes[1]
    fsq1 = f1 ** 2
    fsq2 = f2 ** 2
    ax.scatter(fsq1, fsq2, s=1, alpha=0.3, rasterized=True)
    vmin2 = min(fsq1[fsq1 > 0].min(), fsq2[fsq2 > 0].min())
    vmax2 = max(fsq1.max(), fsq2.max())
    ax.plot([vmin2, vmax2], [vmin2, vmax2], "r-", lw=0.8, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("F² half A")
    ax.set_ylabel("F² half B")
    ax.set_title("F² scatter")
    ax.set_aspect("equal")
    ax.legend()

    fig.tight_layout()
    fig.savefig("sparse_model/split_half_scatter.png", dpi=150)
    print(f"\n  Scatter plot saved to sparse_model/split_half_scatter.png")
else:
    print("  ERROR: No matched reflections between halves!")

print("\nDone.")
