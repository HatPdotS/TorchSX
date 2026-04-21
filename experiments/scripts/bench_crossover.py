"""Find the crossover: frames/batch where GPU compute >= CPU data prep.

Sweeps frames_per_batch from 10 to 500 and measures:
  - CPU time: blosc2 decompress + patch extract + H2D transfer
  - GPU time: forward model + likelihood + backward

Uses the real PixelForwardModel from refine_e2e.py config.
"""

import time
import numpy as np
import torch

from torchsx import SFXStream
from torchsx.merge._hkl import compute_hkl_index
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.scaling import CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs
from torchsx.forward import (
    RefinableGeometry, RelpProfile, PixelForwardModel, PixelLikelihood,
)
from torchsx.forward.batch import ROIBatch, _build_pixel_grids
from torchsx.forward.corrections import estimate_background
from torchsx.io._frame_store import FrameStore
from torchsx.io._frame_extract import ROIExtractor
from torchsx.io._excise import _parse_event_id

# ── Config ──
H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
SPECTRA_DIR = "/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files"
STORE_DIR = "bench_frame_store"
PATCH_SIZE = 11
DEVICE = "cuda"


def build_model_and_data():
    """Load stream, build forward model, prepare frame→obs mapping."""
    print("Loading stream and building model...")
    s = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
    s = s.remap_filenames(
        r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
        H5_ROOT + r"/\1/",
    )
    core = s._core
    hkl_idx = compute_hkl_index(core.hkl)
    device = torch.device(DEVICE)

    # Build model components
    inp = prepare_scaler_inputs(core, n_shells=20)
    orientation = CrystalOrientation(inp.UB)
    crystal_scale = AbsoluteScale(core.n_crystals)
    crystal_bfactor = BScale(core.n_crystals)
    sf = MergedStructureFactor.from_merge(core)
    relp = RelpProfile(n_unique=hkl_idx.n_unique)
    geometry = RefinableGeometry(core.geometry)
    wavelength = 1.2398e4 / core.beam_energy.median().item()

    # Use GaussianSpectrum for benchmarking (avoids PCA loading complexity)
    from torchsx.forward import GaussianSpectrum
    mean_energy_eV = core.beam_energy.median().item()
    spectrum = GaussianSpectrum(
        n_crystals=core.n_crystals,
        mean_energy_eV=mean_energy_eV,
    )

    model = PixelForwardModel(
        structure_factors=sf,
        orientation=orientation,
        crystal_scale=crystal_scale,
        crystal_bfactor=crystal_bfactor,
        relp_profile=relp,
        spectrum=spectrum,
        geometry=geometry,
        wavelength=wavelength,
    )
    model = model.to(device)

    # Freeze everything for clean timing (just measure compute cost)
    for p in model.parameters():
        p.requires_grad_(True)

    likelihood = PixelLikelihood(mode="gaussian")

    # Build frame→obs map
    store = FrameStore(STORE_DIR)
    meta = core.metadata
    filenames = meta["image"]["filename"]
    events = meta["image"]["event"]
    image_ids = core.image_id[core.crystal_id]

    frame_obs = {}
    for obs_i in range(core.n_observations):
        img_id = image_ids[obs_i].item()
        key = (filenames[img_id], _parse_event_id(events[img_id]))
        if key in store:
            frame_obs.setdefault(key, []).append(obs_i)

    frame_keys = list(frame_obs.keys())
    frame_obs_tensors = [
        torch.tensor(frame_obs[k], dtype=torch.long) for k in frame_keys
    ]

    print(f"  {len(frame_keys)} frames, "
          f"{sum(t.numel() for t in frame_obs_tensors)} obs")
    print(f"  Avg obs/frame: "
          f"{sum(t.numel() for t in frame_obs_tensors) / len(frame_keys):.1f}")

    return (model, likelihood, core, hkl_idx, store,
            frame_keys, frame_obs_tensors, device)


def time_cpu_prep(store, core, hkl_idx, frame_keys, frame_obs_tensors,
                  n_frames, device, nthreads=8):
    """Time CPU decompress + extract + H2D transfer for n_frames."""
    extractor = ROIExtractor(store, patch_size=PATCH_SIZE, nthreads=nthreads)
    ph = pw = PATCH_SIZE

    keys = frame_keys[:n_frames]
    obs_lists = frame_obs_tensors[:n_frames]

    # Filter empty
    nonempty = [(k, o) for k, o in zip(keys, obs_lists) if o.numel() > 0]
    if not nonempty:
        return 0.0, 0

    keys_ne, obs_ne = zip(*nonempty)
    obs_idx = torch.cat(obs_ne)
    n_obs = obs_idx.numel()

    fs = core.fs[obs_idx].numpy()
    ss = core.ss[obs_idx].numpy()
    fs_int = np.round(fs).astype(np.int64)
    ss_int = np.round(ss).astype(np.int64)

    centers_per_frame = []
    offset = 0
    for o in obs_ne:
        n = o.numel()
        centers_per_frame.append(
            np.column_stack([ss_int[offset:offset + n],
                             fs_int[offset:offset + n]])
        )
        offset += n

    t0 = time.time()
    patches_np, valid_np = extractor.extract_batch(list(keys_ne), centers_per_frame)

    # H2D transfer + ROIBatch construction
    patches_t = torch.from_numpy(patches_np).float().to(device)
    valid_t = torch.from_numpy(valid_np).to(device)
    crystal_id = core.crystal_id[obs_idx].to(device)
    hkl = core.hkl[obs_idx].to(device)
    hkl_id = hkl_idx.hkl_id[obs_idx].to(device)
    panel_id = core.panel_id[obs_idx].to(device)
    fs_center = core.fs[obs_idx].float().to(device)
    ss_center = core.ss[obs_idx].float().to(device)
    bg = estimate_background(patches_t, valid_t)
    fs_grid, ss_grid, pid_grid = _build_pixel_grids(
        fs_center, ss_center, panel_id, ph, pw)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    extractor.close()
    return elapsed, n_obs


def time_gpu_compute(model, likelihood, core, hkl_idx,
                     frame_obs_tensors, n_frames, device):
    """Time GPU forward + backward for n_frames worth of observations."""
    ph = pw = PATCH_SIZE
    obs_lists = frame_obs_tensors[:n_frames]
    nonempty = [o for o in obs_lists if o.numel() > 0]
    if not nonempty:
        return 0.0, 0

    obs_idx = torch.cat(nonempty)
    n_obs = obs_idx.numel()

    # Build a fake batch with random patches (we just want compute time)
    patches_t = torch.randn(n_obs, ph, pw, device=device)
    valid_t = torch.ones(n_obs, ph, pw, dtype=torch.bool, device=device)
    crystal_id = core.crystal_id[obs_idx].to(device)
    hkl = core.hkl[obs_idx].to(device)
    hkl_id = hkl_idx.hkl_id[obs_idx].to(device)
    panel_id = core.panel_id[obs_idx].to(device)
    fs_center = core.fs[obs_idx].float().to(device)
    ss_center = core.ss[obs_idx].float().to(device)
    bg = torch.zeros(n_obs, device=device)
    fs_grid, ss_grid, pid_grid = _build_pixel_grids(
        fs_center, ss_center, panel_id, ph, pw)

    def _make_batch():
        return ROIBatch(
            patches=patches_t, valid=valid_t,
            crystal_id=crystal_id, hkl_id=hkl_id, hkl=hkl,
            panel_id=panel_id, fs_center=fs_center, ss_center=ss_center,
            background=bg, obs_idx=obs_idx.to(device),
            fs_grid=fs_grid.clone(), ss_grid=ss_grid.clone(),
            pid_grid=pid_grid.clone(),
        )

    # Warmup
    for _ in range(2):
        batch = _make_batch()
        n_pred = model(batch)
        loss = likelihood(n_pred, batch.patches, batch.valid)
        loss.backward()
        model.zero_grad()

    # Timed run (average of 3)
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        batch = _make_batch()
        torch.cuda.synchronize()
        t0 = time.time()
        n_pred = model(batch)
        loss = likelihood(n_pred, batch.patches, batch.valid)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        model.zero_grad()

    return np.median(times), n_obs


def main():
    (model, likelihood, core, hkl_idx, store,
     frame_keys, frame_obs_tensors, device) = build_model_and_data()

    print()
    print("=" * 70)
    print(f"{'frames':>6} {'obs':>7} {'CPU(ms)':>8} {'GPU(ms)':>8} "
          f"{'ratio':>7} {'bound':>8}")
    print("-" * 70)

    # Warmup OS cache
    extractor = ROIExtractor(store, patch_size=PATCH_SIZE, nthreads=8)
    for i in range(min(50, len(frame_keys))):
        cid = store.key_to_id(frame_keys[i])
        _ = extractor._extract_one_frame(
            cid,
            np.array([100], dtype=np.int64),
            np.array([100], dtype=np.int64),
        )
    extractor.close()

    for n_frames in [10, 25, 50, 75, 100, 150, 200, 250, 309]:
        if n_frames > len(frame_keys):
            n_frames = len(frame_keys)

        cpu_time, n_obs_cpu = time_cpu_prep(
            store, core, hkl_idx, frame_keys, frame_obs_tensors,
            n_frames, device)

        gpu_time, n_obs_gpu = time_gpu_compute(
            model, likelihood, core, hkl_idx,
            frame_obs_tensors, n_frames, device)

        ratio = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        bound = "CPU" if cpu_time > gpu_time else "GPU"

        print(f"{n_frames:>6} {n_obs_cpu:>7} {cpu_time*1000:>8.1f} "
              f"{gpu_time*1000:>8.1f} {ratio:>7.2f} {bound:>8}")

    print()
    print("ratio > 1 → CPU-bound (data prep slower than GPU compute)")
    print("ratio < 1 → GPU-bound (GPU compute slower than data prep)")


if __name__ == "__main__":
    main()
