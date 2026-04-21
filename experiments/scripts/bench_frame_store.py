"""Benchmark FrameStore conversion and ROI extraction.

Step 1: Convert ~1000 frames from delay_-45 to a FrameStore
Step 2: Benchmark decompression + patch extraction at various batch sizes
"""

import time
import numpy as np
import torch

from torchsx import SFXStream
from torchsx.io._frame_store import FrameStore
from torchsx.io._frame_extract import ROIExtractor
from torchsx.io._excise import _parse_event_id
from torchsx.merge._hkl import HKLIndex, compute_hkl_index
from torchsx.forward.frame_batch import FrameBatchLoader

H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
STORE_DIR = "bench_frame_store"
N_FRAMES_TARGET = 1000
PATCH_SIZE = 11


def step1_convert():
    """Convert first ~1000 unique frames to FrameStore."""
    print("=" * 60)
    print("STEP 1: Convert H5 → FrameStore")
    print("=" * 60)

    s = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
    s = s.remap_filenames(
        r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
        H5_ROOT + r"/\1/",
    )
    core = s._core
    meta = core.metadata
    filenames = meta["image"]["filename"]
    events = meta["image"]["event"]
    n_images = core.beam_energy.shape[0]

    # Collect unique (filename, event) pairs, take first N_FRAMES_TARGET
    pairs = []
    seen = set()
    for i in range(n_images):
        key = (filenames[i], _parse_event_id(events[i]))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
        if len(pairs) >= N_FRAMES_TARGET:
            break
    pairs.sort()

    print(f"Converting {len(pairs)} frames...")
    t0 = time.time()

    # Resolve data path from geometry
    data_path = core.geometry.data_path if core.geometry else "/data/data"
    store = FrameStore.from_h5_files(
        pairs, STORE_DIR, data_path=data_path, n_threads=8,
    )

    elapsed = time.time() - t0
    print(f"Conversion: {elapsed:.1f}s ({elapsed / len(pairs):.2f}s/frame)")
    print(f"Frames: {len(store)}, shape: {store.frame_shape}, dtype: {store.dtype}")

    import os
    blosc_size = os.path.getsize(os.path.join(STORE_DIR, "frames.blosc2"))
    raw_size = len(store) * np.prod(store.frame_shape) * store.dtype.itemsize
    print(f"Compressed: {blosc_size / 1e9:.2f} GB")
    print(f"Uncompressed: {raw_size / 1e9:.2f} GB")
    print(f"Ratio: {raw_size / blosc_size:.1f}x")

    return store, s


def step2_decompress_benchmark(store):
    """Benchmark raw SChunk decompression at various nthreads."""
    import blosc2

    print()
    print("=" * 60)
    print("STEP 2: SChunk decompression benchmark")
    print("=" * 60)

    n = min(100, len(store))

    for nthreads in [1, 4, 8, 16]:
        schunk = blosc2.open(str(store._schunk_path), mode="r")
        schunk.update_dparams(blosc2.DParams(nthreads=nthreads))

        # Warmup (ensure OS-cached)
        for i in range(min(5, n)):
            _ = schunk.decompress_chunk(i)

        t0 = time.time()
        for i in range(n):
            _ = schunk.decompress_chunk(i)
        elapsed = time.time() - t0
        thr = n * store.frame_shape[0] * store.frame_shape[1] * 2 / elapsed / 1e9
        print(f"  {nthreads:2d}t, {n} frames: {elapsed:.2f}s "
              f"({elapsed / n * 1000:.1f}ms/frame, {thr:.1f} GB/s)")


def step3_extraction_benchmark(store, stream):
    """Benchmark ROIExtractor with realistic observation data."""
    print()
    print("=" * 60)
    print("STEP 3: ROI extraction benchmark (decompress + extract)")
    print("=" * 60)

    core = stream._core
    meta = core.metadata
    filenames = meta["image"]["filename"]
    events = meta["image"]["event"]

    # Build frame→obs map for frames in the store
    image_ids = core.image_id[core.crystal_id]
    frame_obs = {}
    for obs_i in range(core.n_observations):
        img_id = image_ids[obs_i].item()
        key = (filenames[img_id], _parse_event_id(events[img_id]))
        if key in store:
            frame_obs.setdefault(key, []).append(obs_i)

    frame_keys = list(frame_obs.keys())
    print(f"Frames in store with observations: {len(frame_keys)}")
    total_obs = sum(len(v) for v in frame_obs.values())
    print(f"Total observations: {total_obs}")
    print(f"Avg obs/frame: {total_obs / max(len(frame_keys), 1):.1f}")

    for nt in [4, 8, 16]:
        extractor = ROIExtractor(store, patch_size=PATCH_SIZE, nthreads=nt)

        for n_frames in [100, 309]:
            if n_frames > len(frame_keys):
                n_frames = len(frame_keys)

            keys = frame_keys[:n_frames]
            centers_list = []
            for k in keys:
                obs_idx = frame_obs[k]
                ss = np.round(core.ss[obs_idx].numpy()).astype(np.int64)
                fs = np.round(core.fs[obs_idx].numpy()).astype(np.int64)
                centers_list.append(np.column_stack([ss, fs]))

            n_obs = sum(c.shape[0] for c in centers_list)

            t0 = time.time()
            patches, valid = extractor.extract_batch(keys, centers_list)
            elapsed = time.time() - t0

            print(
                f"  {nt}t, {n_frames} frames, {n_obs} obs: "
                f"{elapsed:.2f}s ({elapsed / n_frames * 1000:.1f}ms/frame)"
            )
            print(f"    patches: {patches.shape}, {patches.nbytes / 1e6:.1f} MB")

        extractor.close()


def step4_batch_loader_benchmark(store, stream):
    """Benchmark FrameBatchLoader end-to-end including GPU transfer."""
    print()
    print("=" * 60)
    print("STEP 4: FrameBatchLoader end-to-end benchmark")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    core = stream._core
    hkl_idx = compute_hkl_index(core.hkl)

    loader = FrameBatchLoader(
        frame_store=store,
        core=core,
        hkl_index=hkl_idx,
        patch_size=PATCH_SIZE,
        frames_per_batch=500,
        device=device,
        nthreads=8,
        prefetch=2,
    )

    print(f"Batches per epoch: {len(loader)}")
    print(f"Total observations: {loader.n_observations}")

    # Warm-up: 1 batch
    for batch in loader:
        print(
            f"  Batch: {batch.patches.shape[0]} obs, "
            f"patches {batch.patches.shape}, device={batch.patches.device}"
        )
        break

    # Time 3 epochs (first may be cold, subsequent warm)
    for epoch in range(3):
        t0 = time.time()
        n_batches = 0
        n_obs_total = 0
        for batch in loader:
            n_batches += 1
            n_obs_total += batch.patches.shape[0]
        elapsed = time.time() - t0
        print(f"Epoch {epoch}: {n_batches} batches, {n_obs_total} obs, "
              f"{elapsed:.2f}s ({elapsed / max(n_batches, 1):.2f}s/batch)")

    loader.close()


def main():
    import os
    if os.path.exists(os.path.join(STORE_DIR, "frames.blosc2")):
        print("FrameStore already exists, skipping conversion")
        store = FrameStore(STORE_DIR)
        s = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
        s = s.remap_filenames(
            r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
            H5_ROOT + r"/\1/",
        )
        stream = s
        print(f"Frames: {len(store)}, shape: {store.frame_shape}, dtype: {store.dtype}")
    else:
        store, stream = step1_convert()
    step2_decompress_benchmark(store)
    step3_extraction_benchmark(store, stream)
    step4_batch_loader_benchmark(store, stream)


if __name__ == "__main__":
    main()
