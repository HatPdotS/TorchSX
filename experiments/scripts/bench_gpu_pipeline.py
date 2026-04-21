"""Benchmark all-GPU pipeline: H2D compressed → batch decompress → extract → forward.

The idea: send compressed frames (~10 MB each) over PCIe, decompress on GPU
with nvCOMP batch API, extract patches with torch indexing, run forward model.
No CPU decompression at all.
"""

import time
import numpy as np
import torch


def load_sample_frames(n_frames=200):
    from torchsx import SFXStream
    from torchsx.io._excise import _parse_event_id
    import h5py
    try:
        import bitshuffle.h5  # noqa: F401
    except ImportError:
        pass

    H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
    s = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
    s = s.remap_filenames(
        r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
        H5_ROOT + r"/\1/",
    )
    core = s._core
    meta = core.metadata
    filenames = meta["image"]["filename"]
    events = meta["image"]["event"]
    data_path = core.geometry.data_path

    seen = set()
    pairs = []
    for i in range(len(filenames)):
        key = (filenames[i], _parse_event_id(events[i]))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
        if len(pairs) >= n_frames:
            break

    frames = []
    current_file = None
    current_h5 = None
    for fn, ev in sorted(pairs):
        if fn != current_file:
            if current_h5 is not None:
                current_h5.close()
            current_h5 = h5py.File(fn, "r")
            current_file = fn
        frame = np.asarray(current_h5[data_path][ev], dtype=np.int16)
        frames.append(frame)
    if current_h5 is not None:
        current_h5.close()

    print(f"Loaded {len(frames)} frames, shape {frames[0].shape}")
    return frames


# ──────────────────────────────────────────────────────────────────
# 1. nvCOMP batch encode/decode benchmark
# ──────────────────────────────────────────────────────────────────
def bench_nvcomp_batch(frames):
    import nvidia.nvcomp as nvcomp

    print()
    print("=" * 65)
    print("nvCOMP LZ4 batch encode / decode")
    print("=" * 65)

    H, W = frames[0].shape
    n = len(frames)

    codec = nvcomp.Codec(algorithm="lz4", uncomp_chunk_size=65536)

    # --- Batch encode ---
    arrays = [nvcomp.as_array(f) for f in frames]
    # Warmup
    _ = codec.encode(arrays[:5])

    t0 = time.time()
    compressed = codec.encode(arrays)
    torch.cuda.synchronize()
    enc_time = time.time() - t0
    print(f"Batch encode {n} frames: {enc_time:.3f}s "
          f"({enc_time / n * 1000:.2f}ms/frame)")

    # --- Serial decode baseline ---
    # Warmup
    for _ in range(3):
        _ = codec.decode(compressed[0])

    torch.cuda.synchronize()
    t0 = time.time()
    for c in compressed:
        _ = codec.decode(c)
    torch.cuda.synchronize()
    serial_time = time.time() - t0
    print(f"Serial decode {n} frames: {serial_time:.3f}s "
          f"({serial_time / n * 1000:.2f}ms/frame)")

    # --- Batch decode ---
    # Warmup
    _ = codec.decode(compressed[:5])

    torch.cuda.synchronize()
    t0 = time.time()
    decoded = codec.decode(compressed)
    torch.cuda.synchronize()
    batch_time = time.time() - t0
    thr = n * H * W * 2 / batch_time / 1e9
    print(f"Batch decode {n} frames: {batch_time:.3f}s "
          f"({batch_time / n * 1000:.2f}ms/frame, {thr:.1f} GB/s)")
    print(f"  Speedup vs serial: {serial_time / batch_time:.1f}x")

    # Verify roundtrip
    dec0 = torch.from_dlpack(decoded[0]).view(torch.int16).reshape(H, W)
    ref = torch.from_numpy(frames[0])
    assert torch.equal(dec0.cpu(), ref), "Roundtrip failed!"
    print("  Roundtrip verified OK")

    return compressed, codec


# ──────────────────────────────────────────────────────────────────
# 2. Full GPU pipeline: batch decode → extract patches
# ──────────────────────────────────────────────────────────────────
def bench_gpu_pipeline(frames, compressed, codec):
    print()
    print("=" * 65)
    print("Full GPU pipeline: batch decode → extract patches")
    print("=" * 65)

    device = torch.device("cuda")
    H, W = frames[0].shape
    n = len(frames)
    patch_size = 11
    hh = patch_size // 2
    n_patches_per_frame = 240

    # Pre-generate random centers for each frame
    all_centers_ss = torch.randint(hh, H - hh, (n, n_patches_per_frame),
                                   device=device)
    all_centers_fs = torch.randint(hh, W - hh, (n, n_patches_per_frame),
                                   device=device)
    offsets = torch.arange(-hh, hh + 1, device=device)

    def extract_patches_from_decoded(decoded_list, n_frames):
        """Extract patches from list of decoded GPU arrays."""
        all_patches = []
        for i in range(n_frames):
            frame_t = torch.from_dlpack(decoded_list[i]).view(
                torch.int16).reshape(H, W)
            ss = all_centers_ss[i]
            fs = all_centers_fs[i]
            # Build index grids
            ss_grid = ss[:, None, None] + offsets[None, :, None]
            fs_grid = fs[:, None, None] + offsets[None, None, :]
            ss_grid = ss_grid.expand(-1, patch_size, patch_size)
            fs_grid = fs_grid.expand(-1, patch_size, patch_size)
            patches = frame_t[ss_grid, fs_grid].float()
            all_patches.append(patches)
        return torch.cat(all_patches)

    # Warmup
    dec = codec.decode(compressed[:10])
    _ = extract_patches_from_decoded(dec, 10)

    # Benchmark: batch decode + extract
    for n_frames in [10, 25, 50, 100, 200]:
        if n_frames > n:
            break
        comp_batch = compressed[:n_frames]
        n_obs = n_frames * n_patches_per_frame

        torch.cuda.synchronize()
        t0 = time.time()
        decoded = codec.decode(comp_batch)
        patches = extract_patches_from_decoded(decoded, n_frames)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        print(f"  {n_frames:>4} frames, {n_obs:>6} obs: "
              f"{elapsed * 1000:>7.1f}ms "
              f"({elapsed / n_frames * 1000:.2f}ms/frame)")

    return extract_patches_from_decoded


# ──────────────────────────────────────────────────────────────────
# 3. H2D transfer of compressed data (simulating read from RAM)
# ──────────────────────────────────────────────────────────────────
def bench_h2d_compressed(frames, compressed, codec):
    import nvidia.nvcomp as nvcomp

    print()
    print("=" * 65)
    print("H2D compressed transfer → batch decode → extract")
    print("=" * 65)

    device = torch.device("cuda")
    H, W = frames[0].shape
    n = len(frames)
    patch_size = 11
    hh = patch_size // 2
    n_patches_per_frame = 240
    offsets = torch.arange(-hh, hh + 1, device=device)

    all_centers_ss = torch.randint(hh, H - hh, (n, n_patches_per_frame),
                                   device=device)
    all_centers_fs = torch.randint(hh, W - hh, (n, n_patches_per_frame),
                                   device=device)

    # Get compressed data as CPU numpy arrays (simulating "from RAM")
    cpu_compressed = []
    for c in compressed:
        arr = np.array(torch.from_dlpack(c).cpu())
        cpu_compressed.append(arr)

    total_comp_bytes = sum(a.nbytes for a in cpu_compressed)
    print(f"Total compressed: {total_comp_bytes / 1e6:.1f} MB "
          f"({total_comp_bytes / n / 1e6:.2f} MB/frame)")

    # Benchmark: CPU→GPU transfer of compressed + batch decode + extract
    for n_frames in [50, 100, 200]:
        if n_frames > n:
            break

        batch_cpu = cpu_compressed[:n_frames]
        n_obs = n_frames * n_patches_per_frame

        torch.cuda.synchronize()
        t0 = time.time()

        # H2D transfer: numpy → nvcomp array (triggers H2D)
        gpu_arrays = [nvcomp.as_array(a) for a in batch_cpu]
        t_h2d = time.time() - t0

        # Batch decode
        t1 = time.time()
        decoded = codec.decode(gpu_arrays)
        torch.cuda.synchronize()
        t_dec = time.time() - t1

        # Extract patches
        t2 = time.time()
        all_patches = []
        for i in range(n_frames):
            frame_t = torch.from_dlpack(decoded[i]).view(
                torch.int16).reshape(H, W)
            ss = all_centers_ss[i]
            fs = all_centers_fs[i]
            ss_grid = ss[:, None, None] + offsets[None, :, None]
            fs_grid = fs[:, None, None] + offsets[None, None, :]
            ss_grid = ss_grid.expand(-1, patch_size, patch_size)
            fs_grid = fs_grid.expand(-1, patch_size, patch_size)
            patches = frame_t[ss_grid, fs_grid].float()
            all_patches.append(patches)
        result = torch.cat(all_patches)
        torch.cuda.synchronize()
        t_ext = time.time() - t2

        total = time.time() - t0
        comp_bytes = sum(a.nbytes for a in batch_cpu)

        print(f"  {n_frames} frames, {n_obs} obs:")
        print(f"    H2D {comp_bytes / 1e6:.0f} MB: {t_h2d * 1000:.1f}ms "
              f"({comp_bytes / t_h2d / 1e9:.1f} GB/s)")
        print(f"    Batch decode:    {t_dec * 1000:.1f}ms")
        print(f"    Extract patches: {t_ext * 1000:.1f}ms")
        print(f"    Total:           {total * 1000:.1f}ms "
              f"({total / n_frames * 1000:.2f}ms/frame)")


# ──────────────────────────────────────────────────────────────────
# 4. Comparison summary
# ──────────────────────────────────────────────────────────────────
def bench_cpu_baseline(frames):
    """CPU blosc2 decompress + extract for comparison."""
    import blosc2
    from torchsx.io._excise import _extract_frame_patches

    print()
    print("=" * 65)
    print("CPU baseline: blosc2 8t decompress + extract")
    print("=" * 65)

    H, W = frames[0].shape
    patch_size = 11
    hh = patch_size // 2
    n_patches = 240
    centers_ss = np.random.randint(hh, H - hh, n_patches).astype(np.int64)
    centers_fs = np.random.randint(hh, W - hh, n_patches).astype(np.int64)

    # Compress
    compressed = []
    for f in frames:
        compressed.append(blosc2.compress2(f,
            codec=blosc2.Codec.LZ4, clevel=5,
            filters=[blosc2.Filter.BITSHUFFLE],
            typesize=2, nthreads=1))

    blosc2.set_nthreads(8)
    device = torch.device("cuda")

    for n_frames in [50, 100, 200]:
        if n_frames > len(frames):
            break

        t0 = time.time()
        all_patches = []
        for b in compressed[:n_frames]:
            raw = blosc2.decompress2(b)
            frame = np.frombuffer(raw, dtype=np.int16).reshape(H, W)
            p, v = _extract_frame_patches(frame, centers_ss, centers_fs,
                                          hh, hh, 0.0)
            all_patches.append(p)

        all_p = np.concatenate(all_patches)
        patches_t = torch.from_numpy(all_p).float().to(device)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        print(f"  {n_frames} frames, {n_frames * n_patches} obs: "
              f"{elapsed * 1000:.1f}ms ({elapsed / n_frames * 1000:.2f}ms/frame)")

    blosc2.set_nthreads(1)


def main():
    frames = load_sample_frames(n_frames=200)

    bench_cpu_baseline(frames)

    try:
        compressed, codec = bench_nvcomp_batch(frames)
        bench_gpu_pipeline(frames, compressed, codec)
        bench_h2d_compressed(frames, compressed, codec)
    except Exception as e:
        print(f"\nnvCOMP pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print("""
At 500 frames/batch (~120K obs), projected times:

  CPU path (blosc2 8t serial):
    decompress + extract:  500 × ~5ms     = ~2500ms
    H2D patches (60 MB):                  =    ~5ms
    Total:                                  ~2500ms

  GPU path (nvcomp batch):
    H2D compressed (5 GB):                =  ~400ms  (PCIe 3.0)
    Batch decode on GPU:                  =   ~Xms   (from benchmark)
    Extract patches (torch):              =   ~20ms
    Total:                                  ~420+Xms

  GPU forward + backward:                =   ~54ms   (from crossover bench)
""")


if __name__ == "__main__":
    main()
