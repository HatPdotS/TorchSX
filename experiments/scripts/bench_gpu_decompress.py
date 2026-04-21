"""Benchmark GPU decompression via nvidia.nvcomp vs CPU blosc2.

Tests:
1. blosc2 CPU decode (1 thread, 8 threads, 8 workers × 1 thread)
2. CPU decompress → GPU transfer
3. GPU patch extraction
4. nvCOMP LZ4 GPU decode → zero-copy torch → GPU patch extract
"""

import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor


# ──────────────────────────────────────────────────────────────────────
# Step 0: Read real frames from H5
# ──────────────────────────────────────────────────────────────────────
def load_sample_frames(n_frames=100):
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


# ──────────────────────────────────────────────────────────────────────
# Step 1: blosc2 CPU decode benchmarks
# ──────────────────────────────────────────────────────────────────────
def bench_blosc2_cpu(frames):
    import blosc2

    print()
    print("=" * 60)
    print("blosc2 LZ4+bitshuffle CPU compress/decompress")
    print("=" * 60)

    # Compress with nthreads=1
    compressed = []
    t0 = time.time()
    for frame in frames:
        comp = blosc2.compress2(frame,
            codec=blosc2.Codec.LZ4, clevel=5,
            filters=[blosc2.Filter.BITSHUFFLE],
            typesize=2, nthreads=1)
        compressed.append(comp)
    encode_time = time.time() - t0

    comp_bytes = sum(len(c) for c in compressed)
    raw_bytes = frames[0].nbytes * len(frames)
    print(f"CPU encode (1t) {len(frames)} frames: {encode_time:.2f}s "
          f"({encode_time / len(frames) * 1000:.1f}ms/frame)")
    print(f"Ratio: {raw_bytes / comp_bytes:.1f}x")

    # Decode: 1 thread serial
    blosc2.set_nthreads(1)
    t0 = time.time()
    for comp in compressed:
        _ = blosc2.decompress2(comp)
    elapsed = time.time() - t0
    print(f"CPU decode 1t serial: {elapsed:.2f}s ({elapsed / len(frames) * 1000:.1f}ms/frame)")

    # Decode: 8 threads (blosc2 internal)
    blosc2.set_nthreads(8)
    t0 = time.time()
    for comp in compressed:
        _ = blosc2.decompress2(comp)
    elapsed = time.time() - t0
    print(f"CPU decode 8t serial: {elapsed:.2f}s ({elapsed / len(frames) * 1000:.1f}ms/frame)")
    blosc2.set_nthreads(1)

    # Decode: 8 workers × 1 thread (ThreadPoolExecutor)
    shape = frames[0].shape
    dtype = frames[0].dtype

    def _decode_one(comp):
        buf = blosc2.decompress2(comp)
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    # Warmup
    for c in compressed[:3]:
        _decode_one(c)

    for n_workers in [4, 8, 16]:
        pool = ThreadPoolExecutor(max_workers=n_workers)
        t0 = time.time()
        futures = [pool.submit(_decode_one, c) for c in compressed]
        results = [f.result() for f in futures]
        elapsed = time.time() - t0
        print(f"CPU decode {n_workers}w×1t parallel: {elapsed:.2f}s "
              f"({elapsed / len(frames) * 1000:.1f}ms/frame eff)")
        pool.shutdown()

    return compressed


# ──────────────────────────────────────────────────────────────────────
# Step 2: CPU decompress → GPU transfer
# ──────────────────────────────────────────────────────────────────────
def bench_cpu_to_gpu(frames, compressed):
    import blosc2

    print()
    print("=" * 60)
    print("CPU decompress → GPU transfer")
    print("=" * 60)

    device = torch.device("cuda")
    shape = frames[0].shape
    dtype = frames[0].dtype
    blosc2.set_nthreads(1)

    # Serial: decompress + H2D (unpinned)
    torch.cuda.synchronize()
    t0 = time.time()
    for comp in compressed:
        buf = blosc2.decompress2(comp)
        arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
        t = torch.from_numpy(arr.copy()).to(device)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Serial decompress + H2D (copy): {elapsed:.2f}s ({elapsed / len(frames) * 1000:.1f}ms/frame)")

    # H2D only (pinned)
    pinned = torch.empty(shape, dtype=torch.int16, pin_memory=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for frame in frames:
        pinned[:] = torch.from_numpy(frame)
        t = pinned.to(device, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"H2D pinned only: {elapsed:.2f}s ({elapsed / len(frames) * 1000:.1f}ms/frame)")

    # Best CPU path: multi-worker decompress + patch extract on CPU,
    # then transfer only patches
    print("\n--- Realistic pipeline: CPU decompress+extract, H2D patches only ---")
    from torchsx.io._excise import _extract_frame_patches
    patch_size = 11
    hh = patch_size // 2
    n_patches_per = 240
    H, W = shape

    # Random centers (same for all frames for simplicity)
    centers_ss = np.random.randint(hh, H - hh, n_patches_per).astype(np.int64)
    centers_fs = np.random.randint(hh, W - hh, n_patches_per).astype(np.int64)

    def _decode_and_extract(comp):
        buf = blosc2.decompress2(comp)
        frame = np.frombuffer(buf, dtype=dtype).reshape(shape)
        p, v = _extract_frame_patches(frame, centers_ss, centers_fs, hh, hh, 0.0)
        return p, v

    # Single-threaded
    t0 = time.time()
    patches_list = []
    for comp in compressed:
        p, v = _decode_and_extract(comp)
        patches_list.append(p)
    elapsed = time.time() - t0
    print(f"1w decompress+extract: {elapsed:.2f}s ({elapsed / len(frames) * 1000:.1f}ms/frame)")

    # Multi-worker
    for n_workers in [4, 8]:
        pool = ThreadPoolExecutor(max_workers=n_workers)
        t0 = time.time()
        futures = [pool.submit(_decode_and_extract, c) for c in compressed]
        results = [f.result() for f in futures]
        elapsed_cpu = time.time() - t0

        # Now transfer just patches to GPU
        all_p = np.concatenate([r[0] for r in results])
        torch.cuda.synchronize()
        t0 = time.time()
        patches_t = torch.from_numpy(all_p).float().to(device)
        torch.cuda.synchronize()
        elapsed_h2d = time.time() - t0

        print(f"{n_workers}w decompress+extract: {elapsed_cpu:.2f}s "
              f"({elapsed_cpu / len(frames) * 1000:.1f}ms/frame eff) "
              f"+ H2D {elapsed_h2d * 1000:.1f}ms "
              f"({all_p.nbytes / 1e6:.1f} MB patches)")
        pool.shutdown()


# ──────────────────────────────────────────────────────────────────────
# Step 3: GPU patch extraction from full frame already on GPU
# ──────────────────────────────────────────────────────────────────────
def bench_gpu_patch_extraction(frames):
    print()
    print("=" * 60)
    print("GPU patch extraction (frame already on GPU)")
    print("=" * 60)

    device = torch.device("cuda")
    frame_t = torch.from_numpy(frames[0]).to(device)
    H, W = frame_t.shape
    patch_size = 11
    hh = patch_size // 2
    n_patches = 240

    centers_ss = torch.randint(hh, H - hh, (n_patches,), device=device)
    centers_fs = torch.randint(hh, W - hh, (n_patches,), device=device)
    offsets = torch.arange(-hh, hh + 1, device=device)
    ss_grid = centers_ss[:, None, None] + offsets[None, :, None]
    fs_grid = centers_fs[:, None, None] + offsets[None, None, :]
    ss_grid = ss_grid.expand(-1, patch_size, patch_size)
    fs_grid = fs_grid.expand(-1, patch_size, patch_size)

    for _ in range(10):
        patches = frame_t[ss_grid, fs_grid].float()

    torch.cuda.synchronize()
    n_iters = 100
    t0 = time.time()
    for _ in range(n_iters):
        patches = frame_t[ss_grid, fs_grid].float()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"GPU extract {n_patches} patches × {n_iters}: "
          f"{elapsed * 1000:.1f}ms total, {elapsed / n_iters * 1000:.3f}ms/frame")

    # Simulate 500 frames
    frame_tensors = [torch.from_numpy(f).to(device) for f in frames[:20]]
    torch.cuda.synchronize()
    t0 = time.time()
    all_patches = []
    for rep in range(25):
        for ft in frame_tensors:
            p = ft[ss_grid, fs_grid].float()
            all_patches.append(p)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    result = torch.cat(all_patches)
    print(f"GPU extract 500 frames × {n_patches} patches: {elapsed * 1000:.1f}ms "
          f"({elapsed / 500 * 1000:.3f}ms/frame)")
    print(f"  Total: {result.shape}, {result.nbytes / 1e6:.1f} MB")


# ──────────────────────────────────────────────────────────────────────
# Step 4: nvCOMP LZ4 GPU decode
# ──────────────────────────────────────────────────────────────────────
def bench_nvcomp(frames):
    import nvidia.nvcomp as nvcomp

    print()
    print("=" * 60)
    print("nvCOMP LZ4 GPU decode")
    print("=" * 60)

    device = torch.device("cuda")
    codec = nvcomp.Codec(algorithm="lz4", uncomp_chunk_size=65536)

    # Compress
    compressed = []
    t0 = time.time()
    for frame in frames:
        arr = nvcomp.as_array(frame)
        compressed.append(codec.encode(arr))
    elapsed = time.time() - t0
    print(f"GPU encode {len(frames)} frames: {elapsed:.2f}s "
          f"({elapsed / len(frames) * 1000:.1f}ms/frame)")

    # Warmup decode
    for _ in range(3):
        _ = codec.decode(compressed[0])

    # Benchmark decode only
    torch.cuda.synchronize()
    n_iters = 3
    t0 = time.time()
    for _ in range(n_iters):
        for comp in compressed:
            _ = codec.decode(comp)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_iters
    print(f"GPU decode {len(frames)} frames: {elapsed:.2f}s "
          f"({elapsed / len(frames) * 1000:.2f}ms/frame)")

    # Check zero-copy to torch
    print("\nZero-copy decode → torch:")
    dec = codec.decode(compressed[0])
    H, W = frames[0].shape

    # Try DLPack
    try:
        frame_t = torch.from_dlpack(dec)
        print(f"  from_dlpack: device={frame_t.device}, shape={frame_t.shape}, "
              f"dtype={frame_t.dtype}")
        # Reshape and extract patches
        if frame_t.numel() == H * W * 2:  # bytes
            frame_t = frame_t.view(torch.int16).reshape(H, W)
        elif frame_t.numel() == H * W:
            frame_t = frame_t.reshape(H, W)
        print(f"  Reshaped: {frame_t.shape}, device={frame_t.device}")
        dlpack_ok = True
    except Exception as e:
        print(f"  from_dlpack failed: {e}")
        dlpack_ok = False

    # Try __cuda_array_interface__
    if not dlpack_ok:
        try:
            iface = dec.__cuda_array_interface__
            print(f"  __cuda_array_interface__: shape={iface['shape']}, "
                  f"typestr={iface['typestr']}")
            frame_t = torch.as_tensor(dec, device=device)
            print(f"  torch.as_tensor: {frame_t.shape}, {frame_t.dtype}")
            dlpack_ok = True
        except (AttributeError, Exception) as e:
            print(f"  __cuda_array_interface__ failed: {e}")

    if not dlpack_ok:
        print("  No zero-copy path available — would need CPU roundtrip")
        return compressed, codec, False

    # Benchmark full pipeline: decode → zero-copy torch → extract patches
    print("\n--- Full GPU pipeline: nvcomp decode → torch → extract ---")
    patch_size = 11
    hh = patch_size // 2
    n_patches = 240
    centers_ss = torch.randint(hh, H - hh, (n_patches,), device=device)
    centers_fs = torch.randint(hh, W - hh, (n_patches,), device=device)
    offsets = torch.arange(-hh, hh + 1, device=device)
    ss_grid = centers_ss[:, None, None] + offsets[None, :, None]
    fs_grid = centers_fs[:, None, None] + offsets[None, None, :]
    ss_grid = ss_grid.expand(-1, patch_size, patch_size)
    fs_grid = fs_grid.expand(-1, patch_size, patch_size)

    # Warmup
    for _ in range(3):
        dec = codec.decode(compressed[0])
        ft = torch.from_dlpack(dec)
        if ft.numel() == H * W * 2:
            ft = ft.view(torch.int16).reshape(H, W)
        elif ft.numel() == H * W:
            ft = ft.reshape(H, W)
        _ = ft[ss_grid, fs_grid].float()

    torch.cuda.synchronize()
    t0 = time.time()
    all_patches = []
    for comp in compressed:
        dec = codec.decode(comp)
        ft = torch.from_dlpack(dec)
        if ft.numel() == H * W * 2:
            ft = ft.view(torch.int16).reshape(H, W)
        elif ft.numel() == H * W:
            ft = ft.reshape(H, W)
        p = ft[ss_grid, fs_grid].float()
        all_patches.append(p)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    result = torch.cat(all_patches)
    print(f"nvcomp decode + extract {len(frames)} frames: {elapsed:.2f}s "
          f"({elapsed / len(frames) * 1000:.2f}ms/frame)")
    print(f"Total patches: {result.shape}, already on {result.device}")

    return compressed, codec, True


# ──────────────────────────────────────────────────────────────────────
# Summary comparison for 500-frame batch
# ──────────────────────────────────────────────────────────────────────
def print_summary():
    print()
    print("=" * 60)
    print("SUMMARY: Projected times for 500-frame batch (~120K obs)")
    print("=" * 60)
    print("""
Pipeline A (current CPU path, 8 workers):
  CPU decompress (8w×1t): 500 × 2.6ms / 8  = ~160ms
  CPU extract patches:    500 × 0.5ms / 8   =  ~30ms
  H2D patches only:       120K × 11×11 × 4  =  ~1ms
  Total:                                      ~190ms

Pipeline B (GPU decompress + GPU extract):
  H2D compressed chunks:  500 × 3.5MB / 12GB/s = ~145ms
  GPU decompress:          500 × 4.7ms         = ~2350ms (serial)
  GPU extract patches:     500 × 0.04ms        =   ~20ms
  Total:                                        ~2500ms

Verdict: CPU decompress + CPU extract + H2D patches is 13× faster.
blosc2 bitshuffle gives CPU a huge advantage over raw LZ4.
""")


def main():
    frames = load_sample_frames(n_frames=100)

    compressed = bench_blosc2_cpu(frames)
    bench_cpu_to_gpu(frames, compressed)
    bench_gpu_patch_extraction(frames)

    try:
        bench_nvcomp(frames)
    except Exception as e:
        print(f"\nnvCOMP benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    print_summary()


if __name__ == "__main__":
    main()
