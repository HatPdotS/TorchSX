"""Benchmark decompression when compressed data is already in RAM.

Eliminates I/O from the picture — pure CPU/GPU decompress throughput.
Tests blosc2 (with/without bitshuffle, various thread configs)
and nvCOMP GPU decode from pre-loaded compressed buffers.
"""

import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


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

    print(f"Loaded {len(frames)} frames, shape {frames[0].shape}, "
          f"{frames[0].nbytes / 1e6:.1f} MB/frame")
    return frames


def pre_compress(frames):
    """Pre-compress all frames into in-memory buffers."""
    import blosc2

    result = {}

    # blosc2 LZ4+bitshuffle
    bufs = []
    for frame in frames:
        bufs.append(blosc2.compress2(frame,
            codec=blosc2.Codec.LZ4, clevel=5,
            filters=[blosc2.Filter.BITSHUFFLE],
            typesize=2, nthreads=1))
    result["blosc2_bs"] = bufs
    comp_bytes = sum(len(b) for b in bufs)
    raw_bytes = frames[0].nbytes * len(frames)
    print(f"blosc2 LZ4+bitshuffle: {comp_bytes / 1e6:.0f} MB "
          f"({raw_bytes / comp_bytes:.1f}x)")

    # blosc2 LZ4 only (no bitshuffle)
    bufs = []
    for frame in frames:
        bufs.append(blosc2.compress2(frame,
            codec=blosc2.Codec.LZ4, clevel=5,
            filters=[blosc2.Filter.NOFILTER],
            typesize=2, nthreads=1))
    result["blosc2_nobs"] = bufs
    comp_bytes = sum(len(b) for b in bufs)
    print(f"blosc2 LZ4 (no bitshuffle): {comp_bytes / 1e6:.0f} MB "
          f"({raw_bytes / comp_bytes:.1f}x)")

    # blosc2 ZSTD+bitshuffle (better ratio?)
    bufs = []
    for frame in frames:
        bufs.append(blosc2.compress2(frame,
            codec=blosc2.Codec.ZSTD, clevel=3,
            filters=[blosc2.Filter.BITSHUFFLE],
            typesize=2, nthreads=1))
    result["blosc2_zstd"] = bufs
    comp_bytes = sum(len(b) for b in bufs)
    print(f"blosc2 ZSTD+bitshuffle: {comp_bytes / 1e6:.0f} MB "
          f"({raw_bytes / comp_bytes:.1f}x)")

    return result


def bench_blosc2_from_ram(compressed_dict, shape, n_frames):
    """Benchmark blosc2 decompress from RAM buffers."""
    import blosc2

    print()
    print("=" * 60)
    print("blosc2 decompress from RAM (no I/O)")
    print("=" * 60)

    for label, bufs in compressed_dict.items():
        data = bufs[:n_frames]
        print(f"\n--- {label} ---")

        # Serial, varying nthreads
        for nt in [1, 2, 4, 8, 16]:
            blosc2.set_nthreads(nt)
            # Warmup
            for b in data[:3]:
                _ = blosc2.decompress2(b)
            t0 = time.time()
            for b in data:
                _ = blosc2.decompress2(b)
            elapsed = time.time() - t0
            thr = n_frames * shape[0] * shape[1] * 2 / elapsed / 1e9
            print(f"  serial {nt:2d}t: {elapsed:.3f}s "
                  f"({elapsed / n_frames * 1000:.2f}ms/frame, "
                  f"{thr:.1f} GB/s uncomp)")

        blosc2.set_nthreads(1)

        # ThreadPoolExecutor (GIL test)
        for nw in [4, 8]:
            def _dec(b):
                return blosc2.decompress2(b)

            pool = ThreadPoolExecutor(max_workers=nw)
            # Warmup
            list(pool.map(_dec, data[:3]))
            t0 = time.time()
            list(pool.map(_dec, data))
            elapsed = time.time() - t0
            thr = n_frames * shape[0] * shape[1] * 2 / elapsed / 1e9
            print(f"  ThreadPool {nw}w×1t: {elapsed:.3f}s "
                  f"({elapsed / n_frames * 1000:.2f}ms/frame eff, "
                  f"{thr:.1f} GB/s)")
            pool.shutdown()


def _mp_decode_worker(args):
    """Multiprocessing worker for blosc2 decode."""
    import blosc2
    blosc2.set_nthreads(1)
    buf, shape = args
    raw = blosc2.decompress2(buf)
    # Return just the checksum to avoid IPC of 34 MB frames
    arr = np.frombuffer(raw, dtype=np.int16)
    return arr.sum()  # verification only


def bench_multiprocess(compressed_dict, shape, n_frames):
    """Benchmark multiprocessing decode (bypasses GIL)."""
    import blosc2

    print()
    print("=" * 60)
    print("Multiprocessing decompress (bypasses GIL)")
    print("=" * 60)

    bufs = compressed_dict["blosc2_bs"][:n_frames]
    args = [(b, shape) for b in bufs]

    for nw in [4, 8, 16]:
        ctx = mp.get_context("forkserver")
        t0 = time.time()
        with ctx.Pool(nw) as pool:
            results = pool.map(_mp_decode_worker, args)
        elapsed = time.time() - t0
        thr = n_frames * shape[0] * shape[1] * 2 / elapsed / 1e9
        print(f"  forkserver {nw}w: {elapsed:.3f}s "
              f"({elapsed / n_frames * 1000:.2f}ms/frame eff, "
              f"{thr:.1f} GB/s)")


def bench_nvcomp_from_ram(frames, n_frames):
    """Benchmark nvCOMP GPU decode from RAM buffers."""
    import nvidia.nvcomp as nvcomp

    print()
    print("=" * 60)
    print("nvCOMP LZ4 GPU decode from RAM")
    print("=" * 60)

    data = frames[:n_frames]
    codec = nvcomp.Codec(algorithm="lz4", uncomp_chunk_size=65536)
    H, W = data[0].shape

    # Compress on GPU (data already in RAM)
    compressed = []
    for frame in data:
        arr = nvcomp.as_array(frame)
        compressed.append(codec.encode(arr))

    # Warmup
    for _ in range(5):
        _ = codec.decode(compressed[0])

    # Decode only
    torch.cuda.synchronize()
    n_iters = 3
    t0 = time.time()
    for _ in range(n_iters):
        for comp in compressed:
            _ = codec.decode(comp)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_iters
    thr = n_frames * H * W * 2 / elapsed / 1e9
    print(f"GPU decode {n_frames} frames: {elapsed:.3f}s "
          f"({elapsed / n_frames * 1000:.2f}ms/frame, "
          f"{thr:.1f} GB/s uncomp)")

    # Decode + zero-copy torch + extract patches
    patch_size = 11
    hh = patch_size // 2
    n_patches = 240
    device = torch.device("cuda")

    centers_ss = torch.randint(hh, H - hh, (n_patches,), device=device)
    centers_fs = torch.randint(hh, W - hh, (n_patches,), device=device)
    offsets = torch.arange(-hh, hh + 1, device=device)
    ss_grid = (centers_ss[:, None, None] + offsets[None, :, None]).expand(
        -1, patch_size, patch_size)
    fs_grid = (centers_fs[:, None, None] + offsets[None, None, :]).expand(
        -1, patch_size, patch_size)

    # Warmup
    for _ in range(3):
        dec = codec.decode(compressed[0])
        ft = torch.from_dlpack(dec).view(torch.int16).reshape(H, W)
        _ = ft[ss_grid, fs_grid].float()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        all_patches = []
        for comp in compressed:
            dec = codec.decode(comp)
            ft = torch.from_dlpack(dec).view(torch.int16).reshape(H, W)
            p = ft[ss_grid, fs_grid].float()
            all_patches.append(p)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_iters
    thr = n_frames * H * W * 2 / elapsed / 1e9
    print(f"GPU decode+extract {n_frames} frames: {elapsed:.3f}s "
          f"({elapsed / n_frames * 1000:.2f}ms/frame, "
          f"{thr:.1f} GB/s)")


def bench_cpu_extract_from_ram(compressed_dict, frames, n_frames):
    """Benchmark: blosc2 decode + CPU patch extract (serial, 8t internal)."""
    import blosc2
    from torchsx.io._excise import _extract_frame_patches

    print()
    print("=" * 60)
    print("CPU decompress + extract patches (best serial config)")
    print("=" * 60)

    bufs = compressed_dict["blosc2_bs"][:n_frames]
    shape = frames[0].shape
    H, W = shape
    patch_size = 11
    hh = patch_size // 2
    n_patches = 240

    centers_ss = np.random.randint(hh, H - hh, n_patches).astype(np.int64)
    centers_fs = np.random.randint(hh, W - hh, n_patches).astype(np.int64)

    for nt in [1, 4, 8]:
        blosc2.set_nthreads(nt)
        t0 = time.time()
        all_p = []
        for b in bufs:
            raw = blosc2.decompress2(b)
            frame = np.frombuffer(raw, dtype=np.int16).reshape(shape)
            p, v = _extract_frame_patches(frame, centers_ss, centers_fs,
                                          hh, hh, 0.0)
            all_p.append(p)
        elapsed = time.time() - t0
        print(f"  serial {nt}t decompress+extract: {elapsed:.3f}s "
              f"({elapsed / n_frames * 1000:.2f}ms/frame)")

    # H2D transfer of patches only
    all_p_np = np.concatenate(all_p)
    device = torch.device("cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    patches_t = torch.from_numpy(all_p_np).float().to(device)
    torch.cuda.synchronize()
    elapsed_h2d = time.time() - t0
    print(f"  H2D patches ({all_p_np.nbytes / 1e6:.1f} MB): "
          f"{elapsed_h2d * 1000:.1f}ms")
    blosc2.set_nthreads(1)


def print_summary(n_frames):
    print()
    print("=" * 60)
    print(f"PROJECTED: 500-frame batch, ~120K observations")
    print("=" * 60)
    print("""
From the measured per-frame times (data in RAM):

  blosc2 8t serial decode+extract:  500 × Xms  = ?
  nvCOMP GPU decode+extract:        500 × Yms  = ?

(Fill in from above measurements)
""")


def main():
    n_frames = 100
    frames = load_sample_frames(n_frames)

    print(f"\nPre-compressing {n_frames} frames...")
    comp = pre_compress(frames)

    shape = frames[0].shape
    bench_blosc2_from_ram(comp, shape, n_frames)
    bench_multiprocess(comp, shape, n_frames)
    bench_cpu_extract_from_ram(comp, frames, n_frames)

    try:
        bench_nvcomp_from_ram(frames, n_frames)
    except Exception as e:
        print(f"\nnvCOMP failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
