#!/usr/bin/env python
"""Compress raw Jungfrau HDF5 frames into sparse signal + radial background.

For each frame:
  1. Compute per-frame radial background (scatter_add into radial bins)
  2. Anscombe z-score for Poisson significance
  3. Per-panel erosion: keep signal pixels with >= min_neighbors
  4. Store as sparse COO + 1-D radial profile

Output: one .safetensors file per input HDF5, containing:
  - profiles: (n_frames, n_bins) float32 — radial background per frame
  - signal_idx: (n_total_signal, 2) int32 — (flat_pixel_index, frame_index)
  - signal_val: (n_total_signal,) int32 — observed photon count
  - bin_edges: (n_bins+1,) float32 — radial bin edges
  - bin_idx: (n_good_pixels,) int32 — bin assignment per good pixel
  - good_mask: (total_ss, total_fs) bool — pixel mask
  - metadata: geometry, thresholds, etc. in JSON sidecar

Usage:
  python -m torchsx.cli.sparse_compress \\
      --stream merged_368.stream \\
      --h5 h5files/run0121-s2_inj55_scan_0p5uJ/acq0001.JF06T32V07.h5 \\
      --outdir compressed/ \\
      --z-threshold 2.0 \\
      --min-neighbors 1 \\
      --n-radial-bins 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None


# ── Geometry (canonical implementation in torchsx.geometry) ─────────────────

from torchsx.geometry import Geometry


def parse_geometry_from_stream(path: Path) -> Geometry:
    """Parse CrystFEL geometry from .stream header.

    Backward-compat wrapper — returns a Geometry object (which external
    scripts previously received as a dict).
    """
    return Geometry.from_stream(path)


def build_radial_map(geom) -> np.ndarray:
    """Build (total_ss, total_fs) radial distance map in pixel units.

    Backward-compat wrapper — accepts a Geometry object.
    """
    if isinstance(geom, Geometry):
        return geom.radial_map()
    raise TypeError(f"Expected Geometry, got {type(geom).__name__}")


# ── GPU compression core ───────────────────────────────────────────────────

class SparseCompressor:
    """Precomputed geometry for fast per-frame sparse compression."""

    def __init__(self, geom: Geometry, good: np.ndarray, n_bins: int,
                 z_threshold: float, min_neighbors: int, device: torch.device):
        self.device = device
        self.z_threshold = z_threshold
        self.min_neighbors = min_neighbors
        self.n_bins = n_bins

        self.panel_slices = geom.panel_slices
        self.total_ss, self.total_fs = geom.detector_shape

        # Radial binning
        r_map = geom.radial_map()
        r_good = r_map[good]
        r_max = r_good.max()
        self.bin_edges = np.linspace(0, r_max + 1, n_bins + 1).astype(np.float32)
        bin_idx = np.clip(
            np.digitize(r_good, self.bin_edges) - 1, 0, n_bins - 1
        ).astype(np.int64)

        self.bin_idx = torch.from_numpy(bin_idx).to(device)
        bin_counts = torch.zeros(n_bins, device=device, dtype=torch.float32)
        bin_counts.scatter_add_(0, self.bin_idx,
                                torch.ones(len(bin_idx), device=device))
        self.bin_counts = bin_counts.clamp(min=1)

        self.good = torch.from_numpy(good).to(device)

        # 3x3 erosion kernel (neighbors only, no self)
        self.kernel = torch.ones(1, 1, 3, 3, device=device, dtype=torch.float32)
        self.kernel[0, 0, 1, 1] = 0

    def compress_frame(self, frame: torch.Tensor):
        """Compress a single (total_ss, total_fs) int32 frame.

        Returns
        -------
        profile : (n_bins,) float32 — radial background profile
        signal_flat_idx : (n_signal,) int64 — flat index into good-pixel array
        signal_val : (n_signal,) int32 — observed counts at signal pixels
        """
        frame_good = frame[self.good].float()

        # Radial background
        profile = torch.zeros(self.n_bins, device=self.device, dtype=torch.float32)
        profile.scatter_add_(0, self.bin_idx, frame_good)
        profile /= self.bin_counts
        profile = profile.clamp(min=0)
        bg = profile[self.bin_idx]

        # Anscombe z-score
        z = (torch.sqrt(frame_good + 0.375) - torch.sqrt(bg + 0.375)) / 0.5

        # Scatter z to 2D for erosion
        z_2d = torch.zeros(self.total_ss, self.total_fs, device=self.device)
        z_2d[self.good] = z

        signal_2d = z_2d > self.z_threshold

        # Per-panel erosion
        if self.min_neighbors > 0:
            eroded = torch.zeros_like(signal_2d)
            for ss0, ss1, fs0, fs1 in self.panel_slices:
                panel = signal_2d[ss0:ss1, fs0:fs1].float().unsqueeze(0).unsqueeze(0)
                nb = F.conv2d(panel, self.kernel, padding=1)
                survived = (panel > 0.5) & (nb >= self.min_neighbors)
                eroded[ss0:ss1, fs0:fs1] = survived.squeeze()
            signal_2d = eroded

        # Extract signal from good pixels
        signal_good = signal_2d[self.good]
        signal_flat_idx = signal_good.nonzero(as_tuple=False).squeeze(1)
        signal_val = frame_good[signal_flat_idx].int()

        return profile, signal_flat_idx, signal_val


# ── HDF5 I/O ───────────────────────────────────────────────────────────────

def open_h5(path: Path):
    """Open HDF5 file with bitshuffle filter registered."""
    import bitshuffle.h5  # noqa: F401
    import h5py
    return h5py.File(path, "r")


def find_data_mask_keys(f) -> tuple[str, str]:
    """Auto-detect data and mask dataset paths in Jungfrau HDF5."""
    # Walk to find a 3D dataset (frames) and a 2D bool (mask)
    data_key = None
    mask_key = None
    def visitor(name, obj):
        nonlocal data_key, mask_key
        import h5py
        if isinstance(obj, h5py.Dataset):
            if obj.ndim == 3 and data_key is None:
                data_key = name
            if obj.ndim == 2 and "mask" in name.lower() and mask_key is None:
                mask_key = name
    f.visititems(visitor)
    return data_key, mask_key


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compress Jungfrau HDF5 frames into sparse signal + radial background."
    )
    parser.add_argument("--stream", type=Path, required=True,
                        help="CrystFEL .stream file (for geometry)")
    parser.add_argument("--h5", type=Path, nargs="+", required=True,
                        help="Input HDF5 file(s)")
    parser.add_argument("--outdir", type=Path, default=Path("compressed"),
                        help="Output directory (default: compressed/)")
    parser.add_argument("--z-threshold", type=float, default=2.0,
                        help="Poisson significance threshold (default: 2.0)")
    parser.add_argument("--min-neighbors", type=int, default=1,
                        help="Min signal neighbors for erosion (default: 1, 0=no erosion)")
    parser.add_argument("--n-radial-bins", type=int, default=500,
                        help="Number of radial bins (default: 500)")
    parser.add_argument("--data-key", type=str, default=None,
                        help="HDF5 dataset path for frames (auto-detected if omitted)")
    parser.add_argument("--mask-key", type=str, default=None,
                        help="HDF5 dataset path for pixel mask (auto-detected if omitted)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Frames per GPU batch (default: 100)")

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Sparse Compression ({device}) ===")
    print(f"  z={args.z_threshold}, min_neighbors={args.min_neighbors}, "
          f"bins={args.n_radial_bins}")

    # Parse geometry once
    print(f"\nParsing geometry from {args.stream}...")
    geom = Geometry.from_stream(args.stream)
    print(f"  {geom.n_panels} panels")

    # Load mask from first file
    print(f"Loading pixel mask from {args.h5[0].name}...")
    f0 = open_h5(args.h5[0])
    data_key = args.data_key
    mask_key = args.mask_key
    if data_key is None or mask_key is None:
        dk, mk = find_data_mask_keys(f0)
        data_key = data_key or dk
        mask_key = mask_key or mk
    print(f"  data: {data_key}, mask: {mask_key}")

    good = np.asarray(f0[mask_key][()], dtype=bool)
    f0.close()
    n_good = good.sum()
    print(f"  good pixels: {n_good:,} / {good.size:,}")

    # Build compressor
    compressor = SparseCompressor(
        geom, good, args.n_radial_bins,
        args.z_threshold, args.min_neighbors, device
    )

    # Process each HDF5 file
    for h5_path in args.h5:
        print(f"\n{'='*60}")
        print(f"Processing {h5_path.name}...")
        t0_file = time.time()

        f = open_h5(h5_path)
        dset = f[data_key]
        n_frames = dset.shape[0]
        print(f"  {n_frames} frames, shape={dset.shape}, dtype={dset.dtype}")

        all_profiles = []
        all_signal_idx = []  # flat index into good-pixel array
        all_signal_frame = []  # which frame
        all_signal_val = []

        n_signal_total = 0
        t0 = time.time()

        for start in range(0, n_frames, args.batch_size):
            end = min(start + args.batch_size, n_frames)
            batch = torch.from_numpy(dset[start:end].astype(np.int32)).to(device)

            for fi in range(batch.shape[0]):
                frame_idx = start + fi
                profile, sig_idx, sig_val = compressor.compress_frame(batch[fi])

                all_profiles.append(profile.cpu())
                if sig_idx.numel() > 0:
                    all_signal_idx.append(sig_idx.cpu().int())
                    all_signal_frame.append(
                        torch.full((sig_idx.numel(),), frame_idx, dtype=torch.int32)
                    )
                    all_signal_val.append(sig_val.cpu())
                    n_signal_total += sig_idx.numel()

            elapsed = time.time() - t0
            fps = (end) / elapsed
            print(f"  {end:6d}/{n_frames} frames  "
                  f"{fps:.1f} fps  "
                  f"{n_signal_total:,} signal px total  "
                  f"({n_signal_total / max(end, 1):.0f}/frame avg)",
                  end="\r")

        print()
        f.close()

        dt = time.time() - t0_file
        print(f"  done in {dt:.1f}s ({n_frames/dt:.1f} fps)")

        # Assemble output tensors
        profiles = torch.stack(all_profiles)  # (n_frames, n_bins)

        if all_signal_idx:
            signal_idx = torch.cat(all_signal_idx)    # (n_total,)
            signal_frame = torch.cat(all_signal_frame)  # (n_total,)
            signal_val = torch.cat(all_signal_val)      # (n_total,)
        else:
            signal_idx = torch.zeros(0, dtype=torch.int32)
            signal_frame = torch.zeros(0, dtype=torch.int32)
            signal_val = torch.zeros(0, dtype=torch.int32)

        # Compression stats
        raw_bytes = n_frames * n_good * 4
        compressed_bytes = (
            profiles.numel() * 4  # profiles
            + signal_idx.numel() * 4  # flat indices
            + signal_frame.numel() * 4  # frame indices
            + signal_val.numel() * 4  # values
            + (args.n_radial_bins + 1) * 4  # bin edges
        )
        ratio = raw_bytes / max(compressed_bytes, 1)
        print(f"  raw: {raw_bytes/1e9:.2f} GB, compressed: {compressed_bytes/1e6:.1f} MB, "
              f"ratio: {ratio:.0f}x")
        print(f"  signal pixels: {n_signal_total:,} total, "
              f"{n_signal_total/n_frames:.0f}/frame avg")

        # Save — include parent dir name to avoid collisions across runs
        out_stem = f"{h5_path.parent.name}_{h5_path.stem}"
        out_path = args.outdir / f"{out_stem}.sparse.safetensors"
        meta_path = args.outdir / f"{out_stem}.sparse.json"

        tensors = {
            "profiles": profiles,
            "signal_idx": signal_idx,
            "signal_frame": signal_frame,
            "signal_val": signal_val,
            "bin_edges": torch.from_numpy(compressor.bin_edges),
        }

        if save_file is not None:
            save_file(tensors, str(out_path))
            print(f"  saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        else:
            # Fallback: torch.save
            out_path = out_path.with_suffix(".pt")
            torch.save(tensors, str(out_path))
            print(f"  saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

        # Metadata sidecar
        meta = {
            "source": str(h5_path),
            "stream": str(args.stream),
            "data_key": data_key,
            "mask_key": mask_key,
            "n_frames": n_frames,
            "n_good_pixels": int(n_good),
            "detector_shape": [int(good.shape[0]), int(good.shape[1])],
            "n_radial_bins": args.n_radial_bins,
            "z_threshold": args.z_threshold,
            "min_neighbors": args.min_neighbors,
            "n_signal_total": n_signal_total,
            "compression_ratio": round(ratio, 1),
            "frames_per_sec": round(n_frames / dt, 1),
        }
        with open(meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)
        print(f"  saved {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
