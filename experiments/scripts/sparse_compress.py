#!/usr/bin/env python
"""Fast GPU sparse compression of Jungfrau diffraction frames.

For each frame:
  1. Compute per-frame radial background via scatter_add (GPU)
  2. Anscombe z-score for Poisson significance
  3. Probabilistic erosion: keep signal pixels only if they have
     >= min_neighbors signal neighbors within the same panel
  4. Extract signal pixels as COO sparse tensor

All geometry/bin assignments precomputed once. No scipy, no Python loops.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchsx.geometry import Geometry
from sparse_analysis import (
    load_data,
    STREAM_PATH, H5_PATH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FRAMES = 50
N_RADIAL_BINS = 500
THRESHOLDS = [round(0.5 + i * 0.1, 1) for i in range(96)]  # 0.5 to 10.0 step 0.1
MIN_NEIGHBORS = 1  # minimum signal neighbors to survive erosion


def precompute_binning(r_map: np.ndarray, good: np.ndarray, n_bins: int, device):
    """Precompute bin indices and counts for good pixels (once)."""
    r_good = r_map[good]
    r_max = r_good.max()
    bin_edges = np.linspace(0, r_max + 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(r_good, bin_edges) - 1, 0, n_bins - 1).astype(np.int64)

    bin_idx_gpu = torch.from_numpy(bin_idx).to(device)
    bin_counts = torch.zeros(n_bins, device=device, dtype=torch.float32)
    bin_counts.scatter_add_(0, bin_idx_gpu, torch.ones(len(bin_idx), device=device))
    bin_counts = bin_counts.clamp(min=1)

    return bin_idx_gpu, bin_counts


def get_panel_slices(geom: dict) -> list[tuple[int, int, int, int]]:
    """Return (ss_start, ss_end, fs_start, fs_end) for each panel."""
    return [(p["min_ss"], p["max_ss"] + 1, p["min_fs"], p["max_fs"] + 1)
            for p in geom["panels"]]


def frame_background(frame_good: torch.Tensor, bin_idx: torch.Tensor,
                     bin_counts: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Per-frame radial background on GPU."""
    profile = torch.zeros(n_bins, device=frame_good.device, dtype=torch.float32)
    profile.scatter_add_(0, bin_idx, frame_good)
    profile /= bin_counts
    profile = profile.clamp(min=0)
    return profile[bin_idx]


def anscombe_z(k: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Anscombe z-score: (sqrt(k+3/8) - sqrt(lam+3/8)) / 0.5"""
    return (torch.sqrt(k + 0.375) - torch.sqrt(lam + 0.375)) / 0.5


def per_panel_erosion(signal_2d: torch.Tensor, panel_slices: list,
                      min_neighbors: int = 1) -> torch.Tensor:
    """Erode signal mask per panel using 3x3 neighbor count.

    A signal pixel survives only if it has >= min_neighbors signal neighbors
    within its own panel (8-connected). Panels are processed independently
    so inter-panel gaps don't create false adjacency.

    Parameters
    ----------
    signal_2d : (ss, fs) bool tensor — signal mask on full detector
    panel_slices : list of (ss0, ss1, fs0, fs1) per panel
    min_neighbors : minimum number of signal neighbors to keep pixel

    Returns
    -------
    eroded : (ss, fs) bool tensor
    """
    # 3x3 kernel counting neighbors (center excluded)
    kernel = torch.ones(1, 1, 3, 3, device=signal_2d.device, dtype=torch.float32)
    kernel[0, 0, 1, 1] = 0  # exclude self

    eroded = torch.zeros_like(signal_2d)

    for ss0, ss1, fs0, fs1 in panel_slices:
        panel_mask = signal_2d[ss0:ss1, fs0:fs1].float().unsqueeze(0).unsqueeze(0)
        # Count neighbors with zero-padding (panel boundary = no neighbors)
        neighbor_count = F.conv2d(panel_mask, kernel, padding=1)
        # Keep pixels that are signal AND have enough neighbors
        survived = (panel_mask > 0.5) & (neighbor_count >= min_neighbors)
        eroded[ss0:ss1, fs0:fs1] = survived.squeeze()

    return eroded


def main():
    print(f"=== Fast GPU Sparse Compression with Erosion ({DEVICE}) ===\n")

    # Load geometry
    geom = Geometry.from_stream(STREAM_PATH)
    r_map = geom.radial_map()
    panel_slices = geom.panel_slices
    total_ss, total_fs = geom.detector_shape
    print(f"  {geom.n_panels} panels, detector: {total_ss}x{total_fs}")

    # Load data
    t0 = time.time()
    frames, good = load_data(H5_PATH, N_FRAMES)
    n_good = good.sum()
    print(f"  loaded {N_FRAMES} frames in {time.time()-t0:.1f}s")
    print(f"  good pixels: {n_good:,}, mean: {frames[:, good].mean():.3f}")

    # Precompute binning
    bin_idx_gpu, bin_counts = precompute_binning(r_map, good, N_RADIAL_BINS, DEVICE)
    good_gpu = torch.from_numpy(good).to(DEVICE)

    # Move all frames to GPU
    frames_gpu = torch.from_numpy(frames).to(DEVICE)

    # Process all frames — compare raw threshold vs eroded
    print(f"\nCompressing {N_FRAMES} frames (min_neighbors={MIN_NEIGHBORS})...")
    t0 = time.time()

    raw_bytes = n_good * 4
    thresholds_gpu = torch.tensor(THRESHOLDS, device=DEVICE, dtype=torch.float32)
    n_thresh = len(THRESHOLDS)
    signal_counts_raw = torch.zeros(n_thresh, device=DEVICE, dtype=torch.long)
    signal_counts_eroded = torch.zeros(n_thresh, device=DEVICE, dtype=torch.long)

    # Example data
    example_z_2d = None
    example_eroded = None
    example_raw_mask = None
    example_profile = None

    for i in range(N_FRAMES):
        frame_good = frames_gpu[i][good_gpu].float()

        # Per-frame background
        bg = frame_background(frame_good, bin_idx_gpu, bin_counts, N_RADIAL_BINS)

        # Anscombe z-score (flat, good pixels only)
        z_flat = anscombe_z(frame_good, bg)

        # Scatter z-scores back to 2D for erosion
        z_2d = torch.zeros(total_ss, total_fs, device=DEVICE)
        z_2d[good_gpu] = z_flat

        # Raw counts (no erosion)
        signal_counts_raw += (z_flat.unsqueeze(0) > thresholds_gpu.unsqueeze(1)).sum(dim=1)

        # Per-threshold erosion
        for ti, zt in enumerate(THRESHOLDS):
            signal_mask = z_2d > zt
            eroded = per_panel_erosion(signal_mask, panel_slices, MIN_NEIGHBORS)
            # Count only good pixels that survived
            signal_counts_eroded[ti] += (eroded[good_gpu]).sum()

        if i == 0:
            example_z_2d = z_2d.cpu().numpy()
            # Store example at z=2
            raw_mask = z_2d > 2.0
            example_raw_mask = raw_mask.cpu().numpy()
            example_eroded = per_panel_erosion(raw_mask, panel_slices, MIN_NEIGHBORS).cpu().numpy()
            profile = torch.zeros(N_RADIAL_BINS, device=DEVICE)
            profile.scatter_add_(0, bin_idx_gpu, frame_good)
            profile /= bin_counts
            example_profile = profile.cpu().numpy()

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_FRAMES}")

    dt = time.time() - t0
    fps = N_FRAMES / dt
    print(f"  {dt:.2f}s total, {fps:.1f} frames/sec, {dt/N_FRAMES*1000:.1f} ms/frame")

    raw_np = signal_counts_raw.cpu().numpy().astype(np.float64) / N_FRAMES
    eroded_np = signal_counts_eroded.cpu().numpy().astype(np.float64) / N_FRAMES
    thresholds_np = np.array(THRESHOLDS)

    bg_bytes = N_RADIAL_BINS * 4
    ratios_raw = raw_bytes / np.maximum(bg_bytes + raw_np * 12, 1)
    ratios_eroded = raw_bytes / np.maximum(bg_bytes + eroded_np * 12, 1)

    print(f"\nRaw frame: {raw_bytes/1024:.1f} KB")
    print(f"\n{'z':>5s} | {'raw px':>10s} {'raw ratio':>10s} | {'eroded px':>10s} {'eroded ratio':>12s} | {'removed':>8s}")
    print("-" * 75)
    for zt_show in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        idx = np.argmin(np.abs(thresholds_np - zt_show))
        removed_pct = 100 * (1 - eroded_np[idx] / max(raw_np[idx], 1))
        print(f"{thresholds_np[idx]:5.1f} | {raw_np[idx]:10.0f} {ratios_raw[idx]:9.0f}x | "
              f"{eroded_np[idx]:10.0f} {ratios_eroded[idx]:11.0f}x | {removed_pct:6.1f}%")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Compression ratio: raw vs eroded
    axes[0, 0].semilogy(thresholds_np, ratios_raw, linewidth=2, label="raw threshold")
    axes[0, 0].semilogy(thresholds_np, ratios_eroded, linewidth=2, label=f"+ erosion (>={MIN_NEIGHBORS} neighbors)")
    axes[0, 0].set_xlabel("Significance threshold z")
    axes[0, 0].set_ylabel("Compression ratio")
    axes[0, 0].set_title("Compression ratio vs threshold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, which="both")
    axes[0, 0].set_xlim(0.5, 10)

    # Signal pixel count: raw vs eroded
    axes[0, 1].semilogy(thresholds_np, raw_np, linewidth=2, label="raw")
    axes[0, 1].semilogy(thresholds_np, np.maximum(eroded_np, 0.1), linewidth=2, label="eroded")
    axes[0, 1].set_xlabel("Significance threshold z")
    axes[0, 1].set_ylabel("Signal pixels per frame")
    axes[0, 1].set_title("Signal pixel count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, which="both")
    axes[0, 1].set_xlim(0.5, 10)

    # Fraction removed by erosion
    frac_removed = 1 - eroded_np / np.maximum(raw_np, 1)
    axes[0, 2].plot(thresholds_np, 100 * frac_removed, linewidth=2)
    axes[0, 2].set_xlabel("Significance threshold z")
    axes[0, 2].set_ylabel("% pixels removed by erosion")
    axes[0, 2].set_title("Erosion effectiveness")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(0.5, 10)
    axes[0, 2].set_ylim(0, 100)

    # Example: raw vs eroded at z=2 (zoomed panel region)
    # Find a panel with some signal
    best_panel = 0
    best_count = 0
    for pi, (ss0, ss1, fs0, fs1) in enumerate(panel_slices):
        c = example_raw_mask[ss0:ss1, fs0:fs1].sum()
        if c > best_count:
            best_count = c
            best_panel = pi
    ss0, ss1, fs0, fs1 = panel_slices[best_panel]

    axes[1, 0].imshow(example_raw_mask[ss0:ss1, fs0:fs1], cmap="Reds",
                      aspect="auto", interpolation="nearest")
    axes[1, 0].set_title(f"Raw z>2 (panel {best_panel}, {best_count} px)")

    axes[1, 1].imshow(example_eroded[ss0:ss1, fs0:fs1], cmap="Reds",
                      aspect="auto", interpolation="nearest")
    eroded_count = example_eroded[ss0:ss1, fs0:fs1].sum()
    axes[1, 1].set_title(f"After erosion (panel {best_panel}, {eroded_count} px)")

    # Radial profile
    axes[1, 2].plot(example_profile)
    axes[1, 2].set_xlabel("Radial bin")
    axes[1, 2].set_ylabel("Mean photon count")
    axes[1, 2].set_title("Frame 0 radial profile")

    fig.suptitle(f"Sparse compression with probabilistic erosion ({fps:.0f} fps)", fontsize=14)
    fig.tight_layout()
    fig.savefig("sparse_compress.png", dpi=150)
    print("\nSaved sparse_compress.png")


if __name__ == "__main__":
    main()
