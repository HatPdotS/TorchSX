#!/usr/bin/env python
"""Sparse diffraction image analysis via Poisson significance thresholding.

Loads raw Jungfrau detector frames, estimates per-frame radial background,
computes Poisson significance for each pixel, and sweeps thresholds to
quantify compression via sparse COO representation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson, norm
from scipy.signal import savgol_filter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- paths -------------------------------------------------------------------
STREAM_PATH = Path("merged_368.stream")
H5_PATH = Path(
    "h5files/run0121-s2_inj55_scan_0p5uJ/acq0001.JF06T32V07.h5"
)
DATA_KEY = "data/JF06T32V07/data"
MASK_KEY = "data/JF06T32V07/meta/pixel_mask"
N_FRAMES = 50
N_RADIAL_BINS = 500
THRESHOLDS = [2, 3, 4, 5, 6, 8, 10]

# ── Step 1: geometry → radial distance map ──────────────────────────────────

def parse_geometry_from_stream(path: Path) -> dict:
    """Parse CrystFEL geometry header into global params + panel list."""
    geom_lines: list[str] = []
    in_geom = False
    with open(path) as f:
        for line in f:
            if line.startswith("----- Begin geometry file -----"):
                in_geom = True
                continue
            if line.startswith("----- End geometry file -----"):
                break
            if in_geom:
                geom_lines.append(line)

    import re
    vec_pat = re.compile(
        r"([+-]?\d+\.?\d*(?:e[+-]?\d+)?)x\s+"
        r"([+-]?\d+\.?\d*(?:e[+-]?\d+)?)y\s+"
        r"([+-]?\d+\.?\d*(?:e[+-]?\d+)?)z"
    )
    global_params: dict = {}
    panel_data: dict[str, dict] = {}

    for line in geom_lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()

        if "/" in key:
            pname, prop = key.split("/", 1)
            if pname not in panel_data:
                panel_data[pname] = {"name": pname}
            if prop in ("fs", "ss"):
                m = vec_pat.search(value)
                if m:
                    panel_data[pname][f"{prop}_vec"] = (
                        float(m.group(1)), float(m.group(2)), float(m.group(3))
                    )
            elif prop in ("min_fs", "max_fs", "min_ss", "max_ss"):
                panel_data[pname][prop] = int(value)
            elif prop in ("corner_x", "corner_y", "coffset"):
                panel_data[pname][prop] = float(value)
        else:
            if key == "clen":
                global_params["clen"] = float(value.split()[0])
            elif key == "res":
                global_params["res"] = float(value.split()[0])

    # Sort panels by min_ss
    panels = []
    for pname in sorted(panel_data, key=lambda n: panel_data[n].get("min_ss", 0)):
        panels.append(panel_data[pname])

    return {"panels": panels, **global_params}


def build_radial_map(geom: dict) -> np.ndarray:
    """Build (total_ss, total_fs) radial distance map in pixel units.

    Returns r in pixel units (distance from beam center in detector plane).
    """
    panels = geom["panels"]
    # Determine total detector shape from panel extents
    max_ss = max(p.get("max_ss", 0) for p in panels) + 1
    max_fs = max(p.get("max_fs", 0) for p in panels) + 1

    r_map = np.zeros((max_ss, max_fs), dtype=np.float32)

    for p in panels:
        ss0, ss1 = p["min_ss"], p["max_ss"]
        fs0, fs1 = p["min_fs"], p["max_fs"]
        n_ss = ss1 - ss0 + 1
        n_fs = fs1 - fs0 + 1

        cx = p.get("corner_x", 0.0)
        cy = p.get("corner_y", 0.0)
        fsx, fsy, _ = p.get("fs_vec", (1.0, 0.0, 0.0))
        ssx, ssy, _ = p.get("ss_vec", (0.0, 1.0, 0.0))

        # Local pixel indices
        dfs = np.arange(n_fs, dtype=np.float32)[None, :]  # (1, n_fs)
        dss = np.arange(n_ss, dtype=np.float32)[:, None]  # (n_ss, 1)

        x = cx + dfs * fsx + dss * ssx
        y = cy + dfs * fsy + dss * ssy
        r_map[ss0 : ss1 + 1, fs0 : fs1 + 1] = np.sqrt(x ** 2 + y ** 2)

    return r_map


# ── Step 2: load frames ─────────────────────────────────────────────────────

def load_data(h5_path: Path, n_frames: int):
    """Load frames and pixel mask from HDF5."""
    import bitshuffle.h5  # noqa: F401 — registers HDF5 filter
    import h5py

    with h5py.File(h5_path, "r") as f:
        frames = f[DATA_KEY][:n_frames]  # (N, 16448, 1030), int32
        mask = f[MASK_KEY][()]  # (16448, 1030), bool-like
    # Good pixels: mask == 0 typically means good in Jungfrau convention
    # But the plan says bool mask — treat True as good
    good = np.asarray(mask, dtype=bool)
    return frames, good


# ── Step 3: radial background ───────────────────────────────────────────────

def radial_background(frame: np.ndarray, r_map: np.ndarray, good: np.ndarray,
                      n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate radial background for a single frame.

    Returns
    -------
    bg_2d : (ss, fs) background image
    profile : (n_bins,) 1-D radial profile
    bin_edges : (n_bins+1,) bin edges in pixel radius
    """
    r_good = r_map[good]
    vals = frame[good].astype(np.float64)

    r_max = r_good.max()
    bin_edges = np.linspace(0, r_max + 1, n_bins + 1)
    bin_idx_full = np.digitize(r_map, bin_edges) - 1  # 0-based
    bin_idx_full = np.clip(bin_idx_full, 0, n_bins - 1)

    bin_idx = bin_idx_full[good]

    # Mean per bin
    profile = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(profile, bin_idx, vals)
    np.add.at(counts, bin_idx, 1)
    valid = counts > 0
    profile[valid] /= counts[valid]

    # Smooth — Savitzky-Golay with wide window
    window = min(51, n_bins // 4 * 2 + 1)  # must be odd
    if window >= 5:
        profile = savgol_filter(profile, window, 3)

    # Clip to >= 0
    profile = np.clip(profile, 0, None)

    # Map back to 2D
    bg_2d = profile[bin_idx_full]
    return bg_2d, profile, bin_edges


# ── Step 4: Poisson significance ────────────────────────────────────────────

def poisson_significance(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Compute Poisson z-score for each pixel.

    Uses exact Poisson survival function for small lambda,
    Anscombe approximation for large lambda (faster).
    """
    z = np.full_like(k, -np.inf, dtype=np.float64)
    mask_pos = k > 0  # k=0 is never significant above background

    # For pixels with k >= 1 and lam > 0
    active = mask_pos & (lam > 0)
    if not np.any(active):
        return z

    k_a = k[active].astype(np.float64)
    lam_a = lam[active]

    # Split: exact for lam < 20, Anscombe for lam >= 20
    small = lam_a < 20
    large = ~small

    z_active = np.zeros(k_a.shape, dtype=np.float64)

    if np.any(small):
        # Exact: P(X >= k) = 1 - cdf(k-1, lam) = sf(k-1, lam)
        p_val = poisson.sf(k_a[small] - 1, lam_a[small])
        p_val = np.clip(p_val, 1e-300, 1.0)  # avoid log(0)
        z_active[small] = -norm.ppf(p_val)

    if np.any(large):
        # Anscombe: z ≈ (sqrt(k+3/8) - sqrt(lam+3/8)) / 0.5
        z_active[large] = (
            np.sqrt(k_a[large] + 0.375) - np.sqrt(lam_a[large] + 0.375)
        ) / 0.5

    z[active] = z_active
    return z


# ── Step 5: sweep thresholds ────────────────────────────────────────────────

def analyze_thresholds(frames, good, r_map, n_bins, thresholds):
    """Sweep significance thresholds across frames.

    Returns per-threshold aggregated statistics.
    """
    n_frames = frames.shape[0]
    n_good = good.sum()
    raw_bytes_per_frame = n_good * 4  # int32

    results = {z: {"n_signal": 0, "sparse_bytes": 0} for z in thresholds}
    profiles_example = None

    for i in range(n_frames):
        frame = frames[i]
        bg_2d, profile, bin_edges = radial_background(frame, r_map, good, n_bins)

        if i == 0:
            profiles_example = (frame.copy(), bg_2d.copy(), profile.copy(), bin_edges.copy())

        # Poisson z-scores for good pixels
        k = frame[good].astype(np.float64)
        lam = bg_2d[good]
        z = poisson_significance(k, lam)

        for zt in thresholds:
            n_sig = int(np.sum(z > zt))
            results[zt]["n_signal"] += n_sig
            # COO storage: 2 × int32 (row, col) + 1 × int32 (value) = 12 bytes/pixel
            results[zt]["sparse_bytes"] += n_sig * 12

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{n_frames} frames")

    # Background storage per frame: profile (n_bins × 4 bytes)
    bg_bytes_per_frame = n_bins * 4

    table = []
    for zt in thresholds:
        n_sig_avg = results[zt]["n_signal"] / n_frames
        pct = 100.0 * n_sig_avg / n_good
        sparse_avg = results[zt]["sparse_bytes"] / n_frames
        total_compressed = bg_bytes_per_frame + sparse_avg
        ratio = raw_bytes_per_frame / total_compressed if total_compressed > 0 else float("inf")
        table.append({
            "threshold": zt,
            "n_signal": n_sig_avg,
            "pct_good": pct,
            "sparse_bytes": sparse_avg,
            "bg_bytes": bg_bytes_per_frame,
            "total_bytes": total_compressed,
            "ratio": ratio,
        })

    return table, profiles_example


# ── Step 6: visualization ───────────────────────────────────────────────────

def plot_results(table, example_data, good, r_map, n_bins, thresholds):
    """Generate three diagnostic plots."""
    frame, bg_2d, profile, bin_edges = example_data

    # -- Plot 1: compression ratio vs threshold ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    zts = [r["threshold"] for r in table]
    ratios = [r["ratio"] for r in table]
    ax1.plot(zts, ratios, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Poisson significance threshold (z)")
    ax1.set_ylabel("Compression ratio (raw / compressed)")
    ax1.set_title("Sparse COO compression vs significance threshold")
    ax1.grid(True, alpha=0.3)
    for r in table:
        ax1.annotate(f'{r["ratio"]:.1f}x', (r["threshold"], r["ratio"]),
                     textcoords="offset points", xytext=(5, 10), fontsize=9)
    fig1.tight_layout()
    fig1.savefig("compression_ratio.png", dpi=150)
    print("Saved compression_ratio.png")
    plt.close(fig1)

    # -- Plot 2: example frame decomposition ---
    # Recompute z-scores for frame 0
    k = frame[good].astype(np.float64)
    lam = bg_2d[good]
    z = np.full(frame.shape, np.nan, dtype=np.float64)
    z[good] = poisson_significance(k, lam)

    residual = np.full(frame.shape, np.nan, dtype=np.float32)
    residual[good] = frame[good].astype(np.float32) - bg_2d[good].astype(np.float32)

    signal_mask = np.zeros(frame.shape, dtype=bool)
    signal_mask[good] = z[good] > 4.0  # z>4 for visualization

    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original (clipped for visibility)
    im0 = axes[0, 0].imshow(np.clip(frame, 0, 10), cmap="viridis", aspect="auto",
                             interpolation="nearest")
    axes[0, 0].set_title("Original frame (clipped 0–10)")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # Background
    im1 = axes[0, 1].imshow(np.clip(bg_2d, 0, 5), cmap="viridis", aspect="auto",
                             interpolation="nearest")
    axes[0, 1].set_title("Radial background (clipped 0–5)")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # Residual
    im2 = axes[1, 0].imshow(np.clip(residual, -2, 10), cmap="RdBu_r", aspect="auto",
                             interpolation="nearest")
    axes[1, 0].set_title("Residual (frame − background)")
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)

    # Signal pixels at z>4
    signal_img = np.zeros(frame.shape, dtype=np.float32)
    signal_img[signal_mask] = frame[signal_mask].astype(np.float32)
    im3 = axes[1, 1].imshow(signal_img, cmap="hot", aspect="auto",
                             interpolation="nearest")
    axes[1, 1].set_title(f"Signal pixels (z > 4)")
    plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)

    fig2.suptitle("Frame 0 decomposition", fontsize=14)
    fig2.tight_layout()
    fig2.savefig("frame_decomposition.png", dpi=150)
    print("Saved frame_decomposition.png")
    plt.close(fig2)

    # -- Plot 3: radial profile ---
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(bin_centers, profile, linewidth=1.5)
    ax3.set_xlabel("Radial distance (pixels)")
    ax3.set_ylabel("Mean photon count")
    ax3.set_title("Radial background profile (frame 0)")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, bin_centers[-1])
    fig3.tight_layout()
    fig3.savefig("radial_profile.png", dpi=150)
    print("Saved radial_profile.png")
    plt.close(fig3)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=== Sparse Diffraction Image Analysis ===\n")

    # Step 1: geometry
    print("Loading geometry from stream file...")
    geom = parse_geometry_from_stream(STREAM_PATH)
    print(f"  {len(geom['panels'])} panels, res={geom.get('res', 0):.1f} px/m, "
          f"clen={geom.get('clen', 0):.4f} m")

    print("Building radial distance map...")
    r_map = build_radial_map(geom)
    print(f"  detector shape: {r_map.shape}, r range: {r_map.min():.1f} – {r_map.max():.1f} px")

    # Step 2: load data
    print(f"\nLoading {N_FRAMES} frames from HDF5...")
    frames, good = load_data(H5_PATH, N_FRAMES)
    print(f"  frame shape: {frames.shape}, dtype: {frames.dtype}")
    print(f"  good pixels: {good.sum():,} / {good.size:,} ({100*good.sum()/good.size:.1f}%)")
    print(f"  mean photons/pixel (good): {frames[:, good].mean():.3f}")

    # Step 3–5: analyze
    print(f"\nAnalyzing {N_FRAMES} frames with {N_RADIAL_BINS} radial bins...")
    table, example = analyze_thresholds(frames, good, r_map, N_RADIAL_BINS, THRESHOLDS)

    # Print table
    raw_bytes = good.sum() * 4
    print(f"\n{'z':>5s} | {'n_signal':>12s} | {'% good':>8s} | {'sparse KB':>10s} | "
          f"{'total KB':>10s} | {'ratio':>8s}")
    print("-" * 70)
    for r in table:
        print(f"{r['threshold']:5.0f} | {r['n_signal']:12.0f} | {r['pct_good']:7.3f}% | "
              f"{r['sparse_bytes']/1024:10.1f} | {r['total_bytes']/1024:10.1f} | "
              f"{r['ratio']:7.1f}x")
    print(f"\nRaw frame size (good pixels): {raw_bytes/1024:.1f} KB")

    # Step 6: plots
    print("\nGenerating plots...")
    plot_results(table, example, good, r_map, N_RADIAL_BINS, THRESHOLDS)

    print("\nDone.")


if __name__ == "__main__":
    main()
