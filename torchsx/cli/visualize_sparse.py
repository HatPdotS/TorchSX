#!/usr/bin/env python
"""Visualize sparse compressed data overlaid on detector geometry.

Two-panel figure:
  Left  — Radial background model rasterized per panel
  Right — Signal pixels as colored dots in physical (x, y) mm

Usage:
  python -m torchsx.cli.visualize_sparse \
      --sparse compressed_z1.5/run0121-s2_inj55_scan_0p5uJ_acq0001.JF06T32V07.sparse.safetensors \
      --stream merged_368.stream \
      --frame 42 \
      --output frame_42.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from torchsx.geometry import Geometry


# ── Data loading ─────────────────────────────────────────────────────────

def load_sparse_np(path: Path) -> dict:
    """Load sparse safetensors as numpy arrays (no torch needed for plotting)."""
    from safetensors.numpy import load_file
    return load_file(str(path))


def load_sidecar(sparse_path: Path) -> dict:
    """Load JSON sidecar next to the .sparse.safetensors file."""
    json_path = sparse_path.with_suffix("").with_suffix(".json")  # strip .safetensors → .sparse → .json
    # Actually: foo.sparse.safetensors → foo.sparse.json
    json_path = sparse_path.with_suffix(".json")
    with open(json_path) as f:
        return json.load(f)


def load_pixel_mask(meta: dict) -> np.ndarray:
    """Load good-pixel mask from the source H5 referenced in the sidecar."""
    import bitshuffle.h5  # noqa: F401
    import h5py

    h5_path = Path(meta["source"])
    mask_key = meta["mask_key"]
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f[mask_key][()], dtype=bool)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_frame(
    frame_id: int,
    profiles: np.ndarray,
    signal_idx: np.ndarray,
    signal_val: np.ndarray,
    bin_edges: np.ndarray,
    good_mask: np.ndarray,
    geom: Geometry,
    output_path: Path,
    sparse_name: str,
):
    """Create the two-panel visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    panels = geom.panels
    res = geom.res

    # ── Build bin assignment for good pixels ──
    r_map = geom.radial_map()
    r_good = r_map[good_mask]
    n_bins = profiles.shape[1]
    bin_assign = np.clip(np.digitize(r_good, bin_edges) - 1, 0, n_bins - 1)

    # ── Reverse lookup ──
    good_ss, good_fs = np.where(good_mask)
    good_ss = good_ss.astype(np.int32)
    good_fs = good_fs.astype(np.int32)

    # ── Background profile for this frame ──
    bg_profile = profiles[frame_id]  # (n_bins,)

    # ── Signal pixels for this frame ──
    n_signal = len(signal_idx)

    fig, (ax_bg, ax_sig) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(
        f"{sparse_name}  |  frame {frame_id}  |  {n_signal} signal pixels",
        fontsize=12,
    )

    # ── Left panel: Background model ──
    bg_vmin, bg_vmax = None, None
    for panel, (ss0, ss1, fs0, fs1) in zip(panels, geom.panel_slices):
        n_ss = ss1 - ss0
        n_fs = fs1 - fs0
        # Get good-pixel mask for this panel
        panel_good = good_mask[ss0:ss1, fs0:fs1]

        # Build 2D background image
        bg_img = np.full((n_ss, n_fs), np.nan, dtype=np.float32)
        # Get flat indices for this panel's good pixels
        panel_r = r_map[ss0:ss1, fs0:fs1]
        panel_bins = np.clip(np.digitize(panel_r[panel_good], bin_edges) - 1, 0, n_bins - 1)
        bg_img[panel_good] = bg_profile[panel_bins]

        # Extent from diagonal corners so data[0,0] maps correctly
        corners = panel.corners_mm(res)
        x0, y0 = corners[0]
        x1, y1 = corners[2]
        ext = [y0, y1, x0, x1]

        if bg_vmin is None:
            # Compute global range from profile
            bg_vmin = max(bg_profile[bg_profile > 0].min(), 0.1) if (bg_profile > 0).any() else 0.1
            bg_vmax = bg_profile.max()

        ax_bg.imshow(
            bg_img, origin="lower",
            extent=ext,
            aspect="equal",
            cmap="viridis",
            vmin=bg_vmin, vmax=bg_vmax,
            interpolation="nearest",
        )

    # Panel outlines on background
    for panel in panels:
        corners = panel.corners_mm(res)
        poly = Polygon(corners[:, ::-1], closed=True, fill=False,
                       edgecolor="white", linewidth=0.3, alpha=0.5)
        ax_bg.add_patch(poly)

    ax_bg.set_xlabel("y (mm)")
    ax_bg.set_ylabel("x (mm)")
    ax_bg.set_title("Radial background (photons/pixel)")
    ax_bg.set_aspect("equal")

    # Colorbar for background
    sm_bg = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(vmin=bg_vmin, vmax=bg_vmax))
    fig.colorbar(sm_bg, ax=ax_bg, label="Background (photons/pixel)", shrink=0.8)

    # ── Right panel: Signal pixels ──
    # Panel outlines
    for panel in panels:
        corners = panel.corners_mm(res)
        poly = Polygon(corners[:, ::-1], closed=True, fill=False,
                       edgecolor="gray", linewidth=0.3, alpha=0.7)
        ax_sig.add_patch(poly)

    if n_signal > 0:
        sig_ss = good_ss[signal_idx]
        sig_fs = good_fs[signal_idx]
        sig_x, sig_y = geom.pixel_to_mm(sig_ss, sig_fs)
        counts = signal_val.astype(np.float32)
        counts = np.clip(counts, 1, None)  # floor at 1 for LogNorm
        sc = ax_sig.scatter(
            sig_y, sig_x, c=counts, cmap="hot", norm=LogNorm(), s=0.5,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax_sig, label="Photon counts", shrink=0.8)

    ax_sig.set_xlabel("y (mm)")
    ax_sig.set_ylabel("x (mm)")
    ax_sig.set_title("Signal pixels")
    ax_sig.set_aspect("equal")

    # Match axis limits
    all_corners = geom.all_corners_mm()
    margin = 5  # mm
    xlim = (all_corners[:, 1].min() - margin, all_corners[:, 1].max() + margin)
    ylim = (all_corners[:, 0].min() - margin, all_corners[:, 0].max() + margin)
    for ax in (ax_bg, ax_sig):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize sparse compressed data on detector geometry."
    )
    parser.add_argument("--sparse", type=Path, required=True,
                        help="Path to .sparse.safetensors file")
    parser.add_argument("--stream", type=Path, required=True,
                        help="CrystFEL .stream file (for geometry)")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame index (default: auto-pick frame with most signal)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (SVG or PDF, default: sparse_frame_{N}.svg)")

    args = parser.parse_args()

    # Load sparse data
    print(f"Loading {args.sparse.name}...")
    tensors = load_sparse_np(args.sparse)
    profiles = tensors["profiles"]
    signal_idx = tensors["signal_idx"]
    signal_frame = tensors["signal_frame"]
    signal_val = tensors["signal_val"]
    bin_edges = tensors["bin_edges"]

    n_frames, n_bins = profiles.shape
    print(f"  {n_frames} frames, {n_bins} bins, {len(signal_idx)} total signal pixels")

    # Load sidecar
    meta = load_sidecar(args.sparse)

    # Parse geometry
    print(f"Parsing geometry from {args.stream.name}...")
    geom = Geometry.from_stream(args.stream)
    print(f"  {geom.n_panels} panels, res={geom.res}")

    # Load pixel mask
    print(f"Loading pixel mask from {meta['source']}...")
    good_mask = load_pixel_mask(meta)
    print(f"  good pixels: {good_mask.sum():,} / {good_mask.size:,}")

    # Pick frame
    if args.frame is not None:
        frame_id = args.frame
    else:
        # Auto-pick: frame with most signal pixels
        counts_per_frame = np.bincount(signal_frame, minlength=n_frames)
        frame_id = int(np.argmax(counts_per_frame))
        print(f"  auto-picked frame {frame_id} ({counts_per_frame[frame_id]} signal pixels)")

    # Extract this frame's signal
    frame_mask = signal_frame == frame_id
    frame_sig_idx = signal_idx[frame_mask]
    frame_sig_val = signal_val[frame_mask]
    print(f"  frame {frame_id}: {len(frame_sig_idx)} signal pixels")

    # Output path
    output_path = args.output or Path(f"sparse_frame_{frame_id}.svg")

    plot_frame(
        frame_id=frame_id,
        profiles=profiles,
        signal_idx=frame_sig_idx,
        signal_val=frame_sig_val,
        bin_edges=bin_edges,
        good_mask=good_mask,
        geom=geom,
        output_path=output_path,
        sparse_name=args.sparse.stem,
    )


if __name__ == "__main__":
    main()
