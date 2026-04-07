#!/usr/bin/env python
"""Visualize a raw H5 detector frame on physical detector geometry.

Two-panel figure:
  Left  — Full frame (log scale) with panel outlines
  Right — Zoomed to signal range, masked pixels transparent

Usage:
  python -m torchsx.cli.visualize_h5frame \
      --h5 h5files/run0121-s2_inj55_scan_0p5uJ/acq0001.JF06T32V07.h5 \
      --stream merged_368.stream \
      --frame 42 \
      --output h5_frame_42.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from torchsx.geometry import Geometry


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_h5_frame(
    frame_data: np.ndarray,
    good_mask: np.ndarray,
    geom: Geometry,
    frame_id: int,
    h5_name: str,
    output_path: Path,
    vmax_percentile: float = 99.5,
):
    """Render a raw H5 frame on physical detector geometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Polygon

    panels = geom.panels
    res = geom.res

    # Stats
    good_vals = frame_data[good_mask].astype(np.float32)
    n_good = good_mask.sum()
    median_val = np.median(good_vals)
    max_val = good_vals.max()
    vmax = np.percentile(good_vals[good_vals > 0], vmax_percentile) if (good_vals > 0).any() else 1

    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(
        f"{h5_name}  |  frame {frame_id}  |  "
        f"median={median_val:.0f}  max={max_val:.0f}  good={n_good:,}",
        fontsize=11,
    )

    # Render each panel
    for panel, (ss0, ss1, fs0, fs1) in zip(panels, geom.panel_slices):
        panel_good = good_mask[ss0:ss1, fs0:fs1]
        panel_data = frame_data[ss0:ss1, fs0:fs1].astype(np.float32)
        panel_data[~panel_good] = np.nan

        # Extent from the two diagonal corners of the panel so that
        # data[0,0] (ss=0, fs=0) maps to the correct physical position.
        # corners[0] = (x,y) at (dfs=0,dss=0), corners[2] = at (dfs=n_fs,dss=n_ss)
        corners = panel.corners_mm(res)
        x0, y0 = corners[0]  # position of data row 0, col 0
        x1, y1 = corners[2]  # position of data row max, col max
        # extent = [left, right, bottom, top]; rows→vertical(x), cols→horizontal(y)
        ext = [y0, y1, x0, x1]

        # Log-scale panel
        log_data = panel_data.copy()
        log_data[log_data <= 0] = np.nan  # hide zeros/negatives in log
        ax_log.imshow(
            log_data, origin="lower",
            extent=ext,
            aspect="equal", cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(max_val, 2)),
            interpolation="nearest",
        )

        # Linear-scale panel (clipped to percentile)
        ax_lin.imshow(
            panel_data, origin="lower",
            extent=ext,
            aspect="equal", cmap="inferno",
            vmin=0, vmax=vmax,
            interpolation="nearest",
        )

    # Panel outlines on both
    for ax, color in [(ax_log, "white"), (ax_lin, "white")]:
        for panel in panels:
            corners = panel.corners_mm(res)
            poly = Polygon(corners[:, ::-1], closed=True, fill=False,
                           edgecolor=color, linewidth=0.3, alpha=0.5)
            ax.add_patch(poly)

    # Axis limits
    all_corners = geom.all_corners_mm()
    margin = 5
    xlim = (all_corners[:, 1].min() - margin, all_corners[:, 1].max() + margin)
    ylim = (all_corners[:, 0].min() - margin, all_corners[:, 0].max() + margin)

    for ax in (ax_log, ax_lin):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("x (mm)")
        ax.set_aspect("equal")

    ax_log.set_title("Log scale")
    ax_lin.set_title(f"Linear scale (0–{vmax:.0f})")

    # Colorbars
    sm_log = plt.cm.ScalarMappable(cmap="viridis",
                                    norm=LogNorm(vmin=1, vmax=max(max_val, 2)))
    fig.colorbar(sm_log, ax=ax_log, label="Counts", shrink=0.8)

    sm_lin = plt.cm.ScalarMappable(cmap="inferno",
                                    norm=Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm_lin, ax=ax_lin, label="Counts", shrink=0.8)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a raw H5 detector frame on physical geometry."
    )
    parser.add_argument("--h5", type=Path, required=True,
                        help="Input HDF5 file")
    parser.add_argument("--stream", type=Path, required=True,
                        help="CrystFEL .stream file (for geometry)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index (default: 0)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (PNG/SVG/PDF, default: h5_frame_{N}.png)")
    parser.add_argument("--data-key", type=str, default=None,
                        help="HDF5 dataset path for frames (auto-detected if omitted)")
    parser.add_argument("--mask-key", type=str, default=None,
                        help="HDF5 dataset path for pixel mask (auto-detected if omitted)")

    args = parser.parse_args()

    # Open H5
    import bitshuffle.h5  # noqa: F401
    import h5py

    print(f"Opening {args.h5.name}...")
    f = h5py.File(args.h5, "r")

    # Auto-detect keys
    data_key = args.data_key
    mask_key = args.mask_key
    if data_key is None or mask_key is None:
        from torchsx.cli.sparse_compress import find_data_mask_keys
        dk, mk = find_data_mask_keys(f)
        data_key = data_key or dk
        mask_key = mask_key or mk
    print(f"  data: {data_key}, mask: {mask_key}")

    dset = f[data_key]
    n_frames = dset.shape[0]
    print(f"  {n_frames} frames, shape={dset.shape}")

    frame_id = args.frame
    if frame_id >= n_frames:
        print(f"ERROR: frame {frame_id} out of range (0–{n_frames-1})")
        return

    # Read frame + mask
    print(f"Reading frame {frame_id}...")
    frame_data = dset[frame_id]
    good_mask = np.asarray(f[mask_key][()], dtype=bool)
    f.close()

    # Geometry
    print(f"Parsing geometry from {args.stream.name}...")
    geom = Geometry.from_stream(args.stream)
    print(f"  {geom.n_panels} panels")

    # Output path
    output_path = args.output or Path(f"h5_frame_{frame_id}.svg")

    plot_h5_frame(
        frame_data=frame_data,
        good_mask=good_mask,
        geom=geom,
        frame_id=frame_id,
        h5_name=args.h5.name,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
