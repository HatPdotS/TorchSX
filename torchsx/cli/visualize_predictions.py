#!/usr/bin/env python
"""Visualize predicted spot positions on detector geometry.

Renders green rings at each predicted reflection position (2-sigma relp).
If an H5 file is provided, the raw detector frame is shown underneath.
Otherwise, only panel outlines and predicted spots are drawn.

When --frame is omitted, auto-picks a frame that has indexed crystal(s).
When --h5 is also omitted, picks a random crystal from the stream and
attempts to load its H5 frame from the path recorded in the stream metadata.

Usage:
  # Full: H5 frame + predictions
  python -m torchsx.cli.visualize_predictions \
      --sfx delay_-45.sfx --h5 acq0001.JF06T32V07.h5 --frame 42

  # Auto-pick frame with a crystal
  python -m torchsx.cli.visualize_predictions \
      --sfx delay_-45.sfx --h5 acq0001.JF06T32V07.h5

  # No H5 — spots on geometry outlines only
  python -m torchsx.cli.visualize_predictions --sfx delay_-45.sfx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from torchsx.geometry import Geometry


HC_EV_A = 12398.4  # hc in eV*Å


def _make_linlog_norm(vmin, vlin, vmax):
    """Create a Normalize that is linear [vmin, vlin] → [0, 0.5]
    and logarithmic (vlin, vmax] → (0.5, 1.0].
    """
    from matplotlib.colors import Normalize

    class LinLogNorm(Normalize):
        def __init__(self, vmin_, vlin_, vmax_):
            super().__init__(vmin=vmin_, vmax=vmax_, clip=True)
            self.vlin = float(vlin_)

        def __call__(self, value, clip=None):
            v = np.asarray(value, dtype=float)
            lin_range = max(self.vlin - self.vmin, 1e-12)
            log_range = max(np.log(self.vmax / self.vlin), 1e-12)
            result = np.where(
                v <= self.vlin,
                0.5 * (v - self.vmin) / lin_range,
                0.5 + 0.5 * np.log(np.clip(v, self.vlin, None) / self.vlin) / log_range,
            )
            return np.clip(result, 0, 1)

    return LinLogNorm(vmin, vlin, vmax)


def _image_filename_event(core, img_id: int) -> tuple[str, int]:
    """Return (filename, frame_index) for a given image_id."""
    from torchsx.io._excise import _parse_event_id

    meta = core.metadata
    filenames = meta.get("image", {}).get("filename", [])
    events = meta.get("image", {}).get("event", [])

    fn = filenames[img_id] if img_id < len(filenames) else ""
    if img_id < len(events) and events[img_id]:
        ev = _parse_event_id(events[img_id])
    else:
        ev = img_id
    return fn, ev


def _find_crystals_on_frame(
    core,
    h5_path: Path,
    frame_idx: int,
) -> list[int]:
    """Return crystal indices whose image matches the given H5 file and frame.

    The stream may contain duplicate image entries (from concatenated
    sub-streams).  We pick only the first matching image_id to avoid
    reporting phantom extra crystals.
    """
    from torchsx.io._excise import _parse_event_id

    meta = core.metadata
    filenames = meta.get("image", {}).get("filename", [])
    events = meta.get("image", {}).get("event", [])

    h5_name = h5_path.name

    # Find the first image_id matching this H5 file + frame
    first_image_id = None
    for img_id in range(len(filenames)):
        fn = filenames[img_id]
        if not fn.endswith(h5_name):
            continue
        if img_id < len(events) and events[img_id]:
            ev = _parse_event_id(events[img_id])
        else:
            ev = img_id
        if ev == frame_idx:
            first_image_id = img_id
            break

    if first_image_id is None:
        return []

    # Find crystals on this specific image_id only
    crystal_ids = []
    for c in range(core.n_crystals):
        if core.image_id[c].item() == first_image_id:
            crystal_ids.append(c)
    return crystal_ids


def _auto_pick_frame_for_h5(core, h5_path: Path) -> tuple[int, list[int]]:
    """Find a random frame in the H5 file that has indexed crystal(s).

    Returns (frame_idx, crystal_ids).
    Groups by image_id (not frame number) to avoid duplicates from
    concatenated sub-streams.
    """
    from torchsx.io._excise import _parse_event_id

    meta = core.metadata
    filenames = meta.get("image", {}).get("filename", [])
    events = meta.get("image", {}).get("event", [])
    h5_name = h5_path.name

    # Build image_id → frame_idx for images matching this H5 file
    image_to_frame: dict[int, int] = {}
    for img_id in range(len(filenames)):
        fn = filenames[img_id]
        if not fn.endswith(h5_name):
            continue
        if img_id < len(events) and events[img_id]:
            ev = _parse_event_id(events[img_id])
        else:
            ev = img_id
        image_to_frame[img_id] = ev

    if not image_to_frame:
        return -1, []

    # Group crystals by image_id (not frame) to avoid duplicates
    from collections import defaultdict
    image_crystals: dict[int, list[int]] = defaultdict(list)
    matching_imgs = set(image_to_frame.keys())
    for c in range(core.n_crystals):
        img_id = core.image_id[c].item()
        if img_id in matching_imgs:
            image_crystals[img_id].append(c)

    if not image_crystals:
        return -1, []

    # Pick a random image_id that has crystals
    import random
    img_id = random.choice(list(image_crystals.keys()))
    return image_to_frame[img_id], image_crystals[img_id]


def _pick_random_crystal(core) -> tuple[int, str, int]:
    """Pick a random crystal from the stream.

    Returns (crystal_id, h5_filename, frame_idx).
    """
    import random
    cid = random.randrange(core.n_crystals)
    img_id = core.image_id[cid].item()
    fn, ev = _image_filename_event(core, img_id)
    return cid, fn, ev


def _get_observed_spots(
    core,
    crystal_ids: list[int],
    min_radius_px: float = 3.0,
):
    """Get CrystFEL's observed (fs, ss) positions for the given crystals.

    Returns (fs, ss, pixel_radius) numpy arrays.
    The pixel_radius is the per-crystal profile_rad converted to pixels,
    floored at min_radius_px.
    """
    if not crystal_ids:
        return np.array([]), np.array([]), np.array([])

    all_fs = []
    all_ss = []
    all_radius = []

    for cid in crystal_ids:
        obs_mask = core.crystal_id == cid
        fs = core.fs[obs_mask].numpy()
        ss = core.ss[obs_mask].numpy()

        # Convert profile_rad (Å^-1) to pixel radius: δpx ≈ δs × λ × D × res
        img_id = core.image_id[cid].item()
        energy_eV = core.beam_energy[img_id].item()
        wavelength = HC_EV_A / energy_eV
        sigma = core.profile_rad[cid].item()
        geom = core.geometry
        clen = core.camera_length[img_id].item()
        D = clen if clen > 0 else geom.clen
        two_sigma = 2.0 * sigma
        pixel_radius = max(two_sigma * wavelength * D * geom.res, min_radius_px)
        radius_arr = np.full(fs.shape, pixel_radius)

        all_fs.append(fs)
        all_ss.append(ss)
        all_radius.append(radius_arr)

    return (
        np.concatenate(all_fs),
        np.concatenate(all_ss),
        np.concatenate(all_radius),
    )


def _predict_spots_for_crystals(
    core,
    crystal_ids: list[int],
    geom: Geometry,
    d_min: float = 1.5,
    sigma_cutoff: float = 3.0,
    min_radius_px: float = 3.0,
):
    """Predict reflections using our forward model via predict_reflections.

    Uses the stream's UB matrices (a*/b*/c*) and CrystFEL's profile_rad.
    Returns (fs, ss, pixel_radius) numpy arrays.
    """
    from torchsx.forward.coords import RefinableGeometry
    from torchsx.forward.predict import predict_reflections
    from torchsx.scaling._orientation import CrystalOrientation

    if not crystal_ids:
        return np.array([]), np.array([]), np.array([])

    # Gather per-crystal data
    wavelengths = []
    sigmas = []
    camera_lengths = []
    for cid in crystal_ids:
        img_id = core.image_id[cid].item()
        energy_eV = core.beam_energy[img_id].item()
        wavelengths.append(HC_EV_A / energy_eV)
        sigmas.append(core.profile_rad[cid].item())
        camera_lengths.append(core.camera_length[img_id].item())

    wavelength = float(np.median(wavelengths))
    median_sigma = float(np.median(sigmas))

    # Use per-image camera length if available, else fall back to geometry clen
    avg_clen = float(np.median(camera_lengths))
    D = avg_clen if avg_clen > 0 else geom.clen

    # Build UB for the selected crystals
    cid_tensor = torch.tensor(crystal_ids, dtype=torch.long)

    # Diagnostic: show per-crystal parameters
    for i, cid in enumerate(crystal_ids):
        img_id = core.image_id[cid].item()
        cell_i = core.cell[cid]
        res_i = core.resolution[cid].item()
        n_refl_i = core.n_refl[cid].item()
        print(f"  Crystal {cid} (image {img_id}):")
        print(f"    cell: {cell_i[0]:.2f} {cell_i[1]:.2f} {cell_i[2]:.2f} Å  "
              f"{cell_i[3]:.1f} {cell_i[4]:.1f} {cell_i[5]:.1f}°")
        print(f"    resolution: {res_i:.2f} Å  |  n_refl: {n_refl_i}  |  "
              f"profile_rad: {sigmas[i]:.4f} Å⁻¹")
        print(f"    wavelength: {wavelengths[i]:.4f} Å  |  "
              f"camera_length: {camera_lengths[i]:.4f} m")
    print(f"  Prediction params: d_min={d_min:.2f} Å  sigma_cutoff={sigma_cutoff}  "
          f"clen={D:.4f} m (geom: {geom.clen:.4f} m)")

    UB = torch.stack([
        core.astar[cid_tensor],
        core.bstar[cid_tensor],
        core.cstar[cid_tensor],
    ], dim=-1)  # (n_sel, 3, 3)

    orient = CrystalOrientation(UB)
    ref_geom = RefinableGeometry(geom)

    # Override clen with per-image camera length from predict_refine
    if D != geom.clen:
        ref_geom._clen.fill_(D)

    pred = predict_reflections(
        orient, ref_geom,
        cell=core.cell[cid_tensor],
        d_min=d_min,
        wavelength=wavelength,
        sigma_cutoff=sigma_cutoff,
        relp_sigma=median_sigma,
    )

    n_pred = pred.fs.shape[0]
    if n_pred == 0:
        return np.array([]), np.array([]), np.array([])

    fs_out = pred.fs.numpy()
    ss_out = pred.ss.numpy()

    # Per-spot pixel radius from the crystal's profile_rad
    two_sigma = 2.0 * median_sigma
    pixel_radius = max(two_sigma * wavelength * D * geom.res, min_radius_px)
    radius_out = np.full(n_pred, pixel_radius)

    return fs_out, ss_out, radius_out


def plot_predictions(
    frame_data: np.ndarray | None,
    good_mask: np.ndarray | None,
    geom: Geometry,
    crystfel_fs: np.ndarray,
    crystfel_ss: np.ndarray,
    crystfel_radius_px: np.ndarray,
    our_fs: np.ndarray,
    our_ss: np.ndarray,
    our_radius_px: np.ndarray,
    frame_id: int,
    h5_name: str,
    output_path: Path,
    n_crystals: int,
):
    """Render detector frame with two layers of spot rings.

    Green rings = CrystFEL indexed spots (observed fs/ss from stream).
    Red rings   = our forward-model predictions (UB → recip_to_pixel).

    Color scale: linear from 0 to the 90th percentile of good pixels,
    then logarithmic above that.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    panels = geom.panels
    res = geom.res
    n_crystfel = len(crystfel_fs)
    n_ours = len(our_fs)
    has_frame = frame_data is not None and good_mask is not None

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle(
        f"{h5_name}  |  frame {frame_id}  |  "
        f"{n_crystals} crystal(s)  |  "
        f"green={n_crystfel} CrystFEL  red={n_ours} ours",
        fontsize=11,
    )

    if has_frame:
        good_vals = frame_data[good_mask].astype(np.float32)
        max_val = good_vals.max() if good_vals.size > 0 else 1
        pos_vals = good_vals[good_vals > 0]
        p90 = max(float(np.percentile(pos_vals, 90)) if pos_vals.size > 0 else 1.0, 1.0)
        norm = _make_linlog_norm(vmin=0, vlin=p90, vmax=max(max_val, p90 * 2))

        for panel, (ss0, ss1, fs0, fs1) in zip(panels, geom.panel_slices):
            panel_good = good_mask[ss0:ss1, fs0:fs1]
            panel_data = frame_data[ss0:ss1, fs0:fs1].astype(np.float32)
            panel_data[~panel_good] = np.nan

            corners = panel.corners_mm(res)
            x0, y0 = corners[0]
            x1, y1 = corners[2]
            ext = [y0, y1, x0, x1]

            ax.imshow(
                panel_data, origin="lower",
                extent=ext,
                aspect="equal", cmap="viridis",
                norm=norm,
                interpolation="nearest",
            )

    # Panel outlines
    outline_color = "white" if has_frame else "gray"
    for panel in panels:
        corners = panel.corners_mm(res)
        poly = Polygon(corners[:, ::-1], closed=True, fill=False,
                       edgecolor=outline_color, linewidth=0.3, alpha=0.7)
        ax.add_patch(poly)

    mm_per_pixel = 1000.0 / res
    has_circles = False

    def _add_rings(fs_arr, ss_arr, radius_px_arr, color):
        nonlocal has_circles
        if len(fs_arr) == 0:
            return
        x_mm, y_mm = geom.pixel_to_mm(
            ss_arr.astype(np.float64),
            fs_arr.astype(np.float64),
        )
        radius_mm = radius_px_arr * mm_per_pixel
        for i in range(len(fs_arr)):
            circle = Circle(
                (y_mm[i], x_mm[i]),
                radius_mm[i],
                fill=False,
                edgecolor=color,
                linewidth=1.0,  # scaled below
                alpha=0.8,
            )
            ax.add_patch(circle)
        has_circles = True

    # Draw our predictions first (red, behind), CrystFEL on top (green)
    _add_rings(our_fs, our_ss, our_radius_px, "red")
    _add_rings(crystfel_fs, crystfel_ss, crystfel_radius_px, "lime")

    # Axis limits
    all_corners = geom.all_corners_mm()
    margin = 5
    xlim = (all_corners[:, 1].min() - margin, all_corners[:, 1].max() + margin)
    ylim = (all_corners[:, 0].min() - margin, all_corners[:, 0].max() + margin)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_aspect("equal")
    if has_frame:
        ax.set_title(f"Lin-log (p90={p90:.0f}) | green=CrystFEL | red=ours (2σ relp)")
        ax.set_facecolor("black")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Counts", shrink=0.8)
    else:
        ax.set_title("green=CrystFEL | red=ours (2σ relp)")
        ax.set_facecolor("black")

    fig.tight_layout()

    # Compute DPI so each detector pixel gets at least 1 output pixel.
    total_ss, total_fs = geom.detector_shape
    max_det_px = max(total_ss, total_fs)
    fig_inches = 12
    axis_frac = 0.80
    min_dpi = int(np.ceil(max_det_px / (fig_inches * axis_frac)))
    dpi = max(min_dpi, 150)

    # Set ring linewidth to 1 detector pixel in the output.
    if has_circles:
        data_range_mm = xlim[1] - xlim[0]
        output_px = fig_inches * axis_frac * dpi
        mm_per_output_px = data_range_mm / output_px
        one_det_pixel_mm = 1000.0 / res
        lw_display_px = one_det_pixel_mm / mm_per_output_px
        lw_pt = max(lw_display_px * 72.0 / dpi, 0.01)
        for p in ax.patches:
            if isinstance(p, Circle):
                p.set_linewidth(lw_pt)

    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}  (dpi={dpi})")


def _load_h5_frame(
    h5_path: Path, frame_idx: int,
    data_key: str | None, mask_key: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a single frame + pixel mask from an H5 file."""
    import bitshuffle.h5  # noqa: F401
    import h5py

    f = h5py.File(h5_path, "r")
    if data_key is None or mask_key is None:
        from torchsx.cli.sparse_compress import find_data_mask_keys
        dk, mk = find_data_mask_keys(f)
        data_key = data_key or dk
        mask_key = mask_key or mk

    dset = f[data_key]
    n_frames = dset.shape[0]
    if frame_idx >= n_frames:
        f.close()
        raise ValueError(f"Frame {frame_idx} out of range (0–{n_frames-1})")

    frame_data = dset[frame_idx]
    good_mask = np.asarray(f[mask_key][()], dtype=bool)
    f.close()
    return frame_data, good_mask


def main():
    parser = argparse.ArgumentParser(
        description="Visualize predicted spot positions on detector geometry."
    )
    parser.add_argument("--sfx", type=Path, default=None,
                        help="Path to .sfx file (pre-parsed stream)")
    parser.add_argument("--stream", type=Path, default=None,
                        help="CrystFEL .stream file (alternative to --sfx)")
    parser.add_argument("--h5", type=Path, default=None,
                        help="Input HDF5 file (optional — omit for spots-only)")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame index (auto-picked if omitted)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (PNG/SVG/PDF)")
    parser.add_argument("--data-key", type=str, default=None,
                        help="HDF5 dataset path for frames")
    parser.add_argument("--mask-key", type=str, default=None,
                        help="HDF5 dataset path for pixel mask")
    parser.add_argument("--d-min", type=float, default=1.5,
                        help="Resolution cutoff for our predictions in Angstroms (default: 1.5)")
    parser.add_argument("--min-radius", type=float, default=3.0,
                        help="Minimum circle radius in pixels (default: 3)")

    args = parser.parse_args()

    if args.sfx is None and args.stream is None:
        parser.error("Either --sfx or --stream is required")

    # Load stream/sfx data
    from torchsx import SFXStream

    if args.sfx is not None:
        print(f"Loading {args.sfx.name}...")
        stream = SFXStream.load(args.sfx, merge_friedel=False)
    else:
        print(f"Parsing {args.stream.name}...")
        stream = SFXStream.from_stream(args.stream, merge_friedel=False)

    core = stream._core
    geom = core.geometry
    print(f"  {core.n_crystals:,} crystals, {core.n_observations:,} observations")

    # ── Resolve crystal_ids, h5_path, frame_id ──────────────────────────
    frame_data = None
    good_mask = None
    h5_path = args.h5
    frame_id = args.frame
    crystal_ids: list[int] = []

    if h5_path is not None:
        # H5 file provided
        if frame_id is not None:
            # Both H5 and frame specified — find crystals on that frame
            crystal_ids = _find_crystals_on_frame(core, h5_path, frame_id)
            print(f"Frame {frame_id}: {len(crystal_ids)} crystal(s)")
        else:
            # H5 provided but no frame — auto-pick a frame with crystals
            frame_id, crystal_ids = _auto_pick_frame_for_h5(core, h5_path)
            if frame_id < 0:
                print(f"WARNING: no indexed crystals found for {h5_path.name}")
                frame_id = 0
            else:
                n_refl = sum(core.n_refl[c].item() for c in crystal_ids)
                print(f"Auto-picked frame {frame_id}: "
                      f"{len(crystal_ids)} crystal(s), {n_refl} reflections")

        # Load the H5 frame
        print(f"Reading {h5_path.name} frame {frame_id}...")
        frame_data, good_mask = _load_h5_frame(
            h5_path, frame_id, args.data_key, args.mask_key,
        )
    else:
        # No H5 — pick a random crystal, try to load its H5
        cid, h5_filename, frame_id = _pick_random_crystal(core)
        crystal_ids = [cid]
        n_refl = core.n_refl[cid].item()
        print(f"Auto-picked crystal {cid} ({n_refl} reflections)")
        print(f"  source: {h5_filename}  frame {frame_id}")

        # Try to find and open the H5 file
        h5_candidate = Path(h5_filename)
        if h5_candidate.exists():
            h5_path = h5_candidate
        else:
            # Try relative to stream/sfx location
            base = args.sfx or args.stream
            h5_candidate = base.parent / h5_candidate.name
            if h5_candidate.exists():
                h5_path = h5_candidate

        if h5_path is not None and h5_path.exists():
            print(f"Reading {h5_path.name} frame {frame_id}...")
            try:
                frame_data, good_mask = _load_h5_frame(
                    h5_path, frame_id, args.data_key, args.mask_key,
                )
                # Also find any other crystals on this frame
                crystal_ids = _find_crystals_on_frame(core, h5_path, frame_id)
                if cid not in crystal_ids:
                    crystal_ids.append(cid)
            except Exception as e:
                print(f"  Could not load frame: {e}")
                h5_path = None
        else:
            print(f"  H5 file not found — showing spots only")
            h5_path = None

    # ── Gather spots ──────────────────────────────────────────────────
    empty = np.array([])
    if crystal_ids:
        n_refl_total = sum(core.n_refl[c].item() for c in crystal_ids)
        print(f"{len(crystal_ids)} crystal(s), {n_refl_total} CrystFEL reflections")

        # CrystFEL observed spots (green)
        cf_fs, cf_ss, cf_rad = _get_observed_spots(
            core, crystal_ids, min_radius_px=args.min_radius,
        )
        print(f"  CrystFEL: {len(cf_fs)} spots")

        # Our forward-model predictions (red)
        our_fs, our_ss, our_rad = _predict_spots_for_crystals(
            core, crystal_ids, geom,
            d_min=args.d_min, min_radius_px=args.min_radius,
        )
        print(f"  Ours:     {len(our_fs)} spots")
    else:
        cf_fs = cf_ss = cf_rad = empty
        our_fs = our_ss = our_rad = empty
        print("No crystals found — nothing to display")

    # ── Plot ────────────────────────────────────────────────────────────
    h5_label = h5_path.name if h5_path is not None else "(no H5)"
    output_path = args.output or Path(f"predictions_frame_{frame_id}.png")

    plot_predictions(
        frame_data=frame_data,
        good_mask=good_mask,
        geom=geom,
        crystfel_fs=cf_fs,
        crystfel_ss=cf_ss,
        crystfel_radius_px=cf_rad,
        our_fs=our_fs,
        our_ss=our_ss,
        our_radius_px=our_rad,
        frame_id=frame_id,
        h5_name=h5_label,
        output_path=output_path,
        n_crystals=len(crystal_ids),
    )


if __name__ == "__main__":
    main()
