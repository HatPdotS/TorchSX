""".stream parser for SFX data."""

from __future__ import annotations

import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata


# ---------------------------------------------------------------------------
# .stream parsing
# ---------------------------------------------------------------------------


def _is_valid_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _get_starting_positions(stream_file: str, n_workers: int) -> list[int]:
    """Divide file into n_workers byte segments, each starting at a chunk boundary."""
    with open(stream_file, "r") as f:
        f.seek(0, 2)
        file_size = f.tell()
        starting_points = range(0, file_size, file_size // n_workers)
        clean = []
        for sp in starting_points:
            f.seek(sp)
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("----- Begin chunk -----"):
                    clean.append(f.tell())
                    break
    return sorted(set(clean))


def _read_file_segment(args: tuple[str, int, int]) -> bytes:
    """Read a byte range from file."""
    file_path, start, end = args
    with open(file_path, "rb") as f:
        f.seek(start)
        return f.read(end - start)


def _read_file_in_segments(stream_file: str, n_workers: int) -> list[bytes]:
    """ThreadPool-based parallel file read."""
    positions = _get_starting_positions(stream_file, n_workers)
    with open(stream_file, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
    ends = positions[1:] + [file_size]
    args = [(stream_file, s, e) for s, e in zip(positions, ends)]

    if n_workers <= 1:
        return [_read_file_segment(a) for a in args]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_read_file_segment, args))


def _parse_vec_line(line: str) -> list[float]:
    """Parse 'astar = -0.5133608 -0.2970815 -0.4197558 nm^-1'."""
    parts = line.split("=", 1)[1].split()
    return [float(x) for x in parts if _is_valid_float(x)]


def _parse_segment(segment_bytes: bytes, panel_name_to_id: dict[str, int]) -> dict:
    """Parse a segment of stream file bytes into structured data.

    Returns dict with lists for each field, ready for tensor conversion.
    """
    # Accumulators
    obs = defaultdict(list)  # observation-level
    crys = defaultdict(list)  # crystal-level
    imgs = defaultdict(list)  # image-level
    meta_filenames = []
    meta_events = []

    crystal_idx = 0
    image_idx = 0

    lines = segment_bytes.decode("utf-8", errors="replace").splitlines()
    it = iter(lines)

    while True:
        # Find next chunk
        try:
            line = next(it)
        except StopIteration:
            break

        if not line.startswith("Image filename:"):
            # Skip until we find a chunk start (Image filename line)
            continue

        # --- Parse chunk (image) ---
        filename = line.split(":", 1)[1].strip()
        meta_filenames.append(filename)

        # Event line
        line = next(it)
        if line.startswith("Event:"):
            event_str = line.split(":", 1)[1].strip()
            meta_events.append(event_str)
        else:
            meta_events.append("")

        # Parse chunk key-value pairs until peak list
        chunk_meta = {}
        for line in it:
            if line.startswith("Peaks from peak search"):
                break
            if line.startswith("----- End chunk -----"):
                break
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().split()[0]
                try:
                    chunk_meta[key] = int(val)
                except ValueError:
                    try:
                        chunk_meta[key] = float(val)
                    except ValueError:
                        chunk_meta[key] = val

        # Image-level data
        beam_energy = chunk_meta.get("photon_energy_eV", 0.0)
        n_peaks = chunk_meta.get("num_peaks", 0)
        indexed_by = chunk_meta.get("indexed_by", "none")

        # average_camera_length is in metres in the stream
        avg_clen = chunk_meta.get("average_camera_length", 0.0)

        imgs["beam_energy"].append(float(beam_energy))
        imgs["n_peaks"].append(int(n_peaks))
        imgs["det_shift_x"].append(0.0)  # filled from crystal if available
        imgs["det_shift_y"].append(0.0)
        imgs["camera_length"].append(float(avg_clen))

        current_image_id = image_idx
        image_idx += 1
        crystal_in_image_counter = 0

        # Skip peak list
        for line in it:
            if line.startswith("End of peak list"):
                break

        # Parse crystals in this chunk
        for line in it:
            if line.startswith("----- End chunk -----"):
                break
            if line.startswith("--- Begin crystal"):
                # Cell parameters line
                line = next(it)
                cell_vals = [float(x) for x in line.split() if _is_valid_float(x)]
                # cell_vals = [a, b, c (nm), alpha, beta, gamma (deg)]
                # Convert nm → Å
                cell_a = cell_vals[0] * 10.0
                cell_b = cell_vals[1] * 10.0
                cell_c = cell_vals[2] * 10.0
                cell_alpha = cell_vals[3]
                cell_beta = cell_vals[4]
                cell_gamma = cell_vals[5]

                # Reciprocal vectors (nm^-1) → Å^-1 (÷10)
                line = next(it)
                astar = _parse_vec_line(line)
                line = next(it)
                bstar = _parse_vec_line(line)
                line = next(it)
                cstar = _parse_vec_line(line)
                # Convert nm^-1 → Å^-1
                astar = [v / 10.0 for v in astar]
                bstar = [v / 10.0 for v in bstar]
                cstar = [v / 10.0 for v in cstar]

                # Crystal key-value pairs
                crystal_meta = {}
                for line in it:
                    if line.startswith("Reflections measured after indexing"):
                        break
                    if "=" in line:
                        try:
                            key, _, val = line.partition("=")
                            key = key.strip()
                            val = val.strip()
                            # Handle "predict_refine/det_shift x = 0.000 y = 0.000 mm"
                            if "det_shift" in key and "y =" in val:
                                parts = val.split("y =")
                                crystal_meta["det_shift_x"] = float(parts[0].strip())
                                crystal_meta["det_shift_y"] = float(
                                    parts[1].strip().replace("mm", "").strip()
                                )
                            else:
                                first_val = val.split()[0]
                                try:
                                    crystal_meta[key] = float(first_val)
                                except ValueError:
                                    crystal_meta[key] = first_val
                        except Exception:
                            pass

                # Parse profile_radius (nm^-1 → Å^-1)
                profile_rad_nm = crystal_meta.get("profile_radius", 0.04)
                profile_rad = float(profile_rad_nm) / 10.0  # nm^-1 → Å^-1

                # Parse resolution (nm^-1 → Å)
                res_nm = crystal_meta.get("diffraction_resolution_limit", 0.0)
                resolution_A = 10.0 / float(res_nm) if float(res_nm) > 0 else 999.0

                # Update image det_shift from crystal
                if "det_shift_x" in crystal_meta:
                    imgs["det_shift_x"][-1] = crystal_meta["det_shift_x"]
                    imgs["det_shift_y"][-1] = crystal_meta["det_shift_y"]

                # Skip header line of reflection table
                line = next(it)

                # Parse reflections
                n_refl_count = 0
                for line in it:
                    if line.startswith("End of reflections"):
                        break
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    try:
                        h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                        I_val = float(parts[3])
                        sigma_val = float(parts[4])
                        peak_val = float(parts[5])
                        bg_val = float(parts[6])
                        fs_val = float(parts[7])
                        ss_val = float(parts[8])
                        panel_name = parts[9]
                        pid = panel_name_to_id.get(panel_name, 0)

                        obs["h"].append(h)
                        obs["k"].append(k)
                        obs["l"].append(l)
                        obs["I"].append(I_val)
                        obs["sigma"].append(sigma_val)
                        obs["peak"].append(peak_val)
                        obs["background"].append(bg_val)
                        obs["fs"].append(fs_val)
                        obs["ss"].append(ss_val)
                        obs["panel_id"].append(pid)
                        obs["crystal_id"].append(crystal_idx)
                        n_refl_count += 1
                    except (ValueError, IndexError):
                        continue

                # Store crystal data
                crys["cell"].append([cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma])
                crys["astar"].append(astar)
                crys["bstar"].append(bstar)
                crys["cstar"].append(cstar)
                crys["profile_rad"].append(profile_rad)
                crys["resolution"].append(resolution_A)
                crys["n_refl"].append(n_refl_count)
                crys["image_id"].append(current_image_id)
                crys["crystal_in_image"].append(crystal_in_image_counter)
                crystal_in_image_counter += 1

                crystal_idx += 1

                # Skip to "--- End crystal"
                for line in it:
                    if line.startswith("--- End crystal"):
                        break

    return {
        "obs": dict(obs),
        "crys": dict(crys),
        "imgs": dict(imgs),
        "filenames": meta_filenames,
        "events": meta_events,
        "n_crystals": crystal_idx,
        "n_images": image_idx,
    }


def parse_stream(
    path: str | Path,
    spacegroup: str = "P 1",
    n_workers: int = 1,
) -> _SFXStreamCore:
    """Parse a CrystFEL .stream file into stacked tensors.

    Parameters
    ----------
    path : path to .stream file
    spacegroup : space group string (stored as metadata)
    n_workers : number of parallel workers (default 1 for login node safety)

    Returns
    -------
    _SFXStreamCore with all tensors on CPU
    """
    path = str(path)

    # Parse geometry from file header
    geometry = Geometry.from_stream(path)
    panel_name_to_id = geometry.panel_name_to_id

    # Read file segments
    segments = _read_file_in_segments(path, max(n_workers, 1))

    # Parse segments
    if n_workers <= 1:
        results = [_parse_segment(seg, panel_name_to_id) for seg in segments]
    else:
        # For parallel: pass panel_name_to_id as part of args
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_parse_segment, seg, panel_name_to_id)
                for seg in segments
            ]
            results = [f.result() for f in futures]

    # Assemble results — need to offset crystal_id and image_id across segments
    all_obs = defaultdict(list)
    all_crys = defaultdict(list)
    all_imgs = defaultdict(list)
    all_filenames = []
    all_events = []

    crystal_offset = 0
    image_offset = 0

    for res in results:
        obs = res["obs"]
        crys = res["crys"]
        imgs = res["imgs"]

        # Offset crystal_id in observations
        if obs.get("crystal_id"):
            all_obs["crystal_id"].extend([c + crystal_offset for c in obs["crystal_id"]])
            for key in ("h", "k", "l", "I", "sigma", "peak", "background", "fs", "ss", "panel_id"):
                if key in obs:
                    all_obs[key].extend(obs[key])

        # Offset image_id in crystals
        if crys.get("image_id"):
            all_crys["image_id"].extend([i + image_offset for i in crys["image_id"]])
            for key in ("cell", "astar", "bstar", "cstar", "profile_rad", "resolution", "n_refl", "crystal_in_image"):
                if key in crys:
                    all_crys[key].extend(crys[key])

        # Images
        for key in ("beam_energy", "n_peaks", "det_shift_x", "det_shift_y", "camera_length"):
            if key in imgs:
                all_imgs[key].extend(imgs[key])

        all_filenames.extend(res["filenames"])
        all_events.extend(res["events"])

        crystal_offset += res["n_crystals"]
        image_offset += res["n_images"]

    n_obs = len(all_obs.get("h", []))
    n_crys = len(all_crys.get("cell", []))
    n_imgs = len(all_imgs.get("beam_energy", []))

    if n_obs == 0 or n_crys == 0:
        raise ValueError(f"No indexed crystals found in {path}")

    # Build tensors
    hkl = torch.tensor(
        list(zip(all_obs["h"], all_obs["k"], all_obs["l"])),
        dtype=torch.int16,
    )
    I = torch.tensor(all_obs["I"], dtype=torch.float32)
    sigma = torch.tensor(all_obs["sigma"], dtype=torch.float32)
    background = torch.tensor(all_obs["background"], dtype=torch.float32)
    peak_t = torch.tensor(all_obs["peak"], dtype=torch.float32)
    fs = torch.tensor(all_obs["fs"], dtype=torch.float32)
    ss = torch.tensor(all_obs["ss"], dtype=torch.float32)
    panel_id = torch.tensor(all_obs["panel_id"], dtype=torch.int16)
    crystal_id = torch.tensor(all_obs["crystal_id"], dtype=torch.int32)

    cell = torch.tensor(all_crys["cell"], dtype=torch.float32)
    astar = torch.tensor(all_crys["astar"], dtype=torch.float32)
    bstar = torch.tensor(all_crys["bstar"], dtype=torch.float32)
    cstar = torch.tensor(all_crys["cstar"], dtype=torch.float32)
    profile_rad = torch.tensor(all_crys["profile_rad"], dtype=torch.float32)
    resolution = torch.tensor(all_crys["resolution"], dtype=torch.float32)
    n_refl = torch.tensor(all_crys["n_refl"], dtype=torch.int32)
    image_id_t = torch.tensor(all_crys["image_id"], dtype=torch.int32)
    crystal_in_image = torch.tensor(all_crys["crystal_in_image"], dtype=torch.int32)

    beam_energy = torch.tensor(all_imgs["beam_energy"], dtype=torch.float32)
    det_shift_x = torch.tensor(all_imgs["det_shift_x"], dtype=torch.float32)
    det_shift_y = torch.tensor(all_imgs["det_shift_y"], dtype=torch.float32)
    n_peaks = torch.tensor(all_imgs["n_peaks"], dtype=torch.int32)
    camera_length = torch.tensor(all_imgs["camera_length"], dtype=torch.float32)

    # Warn on constant sigma
    if sigma.numel() > 0 and sigma.std() < 1e-6:
        warnings.warn(
            f"All sigma values are constant ({sigma[0].item():.2f}), "
            "likely set to photon energy. Consider replacing with proper estimates.",
            stacklevel=2,
        )

    # Build metadata
    metadata = StreamMetadata()
    metadata["image"] = {
        "filename": all_filenames,
        "event": all_events,
    }
    metadata["crystal"] = {
        "panel_names": geometry.panel_names,
    }

    core = _SFXStreamCore(
        hkl=hkl,
        I=I,
        sigma=sigma,
        background=background,
        peak=peak_t,
        fs=fs,
        ss=ss,
        panel_id=panel_id,
        crystal_id=crystal_id,
        astar=astar,
        bstar=bstar,
        cstar=cstar,
        cell=cell,
        profile_rad=profile_rad,
        resolution=resolution,
        n_refl=n_refl,
        image_id=image_id_t,
        crystal_in_image=crystal_in_image,
        beam_energy=beam_energy,
        det_shift_x=det_shift_x,
        det_shift_y=det_shift_y,
        n_peaks=n_peaks,
        camera_length=camera_length,
        geometry=geometry,
        metadata=metadata,
        spacegroup=spacegroup,
        source_path=str(path),
    )

    return core
