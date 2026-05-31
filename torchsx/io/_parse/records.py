"""Pass-2 record parsing: line streaming, tensor allocation, direct writes.

Leaf module (numpy + torch only).
"""

from __future__ import annotations

import numpy as np
import torch


def _line_iter(f, end_pos: int | None):
    """Yield decoded lines from ``f`` up to (but not past) ``end_pos`` bytes."""
    while end_pos is None or f.tell() < end_pos:
        b = f.readline()
        if not b:
            return
        yield b.rstrip(b"\r\n").decode("utf-8", "replace")


def _is_valid_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_vec_line(line: str) -> list[float]:
    """Parse ``'astar = -0.513 -0.297 -0.420 nm^-1'`` → ``[-0.513, -0.297, -0.420]``."""
    parts = line.split("=", 1)[1].split()
    return [float(x) for x in parts if _is_valid_float(x)]


# ---------------------------------------------------------------------------
# Pass 2: preallocated tensors + direct writes through numpy views
# ---------------------------------------------------------------------------


def _allocate_tensors(n_obs: int, n_crys: int, n_imgs: int) -> dict[str, torch.Tensor]:
    return {
        # obs
        "hkl":        torch.empty((n_obs, 3), dtype=torch.int16),
        "I":          torch.empty(n_obs, dtype=torch.float32),
        "sigma":      torch.empty(n_obs, dtype=torch.float32),
        "peak":       torch.empty(n_obs, dtype=torch.float32),
        "background": torch.empty(n_obs, dtype=torch.float32),
        "fs":         torch.empty(n_obs, dtype=torch.float32),
        "ss":         torch.empty(n_obs, dtype=torch.float32),
        "panel_id":   torch.empty(n_obs, dtype=torch.int16),
        "crystal_id": torch.empty(n_obs, dtype=torch.int32),
        # crystals
        "cell":              torch.empty((n_crys, 6), dtype=torch.float32),
        "astar":             torch.empty((n_crys, 3), dtype=torch.float32),
        "bstar":             torch.empty((n_crys, 3), dtype=torch.float32),
        "cstar":             torch.empty((n_crys, 3), dtype=torch.float32),
        "profile_rad":       torch.empty(n_crys, dtype=torch.float32),
        "resolution":        torch.empty(n_crys, dtype=torch.float32),
        "n_refl":            torch.empty(n_crys, dtype=torch.int32),
        "image_id":          torch.empty(n_crys, dtype=torch.int32),
        "crystal_in_image":  torch.empty(n_crys, dtype=torch.int32),
        # images
        "beam_energy":   torch.empty(n_imgs, dtype=torch.float32),
        "det_shift_x":   torch.empty(n_imgs, dtype=torch.float32),
        "det_shift_y":   torch.empty(n_imgs, dtype=torch.float32),
        "camera_length": torch.empty(n_imgs, dtype=torch.float32),
        "n_peaks":       torch.empty(n_imgs, dtype=torch.int32),
    }


def _parse_into_views(
    it,
    panel_name_to_id: dict[str, int],
    v: dict[str, np.ndarray],
    filenames: list[str],
    events: list[str],
    crystal_idx: int = 0,
    image_idx: int = 0,
    n_obs: int = 0,
) -> tuple[int, int, int]:
    """Pass 2 core: write directly into preallocated numpy views ``v``.

    Indices ``(crystal_idx, image_idx, n_obs)`` are threaded in/out so
    non-contiguous byte ranges (sampled-crystal mode) can call this multiple
    times with the same views.
    """
    while True:
        try:
            line = next(it)
        except StopIteration:
            break

        if not line.startswith("Image filename:"):
            continue

        # --- chunk / image start ---
        filenames.append(line.split(":", 1)[1].strip())

        try:
            line = next(it)
        except StopIteration:
            break
        if line.startswith("Event:"):
            events.append(line.split(":", 1)[1].strip())
        else:
            events.append("")

        # Chunk-level key/value pairs
        chunk_meta: dict[str, object] = {}
        for line in it:
            if line.startswith("Peaks from peak search"):
                break
            if line.startswith("----- End chunk -----"):
                break
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                vparts = val.strip().split()
                if not vparts:
                    continue
                v0 = vparts[0]
                try:
                    chunk_meta[key] = int(v0)
                except ValueError:
                    try:
                        chunk_meta[key] = float(v0)
                    except ValueError:
                        chunk_meta[key] = v0

        v["beam_energy"][image_idx]   = float(chunk_meta.get("photon_energy_eV", 0.0))
        v["n_peaks"][image_idx]       = int(chunk_meta.get("num_peaks", 0))
        v["det_shift_x"][image_idx]   = 0.0  # may be overwritten by first crystal
        v["det_shift_y"][image_idx]   = 0.0
        v["camera_length"][image_idx] = float(chunk_meta.get("average_camera_length", 0.0))

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
            if not line.startswith("--- Begin crystal"):
                continue

            # Cell parameters (nm → Å on a,b,c; α,β,γ in degrees)
            try:
                line = next(it)
            except StopIteration:
                break
            cell_vals = [float(x) for x in line.split() if _is_valid_float(x)]
            cell_row = v["cell"][crystal_idx]
            cell_row[0] = cell_vals[0] * 10.0
            cell_row[1] = cell_vals[1] * 10.0
            cell_row[2] = cell_vals[2] * 10.0
            cell_row[3] = cell_vals[3]
            cell_row[4] = cell_vals[4]
            cell_row[5] = cell_vals[5]

            # Reciprocal vectors (nm⁻¹ → Å⁻¹)
            for key in ("astar", "bstar", "cstar"):
                line = next(it)
                vec = _parse_vec_line(line)
                row = v[key][crystal_idx]
                row[0] = vec[0] / 10.0
                row[1] = vec[1] / 10.0
                row[2] = vec[2] / 10.0

            # Crystal key/value pairs
            crystal_meta: dict[str, object] = {}
            for line in it:
                if line.startswith("Reflections measured after indexing"):
                    break
                if "=" not in line:
                    continue
                try:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
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

            profile_rad_nm = float(crystal_meta.get("profile_radius", 0.04))
            v["profile_rad"][crystal_idx] = profile_rad_nm / 10.0

            res_nm = float(crystal_meta.get("diffraction_resolution_limit", 0.0))
            v["resolution"][crystal_idx] = 10.0 / res_nm if res_nm > 0 else 999.0

            if "det_shift_x" in crystal_meta:
                v["det_shift_x"][current_image_id] = crystal_meta["det_shift_x"]
                v["det_shift_y"][current_image_id] = crystal_meta["det_shift_y"]

            # Skip reflection-table header
            try:
                next(it)
            except StopIteration:
                break

            # Reflections
            n_refl_count = 0
            hkl_view = v["hkl"]
            I_view = v["I"]
            sig_view = v["sigma"]
            peak_view = v["peak"]
            bg_view = v["background"]
            fs_view = v["fs"]
            ss_view = v["ss"]
            pid_view = v["panel_id"]
            cid_view = v["crystal_id"]

            for line in it:
                if line.startswith("End of reflections"):
                    break
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    h = int(parts[0]); k = int(parts[1]); l = int(parts[2])
                    I_val = float(parts[3]); sigma_val = float(parts[4])
                    peak_val = float(parts[5]); bg_val = float(parts[6])
                    fs_val = float(parts[7]); ss_val = float(parts[8])
                    pid = panel_name_to_id.get(parts[9], 0)
                except (ValueError, IndexError):
                    continue
                hkl_view[n_obs, 0] = h
                hkl_view[n_obs, 1] = k
                hkl_view[n_obs, 2] = l
                I_view[n_obs]   = I_val
                sig_view[n_obs] = sigma_val
                peak_view[n_obs] = peak_val
                bg_view[n_obs]  = bg_val
                fs_view[n_obs]  = fs_val
                ss_view[n_obs]  = ss_val
                pid_view[n_obs] = pid
                cid_view[n_obs] = crystal_idx
                n_refl_count += 1
                n_obs += 1

            v["n_refl"][crystal_idx]           = n_refl_count
            v["image_id"][crystal_idx]         = current_image_id
            v["crystal_in_image"][crystal_idx] = crystal_in_image_counter
            crystal_in_image_counter += 1
            crystal_idx += 1

            # Skip to End crystal
            for line in it:
                if line.startswith("--- End crystal"):
                    break

    return crystal_idx, image_idx, n_obs


_OBS_TRIM_FIELDS = (
    "hkl", "I", "sigma", "peak", "background", "fs", "ss",
    "panel_id", "crystal_id",
)
_CRYS_TRIM_FIELDS = (
    "cell", "astar", "bstar", "cstar",
    "profile_rad", "resolution", "n_refl", "image_id", "crystal_in_image",
)
_IMG_TRIM_FIELDS = (
    "beam_energy", "det_shift_x", "det_shift_y", "camera_length", "n_peaks",
)


def _finalize(
    tensors: dict[str, torch.Tensor],
    n_obs_actual: int,
    n_obs_alloc: int,
    n_crys_actual: int,
    n_crys_alloc: int,
    n_imgs_actual: int,
    n_imgs_alloc: int,
    filenames: list[str],
    events: list[str],
) -> dict:
    """Package tensors + metadata as a segment result.

    Any dimension where the pass-2 count came in below the pre-allocation
    (e.g. malformed chunks without ``Image filename:`` in the sampled path)
    is trimmed via contiguous ``clone`` so the returned tensor size matches
    reality.
    """
    if n_obs_actual != n_obs_alloc:
        for k in _OBS_TRIM_FIELDS:
            tensors[k] = tensors[k][:n_obs_actual].clone()
    if n_crys_actual != n_crys_alloc:
        for k in _CRYS_TRIM_FIELDS:
            tensors[k] = tensors[k][:n_crys_actual].clone()
    if n_imgs_actual != n_imgs_alloc:
        for k in _IMG_TRIM_FIELDS:
            tensors[k] = tensors[k][:n_imgs_actual].clone()

    return {
        **tensors,
        "filenames":  filenames,
        "events":     events,
        "n_crystals": n_crys_actual,
        "n_images":   n_imgs_actual,
        "n_obs":      n_obs_actual,
    }
