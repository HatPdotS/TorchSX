"""Pass-2 record parsing: line streaming, array allocation, direct writes.

Leaf module (numpy + torch only). This is the pure-Python twin of the native
parser (``native/src/parse.rs``): same state machine, same rules for malformed
input, same report. Keep the two in step; ``tests/test_native_parse.py``
checks they agree.

Malformed content is excluded at the smallest unit that keeps the rest valid,
and each exclusion is counted by reason (see :data:`REASONS`):

- a reflection line with fewer than ten fields, a non-numeric field or an hkl
  outside int16 is dropped;
- a crystal whose cell line has fewer than six numbers, whose reciprocal
  vector line has no ``=`` or fewer than three numbers, or that is cut off
  (chunk marker or end of input before ``End of reflections``) is dropped with
  its reflections;
- a non-numeric value of a key that is read keeps the default; a ``det_shift``
  line that does not parse is ignored;
- an ``Image filename:`` line with nothing after it, and a chunk without one,
  are skipped.

A chunk marker (``----- Begin chunk -----`` / ``----- End chunk -----``) ends
whatever image or crystal is open, so no chunk can swallow the next one, and
``--- Begin crystal`` starts a crystal even where a peak section was expected.
"""

from __future__ import annotations

import numpy as np
import torch

REASONS = (
    "reflection_short",
    "reflection_value",
    "reflection_hkl_range",
    "crystal_cell",
    "crystal_vector",
    "crystal_truncated",
    "image_truncated",
    "chunk_no_image",
    "key_value",
    "det_shift",
)

_BEGIN_CHUNK = "----- Begin chunk -----"
_END_CHUNK = "----- End chunk -----"

(_OUTER, _AFTER_FILENAME, _CHUNK_KV, _PEAKS, _CRYSTALS, _CELL, _VECTOR,
 _CRYSTAL_KV, _REFL_HEADER, _REFL, _SKIP_CRYSTAL) = range(11)


def _line_iter(f, end_pos: int | None):
    """Yield ``(offset, line)`` for the lines starting before ``end_pos``.

    Lines are decoded as UTF-8 (invalid bytes replaced) without the trailing
    ``\\r``/``\\n`` run.
    """
    pos = f.tell()
    while end_pos is None or pos < end_pos:
        b = f.readline()
        if not b:
            return
        yield pos, b.rstrip(b"\r\n").decode("utf-8", "replace")
        pos += len(b)


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------

_LEVEL_SHAPES = {
    "obs": {
        "hkl": ((3,), np.int16),
        "I": ((), np.float32),
        "sigma": ((), np.float32),
        "peak": ((), np.float32),
        "background": ((), np.float32),
        "fs": ((), np.float32),
        "ss": ((), np.float32),
        "panel_id": ((), np.int16),
        "crystal_id": ((), np.int32),
    },
    "crystal": {
        "cell": ((6,), np.float32),
        "astar": ((3,), np.float32),
        "bstar": ((3,), np.float32),
        "cstar": ((3,), np.float32),
        "profile_rad": ((), np.float32),
        "resolution": ((), np.float32),
        "n_refl": ((), np.int32),
        "image_id": ((), np.int32),
        "crystal_in_image": ((), np.int32),
    },
    "image": {
        "beam_energy": ((), np.float32),
        "det_shift_x": ((), np.float32),
        "det_shift_y": ((), np.float32),
        "camera_length": ((), np.float32),
        "n_peaks": ((), np.int32),
    },
}


def _allocate_arrays(n_obs: int, n_crys: int, n_imgs: int) -> dict[str, np.ndarray]:
    """Uninitialised column arrays; the counts are capacity hints."""
    sizes = {"obs": n_obs, "crystal": n_crys, "image": n_imgs}
    return {
        name: np.empty((sizes[level],) + shape, dtype)
        for level, fields in _LEVEL_SHAPES.items()
        for name, (shape, dtype) in fields.items()
    }


def _grow(v: dict[str, np.ndarray], level: str, need: int) -> None:
    """Enlarge every array of ``level`` in place in ``v`` to hold ``need`` rows."""
    for name in _LEVEL_SHAPES[level]:
        old = v[name]
        new = np.empty((max(need, 2 * len(old) + 16),) + old.shape[1:], old.dtype)
        new[: len(old)] = old
        v[name] = new


def _finalize(
    v: dict[str, np.ndarray],
    n_obs: int,
    n_crys: int,
    n_imgs: int,
) -> dict[str, torch.Tensor]:
    """Trim the arrays to the parsed counts and wrap them as tensors.

    A trimmed array is copied so the unused capacity is released; an exact
    one is shared with its tensor.
    """
    counts = {"obs": n_obs, "crystal": n_crys, "image": n_imgs}
    out = {}
    for level, fields in _LEVEL_SHAPES.items():
        n = counts[level]
        for name in fields:
            a = v[name]
            out[name] = torch.from_numpy(a if len(a) == n else a[:n].copy())
    return out


# ---------------------------------------------------------------------------
# Number parsing, as the stored values were read before (int(), float())
# ---------------------------------------------------------------------------


def _float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _number_f64(s: str) -> float | None:
    """``float(v)`` of a chunk value stored as ``int(v0)``, else ``float(v0)``."""
    try:
        i = int(s)
    except ValueError:
        return _float(s)
    try:
        return float(i)
    except OverflowError:
        return None


def _number_i32(s: str) -> int | None:
    """``int(v)`` of the same stored value, which must fit an int32."""
    try:
        i = int(s)
    except ValueError:
        f = _float(s)
        if f is None:
            return None
        try:
            i = int(f)
        except (ValueError, OverflowError):
            return None
    return i if -(2**31) <= i < 2**31 else None


def _report_add(report: dict[str, list[int]], reason: str, off: int) -> None:
    entry = report.get(reason)
    if entry is None:
        report[reason] = [1, off]
    else:
        entry[0] += 1
        if off < entry[1]:
            entry[1] = off


# ---------------------------------------------------------------------------
# Pass 2
# ---------------------------------------------------------------------------


def _parse_into_views(
    it,
    panel_name_to_id: dict[str, int],
    v: dict[str, np.ndarray],
    filenames: list[str],
    events: list[str],
    report: dict[str, list[int]],
    crystal_idx: int = 0,
    image_idx: int = 0,
    n_obs: int = 0,
) -> tuple[int, int, int]:
    """Parse ``(offset, line)`` pairs from ``it`` into the arrays of ``v``.

    ``v`` entries are replaced by larger arrays if they fill up. Indices
    ``(crystal_idx, image_idx, n_obs)`` are threaded in/out so separate byte
    ranges (sampled-crystal mode) can call this repeatedly; each range must
    start at a chunk. Exclusions are added to ``report`` as
    ``reason -> [count, first_offset]``.
    """
    st = _OUTER
    chunk_off = None
    chunk_has_image = False
    filename = event = ""
    filename_off = crystal_off = 0
    energy = npeaks = clen = prof = res = None
    det = None
    cur_image = crystal_in_image = obs_start = 0
    cell: list[float] = []
    vecs: list[list[float]] = []

    I_cap = len(v["I"])

    def commit_image():
        nonlocal image_idx, cur_image, crystal_in_image
        be = cl = 0.0
        npk = 0
        if energy is not None:
            be = _number_f64(energy[0])
            if be is None:
                _report_add(report, "key_value", energy[1])
                be = 0.0
        if npeaks is not None:
            npk = _number_i32(npeaks[0])
            if npk is None:
                _report_add(report, "key_value", npeaks[1])
                npk = 0
        if clen is not None:
            cl = _number_f64(clen[0])
            if cl is None:
                _report_add(report, "key_value", clen[1])
                cl = 0.0
        if image_idx >= len(v["beam_energy"]):
            _grow(v, "image", image_idx + 1)
        v["beam_energy"][image_idx] = be
        v["n_peaks"][image_idx] = npk
        v["det_shift_x"][image_idx] = 0.0
        v["det_shift_y"][image_idx] = 0.0
        v["camera_length"][image_idx] = cl
        filenames.append(filename)
        events.append(event)
        cur_image = image_idx
        image_idx += 1
        crystal_in_image = 0

    def commit_crystal():
        nonlocal crystal_idx, crystal_in_image
        p = 0.04
        r = 0.0
        if prof is not None:
            p = _float(prof[0])
            if p is None:
                _report_add(report, "key_value", prof[1])
                p = 0.04
        if res is not None:
            r = _float(res[0])
            if r is None:
                _report_add(report, "key_value", res[1])
                r = 0.0
        if crystal_idx >= len(v["n_refl"]):
            _grow(v, "crystal", crystal_idx + 1)
        row = v["cell"][crystal_idx]
        row[0] = cell[0] * 10.0
        row[1] = cell[1] * 10.0
        row[2] = cell[2] * 10.0
        row[3] = cell[3]
        row[4] = cell[4]
        row[5] = cell[5]
        for key, vec in zip(("astar", "bstar", "cstar"), vecs):
            row = v[key][crystal_idx]
            row[0] = vec[0] / 10.0
            row[1] = vec[1] / 10.0
            row[2] = vec[2] / 10.0
        v["profile_rad"][crystal_idx] = p / 10.0
        v["resolution"][crystal_idx] = 10.0 / r if r > 0 else 999.0
        v["n_refl"][crystal_idx] = n_obs - obs_start
        v["image_id"][crystal_idx] = cur_image
        v["crystal_in_image"][crystal_idx] = crystal_in_image
        crystal_in_image += 1
        if det is not None:
            v["det_shift_x"][cur_image] = det[0]
            v["det_shift_y"][cur_image] = det[1]
        crystal_idx += 1

    for off, line in it:
        marker = line.startswith(_BEGIN_CHUNK) or line.startswith(_END_CHUNK)
        while True:
            if st == _REFL:
                if line.startswith("End of reflections"):
                    commit_crystal()
                    st = _SKIP_CRYSTAL
                    break
                if marker:
                    n_obs = obs_start
                    _report_add(report, "crystal_truncated", crystal_off)
                    st = _OUTER
                    continue
                parts = line.split()
                if len(parts) < 10:
                    _report_add(report, "reflection_short", off)
                    break
                try:
                    h = int(parts[0]); k = int(parts[1]); l = int(parts[2])
                    I_val = float(parts[3]); sigma_val = float(parts[4])
                    peak_val = float(parts[5]); bg_val = float(parts[6])
                    fs_val = float(parts[7]); ss_val = float(parts[8])
                except ValueError:
                    _report_add(report, "reflection_value", off)
                    break
                if not (-32768 <= h <= 32767 and -32768 <= k <= 32767 and -32768 <= l <= 32767):
                    _report_add(report, "reflection_hkl_range", off)
                    break
                if n_obs >= I_cap:
                    _grow(v, "obs", n_obs + 1)
                    I_cap = len(v["I"])
                hkl = v["hkl"]
                hkl[n_obs, 0] = h
                hkl[n_obs, 1] = k
                hkl[n_obs, 2] = l
                v["I"][n_obs] = I_val
                v["sigma"][n_obs] = sigma_val
                v["peak"][n_obs] = peak_val
                v["background"][n_obs] = bg_val
                v["fs"][n_obs] = fs_val
                v["ss"][n_obs] = ss_val
                v["panel_id"][n_obs] = panel_name_to_id.get(parts[9], 0)
                v["crystal_id"][n_obs] = crystal_idx
                n_obs += 1
                break

            if st == _OUTER:
                if line.startswith(_BEGIN_CHUNK):
                    if chunk_off is not None and not chunk_has_image:
                        _report_add(report, "chunk_no_image", chunk_off)
                    chunk_off = off
                    chunk_has_image = False
                elif line.startswith("Image filename:"):
                    chunk_has_image = True
                    filename = line[len("Image filename:"):].strip()
                    filename_off = off
                    st = _AFTER_FILENAME
                break

            if st == _AFTER_FILENAME:
                if marker:
                    _report_add(report, "image_truncated", filename_off)
                    st = _OUTER
                    continue
                event = line[len("Event:"):].strip() if line.startswith("Event:") else ""
                energy = npeaks = clen = None
                st = _CHUNK_KV
                break

            if st == _CHUNK_KV:
                if line.startswith("Peaks from peak search"):
                    commit_image()
                    st = _PEAKS
                    break
                if marker:
                    commit_image()
                    st = _OUTER
                    continue
                if line.startswith("--- Begin crystal"):
                    # No peak section: the crystals still count.
                    commit_image()
                    st = _CRYSTALS
                    continue
                eq = line.find("=")
                if eq >= 0:
                    vparts = line[eq + 1:].split()
                    if vparts:
                        key = line[:eq].strip()
                        if key == "photon_energy_eV":
                            energy = (vparts[0], off)
                        elif key == "num_peaks":
                            npeaks = (vparts[0], off)
                        elif key == "average_camera_length":
                            clen = (vparts[0], off)
                break

            if st == _PEAKS:
                if marker:
                    st = _OUTER
                    continue
                if line.startswith("End of peak list"):
                    st = _CRYSTALS
                elif line.startswith("--- Begin crystal"):
                    st = _CRYSTALS
                    continue
                break

            if st == _CRYSTALS:
                if marker:
                    st = _OUTER
                    continue
                if line.startswith("--- Begin crystal"):
                    crystal_off = off
                    st = _CELL
                break

            if st == _CELL:
                if marker:
                    _report_add(report, "crystal_truncated", crystal_off)
                    st = _OUTER
                    continue
                cell = []
                for tok in line.split():
                    x = _float(tok)
                    if x is not None:
                        cell.append(x)
                        if len(cell) == 6:
                            break
                if len(cell) == 6:
                    vecs = []
                    st = _VECTOR
                else:
                    _report_add(report, "crystal_cell", off)
                    st = _SKIP_CRYSTAL
                break

            if st == _VECTOR:
                if marker:
                    _report_add(report, "crystal_truncated", crystal_off)
                    st = _OUTER
                    continue
                vec = []
                eq = line.find("=")
                if eq >= 0:
                    for tok in line[eq + 1:].split():
                        x = _float(tok)
                        if x is not None:
                            vec.append(x)
                            if len(vec) == 3:
                                break
                if len(vec) < 3:
                    _report_add(report, "crystal_vector", off)
                    st = _SKIP_CRYSTAL
                    break
                vecs.append(vec)
                if len(vecs) == 3:
                    prof = res = det = None
                    st = _CRYSTAL_KV
                break

            if st == _CRYSTAL_KV:
                if line.startswith("Reflections measured after indexing"):
                    st = _REFL_HEADER
                    break
                if marker:
                    _report_add(report, "crystal_truncated", crystal_off)
                    st = _OUTER
                    continue
                eq = line.find("=")
                if eq >= 0:
                    key = line[:eq].strip()
                    val = line[eq + 1:].strip()
                    if "det_shift" in key and "y =" in val:
                        parts = val.split("y =")
                        x = _float(parts[0].strip())
                        y = _float(parts[1].strip().replace("mm", "").strip())
                        if x is None or y is None:
                            _report_add(report, "det_shift", off)
                        else:
                            det = (x, y)
                    else:
                        vs = val.split()
                        if vs:
                            if key == "profile_radius":
                                prof = (vs[0], off)
                            elif key == "diffraction_resolution_limit":
                                res = (vs[0], off)
                break

            if st == _REFL_HEADER:
                if marker:
                    _report_add(report, "crystal_truncated", crystal_off)
                    st = _OUTER
                    continue
                obs_start = n_obs
                st = _REFL
                break

            # _SKIP_CRYSTAL: after a committed or rejected crystal, up to
            # "--- End crystal".
            if marker:
                st = _OUTER
                continue
            if line.startswith("--- End crystal"):
                st = _CRYSTALS
            break

    # End of input closes whatever is open, as a chunk marker would.
    if st == _AFTER_FILENAME:
        _report_add(report, "image_truncated", filename_off)
    elif st == _CHUNK_KV:
        commit_image()
    elif st in (_CELL, _VECTOR, _CRYSTAL_KV, _REFL_HEADER, _REFL):
        if st == _REFL:
            n_obs = obs_start
        _report_add(report, "crystal_truncated", crystal_off)
    if chunk_off is not None and not chunk_has_image:
        _report_add(report, "chunk_no_image", chunk_off)

    return crystal_idx, image_idx, n_obs
