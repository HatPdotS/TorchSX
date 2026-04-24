""".stream parser for SFX data.

Two-pass, memory-bounded design:

1. **Pass 1 — byte-level count.** Scan the file (or per-worker byte range)
   counting images, crystals, and reflection lines. No decode, no parse:
   just ``startswith(b"...")`` checks on raw bytes.
2. **Pass 2 — parse into preallocated tensors.** Allocate final-size
   :class:`torch.Tensor` objects up front, obtain ``numpy`` views of them,
   and have the parser write directly into those views by integer index.

There are no ``array.array`` accumulators and no list-of-lists
intermediaries: the peak memory footprint is essentially the final tensor
payload plus a small transient (current-line string, chunk kv-dict, etc.).

Multi-worker: each worker runs its own two-pass over its byte range (local
OS page cache keeps the second pass hot) and returns final tensors for its
segment; the main process ``torch.cat``s them.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp  # registers shared-memory reducers

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata


# ---------------------------------------------------------------------------
# Chunk indexing (byte offsets for multi-worker split)
# ---------------------------------------------------------------------------


def _index_chunks(path: str) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Scan a stream file and return (header_end, chunks).

    ``chunks`` is a list of
    ``(start_offset, end_offset, n_crystals, n_obs_upper)`` per
    ``----- Begin chunk -----`` block. ``n_obs_upper`` is an upper bound on
    the number of reflection lines — it matches the structural check used by
    the parser (``len(line.split()) >= 10``).

    Single file pass; O(n_chunks) memory.
    """
    B_BEGIN_CHUNK = b"----- Begin chunk -----"
    B_BEGIN_CRY = b"--- Begin crystal"
    B_REFL_BEGIN = b"Reflections measured after indexing"
    B_REFL_END = b"End of reflections"

    header_end: int | None = None
    chunks: list[tuple[int, int, int, int]] = []
    current_start: int | None = None
    current_crystals = 0
    current_obs = 0
    in_refl = False

    with open(path, "rb", buffering=1 << 20) as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if in_refl:
                if line.startswith(B_REFL_END):
                    in_refl = False
                elif len(line.split()) >= 10:
                    current_obs += 1
                continue
            c0 = line[:1]
            if c0 == b"-":
                if line.startswith(B_BEGIN_CHUNK):
                    if current_start is None:
                        header_end = pos
                    else:
                        chunks.append(
                            (current_start, pos, current_crystals, current_obs)
                        )
                    current_start = pos
                    current_crystals = 0
                    current_obs = 0
                elif line.startswith(B_BEGIN_CRY):
                    current_crystals += 1
            elif c0 == b"R" and line.startswith(B_REFL_BEGIN):
                in_refl = True
                f.readline()  # skip reflection-table header
        if current_start is not None:
            end_pos = f.tell()
            chunks.append((current_start, end_pos, current_crystals, current_obs))

    if header_end is None:
        raise ValueError(f"No chunks found in {path}")
    return header_end, chunks


def _find_first_chunk_offset(path: str) -> int:
    """Locate the first ``----- Begin chunk -----`` byte offset."""
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                raise ValueError(f"No chunks found in {path}")
            if line.startswith(b"----- Begin chunk -----"):
                return pos


def _find_chunk_at_or_after(path: str, target: int) -> int:
    """Return the byte offset of the first ``----- Begin chunk -----`` at or
    after ``target``; EOF byte offset if none.

    O(1) lines read in practice (scans from ``target`` forward until the next
    begin-chunk marker), since chunks are typically a few KB – few MB.
    """
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(target)
        if target > 0:
            # May have landed mid-line; discard to reach the next line start.
            f.readline()
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return pos  # EOF
            if line.startswith(b"----- Begin chunk -----"):
                return pos


def _coarse_chunk_split(path: str, n_workers: int) -> list[tuple[int, int]]:
    """Divide the file into ``n_workers`` roughly equal byte ranges aligned
    on ``----- Begin chunk -----`` boundaries.

    Much faster than :func:`_index_chunks` because it only needs to locate
    ``n_workers`` boundaries, not every chunk. ``n_workers`` short scans,
    each finishing within at most one chunk length.
    """
    if n_workers < 1:
        n_workers = 1
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()

    header_end = _find_first_chunk_offset(path)

    # Start of each worker's range: chunk-aligned.
    starts: list[int] = [header_end]
    for i in range(1, n_workers):
        target = header_end + (file_size - header_end) * i // n_workers
        s = _find_chunk_at_or_after(path, target)
        if s > starts[-1]:
            starts.append(s)
    # Build (start, end) ranges; last end = file size.
    ends = starts[1:] + [file_size]
    return list(zip(starts, ends))


def _split_work(
    chunks: list[tuple[int, int, int, int]], n_workers: int
) -> list[tuple[int, int, int, int, int]]:
    """Partition chunk list into ``n_workers`` contiguous ranges.

    Returns per-worker ``(start_byte, end_byte, n_obs, n_crys, n_imgs)``
    aggregates. Balanced by obs count (the dominant term for memory). Each
    range spans complete chunks, so workers never split a chunk.
    """
    if not chunks:
        return []

    def _aggregate(sub):
        n_cry = sum(c[2] for c in sub)
        n_obs = sum(c[3] for c in sub)
        n_imgs = len(sub)
        return (sub[0][0], sub[-1][1], n_obs, n_cry, n_imgs)

    if n_workers <= 1 or len(chunks) == 1:
        return [_aggregate(chunks)]

    total_work = sum(c[3] or 1 for c in chunks)
    target = total_work / n_workers
    ranges: list[tuple[int, int, int, int, int]] = []
    start_idx = 0
    acc = 0
    for i, c in enumerate(chunks):
        acc += c[3] or 1
        if (
            acc >= target
            and len(ranges) < n_workers - 1
            and i < len(chunks) - 1
        ):
            ranges.append(_aggregate(chunks[start_idx : i + 1]))
            start_idx = i + 1
            acc = 0
    ranges.append(_aggregate(chunks[start_idx:]))
    return ranges


# ---------------------------------------------------------------------------
# Pass 1: byte-level counting
# ---------------------------------------------------------------------------


def _count_byte_range(path: str, start: int, end: int) -> tuple[int, int, int]:
    """Count (n_obs_upper, n_crystals, n_images) in ``[start, end)``.

    Pure byte-level scan — no decode, no parse. Very fast (usually IO-bound).

    ``n_obs_upper`` may overcount by the handful of reflection lines the
    second pass skips due to ``len(parts) < 10`` or a parse failure. Callers
    preallocate at this upper bound and slice to the actual count at the end.
    """
    n_obs = 0
    n_crys = 0
    n_imgs = 0
    in_refl = False

    B_IMG = b"Image filename:"
    B_BEGIN_CRY = b"--- Begin crystal"
    B_REFL_BEGIN = b"Reflections measured after indexing"
    B_REFL_END = b"End of reflections"

    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            if in_refl:
                if line.startswith(B_REFL_END):
                    in_refl = False
                else:
                    n_obs += 1
            else:
                # Fast single-byte gate first.
                c0 = line[:1]
                if c0 == b"-":
                    if line.startswith(B_BEGIN_CRY):
                        n_crys += 1
                elif c0 == b"I":
                    if line.startswith(B_IMG):
                        n_imgs += 1
                elif c0 == b"R":
                    if line.startswith(B_REFL_BEGIN):
                        in_refl = True
                        f.readline()  # skip reflection-table header line
    return n_obs, n_crys, n_imgs


# ---------------------------------------------------------------------------
# Line streaming (used by pass 2)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Worker entry points
# ---------------------------------------------------------------------------


def _parse_byte_range(
    path: str,
    start: int,
    end: int,
    panel_name_to_id: dict[str, int],
) -> dict:
    """Two-pass parse of ``[start, end)`` into tensors.

    Pass 1 counts; pass 2 writes directly into preallocated tensors through
    numpy views.
    """
    n_obs_alloc, n_crys, n_imgs = _count_byte_range(path, start, end)
    tensors = _allocate_tensors(n_obs_alloc, n_crys, n_imgs)
    views = {k: t.numpy() for k, t in tensors.items()}

    filenames: list[str] = []
    events: list[str] = []
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        it = _line_iter(f, end)
        n_cry_got, n_img_got, n_obs_got = _parse_into_views(
            it, panel_name_to_id, views, filenames, events
        )

    # Pass 1 counts structural markers; pass 2 counts what it could actually
    # parse. ``_finalize`` trims if any dimension over-allocated (e.g. a
    # malformed chunk with no ``Image filename:``).
    return _finalize(
        tensors, n_obs_got, n_obs_alloc, n_cry_got, n_crys, n_img_got, n_imgs,
        filenames, events,
    )


def _parse_byte_range_args(args: tuple) -> dict:
    """Trampoline for ProcessPoolExecutor (pickles a plain tuple)."""
    return _parse_byte_range(*args)


def _parse_into_shared(
    path: str,
    start: int,
    end: int,
    panel_name_to_id: dict[str, int],
    shared_tensors: dict[str, torch.Tensor],
    obs_offset: int,
    crys_offset: int,
    img_offset: int,
) -> tuple[int, int, int, list[str], list[str]]:
    """Worker: parse ``[start, end)`` and write directly into shared tensors.

    The shared tensors are pre-allocated at the *global* final size by the
    main process and placed in shared memory via ``torch.Tensor.share_memory_``.
    We get ``numpy`` views of them (which share storage) and pass those to
    the same ``_parse_into_views`` used by the serial path, seeded with the
    global offsets so writes land in this worker's slice.
    """
    views = {k: t.numpy() for k, t in shared_tensors.items()}
    filenames: list[str] = []
    events: list[str] = []
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        it = _line_iter(f, end)
        crys_end, img_end, obs_end = _parse_into_views(
            it, panel_name_to_id, views, filenames, events,
            crystal_idx=crys_offset, image_idx=img_offset, n_obs=obs_offset,
        )
    # Report segment-local counts so the main process can validate.
    return (
        obs_end - obs_offset,
        crys_end - crys_offset,
        img_end - img_offset,
        filenames,
        events,
    )


def _parse_into_shared_args(args: tuple) -> tuple:
    return _parse_into_shared(*args)


def _count_byte_range_args(args: tuple) -> tuple[int, int, int]:
    """Trampoline for the parallel count phase."""
    path, start, end = args
    return _count_byte_range(path, start, end)


# ---------------------------------------------------------------------------
# Native (C++) worker trampolines — used when the JIT-compiled extension
# loads successfully. Each worker process loads the extension the first time
# it's called; the compiled binary is cached under
# ``~/.cache/torch_extensions``.
# ---------------------------------------------------------------------------


def _native_count_args(args: tuple) -> tuple[int, int, int]:
    from torchsx.io import _native
    mod = _native.load()
    if mod is None:
        return _count_byte_range(*args)
    path, start, end = args
    return tuple(mod.count_byte_range(path, int(start), int(end)))


def _native_parse_args(args: tuple) -> tuple:
    from torchsx.io import _native
    mod = _native.load()
    path, s, e, panel_name_to_id, tensors, obs_off, crys_off, img_off = args
    if mod is None:
        return _parse_into_shared(
            path, s, e, panel_name_to_id, tensors, obs_off, crys_off, img_off
        )
    res = mod.parse_into_shared(
        path, int(s), int(e), panel_name_to_id,
        tensors["hkl"], tensors["I"], tensors["sigma"], tensors["peak"],
        tensors["background"], tensors["fs"], tensors["ss"], tensors["panel_id"],
        tensors["crystal_id"],
        tensors["cell"], tensors["astar"], tensors["bstar"], tensors["cstar"],
        tensors["profile_rad"], tensors["resolution"], tensors["n_refl"],
        tensors["image_id"], tensors["crystal_in_image"],
        tensors["beam_energy"], tensors["det_shift_x"], tensors["det_shift_y"],
        tensors["camera_length"], tensors["n_peaks"],
        int(obs_off), int(crys_off), int(img_off),
    )
    return tuple(res)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_stream(
    path: str | Path,
    spacegroup: str = "P 1",
    n_workers: int = 1,
    max_crystals: int | None = None,
    seed: int = 0,
) -> _SFXStreamCore:
    """Parse a CrystFEL .stream file into stacked tensors.

    Parameters
    ----------
    path : path to .stream file
    spacegroup : space group string (stored as metadata)
    n_workers : number of parallel worker processes. ``1`` parses serially.
    max_crystals : if set, parse a random subset containing at most this many
        crystals. Serial parse; ``n_workers`` ignored.
    seed : RNG seed for chunk sampling (only used when ``max_crystals`` is set).
    """
    path = str(path)

    if max_crystals is not None:
        return _parse_stream_sampled(path, spacegroup, int(max_crystals), int(seed))

    geometry = Geometry.from_stream(path)
    panel_name_to_id = geometry.panel_name_to_id

    if n_workers <= 1:
        header_end = _find_first_chunk_offset(path)
        with open(path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()

        # Try the native worker in-process (no multiprocessing): pass regular
        # (non-shared) tensors; the C++ code doesn't care about shared_memory
        # state and writes through raw data_ptr regardless.
        from torchsx.io import _native
        native_mod = _native.load()
        if native_mod is not None:
            n_obs, n_crys, n_imgs = native_mod.count_byte_range(
                path, int(header_end), int(file_size)
            )
            tensors = _allocate_tensors(n_obs, n_crys, n_imgs)
            res = native_mod.parse_into_shared(
                path, int(header_end), int(file_size), panel_name_to_id,
                tensors["hkl"], tensors["I"], tensors["sigma"], tensors["peak"],
                tensors["background"], tensors["fs"], tensors["ss"],
                tensors["panel_id"], tensors["crystal_id"],
                tensors["cell"], tensors["astar"], tensors["bstar"],
                tensors["cstar"], tensors["profile_rad"], tensors["resolution"],
                tensors["n_refl"], tensors["image_id"],
                tensors["crystal_in_image"],
                tensors["beam_energy"], tensors["det_shift_x"],
                tensors["det_shift_y"], tensors["camera_length"],
                tensors["n_peaks"],
                0, 0, 0,
            )
            _, _, _, filenames, events = tuple(res)
            return _finalize_shared(
                tensors, list(filenames), list(events), geometry, spacegroup, path
            )

        # Pure-Python fallback
        result = _parse_byte_range(path, header_end, file_size, panel_name_to_id)
        return _assemble_core([result], path, spacegroup, geometry)

    # Multi-worker: fully parallel pipeline with a single long-lived pool.
    # Stage A — main does a tiny O(n_workers) coarse split to find chunk-
    # aligned byte ranges (no full file scan).
    # Stage B — workers count obs/crys/imgs in their range (parallel).
    # Main — allocate shared tensors at final size, compute offsets.
    # Stage C — workers parse directly into shared tensor slices (parallel).
    # Workers are reused across B and C (single spawn round).
    #
    # "spawn" start method — fork-after-threads (torch's internal thread
    # pool) deadlocks.
    # Warm the native extension in main before spawning so the compile
    # happens once (not once per worker). If the extension loads, workers
    # will use the cached .so — zero compile time.
    from torchsx.io import _native
    native_mod = _native.load()
    count_fn = _native_count_args if native_mod is not None else _count_byte_range_args
    parse_fn = _native_parse_args if native_mod is not None else _parse_into_shared_args

    ranges = _coarse_chunk_split(path, n_workers)

    # Start method: "fork" is the fastest — workers inherit main's already-
    # imported torch + compiled native extension, so worker startup costs
    # ~0 instead of ~3s per worker under spawn (which was the dominant
    # overhead at n≥8 on this workload).
    #
    # Fork-after-threads is only unsafe if main has already started torch's
    # thread pool; that happens lazily on the first parallel tensor op. We
    # carefully avoid triggering it in the lead-up to this call: the only
    # torch ops are ``empty`` + ``share_memory_``, neither of which spawns
    # threads. If someone did trigger the thread pool beforehand (e.g. by
    # computing on tensors), the user can opt out via TORCHSX_MP_START.
    start_method = os.environ.get("TORCHSX_MP_START", "fork")
    try:
        ctx = mp.get_context(start_method)
    except ValueError:
        ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        # Stage B: parallel count
        count_args = [(path, s, e) for s, e in ranges]
        counts = list(pool.map(count_fn, count_args))

        n_obs_total = sum(c[0] for c in counts)
        n_crys_total = sum(c[1] for c in counts)
        n_imgs_total = sum(c[2] for c in counts)

        tensors = _allocate_tensors(n_obs_total, n_crys_total, n_imgs_total)
        for t in tensors.values():
            t.share_memory_()

        # Stage C: parallel parse into shared
        parse_args = []
        obs_off = crys_off = img_off = 0
        for (s, e), (n_obs_r, n_crys_r, n_imgs_r) in zip(ranges, counts):
            parse_args.append(
                (path, s, e, panel_name_to_id, tensors, obs_off, crys_off, img_off)
            )
            obs_off += n_obs_r
            crys_off += n_crys_r
            img_off += n_imgs_r

        seg_results = list(pool.map(parse_fn, parse_args))

    # Assemble filenames/events in file order; sanity-check counts.
    total_obs = total_crys = total_imgs = 0
    all_filenames: list[str] = []
    all_events: list[str] = []
    for n_obs_s, n_crys_s, n_imgs_s, filenames, events in seg_results:
        total_obs += n_obs_s
        total_crys += n_crys_s
        total_imgs += n_imgs_s
        all_filenames.extend(filenames)
        all_events.extend(events)

    if total_obs != n_obs_total:
        # Would leave gaps in shared tensors; refuse rather than silently corrupt.
        raise RuntimeError(
            f"Worker obs count {total_obs} != preallocated {n_obs_total}; "
            "shared tensor layout would be corrupt."
        )
    assert total_crys == n_crys_total
    assert total_imgs == n_imgs_total

    return _finalize_shared(
        tensors, all_filenames, all_events, geometry, spacegroup, path
    )


def _finalize_shared(
    tensors: dict[str, torch.Tensor],
    all_filenames: list[str],
    all_events: list[str],
    geometry: Geometry,
    spacegroup: str,
    path: str,
) -> _SFXStreamCore:
    """Wrap already-populated shared tensors as _SFXStreamCore."""
    sigma = tensors["sigma"]
    if sigma.numel() > 0 and sigma.std() < 1e-6:
        warnings.warn(
            f"All sigma values are constant ({sigma[0].item():.2f}), "
            "likely set to photon energy. Consider replacing with proper estimates.",
            stacklevel=2,
        )

    metadata = StreamMetadata()
    metadata["image"] = {"filename": all_filenames, "event": all_events}
    metadata["crystal"] = {"panel_names": geometry.panel_names}

    return _SFXStreamCore(
        hkl=tensors["hkl"],
        I=tensors["I"],
        sigma=tensors["sigma"],
        background=tensors["background"],
        peak=tensors["peak"],
        fs=tensors["fs"],
        ss=tensors["ss"],
        panel_id=tensors["panel_id"],
        crystal_id=tensors["crystal_id"],
        astar=tensors["astar"],
        bstar=tensors["bstar"],
        cstar=tensors["cstar"],
        cell=tensors["cell"],
        profile_rad=tensors["profile_rad"],
        resolution=tensors["resolution"],
        n_refl=tensors["n_refl"],
        image_id=tensors["image_id"],
        crystal_in_image=tensors["crystal_in_image"],
        beam_energy=tensors["beam_energy"],
        det_shift_x=tensors["det_shift_x"],
        det_shift_y=tensors["det_shift_y"],
        n_peaks=tensors["n_peaks"],
        camera_length=tensors["camera_length"],
        geometry=geometry,
        metadata=metadata,
        spacegroup=spacegroup,
        source_path=str(path),
    )


def _parse_stream_sampled(
    path: str,
    spacegroup: str,
    max_crystals: int,
    seed: int,
) -> _SFXStreamCore:
    """Parse a random subset of chunks totalling ``max_crystals`` crystals."""
    print(f"  [sampled] indexing chunks in {path} ...", flush=True)
    _, chunks = _index_chunks(path)  # now (start, end, n_crys, n_obs) per chunk
    total_crystals = sum(c[2] for c in chunks)
    print(
        f"  [sampled] {len(chunks)} chunks, {total_crystals} crystals total; "
        f"sampling {max_crystals} crystals (seed={seed})",
        flush=True,
    )

    if total_crystals < max_crystals:
        raise ValueError(
            f"Only {total_crystals} crystals in stream; cannot sample {max_crystals}"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chunks))
    selected_idx: list[int] = []
    n_selected = 0
    for i in order:
        selected_idx.append(int(i))
        n_selected += chunks[i][2]
        if n_selected >= max_crystals:
            break
    selected_idx.sort()  # preserve file order for read locality
    print(
        f"  [sampled] picked {len(selected_idx)} chunks → {n_selected} crystals",
        flush=True,
    )

    geometry = Geometry.from_stream(path)
    panel_name_to_id = geometry.panel_name_to_id

    # Pass 1: counts already known from _index_chunks (one per chunk).
    n_obs_alloc = 0
    n_crys_total = 0
    n_imgs_total = 0
    for i in selected_idx:
        _, _, n_crys_c, n_obs_c = chunks[i]
        n_obs_alloc += n_obs_c
        n_crys_total += n_crys_c
        n_imgs_total += 1  # one image per chunk

    tensors = _allocate_tensors(n_obs_alloc, n_crys_total, n_imgs_total)
    views = {k: t.numpy() for k, t in tensors.items()}
    filenames: list[str] = []
    events: list[str] = []
    crystal_idx = image_idx = n_obs_got = 0

    # Pass 2: parse into the pre-allocated views
    with open(path, "rb", buffering=1 << 20) as f:
        for i in selected_idx:
            start, end, _, _ = chunks[i]
            f.seek(start)
            it = _line_iter(f, end)
            crystal_idx, image_idx, n_obs_got = _parse_into_views(
                it, panel_name_to_id, views, filenames, events,
                crystal_idx, image_idx, n_obs_got,
            )

    result = _finalize(
        tensors, n_obs_got, n_obs_alloc,
        crystal_idx, n_crys_total,  # crystal_idx holds the actual final count
        image_idx, n_imgs_total,    # image_idx likewise
        filenames, events,
    )
    return _assemble_core([result], path, spacegroup, geometry)


# ---------------------------------------------------------------------------
# Assembly (main process)
# ---------------------------------------------------------------------------


_OBS_FIELDS = (
    "hkl", "I", "sigma", "peak", "background", "fs", "ss",
    "panel_id", "crystal_id",
)
_CRYS_FIELDS_NO_IMAGE = (
    "cell", "astar", "bstar", "cstar",
    "profile_rad", "resolution", "n_refl", "crystal_in_image",
)
_IMG_FIELDS = (
    "beam_energy", "det_shift_x", "det_shift_y", "camera_length", "n_peaks",
)


def _assemble_single(
    res: dict,
    path: str,
    spacegroup: str,
    geometry: Geometry,
) -> _SFXStreamCore:
    """Fast path for single-segment results — no cat, no extra allocation."""
    sigma = res["sigma"]
    if sigma.numel() > 0 and sigma.std() < 1e-6:
        warnings.warn(
            f"All sigma values are constant ({sigma[0].item():.2f}), "
            "likely set to photon energy. Consider replacing with proper estimates.",
            stacklevel=2,
        )

    metadata = StreamMetadata()
    metadata["image"] = {"filename": res["filenames"], "event": res["events"]}
    metadata["crystal"] = {"panel_names": geometry.panel_names}

    return _SFXStreamCore(
        hkl=res["hkl"],
        I=res["I"],
        sigma=res["sigma"],
        background=res["background"],
        peak=res["peak"],
        fs=res["fs"],
        ss=res["ss"],
        panel_id=res["panel_id"],
        crystal_id=res["crystal_id"],
        astar=res["astar"],
        bstar=res["bstar"],
        cstar=res["cstar"],
        cell=res["cell"],
        profile_rad=res["profile_rad"],
        resolution=res["resolution"],
        n_refl=res["n_refl"],
        image_id=res["image_id"],
        crystal_in_image=res["crystal_in_image"],
        beam_energy=res["beam_energy"],
        det_shift_x=res["det_shift_x"],
        det_shift_y=res["det_shift_y"],
        n_peaks=res["n_peaks"],
        camera_length=res["camera_length"],
        geometry=geometry,
        metadata=metadata,
        spacegroup=spacegroup,
        source_path=str(path),
    )


def _assemble_core(
    results: list[dict],
    path: str,
    spacegroup: str,
    geometry: Geometry,
) -> _SFXStreamCore:
    """Concatenate per-segment tensors into a single _SFXStreamCore."""
    if not results:
        raise ValueError(f"No indexed crystals found in {path}")

    # Single-segment fast path (serial parse): reuse the segment's tensors
    # directly. ``torch.cat`` on a single-element list still allocates new
    # storage — skipping it saves one whole tensor copy (~final payload).
    if len(results) == 1:
        return _assemble_single(results[0], path, spacegroup, geometry)

    obs: dict[str, list[torch.Tensor]] = {k: [] for k in _OBS_FIELDS}
    crys: dict[str, list[torch.Tensor]] = {k: [] for k in _CRYS_FIELDS_NO_IMAGE}
    crys_image_id: list[torch.Tensor] = []
    imgs: dict[str, list[torch.Tensor]] = {k: [] for k in _IMG_FIELDS}
    all_filenames: list[str] = []
    all_events: list[str] = []

    crystal_offset = 0
    image_offset = 0

    for res in results:
        cid = res["crystal_id"]
        if crystal_offset:
            cid = cid + crystal_offset
        obs["crystal_id"].append(cid)

        iid = res["image_id"]
        if image_offset:
            iid = iid + image_offset
        crys_image_id.append(iid)

        for k in _OBS_FIELDS:
            if k == "crystal_id":
                continue
            obs[k].append(res[k])
        for k in _CRYS_FIELDS_NO_IMAGE:
            crys[k].append(res[k])
        for k in _IMG_FIELDS:
            imgs[k].append(res[k])

        all_filenames.extend(res["filenames"])
        all_events.extend(res["events"])

        crystal_offset += res["n_crystals"]
        image_offset += res["n_images"]

    obs_cat = {k: torch.cat(v) for k, v in obs.items()}
    crys_cat = {k: torch.cat(v) for k, v in crys.items()}
    crys_cat["image_id"] = torch.cat(crys_image_id)
    imgs_cat = {k: torch.cat(v) for k, v in imgs.items()}

    n_obs = obs_cat["I"].numel()
    n_crys = crys_cat["cell"].shape[0]

    if n_obs == 0 or n_crys == 0:
        raise ValueError(f"No indexed crystals found in {path}")

    sigma = obs_cat["sigma"]
    if sigma.numel() > 0 and sigma.std() < 1e-6:
        warnings.warn(
            f"All sigma values are constant ({sigma[0].item():.2f}), "
            "likely set to photon energy. Consider replacing with proper estimates.",
            stacklevel=2,
        )

    metadata = StreamMetadata()
    metadata["image"] = {"filename": all_filenames, "event": all_events}
    metadata["crystal"] = {"panel_names": geometry.panel_names}

    return _SFXStreamCore(
        hkl=obs_cat["hkl"],
        I=obs_cat["I"],
        sigma=obs_cat["sigma"],
        background=obs_cat["background"],
        peak=obs_cat["peak"],
        fs=obs_cat["fs"],
        ss=obs_cat["ss"],
        panel_id=obs_cat["panel_id"],
        crystal_id=obs_cat["crystal_id"],
        astar=crys_cat["astar"],
        bstar=crys_cat["bstar"],
        cstar=crys_cat["cstar"],
        cell=crys_cat["cell"],
        profile_rad=crys_cat["profile_rad"],
        resolution=crys_cat["resolution"],
        n_refl=crys_cat["n_refl"],
        image_id=crys_cat["image_id"],
        crystal_in_image=crys_cat["crystal_in_image"],
        beam_energy=imgs_cat["beam_energy"],
        det_shift_x=imgs_cat["det_shift_x"],
        det_shift_y=imgs_cat["det_shift_y"],
        n_peaks=imgs_cat["n_peaks"],
        camera_length=imgs_cat["camera_length"],
        geometry=geometry,
        metadata=metadata,
        spacegroup=spacegroup,
        source_path=str(path),
    )
