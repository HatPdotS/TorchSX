"""Public ``parse_stream`` entry point, driving logic, and core assembly.

Depends on the three sibling modules. ``from torchsx.io import _native`` stays
function-local in ``parse_stream`` (warms the extension in the main process
before fork/spawn).
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
from torchsx.io._parse.indexing import (
    _coarse_chunk_split,
    _find_first_chunk_offset,
    _index_chunks_native_or_python,
)
from torchsx.io._parse.records import (
    _allocate_tensors,
    _finalize,
    _line_iter,
    _parse_into_views,
)
from torchsx.io._parse.workers import (
    _count_byte_range_args,
    _native_count_args,
    _native_parse_args,
    _parse_byte_range,
    _parse_into_shared_args,
)


def parse_stream(
    path: str | Path,
    spacegroup: str = "P 1",
    n_workers: int = 1,
    max_crystals: int | None = None,
    seed: int = 0,
    sequential: bool = False,
) -> _SFXStreamCore:
    """Parse a CrystFEL .stream file into stacked tensors.

    Parameters
    ----------
    path : path to .stream file
    spacegroup : space group string (stored as metadata)
    n_workers : number of parallel worker processes for full-file parse;
        also the number of native indexer threads in sampled mode.  Sequential
        head mode forces single-threaded indexing.
    max_crystals : if set, parse a subset containing at most this many
        crystals. Parse phase is serial (``n_workers`` only affects indexing).
    seed : RNG seed for random chunk sampling (only used when
        ``max_crystals`` is set and ``sequential`` is False).
    sequential : if True with ``max_crystals``, take chunks from the file
        start in order and stop scanning as soon as enough crystals are
        collected. Avoids the full-file indexing pass.
    """
    path = str(path)

    if max_crystals is not None:
        if sequential:
            return _parse_stream_head(path, spacegroup, int(max_crystals))
        return _parse_stream_sampled(
            path, spacegroup, int(max_crystals), int(seed), n_workers=int(n_workers),
        )

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


def _parse_stream_head(
    path: str,
    spacegroup: str,
    max_crystals: int,
) -> _SFXStreamCore:
    """Parse the first chunks of a stream until ``max_crystals`` is reached.

    Single-pass index (early-exit) + targeted parse. Avoids scanning the
    whole file just to know its total chunk count.
    """
    print(
        f"  [head] indexing chunks of {path} until ≥{max_crystals} crystals ...",
        flush=True,
    )
    selected = _index_chunks_native_or_python(path, max_crystals=max_crystals)
    if not selected:
        raise ValueError(f"No chunks found in {path}")

    n_obs_alloc = sum(c[3] for c in selected)
    n_crys_total = sum(c[2] for c in selected)
    n_imgs_total = len(selected)  # one image per chunk
    print(
        f"  [head] selected {n_imgs_total} chunks → {n_crys_total} crystals",
        flush=True,
    )

    geometry = Geometry.from_stream(path)
    panel_name_to_id = geometry.panel_name_to_id

    tensors = _allocate_tensors(n_obs_alloc, n_crys_total, n_imgs_total)
    views = {k: t.numpy() for k, t in tensors.items()}
    filenames: list[str] = []
    events: list[str] = []
    crystal_idx = image_idx = n_obs_got = 0

    with open(path, "rb", buffering=1 << 20) as f:
        for start, end, _, _ in selected:
            f.seek(start)
            it = _line_iter(f, end)
            crystal_idx, image_idx, n_obs_got = _parse_into_views(
                it, panel_name_to_id, views, filenames, events,
                crystal_idx, image_idx, n_obs_got,
            )

    result = _finalize(
        tensors, n_obs_got, n_obs_alloc,
        crystal_idx, n_crys_total,
        image_idx, n_imgs_total,
        filenames, events,
    )
    return _assemble_core([result], path, spacegroup, geometry)


def _parse_stream_sampled(
    path: str,
    spacegroup: str,
    max_crystals: int,
    seed: int,
    n_workers: int = 1,
) -> _SFXStreamCore:
    """Parse a random subset of chunks totalling ``max_crystals`` crystals.

    Indexing is parallelised via the native C++ extension when ``n_workers > 1``;
    parse phase is serial.
    """
    print(
        f"  [sampled] indexing chunks in {path} (n_workers={n_workers}) ...",
        flush=True,
    )
    chunks = _index_chunks_native_or_python(path, n_workers=n_workers)
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
