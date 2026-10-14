"""Public ``parse_stream`` entry point, driving logic, and core assembly.

Every mode is index → select chunks → parse. With the native module both steps
run on ``n_workers`` threads in one call each; without it the pure-Python
parser runs serially.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata
from torchsx.io._parse.indexing import (
    _find_first_chunk_offset,
    _index_chunks_native_or_python,
)
from torchsx.io._parse.records import (
    REASONS,
    _allocate_arrays,
    _finalize,
    _line_iter,
    _parse_into_views,
)
from torchsx.io._parse.workers import _parse_byte_range


def parse_stream(
    path: str | Path,
    spacegroup: str = "P 1",
    n_workers: int = 1,
    max_crystals: int | None = None,
    seed: int = 0,
    sequential: bool = False,
) -> _SFXStreamCore:
    """Parse reflections and attach source semantics.

    All full/sampled/native/Python paths converge here. Retaining observed
    indices shares their tensor until canonicalization replaces ``hkl``.
    """
    from torchsx.io._crystfel import read_measurement

    spec = read_measurement(path)
    core = _parse_stream(path, spacegroup, n_workers, max_crystals, seed, sequential)
    return _attach_source(core, path, spec)


def _attach_source(core: _SFXStreamCore, path, spec) -> _SFXStreamCore:
    """Source columns every parsed core carries."""
    from torchsx.measurement import MEASUREMENT_KEY

    # The core is still private to this reader: add columns in place, avoiding
    # repeated whole-tensor validation passes on very large observation tables.
    core.levels["obs"]["observed_hkl"] = core.hkl
    core.levels["crystal"]["source_lattice_index"] = core.crystal_in_image
    core.metadata[MEASUREMENT_KEY] = {"record": spec.to_dict()}
    core.metadata.setdefault("image", {})["source_file"] = [str(Path(path).resolve())] * core.n_images
    core.friedel_merged = False
    return core


def parse_stream_blocks(
    path: str | Path,
    spacegroup: str = "P 1",
    n_workers: int = 1,
    block_obs: int = 4_000_000,
    transform=None,
):
    """:func:`parse_stream` of a whole file, yielded as consecutive cores of
    whole chunks with about ``block_obs`` observations each.

    Concatenating the blocks gives the core :func:`parse_stream` returns, so a
    file of any size is read with memory for a few blocks.  The next block is
    parsed -- and passed through ``transform(core)``, if given, whose result is
    yielded -- on a background thread while the caller works on the current
    one.  Exclusions and a constant sigma are reported once, for the whole file.
    """
    import queue
    import threading

    from torchsx.io import _native
    from torchsx.io._crystfel import read_measurement

    path = str(path)
    spec = read_measurement(path)
    geometry = Geometry.from_stream(path)
    native_mod = _native.load()
    chunks = _index_chunks_native_or_python(path, n_workers=n_workers)
    cum = np.cumsum(chunks[:, 3])
    edges = [0]
    while edges[-1] < len(chunks):
        base = cum[edges[-1] - 1] if edges[-1] else 0
        edges.append(max(edges[-1] + 1, int(np.searchsorted(cum, base + block_obs, side="right"))))
    groups = [chunks[a:b] for a, b in zip(edges[:-1], edges[1:])]

    def parse(group):
        if native_mod is not None:
            cols, filenames, events, report = native_mod.parse_chunks(
                path, np.ascontiguousarray(group, dtype=np.int64),
                geometry.panel_names, max(1, int(n_workers)),
            )
            tensors = {k: torch.from_numpy(a) for k, a in cols.items()}
            report = {k: list(v) for k, v in report.items()}
        else:
            tensors, filenames, events, report = _parse_chunks_python(
                path, group, geometry.panel_name_to_id)
        sigma = tensors["sigma"].double()
        moments = (sigma.numel(), float(sigma.sum()), float((sigma * sigma).sum()))
        core = _attach_source(
            _build_core(tensors, filenames, events, {}, geometry, spacegroup, path, warn=False),
            path, spec)
        return (core if transform is None else transform(core)), report, moments

    q: queue.Queue = queue.Queue(maxsize=1)

    def produce():
        try:
            for group in groups:
                q.put(("ok", parse(group)))
        except BaseException as exc:  # handed to the consumer
            q.put(("err", exc))
            return
        q.put(("end", None))

    threading.Thread(target=produce, daemon=True, name="torchsx-parse").start()
    total: dict[str, list[int]] = {}
    sig_n, sig_s, sig_s2 = 0, 0.0, 0.0
    while True:
        kind, item = q.get()
        if kind == "err":
            raise item
        if kind == "end":
            break
        block, report, (n, s1, s2) = item
        for reason, (count, first) in report.items():
            got = total.setdefault(reason, [0, first])
            got[0] += count
            got[1] = min(got[1], first)
        sig_n, sig_s, sig_s2 = sig_n + n, sig_s + s1, sig_s2 + s2
        yield block
    _warn_report(total, path)
    if sig_n > 1:
        var = max(sig_s2 - sig_s * sig_s / sig_n, 0.0) / (sig_n - 1)
        if var ** 0.5 < 1e-6:
            warnings.warn(
                f"All sigma values are constant ({sig_s / sig_n:.2f}), "
                "likely set to photon energy. Consider replacing with proper estimates.",
                stacklevel=2,
            )


def _parse_stream(
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
    n_workers : threads for native indexing and parsing (the pure-Python
        fallback is serial). Sequential head mode indexes serially.
    max_crystals : if set, parse a subset containing at least this many
        crystals (whole chunks).
    seed : RNG seed for random chunk sampling (only used when
        ``max_crystals`` is set and ``sequential`` is False).
    sequential : if True with ``max_crystals``, take chunks from the file
        start in order and stop scanning as soon as enough crystals are
        collected. Avoids the full-file indexing pass.
    """
    path = str(path)
    geometry = Geometry.from_stream(path)
    from torchsx.io import _native

    native_mod = _native.load()

    if max_crystals is None:
        if native_mod is None:
            header_end = _find_first_chunk_offset(path)
            file_size = Path(path).stat().st_size
            tensors, filenames, events, report = _parse_byte_range(
                path, header_end, file_size, geometry.panel_name_to_id
            )
            return _build_core(tensors, filenames, events, report, geometry, spacegroup, path)
        chunks = _index_chunks_native_or_python(path, n_workers=n_workers)
    elif sequential:
        chunks = _select_head(path, int(max_crystals))
    else:
        chunks = _select_sampled(path, int(max_crystals), int(seed), n_workers)

    if native_mod is not None:
        cols, filenames, events, report = native_mod.parse_chunks(
            path, np.ascontiguousarray(chunks, dtype=np.int64),
            geometry.panel_names, max(1, int(n_workers)),
        )
        tensors = {k: torch.from_numpy(a) for k, a in cols.items()}
        report = {k: list(v) for k, v in report.items()}
    else:
        tensors, filenames, events, report = _parse_chunks_python(
            path, chunks, geometry.panel_name_to_id
        )
    return _build_core(tensors, filenames, events, report, geometry, spacegroup, path)


def _select_head(path: str, max_crystals: int) -> np.ndarray:
    """The first chunks of a stream, up to the one reaching ``max_crystals``.

    Single-pass index with early exit; the rest of the file is not read.
    """
    print(
        f"  [head] indexing chunks of {path} until ≥{max_crystals} crystals ...",
        flush=True,
    )
    selected = _index_chunks_native_or_python(path, max_crystals=max_crystals)
    if not len(selected):
        raise ValueError(f"No chunks found in {path}")
    print(
        f"  [head] selected {len(selected)} chunks → {int(selected[:, 2].sum())} crystals",
        flush=True,
    )
    return selected


def _select_sampled(path: str, max_crystals: int, seed: int, n_workers: int) -> np.ndarray:
    """A random subset of chunks totalling at least ``max_crystals`` crystals,
    in file order."""
    print(
        f"  [sampled] indexing chunks in {path} (n_workers={n_workers}) ...",
        flush=True,
    )
    chunks = _index_chunks_native_or_python(path, n_workers=n_workers)
    total_crystals = int(chunks[:, 2].sum())
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
        n_selected += int(chunks[i, 2])
        if n_selected >= max_crystals:
            break
    selected_idx.sort()  # preserve file order for read locality
    print(
        f"  [sampled] picked {len(selected_idx)} chunks → {n_selected} crystals",
        flush=True,
    )
    return chunks[selected_idx]


def _parse_chunks_python(
    path: str, chunks: np.ndarray, panel_name_to_id: dict[str, int]
) -> tuple[dict[str, torch.Tensor], list[str], list[str], dict[str, list[int]]]:
    """Serial pure-Python parse of the given chunk rows."""
    v = _allocate_arrays(int(chunks[:, 3].sum()), int(chunks[:, 2].sum()), len(chunks))
    filenames: list[str] = []
    events: list[str] = []
    report: dict[str, list[int]] = {}
    crystal_idx = image_idx = n_obs = 0
    with open(path, "rb", buffering=1 << 20) as f:
        for start, end, _, _ in chunks.tolist():
            f.seek(start)
            crystal_idx, image_idx, n_obs = _parse_into_views(
                _line_iter(f, end), panel_name_to_id, v, filenames, events, report,
                crystal_idx, image_idx, n_obs,
            )
    return _finalize(v, n_obs, crystal_idx, image_idx), filenames, events, report


def _warn_report(report: dict[str, list[int]], path: str) -> None:
    """One warning listing what the parser excluded, by reason."""
    if not report:
        return
    parts = [
        f"{report[r][0]} × {r.replace('_', ' ')} (first at byte {report[r][1]})"
        for r in REASONS if r in report
    ]
    warnings.warn(
        f"{path}: parsed with exclusions: " + "; ".join(parts),
        stacklevel=4,
    )


def _build_core(
    tensors: dict[str, torch.Tensor],
    filenames: list[str],
    events: list[str],
    report: dict[str, list[int]],
    geometry: Geometry,
    spacegroup: str,
    path: str,
    warn: bool = True,
) -> _SFXStreamCore:
    """Wrap parsed columns as a ``_SFXStreamCore``."""
    if warn:
        _warn_report(report, path)
    sigma = tensors["sigma"]
    if warn and sigma.numel() > 0 and sigma.std() < 1e-6:
        warnings.warn(
            f"All sigma values are constant ({sigma[0].item():.2f}), "
            "likely set to photon energy. Consider replacing with proper estimates.",
            stacklevel=3,
        )

    metadata = StreamMetadata()
    metadata["image"] = {"filename": filenames, "event": events}
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
