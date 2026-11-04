""".stream parser for SFX data.

Every read is index → select chunks → parse. The native module
(``torchsx_native``, see ``native/``) does both steps on ``n_workers``
threads. The public entry points are :func:`parse_stream` and
:func:`parse_stream_blocks`.

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
whatever image, crystal or reflection table is open, so no chunk can swallow
the next one, and ``--- Begin crystal`` starts a crystal even where a peak
section was expected. ``tests/python_parser`` implements the same rules in
pure Python; the native parser is tested against it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from torchsx._core import SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata

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


def parse_stream(
    path: str | Path,
    spacegroup: str = "P 1",
    n_workers: int = 1,
    max_crystals: int | None = None,
    seed: int = 0,
    sequential: bool = False,
) -> SFXStreamCore:
    """Parse reflections and attach source semantics.

    All full/head/sampled reads converge here. Retaining observed
    indices shares their tensor until canonicalization replaces ``hkl``.
    """
    from torchsx.io._crystfel import read_measurement

    spec = read_measurement(path)
    core = _parse_stream(path, spacegroup, n_workers, max_crystals, seed, sequential)
    return _attach_source(core, path, spec)


def _attach_source(core: SFXStreamCore, path, spec) -> SFXStreamCore:
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

    from torchsx.io._crystfel import read_measurement

    path = str(path)
    spec = read_measurement(path)
    geometry = Geometry.from_stream(path)
    chunks = _index_chunks(path, n_workers=n_workers)
    cum = np.cumsum(chunks[:, 3])
    edges = [0]
    while edges[-1] < len(chunks):
        base = cum[edges[-1] - 1] if edges[-1] else 0
        edges.append(max(edges[-1] + 1, int(np.searchsorted(cum, base + block_obs, side="right"))))
    groups = [chunks[a:b] for a, b in zip(edges[:-1], edges[1:])]

    def parse(group):
        tensors, filenames, events, report = _parse_chunks(path, group, geometry, n_workers)
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
) -> SFXStreamCore:
    """Parse a CrystFEL .stream file into stacked tensors.

    Parameters
    ----------
    path : path to .stream file
    spacegroup : space group string (stored as metadata)
    n_workers : threads for indexing and parsing. Sequential head mode
        indexes serially.
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
    if max_crystals is None:
        chunks = _index_chunks(path, n_workers=n_workers)
    elif sequential:
        chunks = _select_head(path, int(max_crystals))
    else:
        chunks = _select_sampled(path, int(max_crystals), int(seed), n_workers)
    tensors, filenames, events, report = _parse_chunks(path, chunks, geometry, n_workers)
    return _build_core(tensors, filenames, events, report, geometry, spacegroup, path)


def _index_chunks(
    path: str,
    n_workers: int = 1,
    max_crystals: int | None = None,
) -> np.ndarray:
    """Per-chunk ``(start, end, n_crys, n_obs)`` rows as an ``(n, 4)`` int64 array.

    Indexes on ``n_workers`` threads with the GIL released. ``max_crystals``
    selects early-exit head mode, which is serial and may return no rows; a
    full index of a file without chunks raises.
    """
    from torchsx.io import _native

    rows = _native.load().index_chunks(path, max(1, int(n_workers)), max_crystals)
    if max_crystals is None and len(rows) == 0:
        raise ValueError(f"No chunks found in {path}")
    return rows


def _parse_chunks(
    path: str, chunks: np.ndarray, geometry: Geometry, n_workers: int
) -> tuple[dict[str, torch.Tensor], list[str], list[str], dict[str, list[int]]]:
    """Columns, filenames, events and exclusion report of the given chunk rows."""
    from torchsx.io import _native

    cols, filenames, events, report = _native.load().parse_chunks(
        path, np.ascontiguousarray(chunks, dtype=np.int64),
        geometry.panel_names, max(1, int(n_workers)),
    )
    tensors = {k: torch.from_numpy(a) for k, a in cols.items()}
    return tensors, filenames, events, {k: list(v) for k, v in report.items()}


def _select_head(path: str, max_crystals: int) -> np.ndarray:
    """The first chunks of a stream, up to the one reaching ``max_crystals``.

    Single-pass index with early exit; the rest of the file is not read.
    """
    print(
        f"  [head] indexing chunks of {path} until ≥{max_crystals} crystals ...",
        flush=True,
    )
    selected = _index_chunks(path, max_crystals=max_crystals)
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
    chunks = _index_chunks(path, n_workers=n_workers)
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
) -> SFXStreamCore:
    """Wrap parsed columns as a ``SFXStreamCore``."""
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

    return SFXStreamCore(
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
