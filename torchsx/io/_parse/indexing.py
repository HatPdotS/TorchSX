"""Byte-level chunk indexing and pass-1 counting.

Leaf module: no torchsx imports. The native-vs-python index dispatcher keeps
its ``from torchsx.io import _native`` function-local (load timing).

A chunk marker (``----- Begin chunk -----`` / ``----- End chunk -----``) always
ends a reflection table, also when ``End of reflections`` is missing, so a
truncated table never swallows the next chunk. The native indexer applies the
same rule; that is what makes its threaded scan match the serial one.
"""

from __future__ import annotations

import numpy as np

B_BEGIN_CHUNK = b"----- Begin chunk -----"
B_END_CHUNK = b"----- End chunk -----"
B_BEGIN_CRY = b"--- Begin crystal"
B_IMG = b"Image filename:"
B_REFL_BEGIN = b"Reflections measured after indexing"
B_REFL_END = b"End of reflections"


def _is_marker(line: bytes) -> bool:
    return line.startswith(B_BEGIN_CHUNK) or line.startswith(B_END_CHUNK)


def _index_chunks(path: str) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Scan a stream file and return (header_end, chunks).

    ``chunks`` is a list of
    ``(start_offset, end_offset, n_crystals, n_obs_upper)`` per
    ``----- Begin chunk -----`` block. ``n_obs_upper`` counts reflection lines
    with at least ten fields (``len(line.split()) >= 10``), the parser's first
    check, so it bounds the rows the parser keeps.

    Single file pass; O(n_chunks) memory.
    """
    selected = _index_chunks_until(path, None)
    if not selected:
        raise ValueError(f"No chunks found in {path}")
    return selected[0][0], selected


def _find_first_chunk_offset(path: str) -> int:
    """Locate the first ``----- Begin chunk -----`` byte offset."""
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                raise ValueError(f"No chunks found in {path}")
            if line.startswith(B_BEGIN_CHUNK):
                return pos


def _count_byte_range(path: str, start: int, end: int) -> tuple[int, int, int]:
    """Count (n_obs_upper, n_crystals, n_images) in ``[start, end)``.

    Pure byte-level scan — no decode, no parse. ``n_obs_upper`` counts every
    line of every reflection table, so it bounds what the parser keeps; the
    parser grows its arrays if malformed input ever beats these counts.
    """
    n_obs = 0
    n_crys = 0
    n_imgs = 0
    in_refl = False
    header_pending = False

    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            if in_refl:
                if header_pending:
                    header_pending = False
                    if not _is_marker(line):
                        continue
                if line.startswith(B_REFL_END):
                    in_refl = False
                    continue
                if not _is_marker(line):
                    n_obs += 1
                    continue
                in_refl = False
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
                    header_pending = True
    return n_obs, n_crys, n_imgs


def _index_chunks_until(
    path: str, max_crystals: int | None
) -> list[tuple[int, int, int, int]]:
    """Walk the stream from the start, indexing chunks until cumulative
    crystals ≥ ``max_crystals`` (all chunks if ``None``). Returns
    ``(start, end, n_crys, n_obs)`` per chunk, like :func:`_index_chunks`.

    The chunk that crosses the threshold is included, ending at the next
    chunk's start; nothing past it is read.
    """
    selected: list[tuple[int, int, int, int]] = []
    current_start: int | None = None
    current_crystals = 0
    current_obs = 0
    n_crys_cum = 0
    in_refl = False
    header_pending = False

    with open(path, "rb", buffering=1 << 20) as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if in_refl:
                if header_pending:
                    header_pending = False
                    if not _is_marker(line):
                        continue
                if line.startswith(B_REFL_END):
                    in_refl = False
                    continue
                if not _is_marker(line):
                    if len(line.split()) >= 10:
                        current_obs += 1
                    continue
                in_refl = False
            c0 = line[:1]
            if c0 == b"-":
                if line.startswith(B_BEGIN_CHUNK):
                    if current_start is not None:
                        selected.append(
                            (current_start, pos, current_crystals, current_obs)
                        )
                        n_crys_cum += current_crystals
                        if max_crystals is not None and n_crys_cum >= max_crystals:
                            current_start = None  # already flushed
                            break
                    current_start = pos
                    current_crystals = 0
                    current_obs = 0
                elif line.startswith(B_BEGIN_CRY) and current_start is not None:
                    current_crystals += 1
            elif c0 == b"R" and line.startswith(B_REFL_BEGIN):
                in_refl = True
                header_pending = True
        if current_start is not None:
            # Reached EOF — flush the last chunk.
            selected.append((current_start, f.tell(), current_crystals, current_obs))

    return selected


def _index_chunks_native_or_python(
    path: str,
    n_workers: int = 1,
    max_crystals: int | None = None,
) -> np.ndarray:
    """Per-chunk ``(start, end, n_crys, n_obs)`` rows as an ``(n, 4)`` int64 array.

    Uses the native indexer when available (GIL released, ``n_workers``
    threads). ``max_crystals`` selects early-exit head mode, which is serial
    and may return no rows; a full index of a file without chunks raises.
    """
    from torchsx.io import _native

    native_mod = _native.load()
    if native_mod is not None:
        rows = native_mod.index_chunks(path, max(1, int(n_workers)), max_crystals)
    else:
        rows = np.asarray(_index_chunks_until(path, max_crystals), dtype=np.int64).reshape(-1, 4)
    if max_crystals is None and len(rows) == 0:
        raise ValueError(f"No chunks found in {path}")
    return rows
