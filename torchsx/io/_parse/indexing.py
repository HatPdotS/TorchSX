"""Byte-level chunk indexing, offset computation, and pass-1 counting.

Leaf module: no torchsx imports. The native-vs-python index dispatcher keeps
its ``from torchsx.io import _native`` function-local (load timing).
"""

from __future__ import annotations

import os  # noqa: F401  (kept for parity with the original module surface)


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


def _index_chunks_until(
    path: str, max_crystals: int
) -> list[tuple[int, int, int, int]]:
    """Walk the stream from the start, indexing chunks until cumulative
    crystals ≥ ``max_crystals``. Returns the selected chunks as
    ``(start, end, n_crys, n_obs)`` — same shape as :func:`_index_chunks`.

    Stops early; never reads past the last needed chunk.
    """
    B_BEGIN_CHUNK = b"----- Begin chunk -----"
    B_BEGIN_CRY = b"--- Begin crystal"
    B_REFL_BEGIN = b"Reflections measured after indexing"
    B_REFL_END = b"End of reflections"

    selected: list[tuple[int, int, int, int]] = []
    current_start: int | None = None
    current_crystals = 0
    current_obs = 0
    n_crys_cum = 0
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
                    if current_start is not None:
                        selected.append(
                            (current_start, pos, current_crystals, current_obs)
                        )
                        n_crys_cum += current_crystals
                        if n_crys_cum >= max_crystals:
                            current_start = None  # already flushed
                            break
                    current_start = pos
                    current_crystals = 0
                    current_obs = 0
                elif line.startswith(B_BEGIN_CRY):
                    current_crystals += 1
            elif c0 == b"R" and line.startswith(B_REFL_BEGIN):
                in_refl = True
                f.readline()  # skip reflection-table header
        if current_start is not None:
            # Reached EOF — flush the last partially-indexed chunk.
            end_pos = f.tell()
            selected.append(
                (current_start, end_pos, current_crystals, current_obs)
            )

    return selected


def _index_chunks_native_or_python(
    path: str,
    n_workers: int = 1,
    max_crystals: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Return per-chunk ``(start, end, n_crys, n_obs)`` tuples.

    Routes through the C++ extension when available (single call, GIL
    released, multi-threaded for ``n_workers > 1``). Falls back to the
    pure-Python serial walk otherwise.

    ``max_crystals``, if set, enables early-exit head mode (forces
    single-threaded — racing past the threshold wastes I/O).
    """
    from torchsx.io import _native
    native_mod = _native.load()
    if native_mod is not None:
        mc = -1 if max_crystals is None else int(max_crystals)
        nt = 1 if mc > 0 else max(1, int(n_workers))
        return [tuple(t) for t in native_mod.index_chunks(path, nt, mc)]
    if max_crystals is not None:
        return _index_chunks_until(path, int(max_crystals))
    _, chunks = _index_chunks(path)
    return chunks
