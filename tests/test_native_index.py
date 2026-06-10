"""Invariants for the native chunk indexer.

The C++ ``index_chunks`` must produce a chunk list bit-exactly matching
the pure-Python ``_index_chunks`` reference, regardless of how many
threads it ran with. Three cheap invariants pin this down:

1. **Cross-thread invariance** — output is independent of ``n_threads``.
2. **Coverage** — chunks are contiguous and cover [header_end, file_size).
3. **Bit-exact vs. Python** — every tuple equals the Python reference.

The early-exit head-mode path is also covered.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from torchsx.io import _native
from torchsx.io._parser import _find_first_chunk_offset, _index_chunks


def _candidate_streams() -> list[Path]:
    """Return existing .stream files this test can run against."""
    paths: list[Path] = []
    try:
        from tests.conftest import STREAM_PATH
        paths.append(Path(STREAM_PATH))
    except Exception:
        pass
    paths.append(Path(
        "/data/project/cls/standfuss/torchsx/experiments/sfx_scan/streams/delay_-45.stream"
    ))
    return [p for p in paths if p.exists()]


@pytest.fixture(scope="module")
def stream_path() -> Path:
    cands = _candidate_streams()
    if not cands:
        pytest.skip("no real .stream file available")
    return cands[0]


@pytest.fixture(scope="module")
def native():
    mod = _native.load()
    if mod is None:
        pytest.skip("native extension not loaded")
    return mod


@pytest.mark.slow
def test_native_matches_python(stream_path: Path, native):
    """C++ single-threaded == Python serial, bit-exact."""
    py_chunks = _index_chunks(str(stream_path))[1]
    nat_chunks = [tuple(t) for t in native.index_chunks(str(stream_path), 1, -1)]
    assert nat_chunks == py_chunks, (
        f"native single-thread differs from Python: "
        f"py={len(py_chunks)} nat={len(nat_chunks)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
def test_cross_thread_invariance(stream_path: Path, native, n_threads: int):
    """index_chunks output is independent of n_threads."""
    ref = [tuple(t) for t in native.index_chunks(str(stream_path), 1, -1)]
    got = [tuple(t) for t in native.index_chunks(str(stream_path), n_threads, -1)]
    assert got == ref, (
        f"n_threads={n_threads} differs from serial: "
        f"len ref={len(ref)} got={len(got)}"
    )


@pytest.mark.slow
def test_coverage(stream_path: Path, native):
    """Chunks tile [header_end, file_size) with no gaps and no overlaps."""
    file_size = stream_path.stat().st_size
    header_end = _find_first_chunk_offset(str(stream_path))
    chunks = [tuple(t) for t in native.index_chunks(str(stream_path), 4, -1)]
    assert chunks, "no chunks indexed"

    assert chunks[0][0] == header_end
    assert chunks[-1][1] == file_size
    for a, b in zip(chunks, chunks[1:]):
        assert a[1] == b[0], f"gap/overlap at chunks {a} -> {b}"
        assert a[0] < a[1], f"non-positive chunk extent {a}"


@pytest.mark.slow
def test_head_mode_early_exit(stream_path: Path, native):
    """max_crystals>0 stops as soon as the threshold is crossed."""
    target = 100
    chunks = [tuple(t) for t in native.index_chunks(str(stream_path), 1, target)]
    n_crys = sum(c[2] for c in chunks)
    assert n_crys >= target
    # Should have stopped at the FIRST chunk that pushed us over the target,
    # so removing the last chunk drops us below.
    if len(chunks) > 1:
        assert sum(c[2] for c in chunks[:-1]) < target


@pytest.mark.slow
def test_head_mode_matches_python(stream_path: Path, native):
    """Native head-mode bit-exact against pure-Python _index_chunks_until."""
    from torchsx.io._parser import _index_chunks_until

    target = 100
    py = _index_chunks_until(str(stream_path), target)
    nat = [tuple(t) for t in native.index_chunks(str(stream_path), 1, target)]
    assert nat == py
