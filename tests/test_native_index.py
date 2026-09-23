"""Invariants for the native chunk indexer.

``torchsx_native.index_chunks`` must return exactly what the pure-Python
``_index_chunks`` / ``_index_chunks_until`` return, whatever the thread count.
Chunk markers end reflection tables in both, which is what keeps the threaded
scan equal to the serial one even for truncated tables.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from torchsx.io import _native
from torchsx.io._parser import _find_first_chunk_offset, _index_chunks, _index_chunks_until


def _rows(arr) -> list[tuple[int, int, int, int]]:
    return [tuple(int(x) for x in r) for r in arr]


@pytest.fixture(scope="module")
def native():
    mod = _native.load()
    if mod is None:
        pytest.skip("native module not installed")
    return mod


@pytest.mark.slow
def test_native_matches_python(real_stream: Path, native):
    """Native single-threaded == Python serial, exactly."""
    py_chunks = _index_chunks(str(real_stream))[1]
    assert _rows(native.index_chunks(str(real_stream), 1)) == py_chunks


@pytest.mark.slow
@pytest.mark.parametrize("n_threads", [2, 4, 8])
def test_cross_thread_invariance(real_stream: Path, native, n_threads: int):
    ref = _rows(native.index_chunks(str(real_stream), 1))
    assert _rows(native.index_chunks(str(real_stream), n_threads)) == ref


@pytest.mark.slow
def test_coverage(real_stream: Path, native):
    """Chunks tile [header_end, file_size) with no gaps and no overlaps."""
    file_size = real_stream.stat().st_size
    header_end = _find_first_chunk_offset(str(real_stream))
    chunks = _rows(native.index_chunks(str(real_stream), 4))
    assert chunks[0][0] == header_end
    assert chunks[-1][1] == file_size
    for a, b in zip(chunks, chunks[1:]):
        assert a[1] == b[0], f"gap/overlap at chunks {a} -> {b}"
        assert a[0] < a[1], f"non-positive chunk extent {a}"


@pytest.mark.slow
def test_head_mode(real_stream: Path, native):
    """Head mode stops at the first chunk reaching the target, as in Python."""
    target = 100
    chunks = _rows(native.index_chunks(str(real_stream), 1, target))
    assert sum(c[2] for c in chunks) >= target
    assert sum(c[2] for c in chunks[:-1]) < target
    assert chunks == _index_chunks_until(str(real_stream), target)


def _chunk(i: int, table: str) -> str:
    return (
        "----- Begin chunk -----\n"
        f"Image filename: img{i}.h5\n"
        "Peaks from peak search\nEnd of peak list\n"
        "--- Begin crystal\n"
        "Reflections measured after indexing\n"
        "   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n"
        f"{table}"
        "----- End chunk -----\n"
    )


def test_truncated_tables_split_chunks(native, tmp_path: Path):
    """A table without ``End of reflections`` ends at the next chunk marker."""
    line = "   1    2    3 1.0 2.0 3.0 4.0 5.0 6.0 p0\n"
    good = line * 3 + "End of reflections\n--- End crystal\n"
    cut = line * 2  # no End of reflections
    text = "header\n" + "".join(_chunk(i, cut if i % 3 == 1 else good) for i in range(40))
    path = tmp_path / "cut.stream"
    path.write_text(text)
    py = _index_chunks(str(path))[1]
    assert len(py) == 40
    assert [c[3] for c in py[:3]] == [3, 2, 3]
    for n in (1, 2, 3, 7):
        assert _rows(native.index_chunks(str(path), n)) == py
    assert _rows(native.index_chunks(str(path), 1, 5)) == _index_chunks_until(str(path), 5)


def test_empty_index(native, tmp_path: Path):
    path = tmp_path / "none.stream"
    path.write_text("CrystFEL stream format 2.3\nno chunks here\n")
    assert len(native.index_chunks(str(path), 4)) == 0
    with pytest.raises(ValueError, match="No chunks"):
        _index_chunks(str(path))
