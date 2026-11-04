"""The pure-Python ``.stream`` parser: the reference the native parser is
tested against.

``index_chunks`` and ``parse_chunks`` take and return what the
``torchsx_native`` functions of the same names do, so this package can stand
in for the native module (``monkeypatch.setattr(_native, "load", lambda:
python_parser)``) and drive the whole read path. ``parse_byte_range`` is a
second route that needs no index: count, then parse a byte range.
"""

from __future__ import annotations

import numpy as np

from tests.python_parser.indexing import _count_byte_range, _index_chunks_until
from tests.python_parser.records import (
    _allocate_arrays,
    _finalize,
    _line_iter,
    _parse_into_views,
)


def index_chunks(path, n_threads: int = 1, max_crystals: int | None = None) -> np.ndarray:
    """``(n, 4)`` int64 rows ``(start, end, n_crystals, n_obs_upper)`` per chunk."""
    return np.asarray(_index_chunks_until(str(path), max_crystals), dtype=np.int64).reshape(-1, 4)


def parse_chunks(path, chunks, panel_names, n_threads: int = 1):
    """``(columns, filenames, events, report)`` of the given chunk rows."""
    chunks = np.asarray(chunks, dtype=np.int64).reshape(-1, 4)
    v = _allocate_arrays(int(chunks[:, 3].sum()), int(chunks[:, 2].sum()), len(chunks))
    filenames: list[str] = []
    events: list[str] = []
    report: dict[str, list[int]] = {}
    crystal_idx = image_idx = n_obs = 0
    with open(path, "rb", buffering=1 << 20) as f:
        for start, end, _, _ in chunks.tolist():
            f.seek(start)
            crystal_idx, image_idx, n_obs = _parse_into_views(
                _line_iter(f, end), _panel_ids(panel_names), v, filenames, events, report,
                crystal_idx, image_idx, n_obs,
            )
    return _columns(_finalize(v, n_obs, crystal_idx, image_idx)), filenames, events, _report(report)


def parse_byte_range(path, start: int, end: int, panel_names):
    """:func:`parse_chunks` of ``[start, end)`` in two passes: count, then parse."""
    v = _allocate_arrays(*_count_byte_range(str(path), start, end))
    filenames: list[str] = []
    events: list[str] = []
    report: dict[str, list[int]] = {}
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        n_crys, n_imgs, n_obs = _parse_into_views(
            _line_iter(f, end), _panel_ids(panel_names), v, filenames, events, report
        )
    return _columns(_finalize(v, n_obs, n_crys, n_imgs)), filenames, events, _report(report)


def _panel_ids(panel_names) -> dict[str, int]:
    return {name: i for i, name in enumerate(panel_names)}


def _columns(tensors) -> dict[str, np.ndarray]:
    return {k: t.numpy() for k, t in tensors.items()}


def _report(report: dict[str, list[int]]) -> dict[str, tuple[int, int]]:
    return {k: tuple(v) for k, v in report.items()}
