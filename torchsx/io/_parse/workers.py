"""Serial pure-Python parse of a byte range (the fallback when the native
module is missing)."""

from __future__ import annotations

from torchsx.io._parse.indexing import _count_byte_range
from torchsx.io._parse.records import (
    _allocate_arrays,
    _finalize,
    _line_iter,
    _parse_into_views,
)


def _parse_byte_range(
    path: str,
    start: int,
    end: int,
    panel_name_to_id: dict[str, int],
) -> tuple[dict, list[str], list[str], dict[str, list[int]]]:
    """Two-pass parse of ``[start, end)``: count, then parse into arrays.

    Returns ``(tensors, filenames, events, report)``.
    """
    n_obs_alloc, n_crys, n_imgs = _count_byte_range(path, start, end)
    v = _allocate_arrays(n_obs_alloc, n_crys, n_imgs)
    filenames: list[str] = []
    events: list[str] = []
    report: dict[str, list[int]] = {}
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        n_cry_got, n_img_got, n_obs_got = _parse_into_views(
            _line_iter(f, end), panel_name_to_id, v, filenames, events, report
        )
    return _finalize(v, n_obs_got, n_cry_got, n_img_got), filenames, events, report
