""".stream parser for SFX data — compatibility facade.

The implementation lives in the :mod:`torchsx.io._parse` package
(``indexing`` / ``records`` / ``workers`` / ``orchestrator``). This module
re-exports its names so ``from torchsx.io._parser import parse_stream`` and
the private helpers used by tests keep one import path. The public entry point
is :func:`parse_stream`.

Design: index the chunks, select the ones to read, parse them. The native
module (``torchsx_native``, see ``native/``) does both steps on threads; the
pure-Python fallback counts, then parses serially into preallocated arrays.
"""

from __future__ import annotations

# Re-export the full top-level surface (underscore names included; a bare
# ``import *`` would skip them).
from torchsx.io._parse.indexing import (  # noqa: F401
    _count_byte_range,
    _find_first_chunk_offset,
    _index_chunks,
    _index_chunks_native_or_python,
    _index_chunks_until,
)
from torchsx.io._parse.records import (  # noqa: F401
    REASONS,
    _allocate_arrays,
    _finalize,
    _line_iter,
    _parse_into_views,
)
from torchsx.io._parse.workers import _parse_byte_range  # noqa: F401
from torchsx.io._parse.orchestrator import (  # noqa: F401
    _build_core,
    _parse_chunks_python,
    _parse_stream,
    _select_head,
    _select_sampled,
    parse_stream,
)
