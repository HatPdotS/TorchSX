""".stream parser for SFX data — compatibility facade.

The implementation was split into the :mod:`torchsx.io._parse` package
(``indexing`` / ``records`` / ``workers`` / ``orchestrator``) for readability.
This module re-exports every name that used to be defined here, so existing
imports — ``from torchsx.io._parser import parse_stream`` and the private
helpers used by tests and the profiling scripts under ``experiments/`` — keep
working unchanged. The public entry point is :func:`parse_stream`.

Design (unchanged): a two-pass, memory-bounded parser. Pass 1 counts images /
crystals / reflection lines by raw-byte scanning; pass 2 parses directly into
preallocated tensors via numpy views. Multi-worker parses split the file on
chunk boundaries and write into shared-memory tensors; an optional JIT-compiled
C++ extension accelerates both passes.
"""

from __future__ import annotations

# Re-export the full former top-level surface (underscore names included; a
# bare ``import *`` would skip them). Keep this list complete — tests and
# experiment scripts import several of these private helpers by name.
from torchsx.io._parse.indexing import (  # noqa: F401
    _coarse_chunk_split,
    _count_byte_range,
    _find_chunk_at_or_after,
    _find_first_chunk_offset,
    _index_chunks,
    _index_chunks_native_or_python,
    _index_chunks_until,
    _split_work,
)
from torchsx.io._parse.records import (  # noqa: F401
    _CRYS_TRIM_FIELDS,
    _IMG_TRIM_FIELDS,
    _OBS_TRIM_FIELDS,
    _allocate_tensors,
    _finalize,
    _is_valid_float,
    _line_iter,
    _parse_into_views,
    _parse_vec_line,
)
from torchsx.io._parse.workers import (  # noqa: F401
    _count_byte_range_args,
    _native_count_args,
    _native_parse_args,
    _parse_byte_range,
    _parse_byte_range_args,
    _parse_into_shared,
    _parse_into_shared_args,
)
from torchsx.io._parse.orchestrator import (  # noqa: F401
    _CRYS_FIELDS_NO_IMAGE,
    _IMG_FIELDS,
    _OBS_FIELDS,
    _assemble_core,
    _assemble_single,
    _finalize_shared,
    _parse_stream_head,
    _parse_stream_sampled,
    parse_stream,
)
