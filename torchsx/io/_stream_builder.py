"""StreamBuilder: async, chunked construction of a stream core.

A producer (GPU ingestion stage, a reader thread, …) ``initialize``s a builder
from the join structure and then ``add``s :class:`StreamChunk`s — one per
frame / frame-batch — and returns immediately. A background writer thread does
the "registration" (applies running join-key offsets, appends columns,
maintains a ``key → row-range`` index) so the producer never blocks on it,
mirroring how the .stream parser writes into preallocated shared-memory views.
``finalize`` drains the queue and builds the immutable stream core.

This is the unbounded (queue + writer-thread) path; sizes need not be known a
priori. A preallocated-slot variant (truest analogue of the parser) can be
layered on later for the known-capacity case.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import torch

from torchsx._core import BaseStream, JoinSpec, _default_sx_joins
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata
from torchsx.levels import Level


@dataclass
class StreamChunk:
    """A chunk of rows for one or more levels, with chunk-local join keys.

    ``levels`` maps a level name to its column dict; each column is a tensor
    whose leading dim is that level's row count *within this chunk*. Join-key
    columns (``crystal_id``, ``image_id``) are local to the chunk — the builder
    offsets them on registration. ``key`` optionally tags the chunk (e.g. a
    frame id or ``(filename, event)``) for the row-range index.
    """

    levels: dict[str, dict[str, torch.Tensor]]
    key: object | None = None


_STOP = object()


class StreamBuilder:
    """Async accumulator that finalizes into an immutable stream core."""

    def __init__(
        self,
        joins: list[JoinSpec] | None = None,
        *,
        geometry: Geometry | None = None,
        metadata: StreamMetadata | None = None,
        spacegroup: str = "P 1",
        source_path: str = "",
        async_writer: bool = True,
    ):
        self.joins = joins if joins is not None else _default_sx_joins()
        self.geometry = geometry
        self.metadata = metadata if metadata is not None else StreamMetadata()
        self.spacegroup = spacegroup
        self.source_path = source_path

        # parent level → running row count (drives join-key offsetting)
        self._cum: dict[str, int] = {}
        # level → {column → list of chunk tensors}
        self._buffers: dict[str, dict[str, list[torch.Tensor]]] = {}
        # key → {level → (start, stop)} row ranges
        self.index: dict[object, dict[str, tuple[int, int]]] = {}
        self._level_rows: dict[str, int] = {}
        self._error: BaseException | None = None

        self._async = async_writer
        if async_writer:
            self._queue: queue.Queue = queue.Queue()
            self._thread = threading.Thread(target=self._drain, daemon=True)
            self._thread.start()

    @classmethod
    def initialize(cls, **kwargs) -> "StreamBuilder":
        """Alias for the constructor (declares joins/geometry/spacegroup)."""
        return cls(**kwargs)

    # ---- producer-facing API ------------------------------------------

    def add(self, chunk: StreamChunk) -> None:
        """Hand off a chunk. Non-blocking when ``async_writer=True``."""
        if self._async:
            if self._error is not None:
                raise self._error
            self._queue.put(chunk)
        else:
            self._register(chunk)

    # ---- writer side --------------------------------------------------

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self._register(item)
            except BaseException as exc:  # surface to producer/finalize
                self._error = exc
            finally:
                self._queue.task_done()

    def _register(self, chunk: StreamChunk) -> None:
        # Parent levels must be registered before children offset into them;
        # process levels in their chunk order but offset using running counts.
        chunk_rows: dict[str, int] = {}
        for level_name, cols in chunk.levels.items():
            if not cols:
                continue
            n = next(iter(cols.values())).shape[0]
            chunk_rows[level_name] = n

        # apply join-key offsets (child key += cum[parent])
        offset_cols = {ln: dict(cols) for ln, cols in chunk.levels.items()}
        for join in self.joins:
            child = offset_cols.get(join.child_level)
            if child is None or join.key_column not in child:
                continue
            base = self._cum.get(join.parent_level, 0)
            if base:
                child[join.key_column] = child[join.key_column] + base

        # append to buffers + record row ranges
        ranges: dict[str, tuple[int, int]] = {}
        for level_name, cols in offset_cols.items():
            if not cols:
                continue
            buf = self._buffers.setdefault(level_name, {})
            start = self._level_rows.get(level_name, 0)
            for cname, tensor in cols.items():
                buf.setdefault(cname, []).append(tensor)
            stop = start + chunk_rows[level_name]
            self._level_rows[level_name] = stop
            ranges[level_name] = (start, stop)

        # bump running parent counts for the NEXT chunk's offsets
        for level_name, n in chunk_rows.items():
            self._cum[level_name] = self._cum.get(level_name, 0) + n

        if chunk.key is not None:
            self.index[chunk.key] = ranges

    # ---- finalize -----------------------------------------------------

    def finalize(self) -> BaseStream:
        """Drain pending chunks and build the immutable stream core."""
        if self._async:
            self._queue.put(_STOP)
            self._thread.join()
            if self._error is not None:
                raise self._error

        levels: dict[str, Level] = {}
        for level_name, cols in self._buffers.items():
            lvl = Level(n=0 if not cols else None)
            for cname, parts in cols.items():
                lvl[cname] = torch.cat(parts, dim=0)
            levels[level_name] = lvl

        return BaseStream._make(
            levels, self.joins, self.geometry, self.metadata,
            self.spacegroup, self.source_path,
        )
