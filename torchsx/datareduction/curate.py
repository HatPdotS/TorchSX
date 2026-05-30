"""Curate a run: which frames actually contain crystals?

Sweeps the frames of an acquisition file, finds spots, and classifies hits by
peak count.  Cross-checks against the SwissFEL online-monitor ``.dap`` file (a
sparse per-pulse log: ``pulse_id  _  hit_flag  n_peaks  _``) and can optionally
*verify* candidate hits by actually indexing them — the strongest evidence that
a pattern contains a crystal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..geometry import Geometry
from .frames import JungfrauReader
from .indexing import HEWL_CELL, index_spots
from .spotfinding import PeakFinder8Params, find_spots_on_frame, spots_to_q


def parse_dap(path: str | Path) -> dict[int, tuple[bool, int]]:
    """Parse a ``.dap`` online-monitor file → ``{pulse_id: (hit_flag, n_peaks)}``.

    Note the file is **sparse** — only a subset of frames is logged.
    """
    out: dict[int, tuple[bool, int]] = {}
    rows = np.loadtxt(path, dtype=str, ndmin=2)
    for row in rows:
        out[int(row[0])] = (row[2] == "True", int(row[3]))
    return out


@dataclass
class FrameRecord:
    frame: int
    pulse_id: int
    n_peaks: int
    is_good: bool
    is_hit: bool
    dap_hit: bool | None = None
    dap_n_peaks: int | None = None
    indexed: bool | None = None
    n_indexed: int | None = None


def curate_file(
    reader: JungfrauReader,
    geom: Geometry,
    *,
    params: PeakFinder8Params | None = None,
    min_peaks: int = 20,
    stride: int = 1,
    max_frames: int | None = None,
    dap_path: str | Path | None = None,
    verify_index: bool = False,
    initial_cell: torch.Tensor = HEWL_CELL,
    device: str = "cpu",
) -> tuple[list[FrameRecord], dict]:
    """Sweep frames and classify hits.

    Returns ``(records, summary)``.  A frame is a *hit* when its peak count is
    ``>= min_peaks`` (and it is flagged good).  With ``verify_index=True`` each
    hit is additionally run through the indexer.
    """
    params = params or PeakFinder8Params()
    dap = parse_dap(dap_path) if dap_path else None

    n = reader.n_frames if max_frames is None else min(max_frames, reader.n_frames)
    records: list[FrameRecord] = []
    n_scanned = 0
    for i in range(0, n, stride):
        n_scanned += 1
        good = reader.is_good_frame(i)
        pulse = reader.pulse_id(i)
        if not good:
            records.append(FrameRecord(i, pulse, 0, False, False))
            continue
        spots = find_spots_on_frame(reader, i, geom, params)
        n_peaks = int(spots.shape[0])
        is_hit = n_peaks >= min_peaks
        rec = FrameRecord(i, pulse, n_peaks, True, is_hit)
        if dap is not None and pulse in dap:
            rec.dap_hit, rec.dap_n_peaks = dap[pulse]
        if is_hit and verify_index:
            q = spots_to_q(spots, geom)
            res = index_spots(q, initial_cell, device=device)
            rec.indexed = res.success
            rec.n_indexed = res.n_indexed
        records.append(rec)

    hits = [r for r in records if r.is_hit]
    summary = {
        "file": reader.path,
        "n_scanned": n_scanned,
        "n_good": sum(r.is_good for r in records),
        "n_hits": len(hits),
        "hit_rate": len(hits) / max(n_scanned, 1),
        "min_peaks": min_peaks,
    }
    if verify_index:
        summary["n_indexed_hits"] = sum(bool(r.indexed) for r in hits)
    return records, summary


def save_hits(records: list[FrameRecord], path: str | Path) -> None:
    """Write the curated hit list (hits only) to JSON."""
    hits = [asdict(r) for r in records if r.is_hit]
    Path(path).write_text(json.dumps(hits, indent=2))
