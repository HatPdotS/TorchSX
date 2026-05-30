"""Data reduction: raw JUNGFRAU frames → spots → indexed crystals.

The pipeline foundation that turns SwissFEL serial-crystallography frames into
indexed observations, built around the cloned ``peakfinder8`` (spot finding) and
the SDSC PyTorch ``TORO`` indexer.

Typical flow::

    from torchsx.geometry import Geometry
    from torchsx.datareduction import (
        JungfrauReader, find_spots_on_frame, spots_to_q, index_spots, HEWL_CELL,
    )

    geom = Geometry.from_geom_file("16M_v1_distance_93.8.geom")
    reader = JungfrauReader("acq0001.JF06T32V07.h5")
    spots = find_spots_on_frame(reader, 60, geom)          # (M, 4)
    q = spots_to_q(spots, geom)                            # (M, 3) Å⁻¹
    result = index_spots(q, HEWL_CELL)                     # IndexResult
"""

from .frames import JungfrauReader, list_run
from .spotfinding import (
    PeakFinder8Params,
    find_spots,
    find_spots_on_frame,
    spots_to_q,
)
from .indexing import HEWL_CELL, IndexResult, index_spots
from .curate import FrameRecord, curate_file, parse_dap, save_hits
from .to_stream import build_stream, crystal_to_chunk, reciprocal_vectors

__all__ = [
    "JungfrauReader",
    "list_run",
    "PeakFinder8Params",
    "find_spots",
    "find_spots_on_frame",
    "spots_to_q",
    "index_spots",
    "IndexResult",
    "HEWL_CELL",
    "FrameRecord",
    "curate_file",
    "parse_dap",
    "save_hits",
    "build_stream",
    "crystal_to_chunk",
    "reciprocal_vectors",
]
