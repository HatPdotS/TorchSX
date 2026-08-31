"""Data reduction: raw JUNGFRAU frames → spots → indexed crystals.

The pipeline foundation that turns SwissFEL serial-crystallography frames into
indexed observations, built around the cloned ``peakfinder8`` (spot finding) and
the SDSC PyTorch ``TORO`` indexer.

The common currency is the **spot table**, a ``(M, C)`` float tensor whose
columns are defined once in :mod:`._schema`::

    0 PANEL   1 SS   2 FS   3 INTENSITY   [4 PROB]

Two finder families emit this same table:

* :mod:`.spotfinding` — the compiled pf8 / threshold backends (and the
  ``find_spots_on_frame`` dispatcher + reciprocal mapping). Emits ``(M, 4)``.
* :mod:`.peakfinder` — the native-torch reimplementation, the thresholdless
  *probabilistic* finder (``find_peaks_prob``), and the fully-Bayesian
  matched-filter finder (``find_peaks_bayes``), both of which add the ``PROB``
  column → ``(M, 5)``.

Use ``find_spots_on_frame`` as the entry point; it selects a backend.

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
    find_spots,  # low-level pf8 wrapper; importable for back-compat, not in __all__
    find_spots_on_frame,
    spots_to_q,
)
from .peakfinder import (
    TorchPeakFinderParams,
    find_peaks_bayes,
    find_peaks_prob,
    find_spots_torch,
    radial_background,
)
from .spot_profile import SpotProfile
from .refine import (
    RefineResult,
    crystals_from_stream,
    lattice_log_likelihood,
    refine_geometry,
    refine_lattice,
)
from .noise_model import (
    EmpiricalNull,
    NoiseModel,
    calibrate_artefact_mask,
    calibrate_fixed_pattern,
    calibrate_noise,
    calibrate_null,
    calibrate_spot_profile,
)
from .calibration import (
    Calibration,
    EmpiricalNoise,
    PerPixelModel,
    calibrate_empirical_noise,
    calibrate_per_pixel,
    whitened_residual,
)
from .masking import (
    build_detector_mask,
    load_external_mask,
    load_mask,
    save_mask,
)
from .indexing import HEWL_CELL, IndexResult, index_spots
from .curate import FrameRecord, curate_file, parse_dap, save_hits
from .to_stream import build_stream, crystal_to_chunk, reciprocal_vectors

__all__ = [
    "JungfrauReader",
    "list_run",
    "PeakFinder8Params",
    "find_spots_on_frame",
    "find_spots_torch",
    "find_peaks_prob",
    "find_peaks_bayes",
    "radial_background",
    "TorchPeakFinderParams",
    "SpotProfile",
    "NoiseModel",
    "EmpiricalNull",
    "calibrate_artefact_mask",
    "calibrate_fixed_pattern",
    "calibrate_noise",
    "calibrate_null",
    "calibrate_spot_profile",
    "Calibration",
    "EmpiricalNoise",
    "PerPixelModel",
    "calibrate_empirical_noise",
    "calibrate_per_pixel",
    "whitened_residual",
    "build_detector_mask",
    "load_external_mask",
    "load_mask",
    "save_mask",
    "refine_lattice",
    "refine_geometry",
    "crystals_from_stream",
    "lattice_log_likelihood",
    "RefineResult",
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
