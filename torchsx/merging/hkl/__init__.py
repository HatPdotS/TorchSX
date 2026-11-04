"""HKL-level data-prep primitives: indexing, filtering, completeness and
multiplicity, noise model.

Used by ``stream.py`` and by ``observations.prepare_observations``.  This is the
HKL/reflection bookkeeping layer of the merge engine — *not* crystallographic
lattice indexing, which happens upstream in probixi or CrystFEL.
"""

from torchsx.merging.hkl.hkl import (
    HKLIndex,
    compute_hkl_index,
    merge_intensities,
)
from torchsx.merging.hkl.filter import (
    filter_resolution,
    filter_ewald,
    filter_crystals,
    filter_images,
    filter_outliers,
    filter_cell,
)
from torchsx.merging.hkl.stats import (
    completeness,
    multiplicity,
)
from torchsx.merging.hkl.noise import PoissonNoiseModel

__all__ = [
    "HKLIndex",
    "compute_hkl_index",
    "merge_intensities",
    "filter_resolution",
    "filter_ewald",
    "filter_crystals",
    "filter_images",
    "filter_outliers",
    "filter_cell",
    "completeness",
    "multiplicity",
    "PoissonNoiseModel",
]
