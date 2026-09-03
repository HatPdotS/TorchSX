"""HKL-level data-prep primitives: indexing, filtering, shell statistics,
noise model, adjacency.

Used by ``stream.py`` and by ``observations.prepare_observations``.  This is the
HKL/reflection bookkeeping layer of the merge engine — *not* crystallographic
lattice indexing, which happens upstream in probixi or CrystFEL.
"""

from torchsx.merging.hkl.hkl import (
    HKLIndex,
    compute_hkl_index,
    merge_intensities,
    merge_median,
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
    match_halves,
    MatchedHalves,
    ShellStats,
    cc_half,
    cc_star,
    r_split,
    spearman_cc,
    shell_stats,
    format_shell_table,
    write_shell_csv,
)
from torchsx.merging.hkl.noise import (
    PoissonNoiseModel,
    compute_sigma2_LN,
    compute_sigma2_LN_exact,
)
from torchsx.merging.hkl.shell_fit import fit_shell_gengauss_gauss
from torchsx.merging.hkl.adjacency import Adjacency, build_adjacency, dirichlet_energy

__all__ = [
    "HKLIndex",
    "compute_hkl_index",
    "merge_intensities",
    "merge_median",
    "filter_resolution",
    "filter_ewald",
    "filter_crystals",
    "filter_images",
    "filter_outliers",
    "filter_cell",
    "completeness",
    "multiplicity",
    "match_halves",
    "MatchedHalves",
    "ShellStats",
    "cc_half",
    "cc_star",
    "r_split",
    "spearman_cc",
    "shell_stats",
    "format_shell_table",
    "write_shell_csv",
    "PoissonNoiseModel",
    "compute_sigma2_LN",
    "compute_sigma2_LN_exact",
    "fit_shell_gengauss_gauss",
    "Adjacency",
    "build_adjacency",
    "dirichlet_energy",
]
