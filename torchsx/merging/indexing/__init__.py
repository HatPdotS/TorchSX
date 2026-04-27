"""HKL indexing, filtering, shell statistics, noise model, adjacency.

These are data-preparation primitives used by ``stream.py`` and by
``observations.prepare_observations``.  Ported from the legacy
``torchsx.merge`` package; that package will be moved to
``torchsx.deprecated`` once the refactor lands end-to-end.
"""

from torchsx.merging.indexing.hkl import (
    HKLIndex,
    compute_hkl_index,
    merge_intensities,
    merge_median,
)
from torchsx.merging.indexing.filter import (
    filter_resolution,
    filter_crystals,
    filter_images,
    filter_outliers,
    filter_cell,
)
from torchsx.merging.indexing.stats import (
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
from torchsx.merging.indexing.noise import (
    PoissonNoiseModel,
    compute_sigma2_LN,
    compute_sigma2_LN_exact,
)
from torchsx.merging.indexing.shell_fit import fit_shell_gengauss_gauss
from torchsx.merging.indexing.adjacency import Adjacency, build_adjacency, dirichlet_energy

__all__ = [
    "HKLIndex",
    "compute_hkl_index",
    "merge_intensities",
    "merge_median",
    "filter_resolution",
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
