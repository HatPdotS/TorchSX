"""Merge submodule: HKL indexing, merging, filtering, statistics."""

from torchsx.merge._filter import (
    filter_cell,
    filter_crystals,
    filter_images,
    filter_outliers,
    filter_resolution,
)
from torchsx.merge._hkl import HKLIndex, compute_hkl_index, merge_intensities, merge_median
from torchsx.merge._stats import (
    MatchedHalves,
    ShellStats,
    cc_half,
    cc_star,
    completeness,
    format_shell_table,
    match_halves,
    multiplicity,
    r_split,
    shell_stats,
    write_shell_csv,
)
from torchsx.merge._corrections import (
    Correction,
    CorrectionPipeline,
    FixedCorrection,
    ObsInfo,
    ScaleBCorrection,
    PanelCorrection,
)
from torchsx.merge._noise import PoissonNoiseModel
from torchsx.merge._em import EMConfig, fit_em
from torchsx.merge._dstep import (
    DStepResult,
    QuadratureGrid,
    build_softplus_grid,
    d_step,
    fit_I0_poisson,
    fit_alpha,
)
from torchsx.merge._distribution import (
    DistributionComparison,
    DistShellStats,
    HKLDistribution,
    compare_distributions,
    fit_gengauss,
    format_dist_table,
    split_and_compare,
)
from torchsx.merge._structure_factor import MergedStructureFactor
