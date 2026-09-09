"""Multiplicative-scatter density on ``log S`` around ``log_mean``.

The "scatter" layer models the irreducible multiplicative noise that a
scaling correction could not remove — crystal-to-crystal intensity
variation beyond what linear scale + B-factor capture.
"""

from torchsx.merging.scatter.base import ScatterModel
from torchsx.merging.scatter.gamma import GammaLogScatter
from torchsx.merging.scatter.gengauss import (
    GenGaussLogScatter,
    log_alpha_from_log_width,
)
from torchsx.merging.scatter.predicted import PredictedGenGaussScatter
from torchsx.merging.scatter.mixture import PredictedMixtureScatter
from torchsx.merging.scatter.gengauss_recentered import GenGaussLogScatterRecentered

__all__ = [
    "ScatterModel", "GenGaussLogScatter", "GammaLogScatter",
    "GenGaussLogScatterRecentered", "PredictedGenGaussScatter", "PredictedMixtureScatter",
    "log_alpha_from_log_width",
]
