"""Multiplicative-scatter density on ``log S`` around ``log_mean``.

The "scatter" layer models the irreducible multiplicative noise that a
scaling correction could not remove — crystal-to-crystal intensity
variation beyond what linear scale + B-factor capture.
"""

from torchsx.merging.scatter.base import ScatterModel
from torchsx.merging.scatter.gengauss import (
    GenGaussLogScatter,
    log_alpha_from_log_width,
)
from torchsx.merging.scatter.mixture import PredictedMixtureScatter
from torchsx.merging.scatter.predicted import PredictedGenGaussScatter

__all__ = [
    "GenGaussLogScatter",
    "PredictedGenGaussScatter",
    "PredictedMixtureScatter",
    "ScatterModel",
    "log_alpha_from_log_width",
]
