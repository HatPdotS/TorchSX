"""Auxiliary scalar regularisers — penalties that don't belong to a single
loss module (scale, scatter, likelihood, prior) and instead act on the shared
``Parameters`` container.
"""

from torchsx.merging.regularizers.base import Regularizer
from torchsx.merging.regularizers.miller_graph import MillerGraphSmoothness
from torchsx.merging.regularizers.anchor import GaussianAnchor
from torchsx.merging.regularizers.sigma_cauchy import SigmaCauchy

__all__ = [
    "Regularizer",
    "MillerGraphSmoothness",
    "GaussianAnchor",
    "SigmaCauchy",
]
