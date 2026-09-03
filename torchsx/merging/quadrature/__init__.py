"""Latent-signal quadrature: integrates ``S`` out of the joint per observation."""

from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.trapezoidal import TrapezoidalLogGrid
from torchsx.merging.quadrature.mixed_delta import MixedDeltaTrapezoidal
from torchsx.merging.quadrature.moment_matched import MomentMatchedQuadrature
from torchsx.merging.quadrature.delta_at_es import DeltaAtESQuadrature
from torchsx.merging.quadrature.hybrid_log import HybridLogGrid
from torchsx.merging.quadrature.adaptive_hybrid_log import AdaptiveHybridLogGrid
from torchsx.merging.quadrature.adaptive_signif import AdaptiveSignifSplit
from torchsx.merging.quadrature.laplace_saddle import LaplaceSaddleGrid
from torchsx.merging.quadrature.hierarchical_mc import HierarchicalImportanceQuadrature
from torchsx.merging.quadrature.wide_log_grid import WideLogGrid
from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
from torchsx.merging.quadrature.precomputed import (
    PrecomputedGridQuadrature, PrecomputedGridSpec,
)

__all__ = [
    "Quadrature", "TrapezoidalLogGrid", "MixedDeltaTrapezoidal",
    "MomentMatchedQuadrature", "DeltaAtESQuadrature", "HybridLogGrid",
    "AdaptiveHybridLogGrid", "AdaptiveSignifSplit", "LaplaceSaddleGrid",
    "HierarchicalImportanceQuadrature", "WideLogGrid", "FixedLogGrid", "SplitLogGrid",
    "PrecomputedGridQuadrature", "PrecomputedGridSpec",
]
