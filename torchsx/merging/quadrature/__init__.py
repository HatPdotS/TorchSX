"""Latent-signal quadrature: integrates ``S`` out of the joint per observation."""

from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.trapezoidal import TrapezoidalLogGrid
from torchsx.merging.quadrature.mixed_delta import MixedDeltaTrapezoidal
from torchsx.merging.quadrature.moment_matched import MomentMatchedQuadrature
from torchsx.merging.quadrature.delta_at_es import DeltaAtESQuadrature
from torchsx.merging.quadrature.hybrid_log import HybridLogGrid
from torchsx.merging.quadrature.adaptive_hybrid_log import AdaptiveHybridLogGrid
from torchsx.merging.quadrature.adaptive_signif import AdaptiveSignifSplit

__all__ = [
    "Quadrature", "TrapezoidalLogGrid", "MixedDeltaTrapezoidal",
    "MomentMatchedQuadrature", "DeltaAtESQuadrature", "HybridLogGrid",
    "AdaptiveHybridLogGrid", "AdaptiveSignifSplit",
]
