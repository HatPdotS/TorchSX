"""Per-observation log-scale correction models.

A ``ScaleModel`` returns the additive ``log K`` that multiplies the latent
intensity: ``log_mean = log I_true + log K``.  Composite scaling is
implemented by ``CompositeScale`` (sum over a list of ``ScaleModel``).
"""

from torchsx.merging.scale.base import ScaleModel
from torchsx.merging.scale.per_crystal import ScaleBPerCrystal
from torchsx.merging.scale.panel import PanelScale
from torchsx.merging.scale.fixed import FixedScale
from torchsx.merging.scale.composite import CompositeScale
from torchsx.merging.scale.amortized import AmortizedScale

__all__ = [
    "ScaleModel",
    "ScaleBPerCrystal",
    "PanelScale",
    "FixedScale",
    "CompositeScale",
    "AmortizedScale",
]
