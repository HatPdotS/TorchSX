"""Per-observation log-scale correction models.

A ``ScaleModel`` returns the additive ``log K`` that multiplies the latent
intensity: ``log_mean = log I_true + log K``.
"""

from torchsx.merging.scale.amortized import AmortizedScale
from torchsx.merging.scale.base import ScaleModel

__all__ = ["AmortizedScale", "ScaleModel"]
