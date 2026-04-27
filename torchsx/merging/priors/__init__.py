"""Latent-intensity priors on ``log_I0``.

Each concrete prior is an ``nn.Module`` that owns any per-shell / per-HKL
buffers and returns either a per-HKL vector (MAP-style) or a scalar
(distribution-match style) from its ``forward(log_I0)``.  They are wrapped
into the loss by ``torchsx.merging.targets.prior.IntensityPriorTarget``.
"""

from torchsx.merging.priors.base import IntensityPrior
from torchsx.merging.priors.wilson import WilsonPrior
from torchsx.merging.priors.wilson_match import WilsonMatchPrior
from torchsx.merging.priors.flat import FlatPrior

__all__ = ["IntensityPrior", "WilsonPrior", "WilsonMatchPrior", "FlatPrior"]
