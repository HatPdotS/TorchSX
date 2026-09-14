"""SFX merging: observation preparation and the generative-model layers.

The model an observation's intensity is drawn from, in the order the code
composes it::

    ScaleModel x ScatterModel -> Quadrature -> DetectionLikelihood

Each layer is an ``nn.Module`` behind a narrow abstract base.  The latent
per-reflection intensity is integrated out by the quadrature rather than
estimated, so the objective is a marginal evidence; see
:mod:`torchsx.nn.objective`.

Import the layer you need from its own subpackage
(:mod:`torchsx.merging.scale`, :mod:`~torchsx.merging.scatter`,
:mod:`~torchsx.merging.quadrature`, :mod:`~torchsx.merging.likelihood`).
This module re-exports only the observation-preparation surface, which
everything else is built on.
"""

from torchsx.merging.config import MergeConfig
from torchsx.merging.observations import ObsInfo, PreparedObs, prepare_observations
from torchsx.merging.parameters import Parameters

__all__ = [
    "MergeConfig",
    "ObsInfo",
    "Parameters",
    "PreparedObs",
    "prepare_observations",
]
