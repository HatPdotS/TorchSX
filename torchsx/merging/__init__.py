"""Modular SFX merging pipeline.

Replaces ``torchsx.merging_clean``.  The package is organised around the
generative-model layers from ``design_doc.md`` §3:

    IntensityPrior → ScaleModel × ScatterModel → Quadrature → DetectionLikelihood

Each layer is an ``nn.Module`` behind a narrow abstract base; a single
``torchref.refinement.loss_state.LossState`` aggregates them into one
scalar loss that LBFGS minimises.

Public API is populated incrementally as subpackages land.  See
``torchsx.merging.parameters``, ``torchsx.merging.observations``,
``torchsx.merging.config``, and ``torchsx.merging.hkl``.
"""

from torchsx.merging.config import MergeConfig
from torchsx.merging.parameters import Parameters
from torchsx.merging.observations import PreparedObs, ObsInfo, prepare_observations
from torchsx.merging.merger import Merger
from torchsx.merging.result import MergedResult

# Streamlined chain: stream → loss → posterior → ReflectionData.
from torchsx.merging.loss import LogSpaceLoss
from torchsx.merging.posterior import Posterior

# Re-export the scale models most experiment scripts need.
from torchsx.merging.scale import (
    ScaleModel,
    ScaleBPerCrystal,
    PanelScale,
    FixedScale,
    CompositeScale,
)

__all__ = [
    "MergeConfig",
    "Parameters",
    "PreparedObs",
    "ObsInfo",
    "prepare_observations",
    "Merger",
    "MergedResult",
    "LogSpaceLoss",
    "Posterior",
    "ScaleModel",
    "ScaleBPerCrystal",
    "PanelScale",
    "FixedScale",
    "CompositeScale",
]
