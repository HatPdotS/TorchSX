"""Clean-rewrite merging pipeline: single LBFGS minimization of the
Poisson-marginal GenGauss model with explicit regularizers.

Public API
----------
>>> from torchsx.merging_clean import Merger, MergeConfig, ScaleBCorrection
>>> result = Merger(stream, corrections=[ScaleBCorrection()]).merge()
"""

from torchsx.merging_clean.config import MergeConfig
from torchsx.merging_clean.result import MergedResult
from torchsx.merging_clean.corrections import (
    Correction,
    ScaleBCorrection,
    PanelCorrection,
    FixedCorrection,
    ObsInfo,
)
from torchsx.merging_clean.merger import Merger
from torchsx.merging_clean.loss import UnifiedMergeLoss

__all__ = [
    "Merger",
    "MergeConfig",
    "MergedResult",
    "Correction",
    "ScaleBCorrection",
    "PanelCorrection",
    "FixedCorrection",
    "ObsInfo",
    "UnifiedMergeLoss",
]
