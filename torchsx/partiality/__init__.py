"""Partiality modelling: excitation error, profile fitting, orientation refinement."""

from torchsx.partiality._excitation import ExcitationError
from torchsx.partiality._profile import (
    PartialityProfile,
    GaussianProfile,
    GenLorentzProfile,
)
from torchsx.partiality._concordance import RankConcordanceLoss
from torchsx.partiality._refine import (
    refine_orientations,
    RefinementConfig,
    RefinementResult,
)
from torchsx.partiality._correction import PartialityCorrection
from torchsx.partiality._fit import fit_partiality_model
from torchsx.partiality._pairs import PairData, build_pairs, fit_BG
from torchsx.partiality._pipeline import (
    run_partiality_pipeline,
    PipelineConfig,
    PipelineResult,
)

__all__ = [
    "ExcitationError",
    "PartialityProfile",
    "GaussianProfile",
    "GenLorentzProfile",
    "RankConcordanceLoss",
    "refine_orientations",
    "RefinementConfig",
    "RefinementResult",
    "PartialityCorrection",
    "fit_partiality_model",
    "PairData",
    "build_pairs",
    "fit_BG",
    "run_partiality_pipeline",
    "PipelineConfig",
    "PipelineResult",
]
