"""Rank-concordance orientation and wavelength refinement for SFX."""

from partiality._excitation import ExcitationErrorModel
from partiality._concordance import RankConcordanceLoss
from partiality._refine import refine_orientations, RefinementConfig, RefinementResult
from partiality._data import load_and_prepare, PairData
from partiality._diagnostics import evaluate_concordance, plot_diagnostics
