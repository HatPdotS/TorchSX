"""Gating utilities for the per-HKL (α, β) refinement.

Currently exposes the model-free background-significance gate used to
split observations into (strong, weak) for the truncated-MLE fit.
"""
from torchsx.merging.gating.background_significance import compute_bg_gate

__all__ = ["compute_bg_gate"]
