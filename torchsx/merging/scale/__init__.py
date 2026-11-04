"""The amortised scale model: ``log K`` per observation, with ``log_mean = log I_true + log K``."""

from torchsx.merging.scale.amortized import AmortizedScale

__all__ = ["AmortizedScale"]
