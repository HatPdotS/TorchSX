"""The amortised scale model: ``log K`` per observation, with ``log_mean = log I_true + log K``."""

from torchsx.merging.scale.amortized import AmortizedScale
from torchsx.merging.scale.capture import DiskCapture

__all__ = ["AmortizedScale", "DiskCapture"]
