"""Target submodule — scaling targets and loss functions."""

from torchsx.target._kl import KLScalingTarget
from torchsx.target._self_consistent import SelfConsistentTarget

__all__ = ["KLScalingTarget", "SelfConsistentTarget"]
