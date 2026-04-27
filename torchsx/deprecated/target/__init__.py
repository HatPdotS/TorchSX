"""Target submodule — scaling targets and loss functions."""

from torchsx.deprecated.target._kl import KLScalingTarget
from torchsx.deprecated.target._self_consistent import SelfConsistentTarget

__all__ = ["KLScalingTarget", "SelfConsistentTarget"]
