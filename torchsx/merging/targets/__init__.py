"""``torchref.Target`` subclasses that assemble merging-loss components
into scalar targets consumable by ``LossState``.
"""

from torchsx.merging.targets.base import MergingTarget
from torchsx.merging.targets.data import DataNLLTarget
from torchsx.merging.targets.prior import IntensityPriorTarget
from torchsx.merging.targets.regularizer import RegularizerTarget
from torchsx.merging.targets.scale_reg import ScaleRegTarget

__all__ = [
    "MergingTarget",
    "DataNLLTarget",
    "IntensityPriorTarget",
    "RegularizerTarget",
    "ScaleRegTarget",
]
