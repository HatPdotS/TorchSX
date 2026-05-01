"""Soft-center the per-crystal B-factor mean.

Used together with :class:`WilsonAnisoPrior` so that the global resolution
decay flows into the prior's ``U`` rather than being absorbed by the mean
of the per-crystal B in :class:`ScaleBPerCrystal`.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from torchsx.merging.parameters import Parameters
from torchsx.merging.regularizers.base import Regularizer
from torchsx.merging.scale.per_crystal import ScaleBPerCrystal


class ScaleBMeanRegularizer(Regularizer):
    """Penalty ``n_crystals · mean(scale.B)²`` on the shared scale module.

    Reads ``scale.B`` directly via a stored module reference; the gradient
    flows into the per-crystal B-factor parameter of the linked
    :class:`ScaleBPerCrystal`. The ``n_crystals`` factor keeps the weight's
    data-relative magnitude invariant to crystal count.
    """

    def __init__(self, scale: ScaleBPerCrystal) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, params: Parameters) -> Tensor:
        del params  # unused; we read scale.B directly
        return self.scale.B.mean().pow(2) * float(self.scale.n_crystals)
