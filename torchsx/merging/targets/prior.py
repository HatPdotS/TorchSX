"""Wraps an ``IntensityPrior`` into a scalar ``Target`` for LossState."""

from __future__ import annotations

from torch import Tensor

from torchsx.merging.parameters import Parameters
from torchsx.merging.priors.base import IntensityPrior
from torchsx.merging.targets.base import MergingTarget


class IntensityPriorTarget(MergingTarget):
    """Sum the per-HKL prior into a scalar, or pass scalar-returning priors through.

    ``WilsonPrior`` and ``FlatPrior`` return per-HKL vectors; we ``.sum()``
    them.  ``WilsonMatchPrior`` already returns a scalar; ``.sum()`` on a
    scalar is a no-op.
    """

    name: str = "merging/prior"

    def __init__(self, params: Parameters, prior: IntensityPrior) -> None:
        super().__init__()
        self.params = params
        self.prior = prior

    def forward(self) -> Tensor:
        return self.prior(self.params.log_I0).sum()
