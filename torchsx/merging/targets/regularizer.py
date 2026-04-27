"""Wraps a generic ``Regularizer`` into a scalar ``Target``."""

from __future__ import annotations

from torch import Tensor

from torchsx.merging.parameters import Parameters
from torchsx.merging.regularizers.base import Regularizer
from torchsx.merging.targets.base import MergingTarget


class RegularizerTarget(MergingTarget):
    name: str = "merging/reg"

    def __init__(self, params: Parameters, regularizer: Regularizer) -> None:
        super().__init__()
        self.params = params
        self.regularizer = regularizer

    def forward(self) -> Tensor:
        return self.regularizer(self.params)
