"""Calls ``scale_model.regularizer(params)`` as a scalar target."""

from __future__ import annotations

from torch import Tensor

from torchsx.merging.parameters import Parameters
from torchsx.merging.scale.base import ScaleModel
from torchsx.merging.targets.base import MergingTarget


class ScaleRegTarget(MergingTarget):
    """Target wrapping the module-intrinsic regulariser of a ``ScaleModel``.

    Kept separate from ``RegularizerTarget`` because scale models expose
    their own ``.regularizer(params)`` method (ridge on log_G, B, etc.)
    rather than being a plain ``Regularizer`` subclass.
    """

    name: str = "merging/scale_reg"

    def __init__(self, params: Parameters, scale: ScaleModel) -> None:
        super().__init__()
        self.params = params
        self.scale = scale

    def forward(self) -> Tensor:
        return self.scale.regularizer(self.params)
