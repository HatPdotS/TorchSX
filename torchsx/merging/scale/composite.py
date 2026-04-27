"""Sum-of-``ScaleModel`` composite (additive log-scale)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scale.base import ScaleModel


class CompositeScale(ScaleModel):
    """Additive composition of multiple ``ScaleModel`` submodules.

    ``log K = Σ_c submodules[c].log_k(params, meta, obs_range)``

    Corrections are registered as an ``nn.ModuleList`` so their parameters
    flow through ``.parameters()`` for the optimiser.  ``regularizer``
    returns the sum of each submodule's own regulariser.
    """

    def __init__(self, submodules: Iterable[ScaleModel] = ()) -> None:
        super().__init__()
        self.submodules_list = nn.ModuleList(list(submodules))

    def add(self, submodule: ScaleModel) -> None:
        self.submodules_list.append(submodule)

    def log_k(
        self,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        if len(self.submodules_list) == 0:
            # Return a zero tensor shaped by the chunk; use a meta field for length.
            return torch.zeros_like(meta.s_sq)
        total: Tensor | None = None
        for m in self.submodules_list:
            part = m.log_k(params, meta, obs_range=obs_range)
            total = part if total is None else total + part
        assert total is not None
        return total

    def regularizer(self, params: Parameters) -> Tensor:
        if len(self.submodules_list) == 0:
            return torch.zeros(())
        total: Tensor | None = None
        for m in self.submodules_list:
            part = m.regularizer(params)
            total = part if total is None else total + part
        assert total is not None
        return total
