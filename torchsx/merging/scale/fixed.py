"""Fixed (non-trainable) pre-computed log-scale correction."""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scale.base import ScaleModel


class FixedScale(ScaleModel):
    """Pre-computed per-observation log-correction (e.g. from a neural model
    or from partiality-corrected logs).  Not trainable."""

    def __init__(self, log_c_sorted: Tensor) -> None:
        super().__init__()
        # Registered as a buffer (not a Parameter) so LBFGS ignores it.
        self.register_buffer("log_c", log_c_sorted.clone())

    def log_k(
        self,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        del params, meta
        if obs_range is None:
            return self.log_c
        s, e = obs_range
        return self.log_c[s:e]

    def regularizer(self, params: Parameters) -> Tensor:
        del params
        return torch.zeros((), device=self.log_c.device, dtype=self.log_c.dtype)
