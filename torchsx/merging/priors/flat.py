"""No-op flat prior (ablation baseline)."""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.priors.base import IntensityPrior


class FlatPrior(IntensityPrior):
    """Returns a zero per-HKL vector — disables the intensity prior.

    Useful as an ablation baseline: wire it in with nonzero LossState weight
    and verify the data + other regularisers still recover a sensible fit
    without any Wilson-like shrinkage.
    """

    def forward(self, log_I0: Tensor) -> Tensor:
        return torch.zeros_like(log_I0)
