"""Result container returned by ``Merger.merge()``."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class MergedResult:
    hkl: Tensor
    I0: Tensor
    alpha: Tensor
    beta: Tensor
    mu: Tensor            # = log(I0)
    d_spacing: Tensor
    n_obs: Tensor
    sigma_I0: Tensor
    sigma_mu: Tensor

    diagnostics: dict = field(default_factory=dict)

    def to(self, device) -> "MergedResult":
        return MergedResult(
            **{
                k: (v.to(device) if isinstance(v, Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )
