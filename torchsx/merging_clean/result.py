"""Result container returned by `Merger.merge()`."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class MergedResult:
    hkl: Tensor         # (n_valid, 3) int16
    I0: Tensor          # (n_valid,) linear-space intensity
    alpha: Tensor       # (n_valid,) GenGauss width
    beta: Tensor        # (n_valid,) GenGauss shape
    mu: Tensor          # (n_valid,) = log(I0)
    d_spacing: Tensor   # (n_valid,) Å
    n_obs: Tensor       # (n_valid,) observation count per HKL
    sigma_I0: Tensor    # (n_valid,) empirical std error
    sigma_mu: Tensor    # (n_valid,) std error in log-space

    diagnostics: dict = field(default_factory=dict)

    def to(self, device) -> "MergedResult":
        return MergedResult(
            **{k: (v.to(device) if isinstance(v, Tensor) else v)
               for k, v in self.__dict__.items()}
        )
