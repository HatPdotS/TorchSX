"""Abstract base for multiplicative-scatter density modules."""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters


class ScatterModel(nn.Module):
    """Density of ``log S`` around ``log_mean = log(K · I_true)``.

    Concrete implementations must return the log-density evaluated at a
    per-observation / per-grid-point ``eps = log_S − log_mean`` tensor.

    The scatter is stateless with respect to observations; any per-HKL
    parameters (e.g. GenGauss α, β per HKL) are read from ``params`` and
    indexed with ``meta.hkl_idx``.
    """

    @abstractmethod
    def log_prob(
        self,
        eps: Tensor,          # (n_obs, K) — log S − log_mean at each grid point
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:              # (n_obs, K)
        ...
