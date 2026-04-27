"""Abstract base for latent-intensity priors."""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor


class IntensityPrior(nn.Module):
    """Prior on the per-HKL latent log-intensity ``log_I0``.

    Concrete subclasses implement ``forward(log_I0)`` returning either

    * a per-HKL negative-log-prior vector of shape ``(n_valid,)``
      (MAP-style: ``WilsonPrior``, ``FlatPrior``), or
    * a scalar penalty aggregated across HKLs
      (distribution-match: ``WilsonMatchPrior``).

    ``IntensityPriorTarget`` sums the vector form across HKLs before
    applying the LossState weight.
    """

    @abstractmethod
    def forward(self, log_I0: Tensor) -> Tensor:  # pragma: no cover - abstract
        ...
