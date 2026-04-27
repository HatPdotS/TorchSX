"""Abstract base for per-observation scale corrections."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters


class ScaleModel(nn.Module):
    """Maps observation metadata to ``log K`` per observation.

    Subclasses implement
    ``log_k(params, meta, obs_range=None) → (n_obs,)``.

    ``meta`` is already sliced to the current chunk (``obs_info.slice``);
    ``obs_range`` is the original ``(start, end)`` within the full obs
    list — needed only by models that key off the original obs index
    (e.g. ``FixedScale`` with a precomputed vector).

    Module-intrinsic parameter penalties live on ``regularizer(params)``.
    """

    @abstractmethod
    def log_k(
        self,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        ...

    def regularizer(self, params: Parameters) -> Tensor:
        """Scalar penalty on *this module's own* parameters.  Default 0."""
        try:
            p = next(self.parameters())
            return torch.zeros((), device=p.device, dtype=p.dtype)
        except StopIteration:
            return torch.zeros(())
