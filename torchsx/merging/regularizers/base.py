"""Abstract base for auxiliary regularisers on the shared ``Parameters``."""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from torchsx.merging.parameters import Parameters


class Regularizer(nn.Module):
    """Scalar penalty that reads from the shared ``Parameters`` container.

    Use this for penalties that are *not* intrinsic to a particular scale /
    scatter / likelihood / prior module — things like Miller-graph smoothing
    (which involves multiple parameters) or Gaussian anchors on init values.
    Module-intrinsic regularisers (ridge on per-crystal B, ridge on panel
    gain) stay inside their module's ``.regularizer()`` method.
    """

    @abstractmethod
    def forward(self, params: Parameters) -> Tensor:  # pragma: no cover - abstract
        ...
