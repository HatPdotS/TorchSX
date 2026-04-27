"""Gaussian anchor: Σ (params.<target> − anchor)²."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from torchsx.merging.parameters import Parameters
from torchsx.merging.regularizers.base import Regularizer

ParamName = Literal["log_I0", "log_alpha", "log_beta", "pi_logit"]


class GaussianAnchor(Regularizer):
    """Ridge-toward-anchor penalty on any per-HKL ``Parameters`` field.

    ``anchor`` can be a per-HKL tensor (e.g. ``log_alpha_init`` from the
    shell pre-fit) or a scalar (e.g. the zero-intensity floor target).

    Absorbs the three hand-rolled anchors from ``regularizers_only`` in
    the old loss — the α-init anchor, β-init anchor, and the ``log I₀``
    zero-intensity floor pull.
    """

    def __init__(self, target: ParamName, anchor: Tensor | float) -> None:
        super().__init__()
        self.target = target
        if isinstance(anchor, (int, float)):
            self.anchor_is_scalar = True
            self.register_buffer("anchor", torch.tensor(float(anchor)))
        else:
            self.anchor_is_scalar = False
            self.register_buffer("anchor", anchor.clone())

    def forward(self, params: Parameters) -> Tensor:
        values = getattr(params, self.target)
        diff = values - self.anchor
        return (diff * diff).sum()
