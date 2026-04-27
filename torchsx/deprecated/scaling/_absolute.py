"""AbsoluteScale: per-crystal absolute scale factor G_i."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from torchsx._core import _SFXStreamCore


class AbsoluteScale(nn.Module):
    """Per-crystal absolute scale factor G_i.

    I_scaled = I_obs / G_i = I_obs / exp(log_G_i)
    """

    def __init__(self, n_crystals: int):
        super().__init__()
        self.log_G = nn.Parameter(torch.zeros(n_crystals))

    @classmethod
    def from_core(cls, core: _SFXStreamCore) -> AbsoluteScale:
        """Initialize with log_G = 0 (G = 1, no scaling)."""
        return cls(core.n_crystals)

    def forward(self, core: _SFXStreamCore) -> _SFXStreamCore:
        """Return new core with I_scaled = I / exp(log_G[crystal_id])."""
        scale = torch.exp(self.log_G[core.crystal_id.long()])
        I_scaled = core.I / scale
        return dataclasses.replace(core, I=I_scaled)

    def log_scale_per_obs(self, crystal_id: torch.Tensor) -> torch.Tensor:
        """Return log(G_i) per observation for loss computation."""
        return self.log_G[crystal_id.long()]
