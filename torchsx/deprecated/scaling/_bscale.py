"""BScale: per-crystal isotropic B-factor correction."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from torchsx._core import _SFXStreamCore


class BScale(nn.Module):
    """Per-crystal isotropic B-factor: exp(-2 B_i s²).

    I_scaled = I_input / exp(-2 B s²) = I_input * exp(2 B s²)
    """

    def __init__(self, n_crystals: int):
        super().__init__()
        self.B = nn.Parameter(torch.zeros(n_crystals))

    @classmethod
    def from_core(cls, core: _SFXStreamCore) -> BScale:
        """Initialize with B = 0 (no correction)."""
        return cls(core.n_crystals)

    def forward(self, core: _SFXStreamCore, s_sq: torch.Tensor) -> _SFXStreamCore:
        """Return new core with B-factor correction applied.

        Parameters
        ----------
        core : _SFXStreamCore
            Input data (possibly already scaled).
        s_sq : (N_obs,) float32
            Squared reciprocal resolution (1/(2d))².
        """
        correction = torch.exp(2.0 * self.B[core.crystal_id.long()] * s_sq)
        I_scaled = core.I * correction
        return dataclasses.replace(core, I=I_scaled)

    def log_scale_per_obs(
        self, crystal_id: torch.Tensor, s_sq: torch.Tensor,
    ) -> torch.Tensor:
        """Return log correction = -2 B_i s² per observation."""
        return -2.0 * self.B[crystal_id.long()] * s_sq
