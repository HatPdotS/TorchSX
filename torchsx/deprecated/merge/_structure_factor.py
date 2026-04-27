"""MergedStructureFactor: refined merged |F(h)| amplitudes."""

from __future__ import annotations

import torch
from torch import nn, Tensor

from torchsx._core import _SFXStreamCore
from torchsx.deprecated.merge._hkl import compute_hkl_index, merge_intensities


class MergedStructureFactor(nn.Module):
    """Refined merged structure factor amplitudes |F(h)|.

    Internally stores log|F|. Predicts I = |F|² = exp(2 * log|F|).
    """

    def __init__(self, n_unique: int, unique_hkl: torch.Tensor):
        super().__init__()
        self.log_F = nn.Parameter(torch.zeros(n_unique))
        # Global scale absorbs systematic offset between raw CrystFEL
        # intensities and the forward model's physical units.
        # Applied as: I_pred = exp(log_global_scale) * |F|²
        self.log_global_scale = nn.Parameter(torch.zeros(1))
        self.register_buffer("unique_hkl", unique_hkl)

    @classmethod
    def from_merge(cls, core: _SFXStreamCore) -> MergedStructureFactor:
        """Initialize from inverse-variance weighted merge.

        Parameters
        ----------
        core : _SFXStreamCore
            Unmerged SFX data.
        """
        hkl_idx = compute_hkl_index(core.hkl)
        I_merged, _, _ = merge_intensities(
            hkl_idx.hkl_id, core.I, core.sigma, hkl_idx.n_unique,
        )
        # log|F| = 0.5 * log(I), clamp I > 0 to avoid log(0)
        # Use small floor (0.01) so weak reflections can refine to near-zero F²
        log_F_init = 0.5 * torch.log(I_merged.clamp(min=0.01)).float()

        sf = cls(hkl_idx.n_unique, hkl_idx.unique_hkl)
        sf.log_F.data.copy_(log_F_init)
        return sf

    def predict_I(self, hkl_id: torch.Tensor) -> torch.Tensor:
        """Return global_scale * |F(h)|² per observation."""
        return torch.exp(self.log_global_scale + 2.0 * self.log_F[hkl_id])

    def predict_log_I(self, hkl_id: torch.Tensor) -> torch.Tensor:
        """Return log(global_scale * |F(h)|²) per observation."""
        return self.log_global_scale + 2.0 * self.log_F[hkl_id]

    def to_dict(self) -> dict:
        """Return {hkl, F, I} for output (with global scale absorbed)."""
        with torch.no_grad():
            gs = torch.exp(0.5 * self.log_global_scale)  # sqrt(global_scale)
            F = gs * torch.exp(self.log_F)
            return {"hkl": self.unique_hkl, "F": F, "I": F ** 2}
