"""CrystalOrientation: per-crystal orientation refinement via Rodrigues rotation."""

from __future__ import annotations

import torch
from torch import nn


class CrystalOrientation(nn.Module):
    """Per-crystal orientation with refineable Rodrigues perturbation.

    Stores the base UB matrices and a small refineable rotation delta_omega.
    R(delta_omega) @ UB gives the refined orientation.

    Parameters
    ----------
    UB : (n_crystals, 3, 3) float32
        Base orientation matrices (columns = astar, bstar, cstar in A^-1).
    """

    def __init__(self, UB: torch.Tensor):
        super().__init__()
        n_crystals = UB.shape[0]
        self.delta_omega = nn.Parameter(torch.zeros(n_crystals, 3))
        self.register_buffer("UB", UB.float())

    @staticmethod
    def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
        """Build (N, 3, 3) skew-symmetric matrices from (N, 3) vectors."""
        z = torch.zeros_like(v[:, 0])
        # fmt: off
        return torch.stack([
            z,       -v[:, 2],  v[:, 1],
            v[:, 2],  z,       -v[:, 0],
           -v[:, 1],  v[:, 0],  z,
        ], dim=-1).reshape(-1, 3, 3)
        # fmt: on

    def RUB(self) -> torch.Tensor:
        """Compute R(delta_omega) @ UB for all crystals.

        Returns (n_crystals, 3, 3).
        """
        dw = self.delta_omega.clamp(-0.17, 0.17)
        skew = self._skew_symmetric(dw)
        R = torch.linalg.matrix_exp(skew)
        return torch.bmm(R, self.UB)

    def predict_h_lab(
        self, hkl: torch.Tensor, crystal_id: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reciprocal-space position h_lab = RUB @ hkl.

        Parameters
        ----------
        hkl : (N, 3) int or float — Miller indices
        crystal_id : (N,) int — crystal index per observation

        Returns
        -------
        h_lab : (N, 3) float — reciprocal lattice point in lab frame (A^-1)
        """
        RUB = self.RUB()
        RUB_obs = RUB[crystal_id.long()]
        return torch.einsum("nij,nj->ni", RUB_obs, hkl.float())
