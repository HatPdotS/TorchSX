"""Differentiable excitation error model with per-crystal orientation + wavelength."""

from __future__ import annotations

import torch
from torch import nn, Tensor


class ExcitationError(nn.Module):
    """Excitation error with refineable per-crystal orientation and wavelength.

    Computes r = |R(δω) @ UB @ h + s₀(λ)| - 1/λ per observation,
    where δω = (δωx, δωy, 0) and λ = λ₀ + δλ per crystal.

    Only ωx, ωy are refined — ωz is degenerate (doesn't change |r|).

    Parameters
    ----------
    n_crystals : int
    UB : (C, 3, 3) — base orientation matrices
    lambda_0 : (C,) or scalar — initial per-crystal wavelength (Å)
    """

    def __init__(
        self,
        n_crystals: int,
        UB: Tensor,
        lambda_0: Tensor | float,
    ):
        super().__init__()
        self.delta_omega_xy = nn.Parameter(torch.zeros(n_crystals, 2))
        self.delta_lambda = nn.Parameter(torch.zeros(n_crystals))

        self.register_buffer("UB", UB.float())
        if isinstance(lambda_0, (int, float)):
            lambda_0 = torch.full((n_crystals,), float(lambda_0))
        self.register_buffer("lambda_0", lambda_0.float())

    @staticmethod
    def _skew_symmetric(v: Tensor) -> Tensor:
        """(N, 3) → (N, 3, 3) skew-symmetric matrices."""
        z = torch.zeros_like(v[:, 0])
        return torch.stack([
            z,       -v[:, 2],  v[:, 1],
            v[:, 2],  z,       -v[:, 0],
           -v[:, 1],  v[:, 0],  z,
        ], dim=-1).reshape(-1, 3, 3)

    def effective_wavelength(self) -> Tensor:
        """(C,) effective wavelength = λ₀ + δλ, clamped positive."""
        return (self.lambda_0 + self.delta_lambda).clamp(min=0.1)

    def _all_RUB(self) -> Tensor:
        """R(δωx, δωy, 0) @ UB for all crystals. Returns (C, 3, 3)."""
        dw = torch.zeros(
            self.delta_omega_xy.shape[0], 3,
            device=self.delta_omega_xy.device,
            dtype=self.delta_omega_xy.dtype,
        )
        dw[:, :2] = self.delta_omega_xy.clamp(-0.17, 0.17)
        skew = self._skew_symmetric(dw)
        R = torch.linalg.matrix_exp(skew)
        return torch.bmm(R, self.UB)

    def excitation_error(
        self,
        hkl_obs: Tensor,
        crystal_id: Tensor,
    ) -> Tensor:
        """Signed excitation error per observation.

        r = |R(δω) @ UB @ h + s₀(λ)| - 1/λ

        Parameters
        ----------
        hkl_obs : (N, 3) Miller indices
        crystal_id : (N,) crystal index

        Returns (N,) signed r in Å⁻¹.
        """
        cid = crystal_id.long()
        RUB = self._all_RUB()
        RUB_obs = RUB[cid]
        q = torch.einsum("nij,nj->ni", RUB_obs, hkl_obs.float())

        lam = self.effective_wavelength()
        inv_lam = 1.0 / lam
        inv_lam_obs = inv_lam[cid]

        s0 = torch.zeros_like(q)
        s0[:, 2] = inv_lam_obs

        kf = q + s0
        return kf.norm(dim=-1) - inv_lam_obs

    def abs_excitation_error(
        self,
        hkl_obs: Tensor,
        crystal_id: Tensor,
    ) -> Tensor:
        """|r| per observation. Differentiable via torch.abs."""
        return self.excitation_error(hkl_obs, crystal_id).abs()
