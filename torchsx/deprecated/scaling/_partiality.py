"""EwaldPartiality: Ewald-offset partiality model for SFX still shots."""

from __future__ import annotations

import math

import torch
from torch import nn


class EwaldPartiality(nn.Module):
    """Ewald-offset partiality model with resolution-dependent relp width.

    Models the fraction of the Bragg reflection sampled by a still shot
    as a Gaussian function of the excitation error:

        log_p = -r² / (2·σ_eff²)

    where r is the distance of the reciprocal lattice point from the
    Ewald sphere (in Å⁻¹), and the relp width is resolution-dependent:

        σ_eff² = σ_0² + B_relp · s²

    σ_0² captures mosaicity + beam divergence (constant across resolution).
    B_relp · s² captures disorder broadening — high-resolution reflections
    have broader relps due to atomic displacement disorder (the same physics
    as the Debye-Waller factor). This naturally gives higher partiality at
    high resolution, counteracting the tendency to lose high-res signal.

    Parameters
    ----------
    n_crystals : int
        Number of crystals.
    UB : (n_crystals, 3, 3) float32
        Orientation matrices (columns = astar, bstar, cstar in Å⁻¹).
    wavelength : float
        Beam wavelength in Å.
    sigma_0 : float
        Initial base relp width (Å⁻¹). Default 0.05 (wide start).
    B_relp : float
        Initial disorder broadening coefficient (Å). Default 0.5.
    sigma_min : float
        Lower bound on σ_eff (Å⁻¹) to prevent gradient death.
    p_min : float
        Minimum partiality (clamp floor).
    """

    def __init__(
        self,
        n_crystals: int,
        UB: torch.Tensor,
        wavelength: float,
        sigma_0: float = 0.05,
        B_relp: float = 0.5,
        sigma_min: float = 0.002,
        p_min: float = 0.01,
    ):
        super().__init__()
        # Refineable orientation perturbation (Rodrigues vector)
        self.delta_omega = nn.Parameter(torch.zeros(n_crystals, 3))
        # Refineable relp profile: base width + resolution-dependent broadening
        self.log_sigma_0 = nn.Parameter(torch.tensor(math.log(sigma_0)))
        self.log_B_relp = nn.Parameter(torch.tensor(math.log(B_relp)))

        self.sigma_min_sq = sigma_min ** 2

        # Fixed geometry
        self.register_buffer("UB", UB.float())
        inv_wl = 1.0 / wavelength
        s0 = torch.tensor([0.0, 0.0, inv_wl], dtype=torch.float32)
        self.register_buffer("s0", s0)
        self.register_buffer("inv_wavelength", torch.tensor(inv_wl, dtype=torch.float32))

        self.log_p_min = math.log(p_min)

    @staticmethod
    def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
        """Build skew-symmetric matrices from (N, 3) vectors.

        Returns (N, 3, 3) such that [v]× @ u = v × u.
        """
        z = torch.zeros_like(v[:, 0])
        # fmt: off
        return torch.stack([
            z,       -v[:, 2],  v[:, 1],
            v[:, 2],  z,       -v[:, 0],
           -v[:, 1],  v[:, 0],  z,
        ], dim=-1).reshape(-1, 3, 3)
        # fmt: on

    def _all_RUB(self) -> torch.Tensor:
        """Compute R(δω) @ UB for all crystals.

        matrix_exp runs on n_crystals (not n_obs) for efficiency.
        δω is clamped to ±0.17 rad (~10°) to prevent numerical overflow.

        Returns (n_crystals, 3, 3).
        """
        dw = self.delta_omega.clamp(-0.17, 0.17)
        skew = self._skew_symmetric(dw)  # (C, 3, 3)
        R = torch.linalg.matrix_exp(skew)  # (C, 3, 3)
        return torch.bmm(R, self.UB)  # (C, 3, 3)

    def excitation_error(
        self, hkl_obs: torch.Tensor, crystal_id: torch.Tensor,
    ) -> torch.Tensor:
        """Compute excitation error r for each observation.

        r = |R(δω) @ UB @ h + s₀| - 1/λ

        Parameters
        ----------
        hkl_obs : (N_obs, 3) int or float — Miller indices per observation
        crystal_id : (N_obs,) int — crystal index per observation

        Returns
        -------
        r : (N_obs,) float — signed excitation error (Å⁻¹)
        """
        RUB = self._all_RUB()  # (C, 3, 3) — computed per-crystal
        RUB_obs = RUB[crystal_id.long()]  # (N, 3, 3) — cheap index

        h = hkl_obs.float()  # (N, 3)
        q = torch.einsum("nij,nj->ni", RUB_obs, h)  # (N, 3)
        q_plus_s0 = q + self.s0
        return q_plus_s0.norm(dim=-1) - self.inv_wavelength

    def log_partiality(
        self,
        hkl_obs: torch.Tensor,
        crystal_id: torch.Tensor,
        s_sq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute log-partiality for each observation.

        log_p = -r² / (2·σ_eff²), clamped to log(p_min).

        σ_eff² = σ_0² + B_relp · s²  (resolution-dependent relp width)

        Parameters
        ----------
        hkl_obs : (N_obs, 3) Miller indices
        crystal_id : (N_obs,) crystal index
        s_sq : (N_obs,) float, optional — (sinθ/λ)² = (1/(2d))²
            If None, uses constant σ_0 (no resolution dependence).

        Returns (N_obs,) float.
        """
        r = self.excitation_error(hkl_obs, crystal_id)
        sigma_0_sq = torch.exp(self.log_sigma_0) ** 2
        if s_sq is not None:
            B_relp = torch.exp(self.log_B_relp)
            sigma_eff_sq = (sigma_0_sq + B_relp * s_sq).clamp(min=self.sigma_min_sq)
        else:
            sigma_eff_sq = sigma_0_sq.clamp(min=self.sigma_min_sq)
        log_p = -r ** 2 / (2.0 * sigma_eff_sq)
        return log_p.clamp(min=self.log_p_min)

    def log_scale_per_obs(
        self,
        hkl_obs: torch.Tensor,
        crystal_id: torch.Tensor,
        s_sq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return log-partiality per observation."""
        return self.log_partiality(hkl_obs, crystal_id, s_sq=s_sq)
