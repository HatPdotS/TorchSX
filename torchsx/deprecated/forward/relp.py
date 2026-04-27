"""RelpProfile: per-reflection anisotropic Gaussian relp shape."""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor


class RelpProfile(nn.Module):
    """Anisotropic Gaussian relp profile in radial/tangential decomposition.

    R(delta_s; sigma_r, sigma_t) = exp(-ds_r^2 / (2*sigma_r^2) - ds_t^2 / (2*sigma_t^2))

    where ds_r is the radial component (along h_hat) and ds_t is the
    tangential component (perpendicular to h_hat).

    Parameters
    ----------
    n_unique : int
        Number of unique reflections.
    sigma_r_init : float
        Initial radial width in A^-1.
    sigma_t_init : float
        Initial tangential width in A^-1.
    per_reflection : bool
        If True (default), one sigma_r/sigma_t per unique reflection.
        If False, single global sigma_r and sigma_t (2 params total).
    """

    def __init__(
        self,
        n_unique: int,
        sigma_r_init: float = 0.02,
        sigma_t_init: float = 0.02,
        per_reflection: bool = True,
    ):
        super().__init__()
        self.per_reflection = per_reflection
        n = n_unique if per_reflection else 1
        self.log_sigma_r = nn.Parameter(
            torch.full((n,), math.log(sigma_r_init))
        )
        self.log_sigma_t = nn.Parameter(
            torch.full((n,), math.log(sigma_t_init))
        )

    @classmethod
    def from_partiality(
        cls,
        n_unique: int,
        sigma_0: float,
        B_relp: float,
        s_sq: Tensor,
        hkl_id: Tensor,
    ) -> RelpProfile:
        """Initialize from EwaldPartiality's resolution-dependent width.

        sigma_r(hkl) = sqrt(sigma_0^2 + B_relp * s^2)
        sigma_t = sigma_0

        Parameters
        ----------
        s_sq : (N_obs,) — per-observation (1/(2d))^2
        hkl_id : (N_obs,) — maps obs to unique reflection
        """
        # Per-unique s_sq (mean over observations)
        s_sq_unique = torch.zeros(n_unique, device=s_sq.device)
        count = torch.zeros(n_unique, device=s_sq.device)
        s_sq_unique.scatter_add_(0, hkl_id, s_sq.float())
        count.scatter_add_(0, hkl_id, torch.ones_like(s_sq))
        s_sq_unique = s_sq_unique / count.clamp(min=1)

        sigma_r_sq = sigma_0 ** 2 + B_relp * s_sq_unique
        sigma_r = sigma_r_sq.clamp(min=1e-8).sqrt()

        obj = cls(n_unique, sigma_r_init=sigma_0, sigma_t_init=sigma_0)
        obj.log_sigma_r.data.copy_(sigma_r.log())
        return obj

    def get_sigma(self, hkl_id: Tensor) -> tuple[Tensor, Tensor]:
        """Return (sigma_r, sigma_t) for given hkl_id indices.

        Handles both per-reflection and global modes.
        """
        if self.per_reflection:
            return (
                torch.exp(self.log_sigma_r[hkl_id]),
                torch.exp(self.log_sigma_t[hkl_id]),
            )
        else:
            return (
                torch.exp(self.log_sigma_r[0]).expand_as(hkl_id),
                torch.exp(self.log_sigma_t[0]).expand_as(hkl_id),
            )

    def get_log_sigma_r(self, hkl_id: Tensor) -> Tensor:
        """Return log_sigma_r for given hkl_id indices."""
        if self.per_reflection:
            return self.log_sigma_r[hkl_id]
        else:
            return self.log_sigma_r[0].expand_as(hkl_id)

    def evaluate(
        self,
        delta_s: Tensor,
        h_hat: Tensor,
        hkl_id: Tensor,
    ) -> Tensor:
        """Evaluate relp profile at displacement delta_s from relp center.

        Parameters
        ----------
        delta_s : (..., 3) float — displacement vector in reciprocal space (A^-1)
        h_hat : (..., 3) float — unit vector along reciprocal lattice vector
        hkl_id : (...,) int — unique reflection index

        Returns
        -------
        R : (...,) float — profile value in [0, 1]
        """
        if self.per_reflection:
            sigma_r = torch.exp(self.log_sigma_r[hkl_id])  # (...)
            sigma_t = torch.exp(self.log_sigma_t[hkl_id])
        else:
            sigma_r = torch.exp(self.log_sigma_r[0])  # scalar
            sigma_t = torch.exp(self.log_sigma_t[0])

        # Radial component: projection onto h_hat
        ds_r = (delta_s * h_hat).sum(dim=-1)  # (...)
        # Tangential component
        ds_t_vec = delta_s - ds_r.unsqueeze(-1) * h_hat
        ds_t_sq = (ds_t_vec * ds_t_vec).sum(dim=-1)  # (...)

        return torch.exp(
            -ds_r ** 2 / (2.0 * sigma_r ** 2)
            - ds_t_sq / (2.0 * sigma_t ** 2)
        )
