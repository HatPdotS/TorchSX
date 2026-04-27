"""Partiality profiles: log_p = f(|r|, 1/d)."""

from __future__ import annotations

import math
from abc import abstractmethod

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class PartialityProfile(nn.Module):
    """Base class for partiality profiles.

    Subclasses implement ``log_partiality(abs_r, inv_d)`` which returns
    per-observation log(p).  The ``correction`` method returns ``-log(p)``
    for use as a log-space correction.
    """

    @abstractmethod
    def log_partiality(self, abs_r: Tensor, inv_d: Tensor) -> Tensor:
        """Compute log(p) per observation.

        Parameters
        ----------
        abs_r : (N,) |excitation error| in Å⁻¹
        inv_d : (N,) 1/d in Å⁻¹
        """

    def correction(self, abs_r: Tensor, inv_d: Tensor) -> Tensor:
        """Return -log(p) as a log-space correction to subtract from log(I)."""
        return -self.log_partiality(abs_r, inv_d)


class GaussianProfile(PartialityProfile):
    """Gaussian partiality: log_p = -r² / (2·σ_eff²).

    σ_eff² = σ₀² + B_relp · s²

    σ₀ captures mosaicity + beam divergence (constant across resolution).
    B_relp · s² captures disorder broadening — high-resolution reflections
    have broader relps.

    Parameters
    ----------
    sigma_0 : float
        Initial base relp width (Å⁻¹).
    B_relp : float
        Initial disorder broadening coefficient (Å).
    sigma_min : float
        Lower bound on σ_eff (Å⁻¹) to prevent gradient death.
    p_min : float
        Minimum partiality (clamp floor).
    """

    def __init__(
        self,
        sigma_0: float = 0.05,
        B_relp: float = 0.5,
        sigma_min: float = 0.002,
        p_min: float = 0.01,
    ):
        super().__init__()
        self.log_sigma_0 = nn.Parameter(torch.tensor(math.log(sigma_0)))
        self.log_B_relp = nn.Parameter(torch.tensor(math.log(B_relp)))
        self.sigma_min_sq = sigma_min ** 2
        self.log_p_min = math.log(p_min)

    def log_partiality(self, abs_r: Tensor, inv_d: Tensor) -> Tensor:
        sigma_0_sq = torch.exp(self.log_sigma_0) ** 2
        s_sq = (inv_d / 2.0) ** 2
        B_relp = torch.exp(self.log_B_relp)
        sigma_eff_sq = (sigma_0_sq + B_relp * s_sq).clamp(min=self.sigma_min_sq)
        log_p = -(abs_r ** 2) / (2.0 * sigma_eff_sq)
        return log_p.clamp(min=self.log_p_min)


class GenLorentzProfile(PartialityProfile):
    """Generalised Lorentzian partiality with resolution-dependent parameters.

    log(p) = a₀(s) - log(1 + (|r| / γ(s))^α(s))

    where s = 1/d (inverse d-spacing), and γ, α, a₀ are polynomials in s
    normalised to [-1, 1]. All ~12 coefficients are differentiable.

    Parameters
    ----------
    poly_order : int
        Order of polynomials for γ, α, a₀ (default 3 → 4 coefficients each).
    inv_d_range : (float, float)
        Range of 1/d for normalisation. Default (0.05, 1.1) Å⁻¹.
    """

    def __init__(
        self,
        poly_order: int = 3,
        inv_d_range: tuple[float, float] = (0.05, 1.1),
    ):
        super().__init__()
        n = poly_order + 1

        # γ stored in log-space (ensures positivity)
        self.log_gamma_coeffs = nn.Parameter(torch.zeros(n))
        self.log_gamma_coeffs.data[0] = math.log(0.003)

        # α stored raw, softplus applied (ensures > 0)
        self.alpha_coeffs = nn.Parameter(torch.zeros(n))
        self.alpha_coeffs.data[0] = 1.0

        # a₀: offset — captures log(p(0)/⟨p⟩)
        self.a0_coeffs = nn.Parameter(torch.zeros(n))

        self.register_buffer("inv_d_lo", torch.tensor(inv_d_range[0]))
        self.register_buffer("inv_d_hi", torch.tensor(inv_d_range[1]))

    def _normalise(self, inv_d: Tensor) -> Tensor:
        """Map 1/d to [-1, 1]."""
        return 2 * (inv_d - self.inv_d_lo) / (self.inv_d_hi - self.inv_d_lo) - 1

    def _eval_poly(self, coeffs: Tensor, x: Tensor) -> Tensor:
        """Evaluate polynomial c₀ + c₁x + c₂x² + ..."""
        result = coeffs[0]
        x_pow = x
        for i in range(1, len(coeffs)):
            result = result + coeffs[i] * x_pow
            x_pow = x_pow * x
        return result

    def get_params(self, inv_d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (γ, α, a₀) at given 1/d values."""
        x = self._normalise(inv_d)
        gamma = torch.exp(self._eval_poly(self.log_gamma_coeffs, x))
        alpha = F.softplus(self._eval_poly(self.alpha_coeffs, x), beta=2.0)
        a0 = self._eval_poly(self.a0_coeffs, x)
        return gamma, alpha, a0

    def log_partiality(self, abs_r: Tensor, inv_d: Tensor) -> Tensor:
        gamma, alpha, a0 = self.get_params(inv_d)
        # Log-space computation to avoid pow() overflow
        log_ratio = alpha * torch.log((abs_r / gamma.clamp(min=1e-6)).clamp(min=1e-10))
        log_ratio = log_ratio.clamp(max=20)
        return a0 - torch.log1p(torch.exp(log_ratio))
