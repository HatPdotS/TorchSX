"""Per-observation Poisson noise model for SFX integration.

Provides σ²_poisson per observation from CrystFEL ring-integration geometry
(peak counts, background, integration radii) and a gain calibration using
negative-intensity observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def compute_sigma2_LN(alpha: Tensor, beta: Tensor) -> Tensor:
    """Log-space variance of GenGauss noise (first-order approximation):
    σ²_LN = α² · Γ(3/β) / Γ(1/β)."""
    lgamma_1b = torch.lgamma(1.0 / beta)
    lgamma_3b = torch.lgamma(3.0 / beta)
    return alpha ** 2 * torch.exp(lgamma_3b - lgamma_1b)


def compute_sigma2_LN_exact(alpha: Tensor, beta: Tensor) -> Tensor:
    """Exact linear-space variance of exp(ε) for ε ~ GenGauss(0, α, β):
    Var = exp(σ²_log) · (exp(σ²_log) − 1)."""
    sigma2_log = compute_sigma2_LN(alpha, beta)
    return torch.exp(sigma2_log) * torch.expm1(sigma2_log)


@dataclass
class PoissonNoiseModel:
    """Per-observation Poisson noise from CrystFEL ring integration.

    Parameters
    ----------
    r_peak, r_gap, r_bg : integration ring radii in pixels (default 7/9/11 as CrystFEL).
    gain : ADU-to-photon conversion factor; calibrate via ``calibrate_gain``.
    """

    r_peak: float = 7.0
    r_gap: float = 9.0
    r_bg: float = 11.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        self.n_peak = math.pi * self.r_peak ** 2
        self.n_bg = math.pi * (self.r_bg ** 2 - self.r_gap ** 2)

    def sigma2_raw(self, I_obs: Tensor, background: Tensor) -> Tensor:
        bg_contrib = background * self.n_peak * (1.0 + self.n_peak / self.n_bg)
        return (I_obs + bg_contrib).clamp(min=1.0)

    def sigma2(self, I_obs: Tensor, background: Tensor) -> Tensor:
        return self.gain ** 2 * self.sigma2_raw(I_obs, background)

    def calibrate_gain(self, I_obs: Tensor, background: Tensor) -> float:
        """Mandel gain calibration from background-limited (negative) peaks.

        Derivation (n_pk photons in peak ring, λ_bg = per-pixel bg rate):
            Var[I_obs_photon] = n_peak · λ_bg · (1 + n_peak/n_bg)
        In ADU (g = ADU/photon):
            Var[I_obs_ADU] = g · n_peak · bg_ADU_per_px · (1 + n_peak/n_bg)
        Conditioned on I_obs<0 (zero mean),
            ⟨I²_neg⟩ = Var[I_obs_ADU]  ⇒  g = ⟨I²_neg⟩ / denom.
        """
        neg = I_obs < 0
        n_neg = int(neg.sum().item())
        if n_neg < 100:
            print(f"  PoissonNoiseModel: only {n_neg} negative obs, "
                  f"keeping gain={self.gain:.3f}")
            return self.gain

        bg_neg = background[neg].clamp(min=0)
        sigma2_pred_ADU = bg_neg * self.n_peak * (1.0 + self.n_peak / self.n_bg)
        var_ADU = (I_obs[neg] ** 2).mean()
        denom = sigma2_pred_ADU.mean().clamp(min=1e-12)
        g = (var_ADU / denom).item()
        self.gain = g
        print(f"  PoissonNoiseModel: gain={g:.4f} ADU/photon "
              f"[n_peak={self.n_peak:.2f}, n_bg={self.n_bg:.2f}, "
              f"{n_neg} negatives]")
        return g
