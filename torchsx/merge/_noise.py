"""Per-observation Poisson noise model for SFX integration.

Provides σ²_poisson per observation from CrystFEL integration geometry
(peak counts, background, integration radii) and a gain calibration
procedure using negative-intensity observations.

The combined variance model in linear space:

    σ²(j) = I₀(h)² · σ²_LN(h) + g² · σ²_poisson(j)

where:
    σ²_LN(h) = α² · Γ(3/β) / Γ(1/β)  — multiplicative (crystal variation)
    σ²_poisson(j) = I_net + bg · n_peak · (1 + n_peak/n_bg)  — shot noise
    g = gain factor (ADU/photon), calibrated from negatives

I₀ is estimated by minimizing the NLL in linear space:

    NLL(I₀) = Σ_j [(I_obs_j - I₀)² / (2σ²_j) + log(σ²_j) / 2]

The log(σ²) term is essential — it prevents I₀ → 0 when there is
real signal (shrinking σ² carries a penalty), while allowing I₀ = 0
for absent reflections (where the Poisson floor dominates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def compute_sigma2_LN(alpha: Tensor, beta: Tensor) -> Tensor:
    """Log-space variance of GenGauss noise (first-order approximation).

    σ²_LN = α² · Γ(3/β) / Γ(1/β)

    Used in the Newton optimizer for I₀ where numerical stability
    requires the small-σ approximation.  For honest error bars, use
    :func:`compute_sigma2_LN_exact` instead.
    """
    lgamma_1b = torch.lgamma(1.0 / beta)
    lgamma_3b = torch.lgamma(3.0 / beta)
    return alpha ** 2 * torch.exp(lgamma_3b - lgamma_1b)


def compute_sigma2_LN_exact(alpha: Tensor, beta: Tensor) -> Tensor:
    """Exact linear-space variance of the multiplicative factor exp(ε).

    For ε ~ GenGauss(0, α, β), the log-space variance is:

        σ²_log = α² · Γ(3/β) / Γ(1/β)

    The linear-space variance of exp(ε) is:

        Var(exp(ε)) = exp(σ²_log) · (exp(σ²_log) - 1)

    For typical SFX data (α ≈ 1.2, σ²_log ≈ 1.5), this gives ~10×
    larger variance than the first-order approximation — critical for
    honest error bars in refinement.
    """
    sigma2_log = compute_sigma2_LN(alpha, beta)
    return torch.exp(sigma2_log) * torch.expm1(sigma2_log)


@dataclass
class PoissonNoiseModel:
    """Per-observation Poisson noise from CrystFEL ring integration.

    Parameters
    ----------
    r_peak, r_gap, r_bg : integration ring radii in pixels.
        Default (7, 9, 11) matches CrystFEL default.
    gain : ADU-to-photon conversion factor.  Calibrated from negatives
        via :meth:`calibrate_gain`, or set manually.
    """

    r_peak: float = 7.0
    r_gap: float = 9.0
    r_bg: float = 11.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        self.n_peak = math.pi * self.r_peak ** 2
        self.n_bg = math.pi * (self.r_bg ** 2 - self.r_gap ** 2)

    def sigma2_raw(self, I_obs: Tensor, background: Tensor) -> Tensor:
        """Per-observation Poisson variance (gain=1).

        σ² = I_net + bg · n_peak · (1 + n_peak/n_bg)
        """
        bg_contrib = background * self.n_peak * (1.0 + self.n_peak / self.n_bg)
        return (I_obs + bg_contrib).clamp(min=1.0)

    def sigma2(self, I_obs: Tensor, background: Tensor) -> Tensor:
        """Per-observation Poisson variance scaled by gain²."""
        return self.gain ** 2 * self.sigma2_raw(I_obs, background)

    def calibrate_gain(self, I_obs: Tensor, background: Tensor) -> float:
        """Calibrate gain from negative observations.

        For negative I_obs, the signal is zero and I_obs ≈ η (pure noise).
        So <I_neg²> ≈ g² · <σ²_poisson_pred> → g = √(<I²>/<σ²_pred>).
        """
        neg = I_obs < 0
        n_neg = neg.sum().item()
        if n_neg < 100:
            print(f"  PoissonNoiseModel: only {n_neg} negative obs, "
                  f"keeping gain={self.gain:.3f}")
            return self.gain

        sigma2_pred = self.sigma2_raw(I_obs[neg], background[neg])
        sigma2_obs = (I_obs[neg] ** 2).mean()
        g = (sigma2_obs / sigma2_pred.mean()).sqrt().item()
        self.gain = g
        print(f"  PoissonNoiseModel: gain={g:.4f} (g²={g**2:.4f}) "
              f"from {n_neg} negatives")
        return g


def linear_space_I0(
    I_all: Tensor,
    hkl_id_all: Tensor,
    sigma2_poisson: Tensor,
    mu_log: Tensor,
    alpha: Tensor,
    beta: Tensor,
    obs_counts_all: Tensor,
    n_valid: int,
    n_iter: int = 10,
) -> tuple[Tensor, Tensor]:
    """Estimate I₀ per HKL by minimizing NLL in linear space.

    Minimizes:
        NLL(I₀) = Σ_j [(I_obs_j - I₀)² / (2σ²_j) + log(σ²_j) / 2]

    where σ²_j = I₀² · σ²_LN(h) + σ²_poisson(j).

    The log(σ²) term prevents I₀ from collapsing to zero when there is
    signal: reducing I₀ shrinks σ² (tighter model) which costs log-likelihood
    if the residuals don't also shrink.  For absent reflections (I_obs ≈ noise),
    the optimum naturally lands at I₀ ≈ 0.

    Uses ALL observations (positive AND negative).

    Parameters
    ----------
    I_all : (N,) ALL observations (positive AND negative)
    hkl_id_all : (N,) HKL index per observation (contiguous 0..n_valid-1)
    sigma2_poisson : (N,) per-observation Poisson variance (g²-scaled)
    mu_log : (n_valid,) initial log-space location estimates
    alpha, beta : (n_valid,) GenGauss distribution parameters
    obs_counts_all : (n_valid,) observation count per HKL (ALL obs)
    n_valid : number of valid HKLs
    n_iter : Newton-like iterations

    Returns
    -------
    I0 : (n_valid,) linear-space intensity estimates
    sigma_I0 : (n_valid,) standard error of I0
    """
    # Multiplicative variance from GenGauss parameters
    sigma2_LN = compute_sigma2_LN(alpha, beta)

    # Initialize I₀ from log-space estimate
    I0 = torch.exp(mu_log).clamp(min=1e-6)

    # Pre-expand σ²_LN to per-observation (doesn't change across iterations)
    sLN_per_obs = sigma2_LN.repeat_interleave(obs_counts_all)

    for _it in range(n_iter):
        I0_per_obs = I0.repeat_interleave(obs_counts_all)

        # σ²_j = I₀² · σ²_LN + σ²_poisson
        sigma2 = I0_per_obs ** 2 * sLN_per_obs + sigma2_poisson
        sigma2 = sigma2.clamp(min=1e-6)
        inv_s2 = 1.0 / sigma2

        residual = I_all - I0_per_obs

        # NLL gradient w.r.t. I₀:
        #   dNLL/dI₀ = Σ_j [
        #     -(I_obs - I₀)/σ² · 1                    (data fit term)
        #     + I₀·σ²_LN/σ² · (1 - (I_obs-I₀)²/σ²)  (log(σ²) term)
        #   ]
        # = Σ_j [ -r/σ² + I₀·σ²_LN/σ² - I₀·σ²_LN·r²/σ⁴ ]
        # = Σ_j [ (-r + I₀·σ²_LN·(1 - r²/σ²)) / σ² ]
        r_scaled = residual * inv_s2          # r/σ²
        sLN_term = I0_per_obs * sLN_per_obs   # I₀·σ²_LN

        grad_per_obs = -r_scaled + sLN_term * inv_s2 * (1.0 - residual ** 2 * inv_s2)

        # Approximate Hessian (Fisher information): Σ 1/σ² + ...
        # Use simplified: H ≈ Σ (1/σ²) + Σ (2·I₀²·σ²_LN²/σ⁴)
        hess_per_obs = inv_s2 + 2.0 * sLN_term ** 2 * inv_s2 ** 2

        # Aggregate per HKL
        grad = torch.zeros(n_valid).scatter_add_(0, hkl_id_all, grad_per_obs)
        hess = torch.zeros(n_valid).scatter_add_(0, hkl_id_all, hess_per_obs)

        # Newton step with damping
        step = grad / hess.clamp(min=1e-12)
        # Limit step size to prevent instability
        max_step = I0 * 0.5 + 1.0  # can't move more than 50% of current value
        step = step.clamp(-max_step, max_step)
        I0 = (I0 - step).clamp(min=0.0)

    # Standard error from inverse Hessian at convergence
    sigma_I0 = 1.0 / hess.clamp(min=1e-12).sqrt()

    return I0, sigma_I0
