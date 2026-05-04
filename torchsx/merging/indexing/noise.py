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

        Note: assumes the negatives are zero-centred. Real data has a
        positive μ-bias (signal contamination + centroiding), so this
        estimator runs high. Prefer ``calibrate_gain_truncnorm``.
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

    def calibrate_gain_truncnorm(
        self,
        I_obs: Tensor,
        background: Tensor,
        *,
        bg_min: float = 0.1,
        z_min: float = -20.0,
        max_pool: int = 1_000_000,
        max_iter: int = 200,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Bias-aware gain from bg-normalised truncated-Gaussian fit on negatives.

        Each negative ``I_j`` is whitened by its predicted σ at g=1,
        ``z_j = I_j / sqrt(bg_j · n_peak · (1 + n_peak/n_bg))``. The pool of
        ``z`` values truncated to (-∞, 0) is fit by MLE to a Normal(μ, σ);
        the gain is ``g = σ²``. The bias is absorbed into μ instead of
        inflating σ as in the naive Mandel estimator.

        Returns
        -------
        (gain, mu_bias_in_sigma_units)
            ``mu_bias_in_sigma_units`` is the fitted μ in units of the
            predicted σ at g=1 — diagnostic only.
        """
        device = I_obs.device
        dtype = torch.float64

        neg = (I_obs < 0) & (background > bg_min)
        n_neg = int(neg.sum().item())
        if n_neg < 100:
            print(f"  PoissonNoiseModel.truncnorm: only {n_neg} valid negatives, "
                  f"keeping gain={self.gain:.3f}")
            return self.gain, 0.0

        I_neg = I_obs[neg].to(dtype=dtype)
        bg_neg = background[neg].to(dtype=dtype)
        sigma_pred_g1 = (bg_neg * self.n_peak * (1.0 + self.n_peak / self.n_bg)).sqrt()
        z = I_neg / sigma_pred_g1.clamp(min=1e-12)
        z = z[(z > z_min) & (z < 0.0)]

        if z.numel() > max_pool:
            gen = torch.Generator(device=device).manual_seed(seed)
            idx = torch.randperm(z.numel(), generator=gen, device=device)[:max_pool]
            z = z[idx]

        sigma0 = (z.std(unbiased=False) / math.sqrt(1.0 - 2.0 / math.pi)).clamp(min=1e-3)
        mu = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
        log_sigma = torch.tensor(
            float(sigma0.log()), device=device, dtype=dtype, requires_grad=True,
        )

        opt = torch.optim.LBFGS(
            [mu, log_sigma], max_iter=max_iter, history_size=10,
            tolerance_grad=1e-8, tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        # log Φ(x) via the standard normal CDF, computed stably as
        # log(0.5 · erfc(-x / √2)).
        def _log_phi_cdf(x: Tensor) -> Tensor:
            return torch.log(0.5 * torch.erfc(-x / math.sqrt(2.0)).clamp(min=1e-300))

        def closure() -> Tensor:
            opt.zero_grad(set_to_none=False)
            sigma = log_sigma.exp()
            zc = (z - mu) / sigma
            ll_point = -0.5 * zc.pow(2) - 0.5 * math.log(2.0 * math.pi) - log_sigma
            log_trunc = _log_phi_cdf(-mu / sigma)  # Φ(-μ/σ)
            nll = -(ll_point.sum() - z.numel() * log_trunc)
            nll.backward()
            return nll

        opt.step(closure)

        sigma_hat = float(log_sigma.detach().exp().item())
        mu_hat = float(mu.detach().item())
        gain = sigma_hat ** 2
        self.gain = gain
        print(
            f"  PoissonNoiseModel.truncnorm: gain={gain:.4f} ADU/photon  "
            f"σ_z={sigma_hat:.3f}  μ_z={mu_hat:+.3f}σ  "
            f"[n_peak={self.n_peak:.2f}, n_bg={self.n_bg:.2f}, "
            f"{int(z.numel())}/{n_neg} negatives]",
            flush=True,
        )
        return gain, mu_hat
