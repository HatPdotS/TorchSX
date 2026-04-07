"""CompositeScalingLoss: composite lognormal-Gaussian loss for SFX scaling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NoiseModel:
    """Per-shell noise estimates.

    Attributes
    ----------
    sigma_LN : (n_shells,) systematic lognormal width
    sigma_shot : (n_shells,) mean shot noise in log space (√(1/I))
    alpha : (n_shells,) balancing weight for lognormal term
    crossover_I : (n_shells,) intensity where shot == systematic
    """

    sigma_LN: torch.Tensor
    sigma_shot: torch.Tensor
    alpha: torch.Tensor
    crossover_I: torch.Tensor


def _lognormal_variance_ratio(sigma: torch.Tensor) -> torch.Tensor:
    """Compute (exp(2σ²) − 2exp(σ²) + 1) / σ².

    This is the factor by which the Gaussian term (with Poisson-fraction
    weight) is inflated relative to unit chi-squared under lognormal noise.
    At small σ → σ², at large σ → exp(2σ²)/σ².
    """
    s2 = sigma ** 2
    V = torch.exp(2.0 * s2) - 2.0 * torch.exp(s2) + 1.0
    # Avoid 0/0 for tiny sigma: Taylor series gives V/σ² → σ² + O(σ⁴)
    return torch.where(s2 > 1e-6, V / s2, s2)


class CompositeScalingLoss(nn.Module):
    """Composite lognormal-Gaussian loss for SFX scaling.

    The Gaussian weight uses the total variance (Poisson + lognormal)::

        σ²_total(I) = max(I, 1) + I² · (exp(σ_LN²) − 1)
        w_gauss = 1 / σ²_total

    The per-shell ``alpha`` balances the two terms so they contribute
    equally in expectation.  It is estimated automatically by
    :meth:`calibrate` from the data, or can be set manually.

    Parameters
    ----------
    n_shells : int
        Number of resolution shells.
    """

    def __init__(self, n_shells: int):
        super().__init__()
        self.register_buffer("sigma_LN", torch.full((n_shells,), 0.1))
        self.register_buffer("sigma_shot", torch.full((n_shells,), 0.01))
        self.register_buffer("alpha", torch.ones(n_shells))

    def calibrate(
        self,
        I_obs: torch.Tensor,
        log_scale: torch.Tensor,
        shells: torch.Tensor,
        I_min_abs: float = 50.0,
    ) -> NoiseModel:
        """Estimate per-shell noise model and set alpha.

        Returns a :class:`NoiseModel` for inspection.  Also updates the
        ``sigma_LN``, ``sigma_shot``, and ``alpha`` buffers in-place.

        Parameters
        ----------
        I_obs : (N,) observed intensities (may include negatives)
        log_scale : (N,) total log scale per obs
        shells : (N,) resolution shell indices
        I_min_abs : minimum intensity for calibration
        """
        n_shells = self.sigma_LN.shape[0]
        device = self.sigma_LN.device

        pos = I_obs > I_min_abs
        log_I = torch.log(I_obs.clamp(min=1.0))
        log_corrected = log_I - log_scale

        new_sigma_LN = torch.zeros(n_shells, device=device)
        new_sigma_shot = torch.zeros(n_shells, device=device)

        for s in range(n_shells):
            sel = pos & (shells == s)
            if sel.sum() < 10:
                new_sigma_LN[s] = 0.1
                new_sigma_shot[s] = 0.01
                continue
            vals = log_corrected[sel]
            sigma_total_sq = vals.var().item()
            sigma_shot_sq = (1.0 / I_obs[sel].clamp(min=1.0)).mean().item()
            new_sigma_shot[s] = sigma_shot_sq ** 0.5
            new_sigma_LN[s] = max(sigma_total_sq - sigma_shot_sq, 0.0) ** 0.5

        new_sigma_LN = new_sigma_LN.clamp(min=1e-4)
        new_sigma_shot = new_sigma_shot.clamp(min=1e-6)

        # Per-shell alpha: variance inflation factor of the Gaussian term
        # under the calibrated lognormal noise.
        new_alpha = _lognormal_variance_ratio(new_sigma_LN).clamp(min=0.01)

        crossover_I = 1.0 / new_sigma_LN ** 2

        self.sigma_LN.copy_(new_sigma_LN)
        self.sigma_shot.copy_(new_sigma_shot)
        self.alpha.copy_(new_alpha)

        return NoiseModel(
            sigma_LN=new_sigma_LN.clone(),
            sigma_shot=new_sigma_shot.clone(),
            alpha=new_alpha.clone(),
            crossover_I=crossover_I,
        )

    def forward(
        self,
        I_obs: torch.Tensor,
        I_pred: torch.Tensor,
        log_I_pred: torch.Tensor,
        shells: torch.Tensor,
    ) -> torch.Tensor:
        """Compute composite loss.

        Parameters
        ----------
        I_obs : (N_obs,) observed intensities (may be negative)
        I_pred : (N_obs,) predicted intensities (|F|² × scale)
        log_I_pred : (N_obs,) log predicted intensities
        shells : (N_obs,) resolution shell indices
        """
        N = I_obs.shape[0]
        sigma_LN = self.sigma_LN[shells]
        alpha = self.alpha[shells]

        # --- Gaussian term ---
        # Use |I_obs| for variance estimate; keeps weight finite for
        # negative intensities while the residual uses the real I_obs.
        I_var = I_obs.abs().clamp(min=1.0)
        # σ²_total = Poisson + lognormal in linear space
        sigma_total_sq = I_var + I_var ** 2 * (torch.exp(sigma_LN ** 2) - 1.0)
        w_gauss = 1.0 / sigma_total_sq
        gauss = (w_gauss * (I_pred - I_obs) ** 2).sum() / N

        # --- Lognormal term (strong positives only) ---
        I_min = 1.0 / sigma_LN ** 2
        strong = (I_obs > I_min).float()
        n_strong = strong.sum().clamp(min=1.0)
        I_obs_safe = I_obs.clamp(min=1.0)
        w_log = 1.0 / sigma_LN ** 2
        log_obs = torch.log(I_obs_safe)
        lognorm = (alpha * strong * w_log * (log_I_pred - log_obs) ** 2).sum() / n_strong

        return gauss + lognorm
