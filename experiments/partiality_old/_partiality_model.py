"""Differentiable generalised Lorentzian partiality model."""

from __future__ import annotations

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GenLorentzPartiality(nn.Module):
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
        # Initial: log(γ) ≈ log(0.003) ≈ -5.8 (from empirical fits)
        self.log_gamma_coeffs = nn.Parameter(torch.zeros(n))
        self.log_gamma_coeffs.data[0] = math.log(0.003)

        # α stored raw, softplus applied (ensures > 0)
        # Initial: α ≈ 1.0 (between exponential and Lorentzian)
        self.alpha_coeffs = nn.Parameter(torch.zeros(n))
        self.alpha_coeffs.data[0] = 1.0

        # a₀: offset — captures log(p(0)/⟨p⟩), since I₀ from D-step
        # is ⟨p⟩×F², observations at r=0 are brighter than I₀
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
        """Compute log(p) per observation.

        Parameters
        ----------
        abs_r : (N,) |excitation error| in Å⁻¹
        inv_d : (N,) 1/d in Å⁻¹

        Returns (N,) log(p).
        """
        gamma, alpha, a0 = self.get_params(inv_d)
        # Numerical safety: compute in log-space to avoid overflow
        # log(1 + (r/γ)^α) = log(1 + exp(α * log(r/γ)))
        log_ratio = alpha * torch.log((abs_r / gamma.clamp(min=1e-6)).clamp(min=1e-10))
        log_ratio = log_ratio.clamp(max=20)  # exp(20) ≈ 5e8, safe for log1p
        return a0 - torch.log1p(torch.exp(log_ratio))

    def correction(self, abs_r: Tensor, inv_d: Tensor) -> Tensor:
        """Return -log(p) as a log-space correction to subtract from log(I)."""
        return -self.log_partiality(abs_r, inv_d)


def fit_partiality_model(
    I_obs: Tensor,
    I0_per_obs: Tensor,
    sigma2_poi: Tensor,
    sigma2_LN_per_obs: Tensor,
    abs_r: Tensor,
    inv_d: Tensor,
    *,
    n_steps: int = 500,
    lr: float = 1e-3,
    subsample: int = 500_000,
    poly_order: int = 3,
    inv_d_range: tuple[float, float] = (0.05, 1.1),
) -> GenLorentzPartiality:
    """Fit GenLorentz partiality by minimising heteroscedastic Gaussian NLL.

    Loss per obs:
        (I_obs - I₀×p)² / var + log(var)
    where:
        var = σ²_poi + (I₀×p)² × σ²_LN

    Uses ALL observations (including negatives). Adam optimiser.

    Parameters
    ----------
    I_obs : (N,) observed intensities (BG-corrected, can be negative)
    I0_per_obs : (N,) merged I₀ per observation's HKL
    sigma2_poi : (N,) Poisson variance per observation
    sigma2_LN_per_obs : (N,) log-normal variance per observation (from α, β)
    abs_r : (N,) |excitation error| (from refined model)
    inv_d : (N,) 1/d-spacing
    n_steps : optimisation steps
    lr : Adam learning rate
    subsample : max observations per step (random subsample for speed)
    """
    device = I_obs.device
    N = len(I_obs)

    model = GenLorentzPartiality(
        poly_order=poly_order, inv_d_range=inv_d_range,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Precompute valid mask (finite values, I0 > 0)
    valid = torch.isfinite(I_obs) & (I0_per_obs > 1) & torch.isfinite(abs_r)
    valid_idx = torch.where(valid)[0]
    N_valid = len(valid_idx)
    print(f"  Fitting GenLorentz partiality: {N_valid:,} valid obs, "
          f"{model.log_gamma_coeffs.numel() * 3} params, {n_steps} steps")

    for step in range(n_steps):
        optimizer.zero_grad()

        # Subsample for speed
        if N_valid > subsample:
            perm = torch.randperm(N_valid, device=device)[:subsample]
            idx = valid_idx[perm]
        else:
            idx = valid_idx

        I_s = I_obs[idx]
        I0_s = I0_per_obs[idx]
        sp_s = sigma2_poi[idx]
        sln_s = sigma2_LN_per_obs[idx]
        r_s = abs_r[idx]
        invd_s = inv_d[idx]

        # Model prediction
        log_p = model.log_partiality(r_s, invd_s)
        p = torch.exp(log_p)
        I_pred = I0_s * p

        # Heteroscedastic variance
        var = sp_s + I_pred ** 2 * sln_s
        var = var.clamp(min=1e-6)

        # Gaussian NLL
        nll = 0.5 * ((I_s - I_pred) ** 2 / var + torch.log(var))
        loss = nll.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    step {step}: NaN/Inf loss — stopping early")
            break

        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            with torch.no_grad():
                gamma, alpha, a0 = model.get_params(invd_s)
            print(f"    step {step:>4}: loss={loss.item():.4f}  "
                  f"γ_med={gamma.median():.4e}  α_med={alpha.median():.2f}  "
                  f"a0_med={a0.median():.3f}")

    return model
