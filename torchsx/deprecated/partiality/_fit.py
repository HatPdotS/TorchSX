"""Fit partiality profile by minimising heteroscedastic Gaussian NLL."""

from __future__ import annotations

import torch
from torch import Tensor

from ._profile import PartialityProfile, GenLorentzProfile


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
) -> GenLorentzProfile:
    """Fit GenLorentz partiality by minimising heteroscedastic Gaussian NLL.

    Loss per obs:
        (I_obs - I0*p)^2 / var + log(var)
    where:
        var = sigma2_poi + (I0*p)^2 * sigma2_LN

    Uses ALL observations (including negatives). Adam optimiser.

    Parameters
    ----------
    I_obs : (N,) observed intensities (BG-corrected, can be negative)
    I0_per_obs : (N,) merged I0 per observation's HKL
    sigma2_poi : (N,) Poisson variance per observation
    sigma2_LN_per_obs : (N,) log-normal variance per observation (from alpha, beta)
    abs_r : (N,) |excitation error| (from refined model)
    inv_d : (N,) 1/d-spacing
    n_steps : optimisation steps
    lr : Adam learning rate
    subsample : max observations per step (random subsample for speed)
    """
    device = I_obs.device
    N = len(I_obs)

    model = GenLorentzProfile(
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

        log_p = model.log_partiality(r_s, invd_s)
        p = torch.exp(log_p)
        I_pred = I0_s * p

        var = sp_s + I_pred ** 2 * sln_s
        var = var.clamp(min=1e-6)

        nll = 0.5 * ((I_s - I_pred) ** 2 / var + torch.log(var))
        loss = nll.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    step {step}: NaN/Inf loss -- stopping early")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            with torch.no_grad():
                gamma, alpha, a0 = model.get_params(invd_s)
            print(f"    step {step:>4}: loss={loss.item():.4f}  "
                  f"gamma_med={gamma.median():.4e}  alpha_med={alpha.median():.2f}  "
                  f"a0_med={a0.median():.3f}")

    return model
