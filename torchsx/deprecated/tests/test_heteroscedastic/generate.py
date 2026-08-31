"""Generate synthetic SFX observations from the heteroscedastic noise model.

Generative model per observation j of HKL h:

    epsilon_j ~ GenGauss(0, alpha_h, beta_h)
    I_signal  = I0_h * exp(epsilon_j)
    I_obs_j   ~ Poisson(I_signal)         (1 ADU = 1 photon)

The Poisson variance is simply sigma2_poi = I_signal.  For the D-step
estimators we provide sigma2_poi = I_obs (the observed value), which is
the standard estimator used in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SyntheticData:
    """Container for synthetic observations + ground truth."""

    # Ground truth per HKL
    I0_true: Tensor        # (n_hkl,)
    alpha_true: Tensor     # (n_hkl,)
    beta_true: Tensor      # (n_hkl,)

    # Observations (sorted by HKL, contiguous)
    I_obs: Tensor          # (N,) — all observations (can be negative from bg subtraction)
    hkl_id: Tensor         # (N,) long — HKL index [0, n_hkl)
    sigma2_poi: Tensor     # (N,) — Poisson variance per obs
    obs_counts: Tensor     # (n_hkl,) — observations per HKL
    n_hkl: int
    n_obs: int

    # Positive-only arrays
    I_pos: Tensor          # (M,)
    hkl_id_pos: Tensor     # (M,) long
    obs_counts_pos: Tensor # (n_hkl,)


def sample_gengauss(alpha: float, beta: float, size: int,
                    generator: torch.Generator | None = None) -> Tensor:
    """Sample from GenGauss(0, alpha, beta).

    Uses the transformation: if X ~ Gamma(1/beta, 1), then
    sign * alpha * X^(1/beta) ~ GenGauss(0, alpha, beta).
    """
    shape_param = 1.0 / beta
    gamma_samples = torch._standard_gamma(
        torch.full((size,), shape_param), generator=generator,
    )
    abs_eps = alpha * gamma_samples.pow(1.0 / beta)
    signs = 2.0 * torch.bernoulli(torch.full((size,), 0.5), generator=generator) - 1.0
    return signs * abs_eps


def _make_positive_arrays(I_obs, hkl_id, n_hkl):
    """Extract positive observations and sort by hkl_id."""
    pos_mask = I_obs > 0
    I_pos = I_obs[pos_mask]
    hkl_id_pos = hkl_id[pos_mask]
    obs_counts_pos = torch.zeros(n_hkl, dtype=torch.long)
    obs_counts_pos.scatter_add_(
        0, hkl_id_pos, torch.ones_like(hkl_id_pos, dtype=torch.long),
    )
    sort_pos = hkl_id_pos.argsort()
    return I_pos[sort_pos], hkl_id_pos[sort_pos], obs_counts_pos


def generate_synthetic(
    *,
    n_hkl: int = 500,
    obs_per_hkl: int = 50,
    I0_range: tuple[float, float] = (100.0, 10000.0),
    alpha: float = 0.3,
    beta: float = 1.3,
    background: float = 10.0,
    seed: int = 42,
) -> SyntheticData:
    """Generate a synthetic SFX dataset with Poisson noise.

    Parameters
    ----------
    n_hkl : number of unique HKLs
    obs_per_hkl : observations per HKL (uniform)
    I0_range : (min, max) for log-uniform I0 distribution
    alpha : GenGauss width (scalar, same for all HKLs)
    beta : GenGauss shape (scalar, same for all HKLs)
    background : constant background level (photons)
    seed : random seed
    """
    gen = torch.Generator().manual_seed(seed)
    N = n_hkl * obs_per_hkl

    # Ground truth I0: log-uniform
    log_I0 = torch.empty(n_hkl).uniform_(
        torch.tensor(I0_range[0]).log().item(),
        torch.tensor(I0_range[1]).log().item(),
        generator=gen,
    )
    I0_true = torch.exp(log_I0)
    alpha_true = torch.full((n_hkl,), alpha)
    beta_true = torch.full((n_hkl,), beta)

    # Per-observation arrays
    hkl_id = torch.arange(n_hkl).repeat_interleave(obs_per_hkl)
    I0_per_obs = I0_true.repeat_interleave(obs_per_hkl)

    # Multiplicative noise: epsilon ~ GenGauss(0, alpha, beta)
    epsilon = sample_gengauss(alpha, beta, N, generator=gen)

    # True signal before Poisson sampling (photons)
    I_signal = (I0_per_obs * torch.exp(epsilon)).clamp(min=0)

    # Poisson-sample the signal + background, then subtract background
    total_photons = torch.poisson(I_signal + background, generator=gen)
    I_obs = total_photons - background

    # Poisson variance: sigma2 = I_signal + background
    # In practice we estimate from the observed value
    sigma2_poi = I_obs.clamp(min=1.0) + background

    obs_counts = torch.full((n_hkl,), obs_per_hkl, dtype=torch.long)

    I_pos, hkl_id_pos, obs_counts_pos = _make_positive_arrays(I_obs, hkl_id, n_hkl)

    return SyntheticData(
        I0_true=I0_true,
        alpha_true=alpha_true,
        beta_true=beta_true,
        I_obs=I_obs,
        hkl_id=hkl_id,
        sigma2_poi=sigma2_poi,
        obs_counts=obs_counts,
        n_hkl=n_hkl,
        n_obs=N,
        I_pos=I_pos,
        hkl_id_pos=hkl_id_pos,
        obs_counts_pos=obs_counts_pos,
    )


def generate_variable_alpha(
    *,
    n_hkl: int = 500,
    obs_per_hkl: int = 50,
    I0_range: tuple[float, float] = (100.0, 10000.0),
    alpha_range: tuple[float, float] = (0.1, 0.8),
    beta: float = 1.3,
    background: float = 10.0,
    seed: int = 42,
) -> SyntheticData:
    """Generate synthetic data with per-HKL alpha varying uniformly."""
    gen = torch.Generator().manual_seed(seed)
    N = n_hkl * obs_per_hkl

    log_I0 = torch.empty(n_hkl).uniform_(
        torch.tensor(I0_range[0]).log().item(),
        torch.tensor(I0_range[1]).log().item(),
        generator=gen,
    )
    I0_true = torch.exp(log_I0)
    alpha_true = torch.empty(n_hkl).uniform_(
        alpha_range[0], alpha_range[1], generator=gen,
    )
    beta_true = torch.full((n_hkl,), beta)

    hkl_id = torch.arange(n_hkl).repeat_interleave(obs_per_hkl)
    I0_per_obs = I0_true.repeat_interleave(obs_per_hkl)

    # Per-obs GenGauss with per-HKL alpha
    epsilon = torch.zeros(N)
    for h in range(n_hkl):
        lo = h * obs_per_hkl
        hi = lo + obs_per_hkl
        epsilon[lo:hi] = sample_gengauss(
            float(alpha_true[h]), beta, obs_per_hkl, generator=gen,
        )

    I_signal = (I0_per_obs * torch.exp(epsilon)).clamp(min=0)
    total_photons = torch.poisson(I_signal + background, generator=gen)
    I_obs = total_photons - background
    sigma2_poi = I_obs.clamp(min=1.0) + background

    obs_counts = torch.full((n_hkl,), obs_per_hkl, dtype=torch.long)
    I_pos, hkl_id_pos, obs_counts_pos = _make_positive_arrays(I_obs, hkl_id, n_hkl)

    return SyntheticData(
        I0_true=I0_true,
        alpha_true=alpha_true,
        beta_true=beta_true,
        I_obs=I_obs,
        hkl_id=hkl_id,
        sigma2_poi=sigma2_poi,
        obs_counts=obs_counts,
        n_hkl=n_hkl,
        n_obs=N,
        I_pos=I_pos,
        hkl_id_pos=hkl_id_pos,
        obs_counts_pos=obs_counts_pos,
    )
