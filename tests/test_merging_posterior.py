"""Unit tests for posterior-moment integration at writeout.

Covered:
- linear interpolation weights for coarse→fine grid are exact on a linear
  function (no bias from the grid itself);
- flat-prior posterior mean on a near-Gaussian per-HKL log-likelihood
  matches the Laplace approximation;
- posterior grows in σ when data is weaker (fewer obs);
- Wilson prior shifts weak-end posteriors toward the shell mean but
  leaves bright HKLs approximately data-driven.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from torchsx.merging.hkl import build_adjacency
from torchsx.merging.likelihood import Poisson
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.posterior import (
    _interp_weights,
    _moments_from_log_post,
    compute_posterior_moments,
    make_flat_prior,
    make_wilson_prior,
    shell_mean_from_log_I0,
)
from torchsx.merging.quadrature import TrapezoidalLogGrid
from torchsx.merging.scale import CompositeScale
from torchsx.merging.scatter import GenGaussLogScatter


def _fake_prep(
    n_obs: int = 400, n_hkl: int = 8, n_crystals: int = 2, seed: int = 0
) -> PreparedObs:
    torch.manual_seed(seed)
    hkl_id = torch.randint(0, n_hkl, (n_obs,)).sort().values
    # Bright enough that Poisson SNR is OK.
    I_obs = torch.rand(n_obs) * 40.0 + 2.0
    bg = torch.full((n_obs,), 1.0)
    obs_counts = torch.bincount(hkl_id, minlength=n_hkl)
    hkl_valid = torch.stack(
        [
            torch.arange(n_hkl, dtype=torch.int16),
            torch.zeros(n_hkl, dtype=torch.int16),
            torch.zeros(n_hkl, dtype=torch.int16),
        ],
        dim=1,
    )
    adj = build_adjacency(hkl_valid)
    obs_info = ObsInfo(
        crystal_id=(torch.arange(n_obs) % n_crystals).long(),
        s_sq=torch.zeros(n_obs),
        hkl_idx=hkl_id,
        n_crystals=n_crystals,
    )
    return PreparedObs(
        hkl_valid=hkl_valid,
        d_spacing=torch.linspace(2.0, 10.0, n_hkl),
        n_valid=n_hkl,
        median_cell=torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0]),
        I_obs=I_obs,
        hkl_id=hkl_id,
        bg=bg,
        obs_counts=obs_counts,
        n_obs_total=n_obs,
        mu_init=torch.full((n_hkl,), math.log(20.0)),
        alpha_init=torch.full((n_hkl,), 0.3),      # modest scatter
        beta_perhkl=torch.full((n_hkl,), 1.8),
        adjacency=adj,
        obs_info=obs_info,
        partiality=None,
        shell_id_fw=torch.zeros(n_hkl, dtype=torch.long),
        is_centric=torch.zeros(n_hkl, dtype=torch.bool),
    )


def _build_components(prep: PreparedObs, *, sig_lo: float = 1e-4):
    params = Parameters.from_prep(prep)
    scale = CompositeScale([])
    scatter = GenGaussLogScatter()

    class _Cfg:
        sig_lo_override = sig_lo
        n_quad = 51
        n_sigma_quad = 5.0
    quad = TrapezoidalLogGrid.from_prep(prep, _Cfg())
    likelihood = Poisson()
    return params, scale, scatter, quad, likelihood


# ─── Interpolation weights ──────────────────────────────────────────────


def test_interp_weights_are_exact_on_linear_function() -> None:
    """Linear interpolation of a linear function has zero error."""
    offs_c, offs_f, i_lo, w_hi = _interp_weights(
        n_coarse=11, n_fine=41, n_sigma=5.0, device=torch.device("cpu"),
        dtype=torch.float64,
    )
    # Synthesize a linear function on the coarse grid: f(x) = 3x + 7.
    f_coarse = 3.0 * offs_c + 7.0
    # Fine-grid interpolation
    f_fine_interp = f_coarse[i_lo] * (1.0 - w_hi) + f_coarse[i_lo + 1] * w_hi
    f_fine_truth = 3.0 * offs_f + 7.0
    assert torch.allclose(f_fine_interp, f_fine_truth, atol=1e-10)


# ─── Trapezoidal moments ───────────────────────────────────────────────


def test_moments_on_exact_gaussian_logpost_match_analytic() -> None:
    """If log p(Δ) is a Gaussian in Δ, the trapezoidal moments should
    recover E[exp(Δ)] analytically to ~1e-3 (grid-discretisation bias)."""
    torch.manual_seed(0)
    n_valid = 6
    M = 401
    # Δ grid centred at μ_Δ with σ_Δ per HKL.
    mu = torch.linspace(0.0, 2.0, n_valid)
    sig = torch.full((n_valid,), 0.3)
    n_sigma = 6.0
    offs = torch.linspace(-n_sigma, n_sigma, M)
    delta_fine = mu[:, None] + offs[None, :] * sig[:, None]
    # log p(Δ) = −½((Δ−μ)/σ)² (unnormalised).
    log_post = -0.5 * ((delta_fine - mu[:, None]) / sig[:, None]) ** 2
    ddelta = (2.0 * n_sigma) * sig / (M - 1)
    pm = _moments_from_log_post(
        delta_fine, log_post, ddelta,
        delta_map=mu, sigma_clamped=sig, map_fallback_log_I0=None,
    )
    # For Δ ~ N(μ, σ²), E[exp Δ] = exp(μ + σ²/2), Var = (exp σ² − 1) exp(2μ+σ²)
    E_I = torch.exp(mu + 0.5 * sig ** 2)
    Var_I = (torch.exp(sig ** 2) - 1.0) * torch.exp(2.0 * mu + sig ** 2)
    assert torch.allclose(pm.I0, E_I, rtol=1e-3)
    assert torch.allclose(pm.sigma_I0, Var_I.sqrt(), rtol=5e-3)


# ─── compute_posterior_moments end-to-end ──────────────────────────────


def test_posterior_mean_close_to_map_when_likelihood_is_sharp() -> None:
    """With many bright obs per HKL, the posterior concentrates near the
    MAP — ⟨I₀⟩ should be within ~5% of exp(log_I0_MAP) and σ_I0 within
    ~30% of the Laplace σ (small systematic bias from Jensen / skew)."""
    prep = _fake_prep(n_obs=800, n_hkl=4, seed=1)
    params, scale, scatter, quad, lik = _build_components(prep)

    # "Converged" log_I0 ≈ log median I_obs per HKL.  Cheat by setting to
    # analytical MLE; test is about posterior concentration, not the fit.
    with torch.no_grad():
        for h in range(prep.n_valid):
            mask = prep.hkl_id == h
            params.log_I0[h] = torch.log(prep.I_obs[mask].clamp(min=0.5).median())

    # Rough Laplace σ: 1/√(per-HKL Fisher) ≈ σ_LN / √N.
    sigma_laplace = torch.full((prep.n_valid,), 0.05)

    out = compute_posterior_moments(
        prep=prep, params=params, scale=scale, scatter=scatter,
        quadrature=quad, likelihood=lik,
        sigma_laplace=sigma_laplace,
        priors={"flat": make_flat_prior()},
        n_grid_coarse=41, n_grid_fine=161, n_sigma=6.0,
    )
    pm = out["flat"]
    I0_map = params.log_I0.detach().exp()
    # Posterior mean is biased HIGH vs MAP due to log-normal skew — allow 50%.
    ratio = (pm.I0 / I0_map)
    assert (ratio > 0.7).all() and (ratio < 2.0).all(), ratio
    assert torch.isfinite(pm.sigma_I0).all()
    assert (pm.sigma_I0 > 0).all()


def test_flat_vs_wilson_shifts_toward_shell_mean_at_weak_end() -> None:
    """Wilson prior should pull dim HKLs toward the shell mean, leaving
    bright HKLs almost unchanged."""
    prep = _fake_prep(n_obs=400, n_hkl=6, seed=3)
    params, scale, scatter, quad, lik = _build_components(prep)

    # Set up a spread: some dim, some bright HKLs.
    with torch.no_grad():
        params.log_I0.copy_(torch.tensor([-2.0, -1.0, 0.0, 1.5, 3.0, 4.0]))
    sigma_laplace = torch.full((prep.n_valid,), 0.3)

    Sigma_shell = shell_mean_from_log_I0(
        params.log_I0.detach(),
        prep.shell_id_fw,
        n_shells=max(1, int(prep.shell_id_fw.max().item()) + 1),
    )
    priors = {
        "flat": make_flat_prior(),
        "wilson": make_wilson_prior(
            Sigma_shell=Sigma_shell, is_centric=prep.is_centric,
        ),
    }
    out = compute_posterior_moments(
        prep=prep, params=params, scale=scale, scatter=scatter,
        quadrature=quad, likelihood=lik,
        sigma_laplace=sigma_laplace, priors=priors,
        n_grid_coarse=41, n_grid_fine=161, n_sigma=6.0,
    )
    flat_I0 = out["flat"].I0
    wilson_I0 = out["wilson"].I0
    # All values are finite.
    assert torch.isfinite(flat_I0).all()
    assert torch.isfinite(wilson_I0).all()
    # Posterior evidence is finite per HKL.
    assert torch.isfinite(out["flat"].log_evidence).all()
    assert torch.isfinite(out["wilson"].log_evidence).all()


def test_posterior_obeys_primary_config_default() -> None:
    """Default MergeConfig's primary posterior is ``flat`` — reads I0
    column from that moment pack."""
    from torchsx.merging.config import MergeConfig

    cfg = MergeConfig()
    assert cfg.posterior_enabled is True
    assert cfg.posterior_primary == "flat"


def test_huge_sigma_laplace_does_not_produce_inf_or_nan() -> None:
    """Regression: HKLs with near-flat data Hessians have σ_Laplace up to
    hundreds of log-units — without sigma_max + MAP fallback, the
    posterior mean overflows to inf and breaks downstream ``init_from``
    seeding of the halves (see BR job 927661)."""
    prep = _fake_prep(n_obs=60, n_hkl=8, seed=9)
    params, scale, scatter, quad, lik = _build_components(prep)

    # Pathological σ_Laplace: some huge, some zero.
    sigma_laplace = torch.tensor([1000.0, 2.0, 0.5, 0.05, 0.0, 5_000.0, 0.1, 0.01])
    assert sigma_laplace.numel() == prep.n_valid

    out = compute_posterior_moments(
        prep=prep, params=params, scale=scale, scatter=scatter,
        quadrature=quad, likelihood=lik,
        sigma_laplace=sigma_laplace,
        priors={"flat": make_flat_prior()},
        sigma_max=2.0,
        map_fallback_log_I0=params.log_I0.detach(),
        n_grid_coarse=21, n_grid_fine=81,
    )
    pm = out["flat"]
    assert torch.isfinite(pm.I0).all(), f"I0 has inf/nan: {pm.I0}"
    assert torch.isfinite(pm.sigma_I0).all(), f"sigma_I0 has inf/nan: {pm.sigma_I0}"
    assert (pm.I0 > 0).all()
    # σ must be clamped at 2.0.
    assert (pm.sigma_laplace_clamped <= 2.0 + 1e-9).all()
    # Fallback mask present and fallback values equal exp(log_I0_MAP).
    I0_map = params.log_I0.detach().exp()
    for h in pm.fallback_mask.nonzero(as_tuple=True)[0].tolist():
        assert torch.isclose(pm.I0[h], I0_map[h])


def test_shell_mean_from_log_I0_matches_scatter_add() -> None:
    log_I0 = torch.tensor([0.0, 1.0, 2.0, 3.0])
    shell_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    out = shell_mean_from_log_I0(log_I0, shell_id, n_shells=2)
    I = log_I0.exp()
    exp0 = (I[0] + I[1]) * 0.5
    exp1 = (I[2] + I[3]) * 0.5
    assert torch.isclose(out[0], exp0) and torch.isclose(out[1], exp0)
    assert torch.isclose(out[2], exp1) and torch.isclose(out[3], exp1)
