"""D-step: fit per-HKL distribution parameters (I₀, α, β).

The combined observation model:

    ε ~ GenGauss(0, α_h, β_h)
    n_peak ~ Poisson(I₀_h · scale_j · exp(ε) + bg_total_j)
    I_obs_j = n_peak - bg_total_j

I₀ is estimated by maximising the Poisson marginal likelihood via softplus
quadrature (all observations, including negative).  α is estimated from
log-space residuals of positive observations, either per resolution shell
or per HKL with Laplacian smoothing.  β is fixed from the initial shell fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ._distribution import _laplacian_smooth


# ── Result container ──────────────────────────────────────────


@dataclass
class DStepResult:
    """Output of a single D-step iteration."""

    I0: Tensor        # (n_valid,) linear-space intensity
    sigma_I0: Tensor  # (n_valid,) std error (empirical)
    alpha: Tensor     # (n_valid,) GenGauss width (smoothed)
    beta: Tensor      # (n_valid,) GenGauss shape (fixed)


# ── α estimation (MLE, log-space, positive observations) ─────


def fit_alpha(
    I_corrected_pos: Tensor,
    hkl_id_pos: Tensor,
    I0: Tensor,
    beta: Tensor,
    obs_counts_pos: Tensor,
    n_valid: int,
) -> Tensor:
    """Estimate GenGauss α per HKL from log-space residuals.

    α^β = (β / n) Σ |log(I_obs_j / I₀_h)|^β

    Parameters
    ----------
    I_corrected_pos : (M,) positive corrected observations
    hkl_id_pos : (M,) HKL index
    I0 : (n_valid,) current intensity estimates
    beta : (n_valid,) GenGauss shape parameter
    obs_counts_pos : (n_valid,) positive observation counts per HKL
    n_valid : number of valid HKLs

    Returns
    -------
    alpha : (n_valid,) GenGauss width parameter
    """
    dev = hkl_id_pos.device

    valid_hkl = I0 > 1.0

    log_I0 = torch.log(I0.clamp(min=1.0))
    log_I0_per_obs = log_I0.repeat_interleave(obs_counts_pos)
    log_r = torch.log(I_corrected_pos.clamp(min=1e-8)) - log_I0_per_obs

    valid_per_obs = valid_hkl.repeat_interleave(obs_counts_pos).float()
    beta_per_obs = beta.repeat_interleave(obs_counts_pos)

    abs_r_pow_b = log_r.abs().clamp(min=1e-8).pow(beta_per_obs) * valid_per_obs
    sum_pow = torch.zeros(n_valid, device=dev).scatter_add_(
        0, hkl_id_pos, abs_r_pow_b)
    n_valid_obs = torch.zeros(n_valid, device=dev).scatter_add_(
        0, hkl_id_pos, valid_per_obs)
    n_valid_obs = n_valid_obs.clamp(min=1.0)

    alpha_pow_b = beta * sum_pow / n_valid_obs
    alpha = alpha_pow_b.clamp(min=1e-6).pow(1.0 / beta)

    median_alpha = alpha[valid_hkl].median() if valid_hkl.any() else torch.tensor(1.0, device=dev)
    alpha[~valid_hkl] = median_alpha

    return alpha.clamp(min=0.01)


def _fit_alpha_shells(
    I_corrected_pos: Tensor,
    hkl_id_pos: Tensor,
    I0: Tensor,
    beta: Tensor,
    obs_counts_pos: Tensor,
    n_valid: int,
    d_spacing: Tensor,
    n_shells: int = 20,
    smooth_sigma: float = 1.5,
) -> Tensor:
    """Fit alpha per resolution shell from pooled residuals.

    Instead of fitting per-HKL (noisy for weak reflections), pool all
    log-residuals within each resolution shell and compute the shell-level
    MLE: alpha_shell^beta = (beta/N) * sum(|log(I/I0)|^beta).

    The result is Gaussian-smoothed across shells and broadcast back to
    per-HKL alpha values.

    Parameters
    ----------
    I_corrected_pos : (M,) positive corrected observations
    hkl_id_pos : (M,) HKL index for each observation
    I0 : (n_valid,) current intensity estimates
    beta : (n_valid,) GenGauss shape parameter
    obs_counts_pos : (n_valid,) positive obs count per HKL
    n_valid : number of valid HKLs
    d_spacing : (n_valid,) d-spacing in Angstrom
    n_shells : number of resolution shells
    smooth_sigma : Gaussian smoothing width in shell units

    Returns
    -------
    alpha : (n_valid,) per-HKL alpha from shell estimates
    """
    dev = I0.device
    n_shells = min(n_shells, max(3, n_valid // 5))

    rank = d_spacing.argsort().argsort()
    hkl_shell = (rank.float() / n_valid * n_shells).long().clamp(max=n_shells - 1)

    shell_per_obs = hkl_shell.repeat_interleave(obs_counts_pos)
    beta_per_obs = beta.repeat_interleave(obs_counts_pos)

    valid_hkl = I0 > 1.0
    log_I0 = torch.log(I0.clamp(min=1.0))
    log_I0_per_obs = log_I0.repeat_interleave(obs_counts_pos)
    log_r = torch.log(I_corrected_pos.clamp(min=1e-8)) - log_I0_per_obs

    valid_per_obs = valid_hkl.repeat_interleave(obs_counts_pos)

    abs_r_pow_b = log_r.abs().clamp(min=1e-8).pow(beta_per_obs) * valid_per_obs.float()

    shell_sum = torch.zeros(n_shells, device=dev).scatter_add_(
        0, shell_per_obs, abs_r_pow_b)
    shell_count = torch.zeros(n_shells, device=dev).scatter_add_(
        0, shell_per_obs, valid_per_obs.float())
    shell_count = shell_count.clamp(min=1.0)

    beta_sum = torch.zeros(n_shells, device=dev).scatter_add_(
        0, hkl_shell, beta * valid_hkl.float())
    beta_count = torch.zeros(n_shells, device=dev).scatter_add_(
        0, hkl_shell, valid_hkl.float()).clamp(min=1.0)
    shell_beta = beta_sum / beta_count
    shell_beta[shell_beta < 0.5] = 1.3

    alpha_shell_pow_b = shell_beta * shell_sum / shell_count
    alpha_shell = alpha_shell_pow_b.clamp(min=1e-6).pow(1.0 / shell_beta)

    log_alpha_shell = torch.log(alpha_shell.clamp(min=1e-6))
    idx = torch.arange(n_shells, dtype=torch.float32, device=dev)
    smoothed = torch.zeros(n_shells, device=dev)
    for k in range(n_shells):
        w = torch.exp(-0.5 * ((idx - idx[k]) / smooth_sigma) ** 2)
        wc = w * shell_count
        smoothed[k] = (wc * log_alpha_shell).sum() / wc.sum().clamp(min=1e-8)

    alpha = torch.exp(smoothed[hkl_shell])
    return alpha.clamp(min=0.01)


# ── Softplus quadrature grid for Poisson marginal ────────────


def _softplus_inv(y: Tensor) -> Tensor:
    """Numerically stable inverse of softplus: log(exp(y) - 1)."""
    return torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(min=1e-8))))


@dataclass
class QuadratureGrid:
    """Pre-computed integration grid for Poisson marginal likelihood.

    All tensors are I0-independent and can be reused across LBFGS iterations.
    """

    signal: Tensor      # (N_obs, K) signal at grid points
    lam: Tensor         # (N_obs, K) λ = signal + bg
    log_poi: Tensor     # (N_obs, K) log Poisson PMF
    log_jac_dt: Tensor  # (N_obs, K) log(dε/dt) analytical Jacobian
    log_dt: Tensor      # (N_obs,)   log uniform t-spacing


def build_softplus_grid(
    n_ph: Tensor,
    bg: Tensor | float,
    n_quad: int = 51,
    n_sigma: float = 5.0,
    signal_min: float = 0.01,
    signal_lo_override: Tensor | None = None,
) -> QuadratureGrid:
    """Build log-linear quadrature grid via softplus mapping.

    Uniform in t-space, mapped to signal via softplus:

    - t ≪ 0: signal ≈ exp(t) → log-spaced (good ε-resolution near bg)
    - t ≫ 0: signal ≈ t      → linear (good Poisson peak resolution)

    Parameters
    ----------
    n_ph : (N,) peak photon counts per observation (I_obs + bg_total)
    bg : per-obs (N,) or scalar total background in peak region
    n_quad : number of quadrature points
    n_sigma : Poisson coverage in σ units
    signal_min : absolute lower bound on signal
    signal_lo_override : (N,) per-obs lower bound on signal.  When given,
        replaces the default ``signal_max * 0.01`` heuristic.  Use this
        to adapt the grid to the current I₀ estimate so that weak
        reflections are properly resolved.
    """
    dev = n_ph.device

    sigma = n_ph.clamp(min=1).sqrt()
    signal_max = (n_ph - bg + n_sigma * sigma).clamp(min=1.0)
    if signal_lo_override is not None:
        signal_lo = signal_lo_override.clamp(min=signal_min)
        signal_lo = torch.min(signal_lo, signal_max * 0.5)
    else:
        signal_lo = (signal_max * 0.01).clamp(min=signal_min)

    t_hi = _softplus_inv(signal_max)
    t_lo = _softplus_inv(signal_lo)

    frac = torch.linspace(0, 1, n_quad, device=dev)
    dt = (t_hi - t_lo) / (n_quad - 1)
    t = t_lo[:, None] + (t_hi - t_lo)[:, None] * frac[None, :]

    signal = F.softplus(t)
    lam = signal + (bg if isinstance(bg, (int, float)) else bg[:, None])
    lam = lam.clamp(min=1e-6)

    log_nf = torch.lgamma(n_ph + 1)
    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_jac_dt = F.logsigmoid(t) - torch.log(signal)
    log_dt = torch.log(dt)

    return QuadratureGrid(
        signal=signal,
        lam=lam,
        log_poi=log_poi,
        log_jac_dt=log_jac_dt,
        log_dt=log_dt,
    )


# ── I₀ estimation (Poisson marginal MLE) ─────────────────────


def _log_gg_norm(beta: Tensor) -> Tensor:
    """GenGauss normalization: log(β) - log(2) - lgamma(1/β)."""
    return torch.log(beta) - math.log(2) - torch.lgamma(1.0 / beta)


def fit_I0_poisson(
    I_obs: Tensor,
    hkl_id: Tensor,
    scale: Tensor,
    alpha: Tensor,
    beta: Tensor,
    I0_init: Tensor,
    obs_counts: Tensor,
    n_valid: int,
    bg: Tensor | float = 0.0,
    grid: QuadratureGrid | None = None,
    n_quad: int = 51,
    max_iter: int = 50,
) -> tuple[Tensor, Tensor]:
    """Estimate I₀ via Poisson marginal MLE with softplus quadrature.

    Generative model::

        ε ~ GenGauss(0, α_h, β_h)
        n_peak ~ Poisson(I₀_h · scale_j · exp(ε) + bg_j)
        I_obs_j = n_peak - bg_j

    The marginal likelihood p(I_obs | I₀) is evaluated by numerical
    integration over ε using an adaptive softplus grid.  Optimised via
    LBFGS in log(I₀) space.  Uses ALL observations (positive and negative).

    Parameters
    ----------
    I_obs : (N,) raw observations (all, including negative)
    hkl_id : (N,) HKL index
    scale : (N,) per-obs multiplicative scale from corrections
    alpha, beta : (n_valid,) GenGauss parameters
    I0_init : (n_valid,) initial I₀ estimate
    obs_counts : (n_valid,) observations per HKL
    n_valid : number of valid HKLs
    bg : per-obs (N,) or scalar — total background in peak region
        (= bg_per_pixel × n_peak_pixels)
    grid : pre-built QuadratureGrid (reuse across EM iterations)
    n_quad : quadrature points (only used if grid is None)
    max_iter : LBFGS iterations
    """
    dev = I_obs.device

    n_ph = (I_obs + bg).clamp(min=0)

    alpha_po = alpha.repeat_interleave(obs_counts)
    beta_po = beta.repeat_interleave(obs_counts)
    log_scale = torch.log(scale.clamp(min=1e-30))
    log_gg_norm = _log_gg_norm(beta_po)

    # Poisson detection floor per HKL: the weakest detectable I0 is
    # σ_floor = √(mean_bg / n_obs) where mean_bg is the mean total
    # background per observation for that HKL.  Grid signal_lo = 0.5 * σ_floor.
    with torch.no_grad():
        if isinstance(bg, (int, float)):
            bg_per_hkl = torch.full((n_valid,), max(float(bg), 1.0), device=dev)
        else:
            # Mean background per HKL from per-obs bg values
            bg_sum = torch.zeros(n_valid, device=dev).scatter_add_(
                0, hkl_id, bg.clamp(min=0))
            bg_per_hkl = (bg_sum / obs_counts.float().clamp(min=1)).clamp(min=1.0)
        sigma_floor = (bg_per_hkl / obs_counts.float().clamp(min=1)).sqrt()
        sig_lo_hkl = (sigma_floor * 0.5).clamp(min=0.01)
        sig_lo = sig_lo_hkl.repeat_interleave(obs_counts)

    log_I0 = torch.log(I0_init.clamp(min=0.01)).clone().requires_grad_(True)

    if grid is None:
        grid = build_softplus_grid(n_ph, bg, n_quad=n_quad,
                                   signal_lo_override=sig_lo)
    log_poi_jac = grid.log_poi + grid.log_jac_dt + grid.log_dt[:, None]

    optimizer = torch.optim.LBFGS(
        [log_I0], lr=1.0, max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7, tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        I0_po = torch.exp(log_I0).repeat_interleave(obs_counts)
        eps = torch.log(grid.signal / (I0_po[:, None] *
                                        torch.exp(log_scale[:, None])))
        log_gg = (-(eps.abs() / alpha_po[:, None]).pow(beta_po[:, None])
                  + log_gg_norm[:, None])
        log_integrand = log_gg + log_poi_jac
        log_p = torch.logsumexp(log_integrand, dim=1)
        nll = torch.zeros(n_valid, device=dev)
        nll.scatter_add_(0, hkl_id, -log_p)
        loss = nll.sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    I0 = torch.exp(log_I0.detach())

    # σ(I₀): empirical scatter
    I0_po = I0.repeat_interleave(obs_counts)
    resid_sq = (I_obs - I0_po * scale) ** 2
    sum_resid_sq = torch.zeros(n_valid, device=dev).scatter_add_(
        0, hkl_id, resid_sq)
    n = obs_counts.float()
    sigma_I0 = (sum_resid_sq / (n * (n - 1).clamp(min=1))).clamp(min=1e-12).sqrt()

    return I0, sigma_I0


# ── Complete D-step ──────────────────────────────────────────


def d_step(
    # All observations (positive + negative), sorted by HKL
    I_all_corrected: Tensor,
    hkl_id_all: Tensor,
    sigma2_poisson_all: Tensor,
    obs_counts_all: Tensor,
    # Positive observations only, sorted by HKL
    I_pos_corrected: Tensor,
    hkl_id_pos: Tensor,
    obs_counts_pos: Tensor,
    # Current parameters
    I0_prev: Tensor,
    alpha_prev: Tensor,
    beta: Tensor,
    n_valid: int,
    # Iteration count (controls LBFGS max_iter)
    n_newton_iter: int,
    # Smoothing infrastructure
    edge_src: Tensor,
    edge_dst: Tensor,
    n_neighbors: Tensor,
    n_smooth_iter: int,
    smooth_lambda: float,
    # Raw observations + scales for Poisson marginal
    I_obs_all: Tensor | None = None,
    scale_all: Tensor | None = None,
    bg: Tensor | float = 0.0,
    # Resolution-dependent alpha (shell-level)
    d_spacing: Tensor | None = None,
    n_alpha_shells: int = 20,
    alpha_prior_lambda: float = 0.0,
    # Legacy — ignored, kept for backward compat
    i0_method: str = "poisson_log",
    I_obs_pos: Tensor | None = None,
    scale_pos: Tensor | None = None,
) -> DStepResult:
    """Complete D-step: fit I₀ (all obs) and α (pos obs).

    I₀ is estimated via Poisson marginal MLE with softplus quadrature.
    α is estimated per resolution shell (if d_spacing provided and
    alpha_prior_lambda > 0) or per HKL with Laplacian smoothing.

    Parameters
    ----------
    I_all_corrected : (N,) ALL corrected observations
    hkl_id_all : (N,) HKL index per ALL obs
    sigma2_poisson_all : (N,) Poisson variance per ALL obs (unused, kept for compat)
    obs_counts_all : (n_valid,) obs count per HKL (all)
    I_pos_corrected : (M,) positive corrected observations
    hkl_id_pos : (M,) HKL index per positive obs
    obs_counts_pos : (n_valid,) obs count per HKL (positive only)
    I0_prev : (n_valid,) previous I₀ estimate
    alpha_prev : (n_valid,) previous α estimate
    beta : (n_valid,) GenGauss shape (fixed)
    n_valid : number of valid HKLs
    n_newton_iter : LBFGS iteration count
    edge_src, edge_dst, n_neighbors : adjacency graph for smoothing
    n_smooth_iter : Laplacian smoothing iterations
    smooth_lambda : smoothing mixing weight
    I_obs_all : (N,) raw observations (required)
    scale_all : (N,) per-obs scale from corrections
    bg : per-obs (N,) or scalar — total background in peak region
    d_spacing : (n_valid,) d-spacing per HKL (enables shell-level alpha)
    n_alpha_shells : resolution shells for shell-level alpha
    alpha_prior_lambda : >0 enables shell-level alpha; 0 = Laplacian
    """
    # 1. Fit I₀ via Poisson marginal
    I0, sigma_I0 = fit_I0_poisson(
        I_obs_all, hkl_id_all,
        scale_all if scale_all is not None else torch.ones_like(I_obs_all),
        alpha_prev, beta, I0_prev, obs_counts_all, n_valid,
        bg=bg,
        max_iter=max(n_newton_iter, 20),
    )

    # 2. Fit α from log-space residuals of positive observations
    if d_spacing is not None and alpha_prior_lambda > 0:
        alpha_smooth = _fit_alpha_shells(
            I_pos_corrected, hkl_id_pos, I0, beta,
            obs_counts_pos, n_valid,
            d_spacing, n_shells=n_alpha_shells,
        )
    else:
        alpha = fit_alpha(
            I_pos_corrected, hkl_id_pos, I0, beta,
            obs_counts_pos, n_valid,
        )
        log_alpha = torch.log(alpha.clamp(min=1e-6))
        log_alpha_smooth = _laplacian_smooth(
            log_alpha, edge_src, edge_dst, n_neighbors,
            n_smooth_iter, smooth_lambda,
            node_weights=I0,
        )
        alpha_smooth = torch.exp(log_alpha_smooth)

    return DStepResult(
        I0=I0,
        sigma_I0=sigma_I0,
        alpha=alpha_smooth,
        beta=beta,
    )
