"""D-step: fit per-HKL distribution parameters (I₀, α, β) via KL minimisation.

The combined observation model in linear space:

    I_obs_j = I₀_h · exp(ε_j) + η_j

where ε ~ GenGauss(0, α_h, β_h) (multiplicative) and η ~ N(0, σ²_poi_j)
(Poisson shot noise).  The per-observation variance is:

    σ²_j(I₀) = I₀² · σ²_LN(h) + σ²_poisson(j)

I₀ is estimated by minimising the NLL in linear space (all observations,
including negative).  α is estimated from log-space residuals of positive
corrected observations.  β is fixed from the initial shell fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ._noise import compute_sigma2_LN
from ._distribution import _laplacian_smooth


# ── Adaptive quadrature range ────────────────────────────────


def _quadrature_range(beta: float, coverage: float = 0.999) -> float:
    """Quadrature range for GenGauss(0, 1, β) covering *coverage* of its mass.

    For GenGauss f(x) ∝ exp(-|x|^β), the CDF from -R to R equals
    ``gammainc(1/β, R^β)`` (regularised lower incomplete gamma).

    We invert this via bisection on ``torch.special.gammainc``.
    """
    a = torch.tensor(1.0 / beta)
    lo, hi = 1.0, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        frac = float(torch.special.gammainc(a, torch.tensor(mid ** beta)))
        if frac < coverage:
            lo = mid
        else:
            hi = mid
    return min(hi, 15.0)


# ── Result container ──────────────────────────────────────────


@dataclass
class DStepResult:
    """Output of a single D-step iteration."""

    I0: Tensor        # (n_valid,) linear-space intensity
    sigma_I0: Tensor  # (n_valid,) std error from inverse Hessian
    alpha: Tensor     # (n_valid,) GenGauss width (smoothed)
    beta: Tensor      # (n_valid,) GenGauss shape (fixed)


# ── I₀ estimation (Newton NLL, linear space) ─────────────────


def fit_I0_newton(
    I_corrected: Tensor,
    hkl_id: Tensor,
    sigma2_poisson: Tensor,
    sigma2_LN: Tensor,
    I0_init: Tensor,
    obs_counts: Tensor,
    n_valid: int,
    n_iter: int = 10,
) -> tuple[Tensor, Tensor]:
    """Estimate I₀ per HKL by minimising NLL with combined variance.

    NLL(I₀) = Σ_j [(I_obs_j - I₀)² / (2σ²_j) + log(σ²_j) / 2]

    where σ²_j = I₀² · σ²_LN(h) + σ²_poisson(j).

    Uses the first-order σ²_LN approximation for stable Newton convergence.
    The final σ(I₀) is computed empirically from the scatter of observations
    around I₀, which is model-free and captures all variance sources.

    Uses ALL observations (positive AND negative).

    Parameters
    ----------
    I_corrected : (N,) corrected observations (after removing G, B, panel)
    hkl_id : (N,) HKL index per obs (contiguous 0..n_valid-1)
    sigma2_poisson : (N,) per-obs Poisson variance (gain²-scaled)
    sigma2_LN : (n_valid,) first-order multiplicative variance
    I0_init : (n_valid,) initial estimate
    obs_counts : (n_valid,) observation counts per HKL
    n_valid : number of valid HKLs
    n_iter : Newton iterations

    Returns
    -------
    I0, sigma_I0 : (n_valid,) each
    """
    I0 = I0_init.clone().clamp(min=1e-6)

    # Pre-expand σ²_LN to per-observation (constant across iterations)
    sLN_per_obs = sigma2_LN.repeat_interleave(obs_counts)

    for _ in range(n_iter):
        I0_per_obs = I0.repeat_interleave(obs_counts)

        # σ²_j = I₀² · σ²_LN + σ²_poisson
        sigma2 = (I0_per_obs ** 2 * sLN_per_obs + sigma2_poisson).clamp(min=1e-6)
        inv_s2 = 1.0 / sigma2
        residual = I_corrected - I0_per_obs

        # Gradient: dNLL/dI₀ = Σ_j [(-r + I₀·σ²_LN·(1 - r²/σ²)) / σ²]
        sLN_term = I0_per_obs * sLN_per_obs
        grad_per_obs = (
            -residual * inv_s2
            + sLN_term * inv_s2 * (1.0 - residual ** 2 * inv_s2)
        )

        # Hessian ≈ Σ (1/σ²) + Σ (2·I₀²·σ²_LN²/σ⁴)
        hess_per_obs = inv_s2 + 2.0 * sLN_term ** 2 * inv_s2 ** 2

        # Aggregate per HKL
        grad = torch.zeros(n_valid, device=hkl_id.device).scatter_add_(0, hkl_id, grad_per_obs)
        hess = torch.zeros(n_valid, device=hkl_id.device).scatter_add_(0, hkl_id, hess_per_obs)

        # Damped Newton step
        step = grad / hess.clamp(min=1e-12)
        max_step = I0 * 0.5 + 1.0
        step = step.clamp(-max_step, max_step)
        I0 = (I0 - step).clamp(min=0.0)

    # σ(I₀) from empirical scatter — model-free, captures all variance sources
    I0_final = I0.repeat_interleave(obs_counts)
    resid_sq = (I_corrected - I0_final) ** 2

    # Per-HKL: σ²(I₀) = Σ(I_obs - I₀)² / (n · (n - 1))
    sum_resid_sq = torch.zeros(n_valid, device=hkl_id.device).scatter_add_(0, hkl_id, resid_sq)
    n = obs_counts.float()
    # Bessel correction: divide by n*(n-1), clamp for safety
    sigma2_I0 = sum_resid_sq / (n * (n - 1.0).clamp(min=1.0))
    sigma_I0 = sigma2_I0.clamp(min=1e-12).sqrt()

    return I0, sigma_I0


# ── I₀ estimation (exact likelihood via quadrature) ──────────


def _prepare_quadrature(
    alpha: Tensor,
    beta: Tensor,
    obs_counts: Tensor,
    sigma2_poisson: Tensor,
    n_quad: int = 21,
    device: torch.device | str | None = None,
) -> dict:
    """Pre-compute quadrature grid and weights (fully vectorised, no loops).

    Returns a dict with all constants needed by ``_quadrature_nll``.
    """
    import math

    if device is None:
        device = alpha.device

    beta_med = float(beta.median())
    R = _quadrature_range(beta_med)
    n_quad_eff = max(n_quad, round(n_quad * R / 3.0))
    eps_tilde = torch.linspace(-R, R, n_quad_eff, device=device)  # (K,)
    d_eps = float(eps_tilde[1] - eps_tilde[0])

    alpha_po = alpha.repeat_interleave(obs_counts).to(device)  # (N,)
    beta_po = beta.repeat_interleave(obs_counts).to(device)    # (N,)
    inv_2s2 = (0.5 / sigma2_poisson.clamp(min=1e-6)).to(device)
    log_norm_gauss = (-0.5 * torch.log(
        2.0 * math.pi * sigma2_poisson.clamp(min=1e-6)
    )).to(device)

    # GenGauss log-weights: (K, N) — fully vectorised
    log_gg_norm = (
        torch.log(beta_po) - math.log(2.0)
        - torch.lgamma(1.0 / beta_po) + math.log(d_eps)
    )  # (N,)
    abs_eps = eps_tilde.abs()  # (K,)
    # |ε̃_k|^β_j for all (k, j): (K, N)
    log_gg_weights = -(abs_eps[:, None] ** beta_po[None, :]) + log_gg_norm[None, :]

    return dict(
        eps_tilde=eps_tilde,
        alpha_po=alpha_po,
        beta_po=beta_po,
        inv_2s2=inv_2s2,
        log_norm_gauss=log_norm_gauss,
        log_gg_weights=log_gg_weights,
        n_quad_eff=n_quad_eff,
    )


def _quadrature_nll(
    I0_param: Tensor,
    I_corrected: Tensor,
    obs_counts: Tensor,
    eps_tilde: Tensor,
    alpha_po: Tensor,
    log_gg_weights: Tensor,
    inv_2s2: Tensor,
    log_norm_gauss: Tensor,
) -> Tensor:
    """Negative log-marginal-likelihood via quadrature (differentiable).

    Fully vectorised over quadrature points — no Python loops.
    """
    I0_po = I0_param.repeat_interleave(obs_counts)  # (N,)

    # (K, N): scale factor per quadrature point
    s = torch.exp(alpha_po[None, :] * eps_tilde[:, None])  # (K, N)
    predicted = I0_po[None, :] * s                          # (K, N)
    residual = I_corrected[None, :] - predicted             # (K, N)

    log_gauss = -(residual ** 2) * inv_2s2[None, :] + log_norm_gauss[None, :]
    log_joints = log_gauss + log_gg_weights                 # (K, N)

    log_p = torch.logsumexp(log_joints, dim=0)              # (N,)
    return -log_p.sum()


def fit_I0_quadrature(
    I_corrected: Tensor,
    hkl_id: Tensor,
    sigma2_poisson: Tensor,
    alpha: Tensor,
    beta: Tensor,
    I0_init: Tensor,
    obs_counts: Tensor,
    n_valid: int,
    n_quad: int = 21,
    max_iter: int = 20,
) -> tuple[Tensor, Tensor]:
    """Estimate I₀ by maximising the exact marginal likelihood via LBFGS.

    The generative model per observation:

        ε ~ GenGauss(0, α_h, β_h)
        I_obs = I₀_h · exp(ε) + N(0, σ²_poi)

    The marginal likelihood:

        p(I_obs | I₀) = ∫ f_GG(ε; α, β) · N(I_obs; I₀·eᵋ, σ²_poi) dε

    is evaluated via quadrature on standardised ε̃ = ε/α.  The range
    adapts to β so that heavy-tailed distributions (β < 1) are not
    truncated.  Gradients computed by autograd; optimised with LBFGS.

    Uses ALL observations (positive AND negative).
    """
    # ── Pre-compute quadrature grid + weights (vectorised) ──
    q = _prepare_quadrature(alpha, beta, obs_counts, sigma2_poisson, n_quad)

    # ── Optimise I₀ via LBFGS ──
    I0_param = I0_init.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [I0_param], lr=1.0, max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7, tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        loss = _quadrature_nll(
            I0_param, I_corrected, obs_counts,
            q["eps_tilde"], q["alpha_po"],
            q["log_gg_weights"], q["inv_2s2"], q["log_norm_gauss"],
        )
        loss.backward()
        with torch.no_grad():
            I0_param.data.clamp_(min=0.0)
        return loss

    optimizer.step(closure)

    # ── Detach result ──
    I0 = I0_param.detach().clamp(min=0.0)

    # ── σ(I₀): empirical scatter (model-free) ──
    I0_final = I0.repeat_interleave(obs_counts)
    resid_sq = (I_corrected - I0_final) ** 2
    sum_resid_sq = torch.zeros(n_valid, device=hkl_id.device).scatter_add_(0, hkl_id, resid_sq)
    n = obs_counts.float()
    sigma2_I0 = sum_resid_sq / (n * (n - 1.0).clamp(min=1.0))
    sigma_I0 = sigma2_I0.clamp(min=1e-12).sqrt()

    return I0, sigma_I0


# ── α estimation (MLE, log-space, positive observations) ─────


def fit_alpha(
    I_corrected_pos: Tensor,
    hkl_id_pos: Tensor,
    I0: Tensor,
    beta: Tensor,
    obs_counts_pos: Tensor,
    n_valid: int,
    *,
    alpha_prior: Tensor | None = None,
    prior_lambda: Tensor | None = None,
) -> Tensor:
    """Estimate GenGauss α per HKL from log-space residuals.

    r_j = log(I_corrected_pos_j) - log(I₀_h)
    α^β = (β / n) Σ |r_j|^β

    When *alpha_prior* and *prior_lambda* are given, the bare MLE is
    refined into a MAP estimate by minimizing the penalized NLL::

        L(α) = n·β·log(α) + S_β/α^β + (λ/2)·(log α − log α_prior)²

    via a few Newton steps on u = log(α).

    Parameters
    ----------
    I_corrected_pos : (M,) positive corrected observations
    hkl_id_pos : (M,) HKL index
    I0 : (n_valid,) current intensity estimates
    beta : (n_valid,) GenGauss shape parameter
    obs_counts_pos : (n_valid,) positive observation counts per HKL
    n_valid : number of valid HKLs
    alpha_prior : (n_valid,) resolution-dependent prior (optional)
    prior_lambda : (n_valid,) per-HKL penalty weight (optional)

    Returns
    -------
    alpha : (n_valid,) GenGauss width parameter
    """
    dev = hkl_id_pos.device

    # Mask out HKLs with I₀ ≈ 0 — their log-residuals are meaningless.
    valid_hkl = I0 > 1.0

    # Log-space residuals (safe: I₀ > 1 for valid HKLs)
    log_I0 = torch.log(I0.clamp(min=1.0))
    log_I0_per_obs = log_I0.repeat_interleave(obs_counts_pos)
    log_r = torch.log(I_corrected_pos.clamp(min=1e-8)) - log_I0_per_obs

    # Mask per-obs: zero out contributions from invalid HKLs
    valid_per_obs = valid_hkl.repeat_interleave(obs_counts_pos).float()

    # β per observation
    beta_per_obs = beta.repeat_interleave(obs_counts_pos)

    # MLE: α^β = (β/n) Σ |r|^β  (only from valid HKLs)
    abs_r_pow_b = log_r.abs().clamp(min=1e-8).pow(beta_per_obs) * valid_per_obs
    sum_pow = torch.zeros(n_valid, device=dev).scatter_add_(
        0, hkl_id_pos, abs_r_pow_b)
    n_valid_obs = torch.zeros(n_valid, device=dev).scatter_add_(
        0, hkl_id_pos, valid_per_obs)
    n_valid_obs = n_valid_obs.clamp(min=1.0)
    alpha_pow_b = beta * sum_pow / n_valid_obs
    alpha = alpha_pow_b.clamp(min=1e-6).pow(1.0 / beta)

    # Refine to MAP estimate via Newton on the penalized NLL
    if alpha_prior is not None and prior_lambda is not None:
        u = torch.log(alpha.clamp(min=1e-6))
        u_prior = torch.log(alpha_prior.clamp(min=1e-6))
        lam = prior_lambda
        S_b = sum_pow      # Σ|r_j|^β per HKL
        n = n_valid_obs     # obs count per HKL
        b = beta

        for _ in range(3):
            exp_mbu = torch.exp(-b * u)
            # NLL = n·u + exp(-β·u)·S_β + (λ/2)·(u - u_prior)²
            grad = n - b * exp_mbu * S_b + lam * (u - u_prior)
            hess = (b * b * exp_mbu * S_b + lam).clamp(min=1e-8)
            u = u - grad / hess

        alpha = torch.exp(u)

    # For HKLs with I₀ ≈ 0: set to prior or global median
    if alpha_prior is not None:
        alpha[~valid_hkl] = alpha_prior[~valid_hkl]
    else:
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

    # Shell assignment by d_spacing rank (equal count)
    rank = d_spacing.argsort().argsort()
    hkl_shell = (rank.float() / n_valid * n_shells).long().clamp(max=n_shells - 1)

    # Per-obs shell assignment and beta
    shell_per_obs = hkl_shell.repeat_interleave(obs_counts_pos)
    beta_per_obs = beta.repeat_interleave(obs_counts_pos)

    # Log-residuals: r_j = log(I_obs_j / I0_h)
    valid_hkl = I0 > 1.0
    log_I0 = torch.log(I0.clamp(min=1.0))
    log_I0_per_obs = log_I0.repeat_interleave(obs_counts_pos)
    log_r = torch.log(I_corrected_pos.clamp(min=1e-8)) - log_I0_per_obs

    # Mask: only use obs from HKLs with meaningful I0
    valid_per_obs = valid_hkl.repeat_interleave(obs_counts_pos)

    # |r|^beta per observation (masked)
    abs_r_pow_b = log_r.abs().clamp(min=1e-8).pow(beta_per_obs) * valid_per_obs.float()

    # Sum |r|^beta and count per shell
    shell_sum = torch.zeros(n_shells, device=dev).scatter_add_(
        0, shell_per_obs, abs_r_pow_b)
    shell_count = torch.zeros(n_shells, device=dev).scatter_add_(
        0, shell_per_obs, valid_per_obs.float())
    shell_count = shell_count.clamp(min=1.0)

    # Use the median beta per shell for the alpha^beta = beta * mean(|r|^beta)
    shell_beta = torch.zeros(n_shells, device=dev)
    beta_sum = torch.zeros(n_shells, device=dev).scatter_add_(
        0, hkl_shell, beta * valid_hkl.float())
    beta_count = torch.zeros(n_shells, device=dev).scatter_add_(
        0, hkl_shell, valid_hkl.float()).clamp(min=1.0)
    shell_beta = beta_sum / beta_count
    shell_beta[shell_beta < 0.5] = 1.3  # fallback

    # alpha_shell^beta = beta * mean(|r|^beta)
    alpha_shell_pow_b = shell_beta * shell_sum / shell_count
    alpha_shell = alpha_shell_pow_b.clamp(min=1e-6).pow(1.0 / shell_beta)

    # Gaussian smooth across shells
    log_alpha_shell = torch.log(alpha_shell.clamp(min=1e-6))
    idx = torch.arange(n_shells, dtype=torch.float32, device=dev)
    smoothed = torch.zeros(n_shells, device=dev)
    for k in range(n_shells):
        w = torch.exp(-0.5 * ((idx - idx[k]) / smooth_sigma) ** 2)
        # Weight by shell_count for robustness
        wc = w * shell_count
        smoothed[k] = (wc * log_alpha_shell).sum() / wc.sum().clamp(min=1e-8)

    # Broadcast shell alpha to per-HKL
    alpha = torch.exp(smoothed[hkl_shell])
    return alpha.clamp(min=0.01)


def _fit_alpha_resolution_prior(
    alpha_raw: Tensor,
    I0: Tensor,
    d_spacing: Tensor,
    beta: Tensor,
    bg: float = 0.0,
    n_shells: int = 20,
    snr_min: float = 30.0,
    smooth_sigma: float = 1.5,
) -> Tensor:
    """Fit smooth α(resolution) from strong reflections.

    Equal-count resolution shells, I0-weighted mean of log(α) within
    each shell (strong reflections only), Gaussian smoothed across shells.
    Applies Poisson debiasing before building the prior curve: the raw
    MLE α is inflated because |log(I_obs/I0)|^β includes Poisson noise.

    Parameters
    ----------
    alpha_raw : (n_valid,) per-HKL raw alpha
    I0 : (n_valid,) current intensities
    d_spacing : (n_valid,) d-spacing in Å
    beta : (n_valid,) GenGauss shape parameter
    bg : background counts (scalar)
    n_shells : number of resolution shells
    snr_min : minimum I0 to include in prior fitting
    smooth_sigma : Gaussian kernel width in shell units

    Returns
    -------
    alpha_prior : (n_valid,) smooth resolution-dependent alpha
    """
    import math

    dev = alpha_raw.device
    n_valid = len(alpha_raw)
    n_shells = min(n_shells, max(3, n_valid // 5))

    # Poisson correction: α_corrected^β = α_raw^β − β · E[|ε_poi|^β]
    # where E[|ε_poi|^β] ≈ E[|N(0,1)|^β] / (I0+bg)^(β/2)
    b = beta.float()
    E_abs_z_beta = 2.0 ** (b / 2.0) * torch.exp(torch.lgamma((b + 1.0) / 2.0)) / math.sqrt(math.pi)
    lam = (I0 + bg).clamp(min=1.0)
    poisson_correction = E_abs_z_beta / lam.pow(b / 2.0)
    alpha_corrected_pow_b = (alpha_raw.pow(b) - b * poisson_correction).clamp(min=1e-8)
    alpha_corrected = alpha_corrected_pow_b.pow(1.0 / b)

    # Equal-count shell binning by d_spacing rank
    rank = d_spacing.argsort().argsort()
    shell_id = (rank.float() / n_valid * n_shells).long().clamp(max=n_shells - 1)

    strong = I0 > snr_min
    log_alpha = torch.log(alpha_corrected.clamp(min=1e-6))

    # Per-shell I0-weighted mean of log(alpha) from strong reflections
    shell_values = {}
    for sid in range(n_shells):
        mask = (shell_id == sid) & strong
        if mask.sum() < 5:
            continue
        w = I0[mask]
        la = log_alpha[mask]
        shell_values[sid] = float((w * la).sum() / w.sum())

    # Fallback: if no shells have enough strong reflections
    if not shell_values:
        return alpha_raw.clone()

    # Gaussian smooth across shells
    sids = sorted(shell_values.keys())
    vals = [shell_values[s] for s in sids]
    idx_arr = torch.tensor(sids, dtype=torch.float32)
    val_arr = torch.tensor(vals, dtype=torch.float32)

    smoothed = torch.zeros(len(sids))
    for k in range(len(sids)):
        w = torch.exp(-0.5 * ((idx_arr - idx_arr[k]) / smooth_sigma) ** 2)
        smoothed[k] = (w * val_arr).sum() / w.sum()

    # Build lookup: shell_id → smoothed log_alpha
    shell_log_alpha = torch.full((n_shells,), float(smoothed.median()),
                                  device=dev)
    for k, sid in enumerate(sids):
        shell_log_alpha[sid] = smoothed[k].item()

    # Fill gaps by nearest-neighbor interpolation
    filled = set(sids)
    for sid in range(n_shells):
        if sid in filled:
            continue
        # Find nearest filled shell
        best_dist = n_shells
        best_val = float(smoothed.median())
        for fsid in sids:
            dist = abs(sid - fsid)
            if dist < best_dist:
                best_dist = dist
                best_val = shell_log_alpha[fsid].item()
        shell_log_alpha[sid] = best_val

    # Map back to per-HKL
    alpha_prior = torch.exp(shell_log_alpha[shell_id])
    return alpha_prior


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
) -> QuadratureGrid:
    """Build log-linear quadrature grid via softplus mapping.

    Uniform in t-space, mapped to signal via softplus:

    - t ≪ 0: signal ≈ exp(t) → log-spaced (good ε-resolution near bg)
    - t ≫ 0: signal ≈ t      → linear (good Poisson peak resolution)

    Per-obs bounds: signal_lo = max(signal_min, signal_hi × 0.01).

    Parameters
    ----------
    n_ph : (N,) photon counts per observation (I_obs + bg)
    bg : per-obs (N,) or scalar background level
    n_quad : number of quadrature points
    n_sigma : Poisson coverage in σ units
    signal_min : absolute lower bound on signal (minimum I0 supported)
    """
    dev = n_ph.device

    sigma = n_ph.clamp(min=1).sqrt()
    signal_max = (n_ph - bg + n_sigma * sigma).clamp(min=1.0)
    signal_lo = (signal_max * 0.01).clamp(min=signal_min)

    t_hi = _softplus_inv(signal_max)
    t_lo = _softplus_inv(signal_lo)

    frac = torch.linspace(0, 1, n_quad, device=dev)
    dt = (t_hi - t_lo) / (n_quad - 1)
    t = t_lo[:, None] + (t_hi - t_lo)[:, None] * frac[None, :]

    signal = F.softplus(t)
    lam = signal + (bg if isinstance(bg, (int, float)) else bg[:, None])
    lam = lam.clamp(min=1e-6)  # guard against negative bg

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


# ── I₀ estimation (GenGauss log-space MLE) ──────────────────


def _log_gg_norm(beta: Tensor) -> Tensor:
    """GenGauss normalization: log(β) - log(2) - lgamma(1/β)."""
    return torch.log(beta) - math.log(2) - torch.lgamma(1.0 / beta)


def fit_I0_log(
    I_obs: Tensor,
    hkl_id: Tensor,
    scale: Tensor,
    alpha: Tensor,
    beta: Tensor,
    I0_init: Tensor,
    obs_counts: Tensor,
    n_valid: int,
    max_iter: int = 50,
) -> tuple[Tensor, Tensor]:
    """GenGauss MLE in log-space (positive observations only).

    Loss = Σ_j |log(I_obs_j / (I0_h · scale_j)) / α_h|^β_h

    The scale enters the prediction, not the data.
    Only positive I_obs contribute (log undefined for non-positive).

    Parameters
    ----------
    I_obs : (N,) raw observations (all, positive + negative)
    hkl_id : (N,) HKL index per obs
    scale : (N,) per-obs multiplicative scale from corrections
    alpha : (n_valid,) GenGauss width
    beta : (n_valid,) GenGauss shape
    I0_init : (n_valid,) initial I₀ estimate
    obs_counts : (n_valid,) observation counts per HKL
    n_valid : number of valid HKLs
    max_iter : LBFGS iterations

    Returns
    -------
    I0, sigma_I0 : (n_valid,) each
    """
    dev = I_obs.device

    # Filter to positive observations
    pos = I_obs > 0
    log_I_pos = torch.log(I_obs[pos])
    hkl_pos = hkl_id[pos]
    log_scale_pos = torch.log(scale[pos].clamp(min=1e-30))
    alpha_po = alpha.repeat_interleave(obs_counts)[pos]
    beta_po = beta.repeat_interleave(obs_counts)[pos]

    log_I0 = torch.log(I0_init.clamp(min=0.1)).clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [log_I0], lr=1.0, max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7, tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        mu = log_I0[hkl_pos] + log_scale_pos
        r = (log_I_pos - mu).abs().clamp(min=1e-8)
        loss = (r / alpha_po).pow(beta_po).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    I0 = torch.exp(log_I0.detach())

    # σ(I₀): empirical scatter (model-free)
    I0_po = I0.repeat_interleave(obs_counts)
    resid_sq = (I_obs - I0_po * scale) ** 2
    sum_resid_sq = torch.zeros(n_valid, device=dev).scatter_add_(
        0, hkl_id, resid_sq)
    n = obs_counts.float()
    sigma_I0 = (sum_resid_sq / (n * (n - 1).clamp(min=1))).clamp(min=1e-12).sqrt()

    return I0, sigma_I0


# ── I₀ estimation (Poisson marginal via softplus quadrature) ─


def fit_I0_poisson_log(
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
    """Poisson marginal MLE with softplus quadrature + analytical Jacobian.

    Generative model::

        ε ~ GenGauss(0, α_h, β_h)
        n ~ Poisson(I0_h · scale_j · exp(ε) + bg_j)
        I_obs_j = n - bg_j

    Marginal likelihood evaluated via softplus change of variables.
    Uses ALL observations (positive and negative).

    Parameters
    ----------
    I_obs : (N,) raw observations (all)
    hkl_id : (N,) HKL index
    scale : (N,) per-obs multiplicative scale from corrections
    alpha, beta : (n_valid,) GenGauss parameters
    I0_init : (n_valid,) initial I₀ estimate
    obs_counts : (n_valid,) obs per HKL
    n_valid : number of valid HKLs
    bg : per-obs (N,) or scalar background level
    grid : pre-built QuadratureGrid (built from n_ph if None)
    n_quad : quadrature points (only used if grid is None)
    max_iter : LBFGS iterations
    """
    dev = I_obs.device

    n_ph = (I_obs + bg).clamp(min=0)
    if grid is None:
        grid = build_softplus_grid(n_ph, bg, n_quad=n_quad)

    alpha_po = alpha.repeat_interleave(obs_counts)
    beta_po = beta.repeat_interleave(obs_counts)
    log_scale = torch.log(scale.clamp(min=1e-30))

    # GenGauss normalization (per-obs, since beta can vary)
    log_gg_norm = _log_gg_norm(beta_po)  # (N,)

    # Pre-combine I0-independent terms
    log_poi_jac = grid.log_poi + grid.log_jac_dt + grid.log_dt[:, None]

    log_I0 = torch.log(I0_init.clamp(min=0.1)).clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [log_I0], lr=1.0, max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7, tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        I0_po = torch.exp(log_I0).repeat_interleave(obs_counts)
        # ε = log(signal / (I0 · scale))
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

    # σ(I₀): empirical scatter (model-free)
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
    # Newton iteration count
    n_newton_iter: int,
    # Smoothing infrastructure
    edge_src: Tensor,
    edge_dst: Tensor,
    n_neighbors: Tensor,
    n_smooth_iter: int,
    smooth_lambda: float,
    # I₀ estimation method
    i0_method: str = "quadrature",
    # Raw observations + scales (for log / poisson_log methods)
    I_obs_all: Tensor | None = None,
    scale_all: Tensor | None = None,
    I_obs_pos: Tensor | None = None,
    scale_pos: Tensor | None = None,
    bg: Tensor | float = 0.0,
    # Resolution-dependent alpha prior
    d_spacing: Tensor | None = None,
    n_alpha_shells: int = 20,
    alpha_prior_lambda: float = 1.0,
) -> DStepResult:
    """Complete D-step: I₀ (all obs), α via MLE (pos obs), smooth α.

    Parameters
    ----------
    I_all_corrected : (N,) ALL corrected observations
    hkl_id_all : (N,) HKL index per ALL obs
    sigma2_poisson_all : (N,) Poisson variance per ALL obs
    obs_counts_all : (n_valid,) obs count per HKL (all)
    I_pos_corrected : (M,) positive corrected observations
    hkl_id_pos : (M,) HKL index per positive obs
    obs_counts_pos : (n_valid,) obs count per HKL (positive only)
    I0_prev : (n_valid,) previous I₀ estimate
    alpha_prev : (n_valid,) previous α estimate
    beta : (n_valid,) GenGauss shape (fixed)
    n_valid : number of valid HKLs
    n_newton_iter : Newton iterations for I₀
    edge_src, edge_dst, n_neighbors : adjacency graph for smoothing
    n_smooth_iter : Laplacian smoothing iterations
    smooth_lambda : smoothing mixing weight
    i0_method : "quadrature" | "newton" | "log" | "poisson_log"
    I_obs_all : (N,) raw observations (for log/poisson_log methods)
    scale_all : (N,) per-obs scale (for log/poisson_log methods)
    I_obs_pos : (M,) raw positive observations (for log method)
    scale_pos : (M,) per-obs scale for positive obs (for log method)
    bg : per-obs (N,) or scalar background (for poisson_log)
    d_spacing : (n_valid,) d-spacing per HKL (enables resolution alpha prior)
    n_alpha_shells : resolution shells for alpha prior
    alpha_prior_lambda : penalty strength for alpha regularization
    """
    # 1. Fit I₀
    if i0_method == "newton":
        sigma2_LN = compute_sigma2_LN(alpha_prev, beta)
        I0, sigma_I0 = fit_I0_newton(
            I_all_corrected, hkl_id_all, sigma2_poisson_all,
            sigma2_LN, I0_prev, obs_counts_all, n_valid,
            n_iter=n_newton_iter,
        )
    elif i0_method == "log":
        I0, sigma_I0 = fit_I0_log(
            I_obs_all, hkl_id_all,
            scale_all if scale_all is not None else torch.ones_like(I_obs_all),
            alpha_prev, beta, I0_prev, obs_counts_all, n_valid,
            max_iter=n_newton_iter,
        )
    elif i0_method == "poisson_log":
        I0, sigma_I0 = fit_I0_poisson_log(
            I_obs_all, hkl_id_all,
            scale_all if scale_all is not None else torch.ones_like(I_obs_all),
            alpha_prev, beta, I0_prev, obs_counts_all, n_valid,
            bg=bg,
            max_iter=max(n_newton_iter, 20),
        )
    else:  # "quadrature"
        I0, sigma_I0 = fit_I0_quadrature(
            I_all_corrected, hkl_id_all, sigma2_poisson_all,
            alpha_prev, beta, I0_prev, obs_counts_all, n_valid,
            max_iter=n_newton_iter,
        )

    # 3. Fit α from log-space residuals of positive observations
    if d_spacing is not None and alpha_prior_lambda > 0:
        # Shell-level alpha: pool residuals per resolution shell for a
        # robust estimate, then broadcast back to per-HKL.
        alpha_smooth = _fit_alpha_shells(
            I_pos_corrected, hkl_id_pos, I0, beta,
            obs_counts_pos, n_valid,
            d_spacing, n_shells=n_alpha_shells,
        )
    else:
        # Fallback: unregularized MLE + Laplacian smoothing
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


# ══════════════════════════════════════════════════════════════
# Joint optimisation of (I₀, α) via LBFGS + autograd
# ══════════════════════════════════════════════════════════════


def _joint_nll(
    I0_param: Tensor,
    log_alpha_param: Tensor,
    I_corrected: Tensor,
    obs_counts: Tensor,
    eps_tilde: Tensor,
    log_gg_weights: Tensor,
    inv_2s2: Tensor,
    log_norm_gauss: Tensor,
    edge_src: Tensor,
    edge_dst: Tensor,
    reg_lambda_eff: float,
) -> Tensor:
    """Joint NLL with Laplacian regularisation (differentiable).

    Loss = -Σ_j log p(I_obs_j | I₀, α, β, σ²_poi) + λ·Σ_edges (Δlog α)²

    Fully vectorised over quadrature points. Alpha enters through
    exp(alpha_po · ε̃_k); GenGauss weights are alpha-independent.
    """
    alpha_po = torch.exp(log_alpha_param).repeat_interleave(obs_counts)  # (N,)
    I0_po = I0_param.repeat_interleave(obs_counts)                       # (N,)

    # (K, N): vectorised over all quadrature points
    s = torch.exp(alpha_po[None, :] * eps_tilde[:, None])
    predicted = I0_po[None, :] * s
    residual = I_corrected[None, :] - predicted
    log_gauss = -(residual ** 2) * inv_2s2[None, :] + log_norm_gauss[None, :]
    log_joints = log_gauss + log_gg_weights  # (K, N)

    log_p = torch.logsumexp(log_joints, dim=0)
    nll = -log_p.sum()

    # Laplacian regularisation on log(α)
    diff = log_alpha_param[edge_src] - log_alpha_param[edge_dst]
    reg = reg_lambda_eff * 0.5 * (diff ** 2).sum()

    return nll + reg


def fit_joint(
    I_corrected: Tensor,
    hkl_id: Tensor,
    sigma2_poisson: Tensor,
    alpha_init: Tensor,
    beta: Tensor,
    I0_init: Tensor,
    obs_counts: Tensor,
    n_valid: int,
    edge_src: Tensor,
    edge_dst: Tensor,
    reg_lambda: float = 0.1,
    n_quad: int = 21,
    max_iter: int = 30,
    warmup_iter: int = 5,
    device: str | torch.device = "cpu",
) -> DStepResult:
    """Joint LBFGS optimisation of (I₀, log α) with exact quadrature NLL.

    The generative model per observation:

        ε ~ GenGauss(0, α_h, β_h)
        I_obs = I₀_h · exp(ε) + N(0, σ²_poi)

    The marginal likelihood ∫ f_GG(ε) · N(I_obs; I₀·eᵋ, σ²_poi) dε is
    evaluated via quadrature on ε̃ = ε/α with adaptive range based on β.
    Gradients computed by autograd; optimised with LBFGS.  α is
    regularised via a Laplacian penalty on log(α) over the Miller-index
    adjacency graph.

    Stage 1: warm-up — LBFGS on [I₀] only (α detached).
    Stage 2: joint  — LBFGS on [I₀, log α] together.

    Uses ALL observations (positive AND negative).

    Parameters
    ----------
    I_corrected : (N,) corrected observations
    hkl_id : (N,) HKL index per obs
    sigma2_poisson : (N,) per-obs Poisson variance (gain=1)
    alpha_init : (n_valid,) initial α from log-space MLE
    beta : (n_valid,) GenGauss shape (fixed)
    I0_init : (n_valid,) initial I₀ from log-space MLE
    obs_counts : (n_valid,) counts per HKL (all obs)
    n_valid : number of HKLs
    edge_src, edge_dst : (E,) adjacency graph edges
    reg_lambda : regularisation weight (each edge worth reg_lambda obs)
    n_quad : quadrature points
    max_iter : LBFGS iterations for joint stage
    warmup_iter : LBFGS iterations for I₀-only warm-up

    Returns
    -------
    DStepResult(I0, sigma_I0, alpha, beta)
    """
    # ── Move to device ──
    I_corrected = I_corrected.to(device)
    hkl_id = hkl_id.to(device)
    sigma2_poisson = sigma2_poisson.to(device)
    alpha_init = alpha_init.to(device)
    beta = beta.to(device)
    I0_init = I0_init.to(device)
    obs_counts = obs_counts.to(device)
    edge_src = edge_src.to(device)
    edge_dst = edge_dst.to(device)

    # ── Pre-compute quadrature grid + weights (vectorised) ──
    q = _prepare_quadrature(alpha_init, beta, obs_counts, sigma2_poisson,
                            n_quad, device=device)

    # ── Regularisation strength ──
    n_obs = I_corrected.shape[0]
    n_edges = max(edge_src.shape[0], 1)
    reg_lambda_eff = reg_lambda * n_obs / n_edges

    # ── Parameters ──
    I0_param = I0_init.clone().requires_grad_(True)
    log_alpha_param = torch.log(alpha_init.clamp(min=0.01)).clone().requires_grad_(True)

    def _make_closure(params):
        """Build LBFGS closure for the given parameter list."""
        # Determine whether alpha is optimised or detached
        track_alpha = any(p is log_alpha_param for p in params)

        def closure():
            for p in params:
                if p.grad is not None:
                    p.grad = None
            la = log_alpha_param if track_alpha else log_alpha_param.detach()
            loss = _joint_nll(
                I0_param, la,
                I_corrected, obs_counts,
                q["eps_tilde"], q["log_gg_weights"],
                q["inv_2s2"], q["log_norm_gauss"],
                edge_src, edge_dst, reg_lambda_eff,
            )
            loss.backward()
            return loss

        return closure

    # ── Stage 1: warm-up — I₀ only ──
    if warmup_iter > 0:
        opt1 = torch.optim.LBFGS(
            [I0_param], lr=1.0, max_iter=warmup_iter,
            line_search_fn="strong_wolfe",
        )
        opt1.step(_make_closure([I0_param]))

    # ── Stage 2: joint optimisation ──
    # Single LBFGS call with logging via closure counter
    _step_count = [0]
    base_closure = _make_closure([I0_param, log_alpha_param])

    def logging_closure():
        loss = base_closure()
        _step_count[0] += 1
        if _step_count[0] % 20 == 0 or _step_count[0] <= 5:
            with torch.no_grad():
                med_I0 = I0_param.median().item()
                med_a = torch.exp(log_alpha_param).median().item()
                per_obs = loss.item() / n_obs
            print(f"    LBFGS eval {_step_count[0]}: loss/obs={per_obs:.4f} "
                  f"med(I₀)={med_I0:.2f} med(α)={med_a:.4f}", flush=True)
        return loss

    opt2 = torch.optim.LBFGS(
        [I0_param, log_alpha_param], lr=1.0, max_iter=max_iter,
        history_size=50,  # more curvature history for 30k params
        line_search_fn="strong_wolfe",
    )
    opt2.step(logging_closure)

    # ── Extract results ──
    I0 = I0_param.detach()
    alpha = torch.exp(log_alpha_param.detach())

    # ── σ(I₀): empirical scatter (model-free, computed on device) ──
    I0_final = I0.repeat_interleave(obs_counts)
    resid_sq = (I_corrected - I0_final) ** 2
    sum_resid_sq = torch.zeros(n_valid, device=I0.device).scatter_add_(0, hkl_id, resid_sq)
    n = obs_counts.float()
    sigma2_I0 = sum_resid_sq / (n * (n - 1.0).clamp(min=1.0))
    sigma_I0 = sigma2_I0.clamp(min=1e-12).sqrt()

    # Move to CPU
    I0 = I0.cpu()
    alpha = alpha.cpu()
    sigma_I0 = sigma_I0.cpu()

    n_zero = int((I0 < 1.0).sum())
    print(f"    fit_joint: I₀<1: {n_zero}/{n_valid}, "
          f"med(α)={alpha.median():.4f}, med(I₀)={I0.median():.1f}")

    return DStepResult(I0=I0, sigma_I0=sigma_I0, alpha=alpha, beta=beta)
