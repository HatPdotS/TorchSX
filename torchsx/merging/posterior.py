"""Posterior moments of ``I₀`` for writeout.

After LBFGS converges we want ``⟨I₀⟩ = ∫ I₀·p(I₀|data) dI₀`` and
``Var[I₀]``, not just the MAP mode ``exp(log_I0_MAP)``.  The expensive
step is the *inner* signal quadrature ``∫ p_Poi(n|S+b)·p_scatter(log S −
log(I·K)) dS`` — it has to be evaluated at many outer ``log I₀`` points.

We avoid re-running the inner quadrature at every fine outer grid point
by noting that ``f_j(μ) := log p(n_j | μ, α_h, β_h, bg_j)``, where
``μ = log I₀ + log K_j``, is a **smooth 1-D function of μ** (α, β are
fixed at the converged values).  Algorithm:

1. For each observation ``j``, evaluate ``f_j(μ)`` on a **coarse** μ-grid
   of ``K_coarse`` points centred at ``μ_MAP_j = log_I0_MAP[h_j] +
   log_scale_j``, with extent ``± n_sigma · σ_Laplace[h_j]``.  This is the
   only inner-quadrature work — ``K_coarse`` evaluations per obs.
2. At writeout, the fine outer grid ``Δ_m ∈ [Δ_MAP − n_sigma·σ, Δ_MAP +
   n_sigma·σ]`` has ``M_fine`` points.  For each obs ``j``, evaluate
   ``f_j(Δ_m + log_scale_j)`` by **linear interpolation** of the coarse
   table — pure bandwidth, no GenGauss / quadrature re-evaluation.
3. Sum the interpolated per-obs curves into per-HKL log-likelihood curves
   ``L_h(Δ_m) = Σ_{j∈h} f_j(Δ_m + log_scale_j)``.
4. Add each candidate prior ``log π(Δ_m)`` and trapezoidal-integrate in
   ``Δ``:  ``⟨I₀⟩_h = ∫ e^{Δ + L_h + log π} dΔ / ∫ e^{L_h + log π} dΔ``.

Cost:  ``N_obs × K_coarse × K_inner`` inner quadrature evals (instead of
``N_obs × M_fine × K_inner``).  With default ``K_coarse = 31``, ``M_fine =
101``, ``K_inner = 51`` that's a ~3× compute reduction and makes the
outer loop essentially free.  Different priors are cheap swaps on top of
the shared ``L_h`` table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.trapezoidal import TrapezoidalLogGrid
from torchsx.merging.scale.base import ScaleModel
from torchsx.merging.scatter.base import ScatterModel


# Type alias for a prior evaluator: given the per-HKL Δ grid (n_valid, M_fine)
# return log_prior(Δ) of the same shape.  ``None`` means flat log p(Δ)=0.
PriorEvaluator = Callable[[Tensor], Tensor]


@dataclass
class PosteriorMoments:
    """Per-HKL posterior summary."""

    I0: Tensor              # (n_valid,)  posterior mean (MAP fallback where unstable)
    sigma_I0: Tensor        # (n_valid,)  sqrt posterior variance
    log_evidence: Tensor    # (n_valid,)  log ∫ π(Δ)·L(Δ) dΔ
    mu: Tensor              # (n_valid,)  = log(I0)
    sigma_mu: Tensor        # (n_valid,)  = σ(log I0) ≈ sigma_I0 / I0
    # Per-HKL quality diagnostics.
    fallback_mask: Tensor   # (n_valid,) bool — True where posterior blew up and MAP was substituted
    grid_mass_frac: Tensor  # (n_valid,) — fraction of total mass captured inside ±n_sigma·σ
    # posterior mode vs. MAP (Laplace) diagnostic — offset in σ units
    mode_shift: Tensor      # (n_valid,) — (Δ*_post − Δ_MAP) / σ_Laplace
    sigma_laplace_clamped: Tensor  # (n_valid,) — σ after the sigma_max cap
    # Structure-factor moments under the same posterior — bypasses French–Wilson.
    F: Tensor | None = None        # (n_valid,) E[√I0 | data]
    F_sigma: Tensor | None = None  # (n_valid,) sqrt(Var[√I0 | data]) = sqrt(I0 - F²)


def _interp_weights(n_coarse: int, n_fine: int, n_sigma: float, device, dtype):
    """Linear-interpolation bookkeeping on the shared [-n_sigma, n_sigma] axis.

    Returns (offsets_coarse, offsets_fine, i_lo, w_hi) such that for a
    coarse table ``f[i]`` of length ``n_coarse``, the fine value at fine
    offset ``m`` is ``f[i_lo[m]] * (1 − w_hi[m]) + f[i_lo[m] + 1] * w_hi[m]``.
    """
    offsets_coarse = torch.linspace(
        -n_sigma, n_sigma, n_coarse, device=device, dtype=dtype
    )
    offsets_fine = torch.linspace(
        -n_sigma, n_sigma, n_fine, device=device, dtype=dtype
    )
    # Normalised position of each fine point inside the coarse grid, in [0, K-1].
    pos = (offsets_fine - offsets_coarse[0]) / (offsets_coarse[-1] - offsets_coarse[0])
    pos = pos * (n_coarse - 1)
    i_lo = pos.floor().long().clamp(max=n_coarse - 2)
    w_hi = (pos - i_lo.to(dtype)).clamp(min=0.0, max=1.0)
    return offsets_coarse, offsets_fine, i_lo, w_hi


@torch.no_grad()
def _fill_f_coarse(
    *,
    prep: PreparedObs,
    params: Parameters,
    scale: ScaleModel,
    scatter: ScatterModel,
    quadrature: TrapezoidalLogGrid,
    likelihood: DetectionLikelihood,
    delta_map: Tensor,          # (n_valid,)
    sigma_laplace: Tensor,      # (n_valid,)
    offsets_coarse: Tensor,     # (K_coarse,)
    chunk_size: int,
) -> Tensor:
    """Compute ``f_j(μ)`` on the per-obs coarse grid.

    Returns ``f_coarse`` of shape ``(n_obs_total, K_coarse)``.
    """
    n_obs = prep.n_obs_total
    K_coarse = offsets_coarse.numel()
    device, dtype = delta_map.device, delta_map.dtype
    f_coarse = torch.empty((n_obs, K_coarse), device=device, dtype=dtype)

    hkl_id = prep.hkl_id
    bg = prep.bg
    n_ph = (prep.I_obs + bg).clamp(min=0)

    # Chunk over observations.
    C = chunk_size if chunk_size and chunk_size > 0 else n_obs
    for s in range(0, n_obs, C):
        e = min(s + C, n_obs)
        oi_chunk = prep.obs_info.slice(s, e)
        h_chunk = hkl_id[s:e]
        delta_map_chunk = delta_map[h_chunk]                   # (Nc,)
        sigma_chunk = sigma_laplace[h_chunk]                    # (Nc,)

        # Δ-invariant inner-quadrature backbone.
        n_ph_chunk = n_ph[s:e]
        bg_chunk = bg[s:e]
        sig_lo = quadrature._sig_lo_slice(n_ph_chunk, bg_chunk, obs_range=(s, e))
        signal, log_jac_plus_dt = quadrature._build_grid(n_ph_chunk, bg_chunk, sig_lo)
        log_signal = torch.log(signal.clamp(min=1e-12))              # (Nc, K_inner)
        log_lik = likelihood.log_prob_given_signal(
            n_ph_chunk, signal, bg_chunk
        )                                                             # (Nc, K_inner)
        L_backbone = log_lik + log_jac_plus_dt                        # (Nc, K_inner)

        log_scale_chunk = scale.log_k(params, oi_chunk, obs_range=(s, e))  # (Nc,)
        mu_map = delta_map_chunk + log_scale_chunk                    # (Nc,)

        # Evaluate f_j at each coarse offset.
        for i in range(K_coarse):
            mu_i = mu_map + offsets_coarse[i] * sigma_chunk           # (Nc,)
            eps = log_signal - mu_i[:, None]                          # (Nc, K_inner)
            log_scatter = scatter.log_prob(eps, params, oi_chunk)     # (Nc, K_inner)
            log_integrand = L_backbone + log_scatter
            log_p_sig = torch.logsumexp(log_integrand, dim=1)         # (Nc,)
            log_p = likelihood.combine_with_marginal(
                n_ph_chunk, bg_chunk, log_p_sig, params, oi_chunk
            )                                                          # (Nc,)
            f_coarse[s:e, i] = log_p

    return f_coarse


@torch.no_grad()
def _reduce_to_hkl_curves(
    f_coarse: Tensor,           # (n_obs, K_coarse)
    hkl_id: Tensor,             # (n_obs,)
    n_valid: int,
    i_lo: Tensor,               # (M_fine,) long
    w_hi: Tensor,               # (M_fine,)
) -> Tensor:
    """Interpolate ``f_j`` to the fine grid and sum by HKL.

    Returns ``L`` of shape ``(n_valid, M_fine)``.
    """
    M_fine = i_lo.numel()
    device, dtype = f_coarse.device, f_coarse.dtype
    L = torch.zeros((n_valid, M_fine), device=device, dtype=dtype)
    # One column of L at a time to bound peak memory to O(n_obs).
    for m in range(M_fine):
        f_m = (
            f_coarse[:, i_lo[m]] * (1.0 - w_hi[m])
            + f_coarse[:, i_lo[m] + 1] * w_hi[m]
        )
        L[:, m].scatter_add_(0, hkl_id, f_m)
    return L


def _moments_from_log_post(
    delta_fine: Tensor,         # (n_valid, M_fine)
    log_post: Tensor,           # (n_valid, M_fine)
    ddelta: Tensor,             # (n_valid,)  — per-HKL Δ step
    *,
    delta_map: Tensor,          # (n_valid,)
    sigma_clamped: Tensor,      # (n_valid,)
    map_fallback_log_I0: Tensor | None,
    summary: str = "linear_mean",
) -> PosteriorMoments:
    """Trapezoidal integration in Δ for the posterior summary.

    Three summaries (selected by ``summary``):

    * ``"linear_mean"`` (default, legacy)
        ``I0 = E[I | data]``, ``σ_I0 = sqrt(Var[I | data])``.  Integrates
        in linear-I space — biased upward by ``exp(σ²_logI / 2)`` for
        broad posteriors (Jensen).

    * ``"log_mean"``
        ``μ = E[log I | data]``, ``σ_μ = sqrt(Var[log I | data])``,
        ``I0 = exp(μ)`` and ``σ_I0 = σ_μ · I0`` (delta method).
        Unbiased w.r.t. log_I_truth on a window-matched grid; on a wide
        global grid it can be biased downward by mass leaking into the
        flat-likelihood tail.

    * ``"mode"``
        ``μ = argmax_Δ log_post(Δ)`` parabolic-refined to sub-grid
        precision, ``σ_μ = 1/√curvature_at_mode`` (with geometric-
        variance fallback at the boundary).  Robust to long flat tails
        that drag the mean — picks the data-favoured peak even when the
        log_I grid extends far past the informative region.

    Emits the same diagnostics either way.
    """
    log_dD = torch.log(ddelta)                                # (n_valid,)
    log_Z = log_dD + torch.logsumexp(log_post, dim=1)         # log ∫ e^{log_post} dΔ

    if summary == "log_mean":
        # Stable softmax-weighted moments of Δ.
        log_max = log_post.max(dim=1, keepdim=True).values
        w = (log_post - log_max).exp()
        Z_local = w.sum(dim=1, keepdim=True).clamp(min=1e-30)
        p = w / Z_local                                        # (n_valid, M_fine)
        mu = (delta_fine * p).sum(dim=1)                       # (n_valid,)
        mu2 = (delta_fine.pow(2) * p).sum(dim=1)
        sigma_mu = (mu2 - mu.pow(2)).clamp(min=0.0).sqrt()
        I0 = mu.exp()
        sigma_I0 = sigma_mu * I0
    elif summary == "linear_mean":
        log_EI = log_dD + torch.logsumexp(delta_fine + log_post, dim=1) - log_Z
        log_EI2 = log_dD + torch.logsumexp(2.0 * delta_fine + log_post, dim=1) - log_Z
        I0 = log_EI.exp()
        I0_sq_mean = log_EI2.exp()
        var_I0 = (I0_sq_mean - I0.pow(2)).clamp(min=0.0)
        sigma_I0 = var_I0.sqrt()
        mu = I0.clamp(min=1e-12).log()
        sigma_mu = sigma_I0 / I0.clamp(min=1e-12)
        # Structure-factor moments F = √I₀ — softmax-weighted mean over the
        # same grid as I0, but computed in linear F space so that the variance
        # never has to subtract two overflowed inf's (which is what kills the
        # naive logsumexp-based formulation when the global log_I grid spans
        # ~9 decades).  E[F] = ∫ exp(½Δ) p(Δ) dΔ and Var[F] = E[F²] − E[F]².
        log_max_p = log_post.max(dim=1, keepdim=True).values
        p_norm_F = (log_post - log_max_p).exp()
        Zp = p_norm_F.sum(dim=1, keepdim=True).clamp(min=1e-30)
        p_norm_F = p_norm_F / Zp                       # (n_valid, K)
        F_grid = (0.5 * delta_fine).exp()              # √I_grid (bounded by √S_max)
        F_mean = (F_grid * p_norm_F).sum(dim=1)
        F2_mean = (F_grid.pow(2) * p_norm_F).sum(dim=1)
        F_var = (F2_mean - F_mean.pow(2)).clamp(min=0.0)
        F_sigma_val = F_var.sqrt()
    elif summary == "mode":
        # Per-HKL posterior mode = argmax(log_post), parabolic-refined to
        # sub-grid precision using the two grid neighbours.  Robust to long
        # flat tails that drag the mean (linear or geometric) — picks the
        # data-favoured peak even when the global log_I grid extends far
        # past the informative region.  σ_logI from the local parabolic
        # curvature; geometric-variance fallback at the boundary.
        argmax_m = log_post.argmax(dim=1)                          # (n_valid,)
        M_fine = log_post.shape[1]
        interior = (argmax_m > 0) & (argmax_m < M_fine - 1)
        k_int = argmax_m.clamp(min=1, max=M_fine - 2)
        y_lo = log_post.gather(1, (k_int - 1)[:, None]).squeeze(1)
        y_md = log_post.gather(1, k_int[:, None]).squeeze(1)
        y_hi = log_post.gather(1, (k_int + 1)[:, None]).squeeze(1)
        # log_post is concave at its argmax → curvature > 0 inside the support.
        curvature = 2.0 * y_md - y_lo - y_hi
        use_para = interior & (curvature > 1e-12)
        refine = 0.5 * (y_hi - y_lo) / curvature.clamp(min=1e-12)
        refine = torch.where(
            use_para, refine.clamp(min=-0.5, max=0.5), torch.zeros_like(refine)
        )
        delta_mode_int = delta_fine.gather(1, k_int[:, None]).squeeze(1)
        mu = delta_mode_int + refine * ddelta

        # σ_logI: parabolic 1/√curvature where well-defined, else the
        # softmax-weighted standard deviation (always finite, bounded by
        # the grid extent).
        log_max = log_post.max(dim=1, keepdim=True).values
        w = (log_post - log_max).exp()
        Z_local = w.sum(dim=1, keepdim=True).clamp(min=1e-30)
        p = w / Z_local
        mu_geo = (delta_fine * p).sum(dim=1)
        mu2_geo = (delta_fine.pow(2) * p).sum(dim=1)
        sigma_geo = (mu2_geo - mu_geo.pow(2)).clamp(min=0.0).sqrt()
        sigma_para = (ddelta.pow(2) / curvature.clamp(min=1e-12)).sqrt()
        sigma_mu = torch.where(use_para, sigma_para, sigma_geo)
        I0 = mu.exp()
        sigma_I0 = sigma_mu * I0
    else:
        raise ValueError(f"Unknown posterior summary: {summary!r}")

    # ── Sub-grid σ floor ─────────────────────────────────────────
    # When the actual posterior is sharper than the integration grid step
    # (typical for strong, well-measured HKLs whose Laplace σ has been
    # clamped at ``sigma_max``), the linear-mean variance can underflow to
    # exactly zero (E[I²] - E[I]² ≈ 0 with all mass on one node) and the
    # parabolic-curvature σ in the mode branch can become arbitrarily
    # small.  Floor σ_μ at half a grid step in log_I — below that the
    # discretisation makes the reported σ meaningless, and at zero the
    # downstream consumer (phenix.iotbx) silently drops the reflection.
    sigma_mu_floor = ddelta * 0.5
    sigma_mu = torch.maximum(sigma_mu, sigma_mu_floor)
    sigma_I0 = torch.maximum(sigma_I0, sigma_mu_floor * I0)

    # ── Diagnostics ──────────────────────────────────────────────
    # Posterior mode (argmax index) → shift from MAP in σ-units.
    argmax_m = log_post.argmax(dim=1)                          # (n_valid,)
    delta_mode = delta_fine.gather(1, argmax_m[:, None]).squeeze(1)
    mode_shift = (delta_mode - delta_map) / sigma_clamped.clamp(min=1e-12)

    # Mass fraction inside the central ±3σ of the (±n_sigma·σ) grid.
    M_fine = delta_fine.shape[1]
    lo = M_fine // 4
    hi = M_fine - lo
    log_max = log_post.max(dim=1, keepdim=True).values
    mass_unnorm = (log_post - log_max).exp().sum(dim=1)
    mass_central = (log_post[:, lo:hi] - log_max).exp().sum(dim=1)
    grid_mass_frac = (mass_central / mass_unnorm.clamp(min=1e-30)).clamp(0.0, 1.0)

    # ── MAP fallback on non-finite moments ─────────────────────
    bad = ~(torch.isfinite(I0) & torch.isfinite(sigma_I0))
    fallback_mask = bad.clone()
    if map_fallback_log_I0 is not None and bad.any():
        mu_fallback = map_fallback_log_I0
        I0_fallback = mu_fallback.exp()
        sigma_fallback = sigma_clamped * I0_fallback           # delta-method
        mu = torch.where(bad, mu_fallback, mu)
        sigma_mu = torch.where(bad, sigma_clamped, sigma_mu)
        I0 = torch.where(bad, I0_fallback, I0)
        sigma_I0 = torch.where(bad, sigma_fallback, sigma_I0)

    F_out = None
    F_sigma_out = None
    if summary == "linear_mean":
        F_out = F_mean
        F_sigma_out = F_sigma_val
        if map_fallback_log_I0 is not None and bad.any():
            F_fallback = (0.5 * map_fallback_log_I0).exp()
            F_out = torch.where(bad, F_fallback, F_out)
            F_sigma_out = torch.where(
                bad, sigma_clamped * 0.5 * F_fallback, F_sigma_out,
            )

    return PosteriorMoments(
        I0=I0, sigma_I0=sigma_I0, log_evidence=log_Z, mu=mu, sigma_mu=sigma_mu,
        fallback_mask=fallback_mask,
        grid_mass_frac=grid_mass_frac,
        mode_shift=mode_shift,
        sigma_laplace_clamped=sigma_clamped,
        F=F_out, F_sigma=F_sigma_out,
    )


# ── Prior evaluators ───────────────────────────────────────────────────


def make_flat_prior() -> PriorEvaluator:
    """Flat improper prior — log π(Δ) = 0 everywhere."""

    def _flat(delta_fine: Tensor) -> Tensor:
        return torch.zeros_like(delta_fine)

    return _flat


def make_wilson_prior(
    *,
    Sigma_shell: Tensor,        # (n_valid,)  fixed per-HKL shell mean ⟨I⟩
    is_centric: Tensor,         # (n_valid,)  bool
) -> PriorEvaluator:
    """Wilson log-prior on ``log I₀`` with a *fixed* Σ_h (classical FW).

    ``Σ_h`` is the converged shell-arithmetic-mean of ``exp(log_I0_MAP)``
    computed once at writeout — it does *not* vary with the outer
    integration variable Δ, so the prior is strictly per-HKL and can be
    interpolated/broadcast like any fixed column.

    Negative log-prior density per the standard derivation (see
    ``priors.wilson.WilsonPrior``):

        acentric:  −log p_Δ = e^Δ/Σ + log Σ − Δ
        centric:   −log p_Δ = e^Δ/(2Σ) + ½·log Σ − ½·Δ

    returned here as **log p** (positive-of-likelihood), so the caller can
    add it to the data log-likelihood with the same sign convention.
    """
    log_Sigma = Sigma_shell.clamp(min=1e-12).log()               # (n_valid,)
    inv_Sigma = 1.0 / Sigma_shell.clamp(min=1e-12)               # (n_valid,)
    centric = is_centric.bool()

    def _wilson(delta_fine: Tensor) -> Tensor:
        # delta_fine: (n_valid, M_fine)
        I = delta_fine.exp()
        nlp_acen = I * inv_Sigma[:, None] + log_Sigma[:, None] - delta_fine
        nlp_cent = 0.5 * (I * inv_Sigma[:, None] + log_Sigma[:, None] - delta_fine)
        nlp = torch.where(centric[:, None], nlp_cent, nlp_acen)
        return -nlp

    return _wilson


def shell_mean_from_log_I0(
    log_I0: Tensor,                # (n_valid,)
    shell_id: Tensor,              # (n_valid,)
    n_shells: int,
    eps: float = 1e-6,
) -> Tensor:
    """Arithmetic shell-mean ``Σ_h = ⟨I⟩_{shell(h)}``, fixed at writeout."""
    I = log_I0.exp()
    shell_sum = torch.zeros(
        n_shells, device=I.device, dtype=I.dtype
    ).scatter_add(0, shell_id, I)
    shell_cnt = torch.bincount(shell_id, minlength=n_shells).to(I.dtype)
    Sigma_per_shell = shell_sum / shell_cnt.clamp(min=1.0)
    return Sigma_per_shell[shell_id].clamp(min=eps)


# ── Entry point ────────────────────────────────────────────────────────


@torch.no_grad()
def compute_posterior_moments(
    *,
    prep: PreparedObs,
    params: Parameters,
    scale: ScaleModel,
    scatter: ScatterModel,
    quadrature: TrapezoidalLogGrid,
    likelihood: DetectionLikelihood,
    sigma_laplace: Tensor,
    priors: Mapping[str, PriorEvaluator],
    n_grid_coarse: int = 31,
    n_grid_fine: int = 101,
    n_sigma: float = 6.0,
    obs_chunk_size: int = 500_000,
    sigma_max: float = 2.0,
    map_fallback_log_I0: Tensor | None = None,
    summary: str = "linear_mean",
) -> dict[str, PosteriorMoments]:
    """Per-HKL posterior mean / std of ``I₀`` under each named prior.

    Parameters
    ----------
    prep, params, scale, scatter, quadrature, likelihood
        Converged components — α, β, scale/B, π_h, σ_lo, etc. are frozen.
    sigma_laplace
        Per-HKL Laplace σ on ``log_I0`` (``1/√H_diag``).  Sets the width
        of the outer integration grid.
    priors
        ``{name: PriorEvaluator}``.  Keys become the output dict keys.
        See :func:`make_flat_prior`, :func:`make_wilson_prior`.
    n_grid_coarse, n_grid_fine, n_sigma
        Coarse grid drives the quadrature cost, fine grid drives the
        integration precision.  Both span ``± n_sigma · σ_Laplace`` in Δ.
    obs_chunk_size
        Memory cap for the per-obs coarse-grid fill.
    sigma_max
        Upper clamp on ``σ_Laplace`` (in log-I₀).  HKLs with tiny data
        Hessians have σ_Laplace → ∞; without a cap the integration grid
        spans thousands of log-units and exp(Δ) overflows.  Default 2.0
        corresponds to a ~7.4× multiplicative uncertainty factor — far
        wider than any plausibly informative HKL.
    map_fallback_log_I0
        Optional per-HKL ``log(I₀_MAP)`` to substitute for any HKL where
        the posterior integration returns non-finite moments (e.g. inf
        from an underconstrained HKL whose grid still overflowed, or
        NaN from log(0) when the data likelihood vanishes everywhere on
        the grid).  If ``None``, non-finite entries are left in place.

    Returns
    -------
    ``{name: PosteriorMoments}``.  Each moment pack additionally carries
    a ``fallback_mask`` buffer identifying HKLs that were substituted.
    """
    device = params.log_I0.device
    dtype = params.log_I0.dtype
    n_valid = prep.n_valid

    delta_map = params.log_I0.detach()
    # Clamp σ on both ends:
    #   min=1e-3  — avoid zero-width grids on degenerate HKLs
    #   max=sigma_max — avoid blow-up from near-flat data Hessians whose
    #                    σ_Laplace can reach hundreds or thousands of
    #                    log-units, making ±n_sigma·σ cover thousands of
    #                    orders of magnitude in I₀ and overflowing exp.
    sigma = sigma_laplace.detach().clamp(min=1e-3, max=float(sigma_max))

    # Shared offset grids + interpolation weights.
    offsets_coarse, offsets_fine, i_lo, w_hi = _interp_weights(
        n_grid_coarse, n_grid_fine, n_sigma, device=device, dtype=dtype
    )

    # Step 1: coarse per-obs log-likelihood table.
    f_coarse = _fill_f_coarse(
        prep=prep, params=params, scale=scale, scatter=scatter,
        quadrature=quadrature, likelihood=likelihood,
        delta_map=delta_map, sigma_laplace=sigma,
        offsets_coarse=offsets_coarse, chunk_size=obs_chunk_size,
    )

    # Step 2: fine-grid per-HKL log-likelihood curve.
    L = _reduce_to_hkl_curves(f_coarse, prep.hkl_id, n_valid, i_lo, w_hi)

    # Per-HKL fine Δ grid.
    delta_fine = delta_map[:, None] + offsets_fine[None, :] * sigma[:, None]
    ddelta = (2.0 * n_sigma) * sigma / (n_grid_fine - 1)

    # Step 3: apply each prior and integrate.
    results: dict[str, PosteriorMoments] = {}
    for name, prior_fn in priors.items():
        log_prior = prior_fn(delta_fine) if prior_fn is not None else torch.zeros_like(L)
        log_post = L + log_prior
        results[name] = _moments_from_log_post(
            delta_fine, log_post, ddelta,
            delta_map=delta_map,
            sigma_clamped=sigma,
            map_fallback_log_I0=map_fallback_log_I0,
            summary=summary,
        )
    return results


@torch.no_grad()
def _fill_f_coarse_per_call(
    *,
    prep: PreparedObs,
    params: Parameters,
    scale: ScaleModel,
    scatter: ScatterModel,
    quadrature,                 # any Quadrature with marginalize_signal
    likelihood: DetectionLikelihood,
    delta_map: Tensor,          # (n_valid,)
    sigma_laplace: Tensor,      # (n_valid,)
    offsets_coarse: Tensor,     # (K_coarse,)
    chunk_size: int,
) -> Tensor:
    """Fill ``f_j(μ)`` by calling ``quadrature.marginalize_signal`` per offset.

    Drop-in alternative to :func:`_fill_f_coarse` for quadratures where the
    inner-grid node positions depend on trial μ (i.e. on log_mean) and the
    cached-backbone optimisation does not apply.  ``LaplaceSaddleGrid``
    relocates its nodes around the joint saddle every call, so each μ
    offset needs a fresh ``marginalize_signal`` evaluation.

    Cost: ``K_coarse × ceil(n_obs / chunk_size)`` quadrature calls (vs the
    cached version's 1 backbone build + ``K_coarse`` scatter evaluations).
    Typical 2–3× slower but works with any Quadrature.
    """
    n_obs = prep.n_obs_total
    K_coarse = offsets_coarse.numel()
    device, dtype = delta_map.device, delta_map.dtype
    f_coarse = torch.empty((n_obs, K_coarse), device=device, dtype=dtype)

    hkl_id = prep.hkl_id
    bg = prep.bg
    n_ph = (prep.I_obs + bg).clamp(min=0)

    C = chunk_size if chunk_size and chunk_size > 0 else n_obs
    for s in range(0, n_obs, C):
        e = min(s + C, n_obs)
        oi_chunk = prep.obs_info.slice(s, e)
        h_chunk = hkl_id[s:e]
        delta_map_chunk = delta_map[h_chunk]                              # (Nc,)
        sigma_chunk = sigma_laplace[h_chunk]                              # (Nc,)
        n_ph_chunk = n_ph[s:e]
        bg_chunk = bg[s:e]

        log_scale_chunk = scale.log_k(params, oi_chunk, obs_range=(s, e))  # (Nc,)
        mu_map = delta_map_chunk + log_scale_chunk                         # (Nc,)

        for i in range(K_coarse):
            mu_i = mu_map + offsets_coarse[i] * sigma_chunk                # (Nc,)
            log_p_sig = quadrature.marginalize_signal(
                scatter, likelihood, mu_i, n_ph_chunk, bg_chunk,
                params, oi_chunk, obs_range=(s, e),
            )                                                              # (Nc,)
            log_p = likelihood.combine_with_marginal(
                n_ph_chunk, bg_chunk, log_p_sig, params, oi_chunk
            )
            f_coarse[s:e, i] = log_p

    return f_coarse


@torch.no_grad()
def compute_posterior_moments_per_call(
    *,
    prep: PreparedObs,
    params: Parameters,
    scale: ScaleModel,
    scatter: ScatterModel,
    quadrature,
    likelihood: DetectionLikelihood,
    sigma_laplace: Tensor,
    priors: Mapping[str, PriorEvaluator],
    n_grid_coarse: int = 31,
    n_grid_fine: int = 101,
    n_sigma: float = 6.0,
    obs_chunk_size: int = 500_000,
    sigma_max: float = 2.0,
    map_fallback_log_I0: Tensor | None = None,
    summary: str = "linear_mean",
) -> dict[str, PosteriorMoments]:
    """Posterior moments for a quadrature whose grid is trial-μ-dependent.

    Same outer machinery as :func:`compute_posterior_moments` (coarse-fine
    Δ grid, prior evaluators, trapezoidal integration) but with the
    cached-backbone ``_fill_f_coarse`` replaced by per-offset
    ``marginalize_signal`` calls.  Use this with ``LaplaceSaddleGrid`` (or
    any Quadrature where node positions depend on log_mean).
    """
    device = params.log_I0.device
    dtype = params.log_I0.dtype
    n_valid = prep.n_valid

    delta_map = params.log_I0.detach()
    sigma = sigma_laplace.detach().clamp(min=1e-3, max=float(sigma_max))

    offsets_coarse, offsets_fine, i_lo, w_hi = _interp_weights(
        n_grid_coarse, n_grid_fine, n_sigma, device=device, dtype=dtype
    )

    f_coarse = _fill_f_coarse_per_call(
        prep=prep, params=params, scale=scale, scatter=scatter,
        quadrature=quadrature, likelihood=likelihood,
        delta_map=delta_map, sigma_laplace=sigma,
        offsets_coarse=offsets_coarse, chunk_size=obs_chunk_size,
    )

    L = _reduce_to_hkl_curves(f_coarse, prep.hkl_id, n_valid, i_lo, w_hi)

    delta_fine = delta_map[:, None] + offsets_fine[None, :] * sigma[:, None]
    ddelta = (2.0 * n_sigma) * sigma / (n_grid_fine - 1)

    results: dict[str, PosteriorMoments] = {}
    for name, prior_fn in priors.items():
        log_prior = prior_fn(delta_fine) if prior_fn is not None else torch.zeros_like(L)
        log_post = L + log_prior
        results[name] = _moments_from_log_post(
            delta_fine, log_post, ddelta,
            delta_map=delta_map,
            sigma_clamped=sigma,
            map_fallback_log_I0=map_fallback_log_I0,
            summary=summary,
        )
    return results


@torch.no_grad()
def compute_posterior_moments_global_grid(
    *,
    prep: PreparedObs,
    params: Parameters,
    scale: ScaleModel,
    scatter: ScatterModel,
    quadrature,
    likelihood: DetectionLikelihood,
    sigma_laplace: Tensor,
    priors: Mapping[str, PriorEvaluator],
    log_I_grid: Tensor,                # (K_post,) — fixed candidate log_I0 grid
    obs_chunk_size: int = 500_000,
    sigma_max: float = 2.0,            # only used for the mode_shift diagnostic
    map_fallback_log_I0: Tensor | None = None,
    summary: str = "linear_mean",
) -> dict[str, PosteriorMoments]:
    """Posterior moments on a fixed global log_I grid.

    Replaces the MAP-centered ``delta_map ± n_sigma·σ_laplace`` window
    used by :func:`compute_posterior_moments_per_call` with a single
    fixed grid (``log_I_grid``) shared across every HKL — no Laplace
    approximation in the posterior path, no per-HKL window adaptation.
    The quadrature decides the grid; the data decides the posterior
    shape.

    Cost: ``K_post × ⌈n_obs / chunk⌉`` ``marginalize_signal`` calls
    (vs the per-call version's ``K_coarse = 31``).  Use when σ_laplace
    is unreliable for weak HKLs (flat marginal NLL → curvature ≈ 0) or
    when you want the dlog_I integration on the same grid as the inner
    likelihood marginal — e.g. ``log_I_grid = quadrature.log_S_grid``
    for ``FixedLogGrid``.
    """
    device = params.log_I0.device
    dtype = params.log_I0.dtype
    n_valid = prep.n_valid
    n_obs = prep.n_obs_total
    K_post = log_I_grid.shape[0]

    log_I_grid = log_I_grid.to(device=device, dtype=dtype)
    delta_map = params.log_I0.detach()
    sigma_clamped = sigma_laplace.detach().clamp(min=1e-3, max=float(sigma_max))

    # Per-HKL log marginal data likelihood at every grid node.
    L = torch.zeros((n_valid, K_post), device=device, dtype=dtype)

    hkl_id = prep.hkl_id
    bg = prep.bg
    n_ph = (prep.I_obs + bg).clamp(min=0)

    C = obs_chunk_size if obs_chunk_size and obs_chunk_size > 0 else n_obs
    n_obs_nan = 0
    n_obs_posinf = 0
    for s in range(0, n_obs, C):
        e = min(s + C, n_obs)
        oi_chunk = prep.obs_info.slice(s, e)
        h_chunk = hkl_id[s:e]
        n_ph_chunk = n_ph[s:e]
        bg_chunk = bg[s:e]
        log_scale_chunk = scale.log_k(params, oi_chunk, obs_range=(s, e))   # (Nc,)

        for k in range(K_post):
            mu_k = log_I_grid[k] + log_scale_chunk                          # (Nc,)
            log_p_sig = quadrature.marginalize_signal(
                scatter, likelihood, mu_k, n_ph_chunk, bg_chunk,
                params, oi_chunk, obs_range=(s, e),
            )                                                                # (Nc,)
            log_p = likelihood.combine_with_marginal(
                n_ph_chunk, bg_chunk, log_p_sig, params, oi_chunk
            )                                                                # (Nc,)
            # Sanitise per-obs log_p before scatter_add: NaN or +inf would
            # contaminate the per-HKL sum and turn every downstream logsumexp
            # into NaN.  Replace with -inf (no support at that grid point).
            bad_lp = ~torch.isfinite(log_p) | (log_p > 0)
            if bad_lp.any():
                n_obs_nan += int(torch.isnan(log_p).sum().item())
                n_obs_posinf += int((log_p > 0).sum().item())
                log_p = torch.where(
                    bad_lp,
                    torch.full_like(log_p, float("-inf")),
                    log_p,
                )
            L[:, k].scatter_add_(0, h_chunk, log_p)

    if n_obs_nan or n_obs_posinf:
        print(
            f"  [posterior/global_grid] sanitised log_p: "
            f"{n_obs_nan} NaN, {n_obs_posinf} positive (out of "
            f"{n_obs * K_post} total chunked log_p evaluations)",
            flush=True,
        )

    # Posterior moments on the global grid (uniform step, identical for all HKLs).
    delta_fine = log_I_grid[None, :].expand(n_valid, -1).contiguous()        # (n_valid, K_post)
    step = float((log_I_grid[-1] - log_I_grid[0]) / (K_post - 1))
    ddelta = torch.full((n_valid,), step, device=device, dtype=dtype)

    results: dict[str, PosteriorMoments] = {}
    for name, prior_fn in priors.items():
        log_prior = prior_fn(delta_fine) if prior_fn is not None else torch.zeros_like(L)
        log_post = L + log_prior
        n_lp_nan = int(torch.isnan(log_post).sum().item())
        n_lp_posinf = int((log_post == float("inf")).sum().item())
        n_blank = int(((log_post == float("-inf")) | torch.isnan(log_post))
                      .all(dim=1).sum().item())
        if n_lp_nan or n_lp_posinf or n_blank:
            print(
                f"  [posterior/{name}] log_post: "
                f"{n_lp_nan} NaN, {n_lp_posinf} +inf entries; "
                f"{n_blank}/{n_valid} HKLs are entirely -inf/NaN",
                flush=True,
            )
        results[name] = _moments_from_log_post(
            delta_fine, log_post, ddelta,
            delta_map=delta_map,
            sigma_clamped=sigma_clamped,
            map_fallback_log_I0=map_fallback_log_I0,
            summary=summary,
        )
    return results


# ── Streamlined Posterior object ───────────────────────────────────────


def _fit_wilson_aniso_posthoc(
    prep: PreparedObs, log_I0_map: Tensor, *, n_steps: int = 80,
):
    """Fit (log_I_mean, U) on MAP log_I0 with log_I0 frozen.

    Used when ``prior='wilson_aniso'`` is requested but no active prior
    was registered during LBFGS.  Cheap: 7 params, no data targets — just
    the prior NLP at the converged MAP.
    """
    from dataclasses import replace
    from torchsx.merging.priors import WilsonAnisoPrior
    from torchsx.merging.config import MergeConfig

    cfg = MergeConfig()
    prep_seeded = replace(prep, mu_init=log_I0_map.detach().cpu())
    prior = WilsonAnisoPrior.from_prep(prep_seeded, cfg).to(log_I0_map.device)
    opt = torch.optim.LBFGS(
        [prior.log_I_mean, prior.U],
        max_iter=n_steps, line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=False)
        loss = prior(log_I0_map.detach()).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return prior


class Posterior:
    """Per-HKL posterior moments on a MAP-centered Δ-window per HKL.

    Build via :meth:`LogSpaceLoss.posterior` (or directly with
    :meth:`from_loss`).  Wraps the per-call MAP-centered integration path
    (:func:`compute_posterior_moments_per_call`): each HKL's Δ-grid spans
    ``delta_map ± n_sigma · sigma_laplace`` so the integration tracks the
    data peak — same approach the n513 reference fit used.  Returns a
    :class:`torchref.ReflectionData` from :meth:`integrate`.

    Both ``I/I_sigma`` and ``F/F_sigma`` are populated, so loading the
    written MTZ back into ``ReflectionData.load_mtz`` will not auto-trigger
    French–Wilson — F/σ_F come directly from posterior moments.
    """

    def __init__(
        self, loss, prior: str, moments: PosteriorMoments,
        wilson_aniso_prior=None,
    ) -> None:
        self._loss = loss
        self._prior = prior
        self._moments = moments
        self._wilson_aniso_prior = wilson_aniso_prior

    @classmethod
    def from_loss(
        cls,
        loss,
        prior: str = "flat",
        *,
        hessian_filter: str | None = None,
    ) -> "Posterior":
        """Build a Posterior from a refined LogSpaceLoss.

        ``hessian_filter`` selects which targets contribute to the Laplace
        σ that sets the integration grid extent.  Default (``None``) uses
        the full LBFGS objective — same Hessian as ``LossState.refine``.
        Pass ``"merging/data"`` to size the grid by the data-only
        likelihood curvature, so the posterior σ reflects what the data
        alone could constrain (the MAP still benefits from prior shrinkage
        but the reported uncertainty is no longer dominated by the
        prior's curvature at prior-driven HKLs).
        """
        device = loss._params.log_I0.device

        prior_fn: PriorEvaluator
        wp = None
        if prior == "flat":
            prior_fn = make_flat_prior()
        elif prior == "wilson":
            prep = loss._prep
            Sigma_shell = shell_mean_from_log_I0(
                loss._params.log_I0.detach(),
                prep.shell_id_fw.to(device),
                n_shells=int(prep.shell_id_fw.max().item()) + 1
                if prep.n_valid > 0 else 1,
                eps=1e-6,
            )
            prior_fn = make_wilson_prior(
                Sigma_shell=Sigma_shell,
                is_centric=prep.is_centric.to(device),
            )
        elif prior == "wilson_aniso":
            prep = loss._prep
            wp = getattr(loss, "_wilson_aniso_prior", None)
            if wp is None:
                wp = _fit_wilson_aniso_posthoc(prep, loss._params.log_I0)
            Sigma_h = wp.log_sigma().detach().exp().clamp(min=1e-6)
            prior_fn = make_wilson_prior(
                Sigma_shell=Sigma_h,                       # per-HKL anisotropic
                is_centric=prep.is_centric.to(device),
            )
        else:
            raise ValueError(
                f"Unknown prior {prior!r}; expected 'flat', 'wilson', or 'wilson_aniso'"
            )

        sigma_laplace = loss._hessian_diag_log_I0(
            targets_filter=hessian_filter,
        )  # (n_valid,)

        # MAP-centered Δ-grid: for each HKL the integration spans
        # ``delta_map ± n_sigma · sigma_laplace`` (clamped to sigma_max),
        # giving a tight log-space trapezoidal integral over the data
        # peak.  Inner signal marginal still uses FixedLogGrid (Triton).
        results = compute_posterior_moments_per_call(
            prep=loss._prep,
            params=loss._params,
            scale=loss._scale,
            scatter=loss._scatter,
            quadrature=loss._quadrature,
            likelihood=loss._likelihood,
            sigma_laplace=sigma_laplace,
            priors={prior: prior_fn},
            n_grid_coarse=31,
            n_grid_fine=101,
            n_sigma=6.0,
            obs_chunk_size=loss._obs_chunk_size,
            sigma_max=2.0,
            map_fallback_log_I0=loss._params.log_I0.detach(),
            summary="linear_mean",
        )
        return cls(loss, prior, results[prior], wilson_aniso_prior=wp)

    # ── moment accessors ──
    @property
    def I0(self) -> Tensor:
        return self._moments.I0

    @property
    def sigma_I0(self) -> Tensor:
        return self._moments.sigma_I0

    @property
    def F(self) -> Tensor:
        if self._moments.F is None:
            raise RuntimeError(
                "Posterior was built with a non-linear-mean summary; "
                "F/F_sigma are only available for summary='linear_mean'."
            )
        return self._moments.F

    @property
    def F_sigma(self) -> Tensor:
        if self._moments.F_sigma is None:
            raise RuntimeError(
                "Posterior was built with a non-linear-mean summary; "
                "F/F_sigma are only available for summary='linear_mean'."
            )
        return self._moments.F_sigma

    @property
    def prior(self) -> str:
        return self._prior

    @property
    def moments(self) -> PosteriorMoments:
        return self._moments

    @property
    def log_I_mean(self) -> float | None:
        """Fitted Wilson plot intercept (only meaningful for ``prior='wilson_aniso'``)."""
        if self._wilson_aniso_prior is None:
            return None
        return float(self._wilson_aniso_prior.log_I_mean.detach().item())

    @property
    def U(self) -> Tensor | None:
        """Fitted anisotropic decay 6-vector (only meaningful for ``prior='wilson_aniso'``)."""
        if self._wilson_aniso_prior is None:
            return None
        return self._wilson_aniso_prior.U.detach().clone()

    def integrate(self):
        """Build a :class:`torchref.ReflectionData` from this posterior.

        Both ``I/I_sigma`` (intensity moments) and ``F/F_sigma`` (square-root
        moments) are populated — no French–Wilson is needed downstream.
        """
        from torchref import ReflectionData
        from torchref.symmetry import Cell, SpaceGroup

        prep = self._loss._prep
        m = self._moments

        median_cell = prep.median_cell.detach().cpu().tolist()
        sg = self._loss._stream._core.spacegroup or "P 1"

        cell = Cell(median_cell)
        spacegroup = SpaceGroup(sg)

        # Skip torchref's resolution-binned R-free auto-generator; it
        # requires a populated `masks()` we do not pre-build here.
        # Downstream phenix uses an external r_free.mtz anyway, so the
        # column written here is informational only.
        rfree_flags = torch.zeros(prep.n_valid, dtype=torch.bool)

        # Final safety net: substitute MAP-square-root for any HKL whose
        # posterior moments came back non-finite or non-positive.  Tells
        # the user how many fell through so they can inspect upstream.
        F_cpu = m.F.detach().cpu().float()
        F_sigma_cpu = m.F_sigma.detach().cpu().float()
        I0_cpu = m.I0.detach().cpu().float()
        I0_sigma_cpu = m.sigma_I0.detach().cpu().float()

        log_I0_map = self._loss._params.log_I0.detach().cpu().float()
        F_map = (0.5 * log_I0_map).exp().clamp(min=1e-6)
        I0_map = log_I0_map.exp().clamp(min=1e-12)
        # σ_F ≈ ½ · I0 · σ_logI ≈ ½ · F · σ_logI; σ_I ≈ I0 · σ_logI.
        # Use a flat 0.1 σ_logI floor when no posterior σ is available.
        sigma_logI_floor = torch.full_like(F_map, 0.1)

        bad_F = ~torch.isfinite(F_cpu) | (F_cpu <= 0)
        # Also treat F_sigma == 0 as "bad": the linear-mean F_var = E[F²] − E[F]²
        # is subject to fp32 catastrophic cancellation when the posterior is
        # narrow (e.g. low-count Poisson at correct gain on Rhodopsin/tubulin
        # — see experiments/integration_importance/RESULTS.md).  A literal zero
        # makes phenix treat the reflection as infinitely precise.
        bad_Fs = ~torch.isfinite(F_sigma_cpu) | (F_sigma_cpu <= 0)
        bad_I = ~torch.isfinite(I0_cpu) | (I0_cpu <= 0)
        bad_Is = ~torch.isfinite(I0_sigma_cpu) | (I0_sigma_cpu < 0)
        n_bad = int((bad_F | bad_Fs | bad_I | bad_Is).sum().item())
        if n_bad:
            print(
                f"  [posterior/{self._prior}] {n_bad}/{prep.n_valid} HKLs "
                "with non-finite/non-positive posterior moments; "
                "substituting MAP-square-root.",
                flush=True,
            )
        F_cpu = torch.where(bad_F, F_map, F_cpu)
        I0_cpu = torch.where(bad_I, I0_map, I0_cpu)
        I0_sigma_cpu = torch.where(bad_Is, I0_map * sigma_logI_floor, I0_sigma_cpu)
        # Prefer the delta-method fallback (F_σ ≈ σ_I / 2F) over the flat
        # σ_logI floor, since I0_sigma_cpu has already been clamped to its
        # own sub-grid floor and carries the actual posterior width.
        F_fallback = 0.5 * I0_sigma_cpu / F_cpu.clamp(min=1e-6)
        F_sigma_cpu = torch.where(bad_Fs, F_fallback, F_sigma_cpu)

        data = ReflectionData.from_tensors(
            hkl=prep.hkl_valid.detach().cpu().int(),
            F=F_cpu,
            F_sigma=F_sigma_cpu,
            cell=cell,
            spacegroup=spacegroup,
            rfree_flags=rfree_flags,
        )
        data.I = I0_cpu
        data.I_sigma = I0_sigma_cpu
        return data


def build_map_only_data(loss, *, hessian_filter: str = "merging/data"):
    """Write the MTZ directly from MAP I0 + Laplace σ — skip integration.

    I0      = exp(log_I0_MAP)                       — MAP estimate
    σ_log_I = loss._hessian_diag_log_I0(filter)     — Laplace σ in log-space
    σ_I0    = σ_log_I · I0                          — delta method
    F       = sqrt(I0) = exp(0.5 · log_I0_MAP)
    σ_F     = σ_I0 / (2·F) = ½ · σ_log_I · F

    Default ``hessian_filter='merging/data'`` gives σ from the likelihood
    curvature alone (no prior shrinkage).  The MAP estimate itself is the
    loss argmin, so it still reflects whatever priors are in the loss.
    """
    from torchref import ReflectionData
    from torchref.symmetry import Cell, SpaceGroup

    prep = loss._prep
    median_cell = prep.median_cell.detach().cpu().tolist()
    sg = loss._stream._core.spacegroup or "P 1"
    cell = Cell(median_cell)
    spacegroup = SpaceGroup(sg)

    log_I0 = loss._params.log_I0.detach()
    sigma_log_I0 = loss._hessian_diag_log_I0(targets_filter=hessian_filter)

    I0 = log_I0.exp()
    sigma_I0 = (sigma_log_I0 * I0).clamp(min=0.0)
    F = (0.5 * log_I0).exp().clamp(min=1e-6)
    sigma_F = 0.5 * sigma_log_I0 * F

    rfree_flags = torch.zeros(prep.n_valid, dtype=torch.bool)
    data = ReflectionData.from_tensors(
        hkl=prep.hkl_valid.detach().cpu().int(),
        F=F.detach().cpu().float(),
        F_sigma=sigma_F.detach().cpu().float(),
        cell=cell,
        spacegroup=spacegroup,
        rfree_flags=rfree_flags,
    )
    data.I = I0.detach().cpu().float()
    data.I_sigma = sigma_I0.detach().cpu().float()
    return data
