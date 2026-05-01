"""Hierarchical importance-sampling quadrature.

Per-observation Population-Monte-Carlo / iterative-importance-sampling
marginaliser.  Operates entirely in ``log_S`` coordinates.

Algorithm (per obs, vectorised over the M-axis):

1. Iter 0 — draw ``n_per_iter`` samples from ``GenGauss(log_mean, α_h, β_h)``
   (the prior itself).  Compute ``log q_0`` via ``GenGaussLogScatter.log_prob``.
2. Iters k = 1..L-1 — compute weighted mean μ_k and std σ_k of all
   accumulated samples (weights = softmax of importance log-weights),
   clamp σ_k ∈ [sigma_min_frac·α_h, sigma_max_frac·α_h], draw
   ``n_per_iter`` from ``Normal(μ_k, σ_k)``.
3. For each iter k, the unbiased per-batch IS estimator is
   ``log_Z_k = logsumexp(log_prior + log_lik − log_q_k) − log(n_per_iter)``.
4. Final marginal: ``log_Z = logsumexp(log_Z_0..log_Z_{L-1}) − log(L)``.

Sample positions, proposal mean/std, and proposal log-densities live in
``torch.no_grad()`` (matches ``LaplaceSaddleGrid.detach_grid=True``).
Gradients flow only through ``scatter.log_prob`` and
``likelihood.log_prob_given_signal`` evaluated at the (detached) samples.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.trapezoidal import (
    TrapezoidalLogGrid, _compute_sig_lo_per_obs,
)
from torchsx.merging.scatter.base import ScatterModel


# Hard cap on |log_S| nodes — exp(50) ≈ 5e21 stays well below FP64 overflow.
_LOG_S_ABS_MAX = 50.0
_LOG_2PI = math.log(2.0 * math.pi)


def _gengauss_sample(
    log_mean: Tensor,    # (M,)
    alpha: Tensor,       # (M,)
    beta: Tensor,        # (M,)
    n_samples: int,
) -> Tensor:             # (M, n_samples)  log_S samples
    """Sample log_S from GenGauss(log_mean, α, β), per-obs vectorised.

    Uses the standard recipe: |eps/α|^β ~ Gamma(1/β, 1) with random sign.
    """
    M = log_mean.shape[0]
    shape = (M, n_samples)
    inv_beta = (1.0 / beta).unsqueeze(1).expand(shape).contiguous()
    rate = torch.ones_like(inv_beta)
    Y = torch.distributions.Gamma(inv_beta, rate).sample()           # (M, N)
    pow_Y = Y.pow(1.0 / beta.unsqueeze(1))                            # |eps/α|
    sign = torch.where(
        torch.rand(shape, device=log_mean.device, dtype=log_mean.dtype) < 0.5,
        torch.full(shape, -1.0, device=log_mean.device, dtype=log_mean.dtype),
        torch.full(shape, 1.0, device=log_mean.device, dtype=log_mean.dtype),
    )
    eps = alpha.unsqueeze(1) * pow_Y * sign                           # (M, N)
    return log_mean.unsqueeze(1) + eps


def _gauss_log_prob(s: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """log Normal(s; μ, σ) — broadcasts; output shape = broadcast of inputs."""
    z = (s - mu) / sigma
    return -0.5 * z * z - sigma.log() - 0.5 * _LOG_2PI


class HierarchicalImportanceQuadrature(Quadrature):
    """Iterative IS marginaliser of ``log ∫ p(n|S,bg)·p_GG(log S − log_mean) dlog S``.

    Parameters
    ----------
    n_iterations
        Number of proposal iterations L (default 10).  Iter 0 is the
        prior-anchored GenGauss seed; iters 1..L-1 are adaptive Gaussian.
    n_per_iter
        Samples drawn per iteration N (default 11).  Total pool = L·N.
    sigma_min_frac, sigma_max_frac
        Adaptive Gaussian σ clamped to [this·α_h, this·α_h] per obs.
        Defaults (0.1, 2.0) prevent collapse / blow-up.
    seed_base
        If not None, ``torch.manual_seed(seed_base)`` is set at construction
        for reproducibility.  Each forward pass draws fresh samples (no
        per-call seed pinning).
    log_S_abs_max
        Hard clamp on |log_S| samples to keep exp() well below FP overflow.
    detach_grid
        If True (default), sample positions and proposal moments are
        detached from autograd; gradients flow only through the integrand
        evaluation.  Always-on for v1.
    """

    def __init__(
        self,
        n_iterations: int = 10,
        n_per_iter: int = 11,
        sigma_min_frac: float = 0.1,
        sigma_max_frac: float = 2.0,
        seed_base: int | None = None,
        log_S_abs_max: float = _LOG_S_ABS_MAX,
        detach_grid: bool = True,
        sig_lo_per_obs: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_iterations = int(n_iterations)
        self.n_per_iter = int(n_per_iter)
        self.sigma_min_frac = float(sigma_min_frac)
        self.sigma_max_frac = float(sigma_max_frac)
        self.log_S_abs_max = float(log_S_abs_max)
        self.detach_grid = bool(detach_grid)
        if seed_base is not None:
            torch.manual_seed(int(seed_base))

        # Inner softplus S-grid used solely by ``compute_posterior_moments``,
        # which calls ``quadrature._sig_lo_slice`` and ``_build_grid`` to set
        # up the per-obs μ-coarse scan.  Marginalisation itself uses the IS
        # sampler — this grid never feeds into ``marginalize_signal``.
        self.s_grid = TrapezoidalLogGrid(sig_lo_per_obs=sig_lo_per_obs)

        # Diagnostic — populated each forward pass.  (M,) tensor of ESS.
        self._last_ess: Tensor | None = None

    @classmethod
    def from_prep(
        cls, prep: PreparedObs, config: MergeConfig
    ) -> "HierarchicalImportanceQuadrature":
        sig_lo = _compute_sig_lo_per_obs(
            n_valid=prep.n_valid,
            hkl_id=prep.hkl_id,
            bg=prep.bg,
            obs_counts=prep.obs_counts,
            override=config.sig_lo_override,
        )
        return cls(
            n_iterations=getattr(config, "hier_n_iterations", 10),
            n_per_iter=getattr(config, "hier_n_per_iter", 11),
            sigma_min_frac=getattr(config, "hier_sigma_min_frac", 0.1),
            sigma_max_frac=getattr(config, "hier_sigma_max_frac", 2.0),
            seed_base=getattr(config, "hier_seed_base", None),
            log_S_abs_max=getattr(config, "hier_log_S_abs_max", _LOG_S_ABS_MAX),
            detach_grid=getattr(config, "hier_detach_grid", True),
            sig_lo_per_obs=sig_lo,
        )

    # Posterior moment code (torchsx.merging.posterior) calls these on the
    # quadrature; forward to the inner softplus grid (HybridLogGrid pattern).
    def _sig_lo_slice(self, n_ph, bg, obs_range):
        return self.s_grid._sig_lo_slice(n_ph, bg, obs_range)

    def _build_grid(self, n_ph, bg, sig_lo_per_obs):
        return self.s_grid._build_grid(n_ph, bg, sig_lo_per_obs)

    # ── Quadrature interface ───────────────────────────────────────────
    def marginalize_signal(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,        # (M,)
        n_ph: Tensor,            # (M,)
        bg: Tensor,              # (M,)
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:                  # (M,)
        device = log_mean.device
        dtype = log_mean.dtype
        M = log_mean.shape[0]
        L = self.n_iterations
        N = self.n_per_iter

        # Per-obs α_h, β_h via meta.hkl_idx (matches LaplaceSaddleGrid).
        h = meta.hkl_idx
        alpha = params.log_alpha[h].exp().clamp(min=1e-8)
        beta = params.log_beta[h].exp().clamp(min=0.5, max=4.0)

        log_mean_d = log_mean.detach() if self.detach_grid else log_mean
        alpha_d = alpha.detach() if self.detach_grid else alpha
        beta_d = beta.detach() if self.detach_grid else beta

        # Pool buffers — sample positions, log_q at own draws, IS log-weights.
        samples_pool = torch.empty((M, L * N), device=device, dtype=dtype)
        log_q_pool = torch.empty((M, L * N), device=device, dtype=dtype)
        log_w_pool = torch.empty((M, L * N), device=device, dtype=dtype)
        log_Z_per_iter = torch.empty((M, L), device=device, dtype=dtype)

        sigma_min = self.sigma_min_frac * alpha_d                    # (M,)
        sigma_max = self.sigma_max_frac * alpha_d                    # (M,)
        max_clamp = self.log_S_abs_max

        for k in range(L):
            # ── Draw iter-k samples and evaluate log q_k at them ───────
            ctx = torch.no_grad() if self.detach_grid else _NullCtx()
            with ctx:
                if k == 0:
                    sk = _gengauss_sample(log_mean_d, alpha_d, beta_d, N)
                    sk = sk.clamp(min=-max_clamp, max=max_clamp)
                    # log q_0 = log_prior at sample (proposal IS the prior).
                    # Evaluated inside no_grad — values used only as a fixed
                    # offset in the IS weight; the downstream integrand runs
                    # scatter.log_prob again outside no_grad with live params.
                    eps_k_d = sk - log_mean_d[:, None]
                    log_q_k = scatter.log_prob(eps_k_d, params, meta)
                else:
                    # Weighted moments of the accumulated pool (iters 0..k-1).
                    pool_end = k * N
                    s_pool = samples_pool[:, :pool_end]              # (M, k·N)
                    lw_pool = log_w_pool[:, :pool_end]               # (M, k·N)
                    lw_max = lw_pool.max(dim=1, keepdim=True).values
                    w = torch.exp(lw_pool - lw_max)                  # unnormalised
                    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-30)
                    w_norm = w / w_sum                               # (M, k·N)
                    mu_k = (w_norm * s_pool).sum(dim=1)              # (M,)
                    var_k = (w_norm * (s_pool - mu_k[:, None]) ** 2).sum(dim=1)
                    sigma_k = var_k.clamp(min=1e-12).sqrt()          # (M,)
                    # Degenerate fallback: if std collapsed, use prior width α.
                    sigma_k = torch.where(
                        sigma_k < 1e-6, alpha_d, sigma_k,
                    )
                    sigma_k = sigma_k.clamp(min=sigma_min, max=sigma_max)
                    # Sample from N(μ_k, σ_k) per obs.
                    z = torch.randn((M, N), device=device, dtype=dtype)
                    sk = mu_k[:, None] + sigma_k[:, None] * z         # (M, N)
                    sk = sk.clamp(min=-max_clamp, max=max_clamp)
                    log_q_k = _gauss_log_prob(
                        sk, mu_k[:, None], sigma_k[:, None],
                    )                                                 # (M, N)

            # ── Integrand at iter-k samples (gradients flow here) ────────
            # eps uses live log_mean so gradients propagate even though sk is detached.
            eps_k = sk - log_mean[:, None]                           # (M, N)
            S_k = sk.exp()                                            # (M, N) — sk detached so no grad through exp
            log_prior_k = scatter.log_prob(eps_k, params, meta)      # (M, N)
            log_lik_k = likelihood.log_prob_given_signal(n_ph, S_k, bg)  # (M, N)
            log_w_k = log_prior_k + log_lik_k - log_q_k              # (M, N)

            # Per-iter IS estimator (unbiased given q_k).
            log_Z_per_iter[:, k] = (
                torch.logsumexp(log_w_k, dim=1) - math.log(N)
            )

            # Update pool (detached — for proposal moments only).
            samples_pool[:, k * N:(k + 1) * N] = sk.detach()
            log_q_pool[:, k * N:(k + 1) * N] = log_q_k.detach()
            log_w_pool[:, k * N:(k + 1) * N] = log_w_k.detach()

        # ── Final combiner: average of L unbiased per-iter IS estimates ─
        log_Z = torch.logsumexp(log_Z_per_iter, dim=1) - math.log(L)

        # ── ESS diagnostic over the full pool (M,) ──────────────────────
        with torch.no_grad():
            lse1 = torch.logsumexp(log_w_pool, dim=1)
            lse2 = torch.logsumexp(2.0 * log_w_pool, dim=1)
            self._last_ess = torch.exp(2.0 * lse1 - lse2).detach()

        return log_Z


class _NullCtx:
    """No-op context manager — replaces ``torch.no_grad()`` when detach_grid=False."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
