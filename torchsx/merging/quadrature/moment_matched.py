"""Moment-matched Gaussian "quadrature" — no integration.

The exact GG marginal ``∫ Poisson(n; K·I·S + bg) · p(S) dS`` has an MLE
for I that's biased on weak HKLs (see
``experiments/synthetic_gengauss/compare_linear_vs_gengauss.py``).
The bias is structural in the integral; reparameterizing the prior
(e.g. recentering so E[S]=1) doesn't fix it.

The fix: replace the exact marginal with its first-two-moments
Gaussian approximation:

    n ~ Normal(μ, σ²)
      μ  = K·I·E[S] + bg
      σ² = K·I·E[S] + bg + (K·I)² · Var(S)
         = μ + (K·I)² · Var(S)

The MLE for I under this Gaussian solves
    Σ K·E[S]·(n − μ)/σ² = 0
which is asymptotically equivalent to the linear estimator
``mean((n − bg)/K) / E[S]`` — unbiased on every HKL regardless of SNR.

α, β still enter through E[S(α,β)] and Var(S(α,β)) so they remain
identifiable from the data variance.

This module bypasses the latent-S grid entirely; it just evaluates a
Gaussian log-PMF per observation.  The ``scatter`` and ``likelihood``
arguments to ``marginalize_signal`` are accepted for API compatibility
but ignored.

Prototype: E[S] and Var(S) are computed once at init time from a
reference (α_ref, β_ref).  The natural extension is per-HKL
``E[S(α_h, β_h)]`` via a fixed-grid integral (differentiable).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.scatter.base import ScatterModel


def compute_s_moments(alpha: float, beta: float, n_grid: int = 200_001) -> tuple[float, float]:
    """Returns (E[S], Var(S)) for S = exp(ε), ε ~ GenGauss(0, α, β)."""
    import numpy as np
    xs = np.linspace(-15.0, 15.0, n_grid)
    dx = float(xs[1] - xs[0])
    norm = beta / (2.0 * alpha * math.gamma(1.0 / beta))
    p = norm * np.exp(-(np.abs(xs) / alpha) ** beta)
    p = p / (p.sum() * dx)
    e_s = float((np.exp(xs) * p).sum() * dx)
    e_s2 = float((np.exp(2 * xs) * p).sum() * dx)
    var_s = max(e_s2 - e_s ** 2, 0.0)
    return e_s, var_s


class MomentMatchedQuadrature(Quadrature):
    """Gaussian likelihood with mean+variance from compound Poisson-GG.

    Returns log ``Normal(n; μ, σ²)`` per observation, where μ and σ² are
    the exact first two moments of the compound Poisson-GenGauss
    distribution.  No latent-signal integration is performed.

    Parameters
    ----------
    alpha_ref, beta_ref : reference (α, β) for the global E[S] and
        Var(S).  On synthetic data the truth values; in production the
        median of the converged GG-MLE α, β.
    """

    def __init__(self, alpha_ref: float = 1.0, beta_ref: float = 1.7) -> None:
        super().__init__()
        e_s, var_s = compute_s_moments(alpha_ref, beta_ref)
        self.register_buffer("e_s", torch.tensor(e_s))
        self.register_buffer("var_s", torch.tensor(var_s))
        self.alpha_ref = float(alpha_ref)
        self.beta_ref = float(beta_ref)

    @classmethod
    def from_prep(cls, prep: PreparedObs, config) -> "MomentMatchedQuadrature":
        # Defaults to truth values for synthetic; override at fit time.
        return cls(alpha_ref=1.0, beta_ref=1.7)

    # Forward quadrature-API methods so the posterior-moments pipeline
    # (which still uses the trapezoidal grid) keeps working alongside.
    def _sig_lo_slice(self, *args, **kwargs):
        raise NotImplementedError(
            "MomentMatchedQuadrature is not a grid quadrature; "
            "compute_posterior_moments() is incompatible. "
            "Use the I0_map column directly."
        )

    def _build_grid(self, *args, **kwargs):
        raise NotImplementedError("Same as _sig_lo_slice — no grid here.")

    def marginalize_signal(
        self,
        scatter: ScatterModel,            # unused
        likelihood: DetectionLikelihood,   # unused
        log_mean: Tensor,                  # log(K · I)  (n_obs,)
        n_ph: Tensor,                      # photon counts  (n_obs,)
        bg: Tensor,                        # background photons  (n_obs,)
        params: Parameters,                # unused
        meta: ObsInfo,                     # unused
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        self._reject_unmodelled_bg(likelihood, variance=True)
        del scatter, likelihood, params, meta, obs_range
        K_I = log_mean.exp()                               # (n_obs,)  K · I
        mu = K_I * self.e_s + bg                            # mean of n
        var_compound = K_I * K_I * self.var_s               # Var(K·I·S)
        var_total = mu + var_compound                       # E[Var(n|S)]+Var(E[n|S])
        # Floor for numerical safety (very weak HKLs can give tiny var).
        var_total = var_total.clamp(min=1.0)
        # log Normal PDF
        diff = n_ph - mu
        log_p = -0.5 * (diff * diff / var_total + torch.log(2.0 * math.pi * var_total))
        return log_p
