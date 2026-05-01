"""Brute-force uniform log_S grid: ±n_sigma·α centered on log_mean.

The simplest possible reference quadrature — no adaptation, no
data-driven points, no bias-shaping tricks.  Per-obs grid:

    log_S = log_mean + α_h · u,    u ∈ [-n_sigma, +n_sigma], uniform

with trapezoidal weights ``α_h · Δu`` (half on the endpoints).  At the
default ``n_sigma=12, n_points=10001`` (Δu ≈ 2.4e-3) the discretisation
error is far below MC noise; this is what the integrand wants the
quadrature to look like, used as a gold-standard reference to test
whether the "weak-HKL upward bias" is algorithmic or genuinely the
data MLE.
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


_LOG_S_ABS_MAX = 50.0


class WideLogGrid(Quadrature):
    """Uniform log_S grid spanning ``log_mean ± n_sigma·α`` per obs.

    The grid follows ``log_mean`` (it is per-call, not fixed), so the
    posterior-moment µ-coarse scan must use the same wide grid at each
    candidate µ.  ``requires_per_call_posterior`` routes the merger to
    ``compute_posterior_moments_per_call`` — which calls
    ``marginalize_signal`` per offset, keeping MAP and posterior on the
    same grid (otherwise the µ-scan would silently fall back to the
    softplus inner grid via ``_sig_lo_slice``/``_build_grid`` and the
    posterior MAP would inflate, see fix24/synthetic_gengauss notes).

    Parameters
    ----------
    n_sigma
        Half-width of the grid in α units (default 12 → covers GG tails
        far past where the integrand has any mass).
    n_points
        Number of nodes (default 10001 → Δu ≈ 2.4e-3 at n_sigma=12).
    sig_lo_per_obs
        Optional precomputed buffer; only needed by the posterior moment
        code (forwarded to an inner ``TrapezoidalLogGrid``).
    """

    requires_per_call_posterior: bool = True

    def __init__(
        self,
        n_sigma: float = 12.0,
        n_points: int = 10001,
        sig_lo_per_obs: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_sigma = float(n_sigma)
        self.n_points = int(n_points)
        u = torch.linspace(-self.n_sigma, self.n_sigma, self.n_points)
        self.register_buffer("u", u)
        # Inner softplus grid used solely by ``compute_posterior_moments`` —
        # never touched by ``marginalize_signal``.
        self.s_grid = TrapezoidalLogGrid(sig_lo_per_obs=sig_lo_per_obs)

    @classmethod
    def from_prep(
        cls, prep: PreparedObs, config: MergeConfig,
    ) -> "WideLogGrid":
        sig_lo = _compute_sig_lo_per_obs(
            n_valid=prep.n_valid, hkl_id=prep.hkl_id, bg=prep.bg,
            obs_counts=prep.obs_counts, override=config.sig_lo_override,
        )
        return cls(
            n_sigma=getattr(config, "wide_n_sigma", 12.0),
            n_points=getattr(config, "wide_n_points", 10001),
            sig_lo_per_obs=sig_lo,
        )

    def _sig_lo_slice(self, n_ph, bg, obs_range):
        return self.s_grid._sig_lo_slice(n_ph, bg, obs_range)

    def _build_grid(self, n_ph, bg, sig_lo_per_obs):
        return self.s_grid._build_grid(n_ph, bg, sig_lo_per_obs)

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
        h = meta.hkl_idx
        log_alpha = params.log_alpha[h].clamp(min=-6)  # (M,)
        alpha = params.log_alpha[h].exp()        # (M,)

        # log_S = log_mean + α · u   →  (M, K)
        log_S = log_mean[:, None] + alpha[:, None] * self.u[None, :]
        log_S = log_S.clamp(min=-_LOG_S_ABS_MAX, max=_LOG_S_ABS_MAX)
        eps = log_S - log_mean[:, None]
        S = log_S.exp()

        log_scatter = scatter.log_prob(eps, params, meta)             # (M, K)
        log_lik = likelihood.log_prob_given_signal(n_ph, S, bg)       # (M, K)

        # Trapezoidal weights in log_S: dlog_S = α · du, with α per-obs.
        # Uniform u with Δu = (2·n_sigma)/(K-1).
        du = (2.0 * self.n_sigma) / (self.n_points - 1)
        log_w_int = log_alpha + math.log(du)                   # (M,) interior
        log_w_end = log_w_int - math.log(2.0)                         # (M,) endpoints
        log_w = log_w_int[:, None].expand(-1, self.n_points).contiguous()
        log_w[:, 0] = log_w_end
        log_w[:, -1] = log_w_end

        return torch.logsumexp(log_scatter + log_lik + log_w, dim=1)
