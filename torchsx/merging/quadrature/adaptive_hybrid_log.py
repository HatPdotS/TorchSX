"""Adaptive hybrid quadrature for GenGauss log-scatter.

This variant keeps the observation-driven softplus S-grid from
``TrapezoidalLogGrid`` and unions it with a model-driven log-S grid centered
on the current ``log_mean``.  Unlike ``HybridLogGrid``, the added log grid is
not tied to a fixed absolute ``log S`` support, so small trial intensities do
not create an artificial lower-bound mode.
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
    TrapezoidalLogGrid,
    _compute_sig_lo_per_obs,
)
from torchsx.merging.scatter.base import ScatterModel


class AdaptiveHybridLogGrid(Quadrature):
    """Union of observation-driven S nodes and model-centered log-S nodes.

    The adaptive branch uses

        ``log_S = log_mean + alpha_h * u``

    with ``u`` spanning a conservative GenGauss tail range.  Grid construction
    is detached by default: the grid follows the current parameters, but
    gradients flow through the integrand rather than through the moving
    quadrature nodes.
    """

    def __init__(
        self,
        n_quad_S: int = 51,
        n_quad_adaptive: int = 51,
        tail_tol: float = 1e-8,
        u_max_cap: float = 30.0,
        n_sigma: float = 5.0,
        signal_min: float = 0.01,
        sig_lo_per_obs: Tensor | None = None,
        detach_grid: bool = True,
    ) -> None:
        super().__init__()
        self.s_grid = TrapezoidalLogGrid(
            n_quad=n_quad_S,
            n_sigma=n_sigma,
            signal_min=signal_min,
            sig_lo_per_obs=sig_lo_per_obs,
        )
        self.n_quad_adaptive = int(n_quad_adaptive)
        self.tail_tol = float(tail_tol)
        self.u_max_cap = float(u_max_cap)
        self.detach_grid = bool(detach_grid)
        self.register_buffer(
            "u_unit", torch.linspace(-1.0, 1.0, self.n_quad_adaptive)
        )

    @classmethod
    def from_prep(
        cls, prep: PreparedObs, config: MergeConfig
    ) -> "AdaptiveHybridLogGrid":
        sig_lo = _compute_sig_lo_per_obs(
            n_valid=prep.n_valid,
            hkl_id=prep.hkl_id,
            bg=prep.bg,
            obs_counts=prep.obs_counts,
            override=config.sig_lo_override,
        )
        return cls(
            n_quad_S=config.n_quad,
            n_quad_adaptive=getattr(config, "n_quad_adaptive", 51),
            tail_tol=getattr(config, "adaptive_tail_tol", 1e-8),
            u_max_cap=getattr(config, "adaptive_u_max_cap", 30.0),
            n_sigma=config.n_sigma_quad,
            signal_min=0.01,
            sig_lo_per_obs=sig_lo,
            detach_grid=getattr(config, "adaptive_detach_grid", True),
        )

    def _sig_lo_slice(self, n_ph, bg, obs_range):
        return self.s_grid._sig_lo_slice(n_ph, bg, obs_range)

    def _build_grid(self, n_ph, bg, sig_lo_per_obs):
        return self.s_grid._build_grid(n_ph, bg, sig_lo_per_obs)

    def _adaptive_log_grid(
        self, log_mean: Tensor, params: Parameters, meta: ObsInfo
    ) -> Tensor:
        h = meta.hkl_idx
        alpha = params.alpha[h].clamp(min=1e-8)
        beta = params.beta[h].clamp(min=1e-6)

        # For exp(-|u|^beta), this conservative envelope makes the pointwise
        # tail density <= tail_tol before normalization.  It is intentionally
        # simple and monotone; the cap protects very small beta during fitting.
        tail = max(self.tail_tol, 1e-300)
        u_max = torch.exp(torch.log(torch.full_like(beta, -math.log(tail))) / beta)
        u_max = u_max.clamp(min=3.0, max=self.u_max_cap)

        log_mean_g = log_mean
        alpha_g = alpha
        u_max_g = u_max
        if self.detach_grid:
            log_mean_g = log_mean_g.detach()
            alpha_g = alpha_g.detach()
            u_max_g = u_max_g.detach()

        u = self.u_unit.to(dtype=log_mean.dtype, device=log_mean.device)
        return log_mean_g[:, None] + alpha_g[:, None] * u_max_g[:, None] * u[None, :]

    def marginalize_signal(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,
        n_ph: Tensor,
        bg: Tensor,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        sig_lo = self.s_grid._sig_lo_slice(n_ph, bg, obs_range)
        signal_S, _ = self.s_grid._build_grid(n_ph, bg, sig_lo)
        log_S_S = torch.log(signal_S.clamp(min=1e-12))

        log_S_A = self._adaptive_log_grid(log_mean, params, meta)
        log_S_all = torch.cat([log_S_S, log_S_A], dim=1)
        log_S_sorted, _ = log_S_all.sort(dim=1)

        diffs = log_S_sorted[:, 1:] - log_S_sorted[:, :-1]
        w = torch.zeros_like(log_S_sorted)
        w[:, 0] = diffs[:, 0] * 0.5
        w[:, -1] = diffs[:, -1] * 0.5
        w[:, 1:-1] = (diffs[:, :-1] + diffs[:, 1:]) * 0.5
        log_w = torch.log(w.clamp(min=1e-30))

        S = torch.exp(log_S_sorted)
        eps = log_S_sorted - log_mean[:, None]
        log_scatter = scatter.log_prob(eps, params, meta)
        log_lik = likelihood.log_prob_given_signal(n_ph, S, bg)

        return torch.logsumexp(log_scatter + log_lik + log_w, dim=1)
