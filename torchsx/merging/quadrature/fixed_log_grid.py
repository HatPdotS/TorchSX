"""Fixed-range uniform log_S grid: no centering on log_mean, n, or bg.

Both ``TrapezoidalLogGrid`` (centered on the Poisson MLE ``n - bg``) and
``WideLogGrid`` (centered on the gengauss prior mode ``log_mean``) bake
in an assumption about where the integrand has mass.  This quadrature
makes no such assumption — the grid is the same tensor for every
observation and every µ-coarse posterior offset:

    log_S ∈ [log(S_min), log(S_max)],  uniform, K nodes

with trapezoidal weights ``Δlog_S`` (half on the endpoints).  At the
default ``S_min=1e-3, S_max=1e6, n_points=10001`` the spacing is
``Δlog_S ≈ 2.07e-3`` over a 9-decade window — wide enough that the
likelihood × scatter integrand is numerically zero at both ends for any
realistic ``log_mean``, and dense enough that the discretisation error
sits below MC noise.  Used as a reference to test whether the
centering choice in the other grids biases the merged intensities.
"""
from __future__ import annotations

import math
import os

import torch
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.likelihood.poisson import Poisson
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.trapezoidal import (
    TrapezoidalLogGrid, _compute_sig_lo_per_obs,
)
from torchsx.merging.scatter.base import ScatterModel
from torchsx.merging.scatter.gengauss import GenGaussLogScatter

# Triton fast path is optional — falls back to eager when unavailable.
try:
    from torchsx.merging.quadrature._fixed_log_grid_triton import (
        HAVE_TRITON, marginalize_signal_triton,
    )
except ImportError:
    HAVE_TRITON = False
    marginalize_signal_triton = None  # type: ignore[assignment]


# Set ``TORCHSX_FIXEDLOGGRID_NO_TRITON=1`` to force the eager fallback
# (e.g. for debugging numerical drift between paths).
_DISABLE_TRITON = os.environ.get("TORCHSX_FIXEDLOGGRID_NO_TRITON", "") == "1"


class FixedLogGrid(Quadrature):
    """Uniform log_S grid in absolute coordinates ``[log(S_min), log(S_max)]``.

    The grid is independent of ``log_mean``, ``n_ph``, ``bg``, and any
    iterate — same K nodes for every observation, every µ-coarse offset.
    ``requires_per_call_posterior = True`` so the merger routes posterior
    moment computation through ``compute_posterior_moments_per_call``,
    which keeps the MAP marginal and every posterior offset on this same
    fixed grid (no fallback to the softplus inner grid).

    Parameters
    ----------
    S_min, S_max
        Linear-S range of the integration window (default ``1e-3`` to
        ``1e6`` — 9 decades, ample for BR-scale data).
    n_points
        Number of nodes (default 10001 → ``Δlog_S ≈ 2.07e-3``).
    sig_lo_per_obs
        Optional precomputed buffer; only forwarded to an inner
        ``TrapezoidalLogGrid`` so legacy ``_sig_lo_slice`` /
        ``_build_grid`` callers keep working.  Never touched by
        ``marginalize_signal``.
    """

    requires_per_call_posterior: bool = True

    def __init__(
        self,
        S_min: float = 1e-3,
        S_max: float = 1e6,
        n_points: int = 10001,
        sig_lo_per_obs: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not (S_min > 0.0 and S_max > S_min):
            raise ValueError(f"need 0 < S_min < S_max, got {S_min=} {S_max=}")
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.n_points = int(n_points)
        log_S = torch.linspace(
            math.log(self.S_min), math.log(self.S_max), self.n_points,
        )
        self.register_buffer("log_S_grid", log_S)
        self.register_buffer("S_grid", log_S.exp())
        self.dlog_S = float(
            (math.log(self.S_max) - math.log(self.S_min)) / (self.n_points - 1)
        )
        self.s_grid = TrapezoidalLogGrid(sig_lo_per_obs=sig_lo_per_obs)

    @classmethod
    def from_prep(
        cls, prep: PreparedObs, config: MergeConfig,
    ) -> "FixedLogGrid":
        sig_lo = _compute_sig_lo_per_obs(
            n_valid=prep.n_valid, hkl_id=prep.hkl_id, bg=prep.bg,
            obs_counts=prep.obs_counts, override=config.sig_lo_override,
        )
        return cls(
            S_min=getattr(config, "fixed_log_S_min", 1e-3),
            S_max=getattr(config, "fixed_log_S_max", 1e6),
            n_points=getattr(config, "fixed_log_n_points", 10001),
            sig_lo_per_obs=sig_lo,
        )

    def _sig_lo_slice(self, n_ph, bg, obs_range):
        return self.s_grid._sig_lo_slice(n_ph, bg, obs_range)

    def _build_grid(self, n_ph, bg, sig_lo_per_obs):
        return self.s_grid._build_grid(n_ph, bg, sig_lo_per_obs)

    def _trap_log_w(self, ref: Tensor, K: int) -> Tensor:
        """Trapezoidal log-weights with half-weight endpoints."""
        log_w = ref.new_full((K,), math.log(self.dlog_S))
        log_w[0] = log_w[0] - math.log(2.0)
        log_w[-1] = log_w[-1] - math.log(2.0)
        return log_w

    def _can_use_triton(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,
    ) -> bool:
        """Triton path is valid only for the (GenGauss × Poisson) workload
        on a CUDA tensor with Triton installed."""
        return (
            HAVE_TRITON
            and not _DISABLE_TRITON
            and log_mean.is_cuda
            and isinstance(scatter, GenGaussLogScatter)
            and isinstance(likelihood, Poisson)
        )

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
        K = self.n_points

        if self._can_use_triton(scatter, likelihood, log_mean):
            log_alpha_obs = params.log_alpha[meta.hkl_idx]
            log_beta_obs = params.log_beta[meta.hkl_idx]
            log_w = self._trap_log_w(self.log_S_grid, K)
            return marginalize_signal_triton(
                log_mean, n_ph, bg, log_alpha_obs, log_beta_obs,
                self.log_S_grid, log_w,
            )

        # Eager fallback — bit-exact w.r.t. the original implementation.
        M = log_mean.shape[0]
        log_S = self.log_S_grid[None, :].expand(M, K)        # (M, K)
        S     = self.S_grid[None, :].expand(M, K)            # (M, K)
        eps   = log_S - log_mean[:, None]                    # (M, K)

        log_scatter = scatter.log_prob(eps, params, meta)            # (M, K)
        log_lik = likelihood.log_prob_given_signal(n_ph, S, bg)      # (M, K)

        log_w = self._trap_log_w(log_scatter, K)
        return torch.logsumexp(log_scatter + log_lik + log_w[None, :], dim=1)
