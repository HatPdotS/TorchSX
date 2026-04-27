"""Hybrid quadrature: data-driven S-grid ∪ fixed log-S grid covering GG support.

The default ``TrapezoidalLogGrid`` builds a per-obs softplus grid in
``S`` between ``signal_lo = (n − bg).clamp(min=signal_min)`` and
``signal_max = (n − bg + n_sigma·√n).clamp(min=1)``. That grid follows
the linear estimate ``(n − bg)/K``, so it covers the Poisson-likelihood
peak well — but for any trial intensity ``I`` whose ``log_mean = log(K·I)``
falls outside that S-window, the GG-prior mode at ``S = K·I`` is
**outside** the grid. The integral undercounts the marginal at the
true ``I``, biasing the MLE upward (synthetic test:
``I_mle / I_truth`` rises from 2.9× at bg=0.1 to 24.7× at bg=100 with
``I_truth=0.01``, ``α=1``, ``β=2``).

The hybrid here unions that data-driven S-grid with a **fixed log-S
grid** spanning the GG generative support (default
``log S ∈ [−8, 8]``), then computes proper trapezoidal weights on the
sorted union (each node gets ``(s_{i+1} − s_{i−1})/2``). The data-driven
nodes catch the Poisson likelihood peak; the fixed log-S nodes catch
the GG prior mode regardless of trial ``I``. Synthetic test with the
same setup gives ``I_mle / I_truth`` of {1.01, 1.00, 1.00, 1.05} across
bg ∈ {0.1, 1, 10, 100} — better than the linear estimator at high bg.

Cost: per-obs grid size grows from ``n_quad_S`` to ``n_quad_S +
n_quad_log`` (~2×). All ops are vectorised; ``log_prob_given_signal``
and ``scatter.log_prob`` are evaluated at the full union.
"""
from __future__ import annotations

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


class HybridLogGrid(Quadrature):
    """Hybrid quadrature: torchsx S-grid ∪ fixed log-S prior grid.

    Parameters
    ----------
    n_quad_S : nodes in the data-driven softplus S-grid (default 51).
    n_quad_log : nodes in the fixed log-S grid (default 51).
    log_S_lo, log_S_hi : range of the fixed log-S grid in log-S
        coordinates. Defaults [-8, 8] cover GG(0, α≤2, β≥1) tails to
        ~e^−5 of the peak.
    n_sigma : upper-bound coverage for the S-grid (in √n units).
    signal_min : absolute lower bound on the S-grid signal_lo.
    sig_lo_per_obs : optional precomputed per-obs lower bound buffer
        (passed through to the inner ``TrapezoidalLogGrid``).
    """

    def __init__(
        self,
        n_quad_S: int = 51,
        n_quad_log: int = 51,
        log_S_lo: float = -8.0,
        log_S_hi: float = 8.0,
        n_sigma: float = 5.0,
        signal_min: float = 0.01,
        sig_lo_per_obs: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.s_grid = TrapezoidalLogGrid(
            n_quad=n_quad_S,
            n_sigma=n_sigma,
            signal_min=signal_min,
            sig_lo_per_obs=sig_lo_per_obs,
        )
        self.n_quad_log = int(n_quad_log)
        self.register_buffer(
            "log_S_prior_grid",
            torch.linspace(float(log_S_lo), float(log_S_hi), n_quad_log),
        )

    @classmethod
    def from_prep(cls, prep: PreparedObs, config: MergeConfig) -> "HybridLogGrid":
        sig_lo = _compute_sig_lo_per_obs(
            n_valid=prep.n_valid,
            hkl_id=prep.hkl_id,
            bg=prep.bg,
            obs_counts=prep.obs_counts,
            override=config.sig_lo_override,
        )
        return cls(
            n_quad_S=config.n_quad,
            n_quad_log=getattr(config, "n_quad_log", 51),
            log_S_lo=getattr(config, "log_S_lo", -8.0),
            log_S_hi=getattr(config, "log_S_hi", 8.0),
            n_sigma=config.n_sigma_quad,
            signal_min=0.01,
            sig_lo_per_obs=sig_lo,
        )

    # Posterior code calls these on TrapezoidalLogGrid — forward to the inner.
    def _sig_lo_slice(self, n_ph, bg, obs_range):
        return self.s_grid._sig_lo_slice(n_ph, bg, obs_range)

    def _build_grid(self, n_ph, bg, sig_lo_per_obs):
        return self.s_grid._build_grid(n_ph, bg, sig_lo_per_obs)

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
        # Data-driven S-grid (just the node positions; weights recomputed below).
        sig_lo = self.s_grid._sig_lo_slice(n_ph, bg, obs_range)
        signal_S, _ = self.s_grid._build_grid(n_ph, bg, sig_lo)        # (M, K_S)
        log_S_S = torch.log(signal_S.clamp(min=1e-12))

        # Fixed log-S grid, broadcast per obs.
        M = n_ph.shape[0]
        log_S_L = self.log_S_prior_grid.expand(M, -1)                  # (M, K_log)

        # Sorted union and neighbour-distance trapezoidal weights.
        log_S_all = torch.cat([log_S_S, log_S_L], dim=1)
        log_S_sorted, _ = log_S_all.sort(dim=1)
        diffs = log_S_sorted[:, 1:] - log_S_sorted[:, :-1]             # (M, K−1)
        w = torch.zeros_like(log_S_sorted)
        w[:, 0]    = diffs[:, 0] * 0.5
        w[:, -1]   = diffs[:, -1] * 0.5
        w[:, 1:-1] = (diffs[:, :-1] + diffs[:, 1:]) * 0.5
        log_w = torch.log(w.clamp(min=1e-30))

        # Integrand at every node.
        S = torch.exp(log_S_sorted)
        eps = log_S_sorted - log_mean[:, None]
        log_scatter = scatter.log_prob(eps, params, meta)              # (M, K)
        log_lik = likelihood.log_prob_given_signal(n_ph, S, bg)        # (M, K)

        return torch.logsumexp(log_scatter + log_lik + log_w, dim=1)
