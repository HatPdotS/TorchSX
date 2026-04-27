"""Per-obs adaptive marginal: delta-at-E[S] for strong obs, hybrid for weak.

Per observation, compute a significance ``signif = (n − bg)/√max(n, bg)``.

* If ``signif > thresh`` (default 5) the Poisson likelihood is sharply
  peaked in S → the GG marginal collapses to a single Poisson evaluation
  at ``S = E[S]`` (the delta-at-E[S] approximation). Cheap, sharp,
  unbiased — and crucially, has **no plateau** at small trial I.
* Otherwise (weak obs, near-bg) use the full ``HybridLogGrid`` marginal.
  These are the obs where the GG actually constrains the integral; they
  also carry the bimodal tail in I-space, which production must break
  with a Wilson-match prior.

This isolates the bimodality to the obs that genuinely produce it. When
an HKL has *any* strong obs the truth basin dominates the marginal and
the diverged-cluster artifact disappears entirely.

Computation: both paths are evaluated for all obs, then a per-obs mask
selects between them. Future optimisation can subset the hybrid call
to weak obs only (saves ~K_grid × n_strong work).
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
from torchsx.merging.quadrature.delta_at_es import compute_e_s
from torchsx.merging.quadrature.hybrid_log import HybridLogGrid
from torchsx.merging.quadrature.trapezoidal import _compute_sig_lo_per_obs
from torchsx.merging.scatter.base import ScatterModel


class AdaptiveSignifSplit(Quadrature):
    """Per-obs split between delta-at-E[S] (strong) and HybridLogGrid (weak)."""

    def __init__(
        self,
        signif_thresh: float = 5.0,
        alpha_ref: float = 1.0,
        beta_ref: float = 1.7,
        n_quad_S: int = 51,
        n_quad_log: int = 51,
        log_S_lo: float = -8.0,
        log_S_hi: float = 8.0,
        n_sigma: float = 5.0,
        signal_min: float = 0.01,
        sig_lo_per_obs: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.signif_thresh = float(signif_thresh)
        self.alpha_ref = float(alpha_ref)
        self.beta_ref = float(beta_ref)
        self.register_buffer("e_s", torch.tensor(compute_e_s(alpha_ref, beta_ref)))
        self.hybrid = HybridLogGrid(
            n_quad_S=n_quad_S, n_quad_log=n_quad_log,
            log_S_lo=log_S_lo, log_S_hi=log_S_hi,
            n_sigma=n_sigma, signal_min=signal_min,
            sig_lo_per_obs=sig_lo_per_obs,
        )

    @classmethod
    def from_prep(cls, prep: PreparedObs, config: MergeConfig) -> "AdaptiveSignifSplit":
        sig_lo = _compute_sig_lo_per_obs(
            n_valid=prep.n_valid,
            hkl_id=prep.hkl_id,
            bg=prep.bg,
            obs_counts=prep.obs_counts,
            override=config.sig_lo_override,
        )
        return cls(
            signif_thresh=getattr(config, "signif_thresh", 5.0),
            alpha_ref=getattr(config, "alpha_ref", 1.0),
            beta_ref=getattr(config, "beta_ref", 1.7),
            n_quad_S=config.n_quad,
            n_quad_log=getattr(config, "n_quad_log", 51),
            log_S_lo=getattr(config, "log_S_lo", -8.0),
            log_S_hi=getattr(config, "log_S_hi", 8.0),
            n_sigma=config.n_sigma_quad,
            signal_min=0.01,
            sig_lo_per_obs=sig_lo,
        )

    # Forward grid-construction calls to the inner hybrid.s_grid for posterior.
    def _sig_lo_slice(self, n_ph, bg, obs_range):
        return self.hybrid.s_grid._sig_lo_slice(n_ph, bg, obs_range)

    def _build_grid(self, n_ph, bg, sig_lo_per_obs):
        return self.hybrid.s_grid._build_grid(n_ph, bg, sig_lo_per_obs)

    def classify(self, n_ph: Tensor, bg: Tensor) -> Tensor:
        """Return per-obs is_strong mask. n_ph and bg are both (M,)."""
        denom = torch.maximum(n_ph, bg).clamp(min=1.0).sqrt()
        signif = (n_ph - bg) / denom
        return signif > self.signif_thresh

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
        is_strong = self.classify(n_ph, bg)

        # Strong path: delta-at-E[S]; S = K·I·E[S].
        S_strong = (log_mean + torch.log(self.e_s)).exp()                  # (M,)
        log_p_strong = likelihood.log_prob_given_signal(
            n_ph, S_strong[:, None], bg
        )[:, 0]                                                            # (M,)

        # Weak path: full hybrid marginal.
        log_p_weak = self.hybrid.marginalize_signal(
            scatter, likelihood, log_mean, n_ph, bg, params, meta, obs_range
        )                                                                  # (M,)

        return torch.where(is_strong, log_p_strong, log_p_weak)
