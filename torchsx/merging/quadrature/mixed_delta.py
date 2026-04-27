"""Mixed-quadrature: GenGauss-scatter integral + no-scatter delta at S=E[S].

Two-hypothesis marginal likelihood:

    L(n | I) = (1 − π) · ∫ Poisson(n; K·I·S + bg) · GenGauss(log S; α, β) dS
             +    π    · Poisson(n; K·I·E[S] + bg)

Both hypotheses target the same I (= latent median intensity).  The delta
point is anchored at ``S = E[S]`` (NOT ``S = 1``) so that the no-scatter
hypothesis estimates the median consistent with the scatter hypothesis,
not the per-obs mean.  Without this anchor the two hypotheses would
target different quantities and mixing them would give an inconsistent
estimator.

Why this exists
---------------
On strong HKLs the GenGauss marginal MLE recovers I_true correctly.
On weak HKLs (low SNR) the marginal collapses toward a moment-matching
estimator (see ``experiments/synthetic_gengauss/compare_linear_vs_gengauss.py``):
the bias on the weakest decile is ~80 %.  The no-scatter hypothesis is
robust on weak data — it's the linear/moment-matching estimator with the
E[S]-correction baked in.  A small (~10 %) mixture of it pulls the MLE
back toward the unbiased linear answer on weak data without disturbing
strong-HKL recovery.

Design notes
------------
- π is a global scalar for now (prototype).  Per-HKL π_h with empirical
  Bayes shrinkage is the natural follow-up.
- E[S] is provided as a config-time constant (computed externally from
  the assumed/median α, β).  A future revision can compute it per HKL
  from learnable α_h, β_h via a frozen numerical quadrature.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.trapezoidal import TrapezoidalLogGrid
from torchsx.merging.scatter.base import ScatterModel


class MixedDeltaTrapezoidal(Quadrature):
    """Trapezoidal GenGauss quadrature mixed with a no-scatter delta.

    Parameters
    ----------
    base : underlying continuous quadrature (typically a TrapezoidalLogGrid).
    e_s : E[S] under the GenGauss scatter prior (a single global float for
        the prototype; computed externally as ``E[exp(ε)]`` under the
        median (α, β) of the dataset).
    pi_no_scatter : mixture weight on the no-scatter hypothesis ∈ (0, 1).
    """

    def __init__(
        self,
        base: TrapezoidalLogGrid,
        e_s: float,
        pi_no_scatter: float = 0.1,
    ) -> None:
        super().__init__()
        if not (0.0 < pi_no_scatter < 1.0):
            raise ValueError(f"pi_no_scatter must be in (0,1), got {pi_no_scatter}")
        if e_s <= 0.0:
            raise ValueError(f"e_s must be positive, got {e_s}")
        self.base = base
        self.register_buffer("log_e_s", torch.tensor(math.log(e_s)))
        self.register_buffer(
            "log_pi", torch.tensor(math.log(pi_no_scatter))
        )
        self.register_buffer(
            "log_1mpi", torch.tensor(math.log1p(-pi_no_scatter))
        )

    @classmethod
    def wrap(
        cls,
        prep: PreparedObs,
        config,
        e_s: float,
        pi_no_scatter: float = 0.1,
    ) -> "MixedDeltaTrapezoidal":
        base = TrapezoidalLogGrid.from_prep(prep, config)
        return cls(base=base, e_s=e_s, pi_no_scatter=pi_no_scatter)

    # ── Forward grid/sig_lo helpers so the posterior pipeline (which
    #    expects a TrapezoidalLogGrid interface) keeps working.  Those
    #    methods are only used by the posterior moment computation, which
    #    treats the marginal as the GG-only branch — that's fine, the
    #    delta term contributes a constant in I and doesn't affect the
    #    mode/curvature.
    def _sig_lo_slice(self, *args, **kwargs):
        return self.base._sig_lo_slice(*args, **kwargs)

    def _build_grid(self, *args, **kwargs):
        return self.base._build_grid(*args, **kwargs)

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
        # ── Scatter hypothesis: full GenGauss integral via base quadrature ──
        log_L_gg = self.base.marginalize_signal(
            scatter, likelihood, log_mean, n_ph, bg, params, meta, obs_range
        )                                                       # (M,)

        # ── No-scatter hypothesis: Poisson at S = E[S] (single point) ──
        # log p(n | S=E[S], bg) = Poisson(n; K·I·E[S] + bg)
        # Using likelihood.log_prob_given_signal with S = (K·I)·E[S] = exp(log_mean + log_e_s)
        S_es = torch.exp(log_mean + self.log_e_s)               # (M,)  = K·I·E[S]
        # The detection-likelihood API takes S of shape (M, K).  Make it (M, 1).
        log_L_lin_full = likelihood.log_prob_given_signal(
            n_ph, S_es[:, None], bg
        )                                                       # (M, 1)
        log_L_lin = log_L_lin_full[:, 0]                         # (M,)

        # ── Mixture in log-space ──
        # log[(1-π) e^A + π e^B] = logsumexp([log(1-π)+A, log π+B])
        return torch.logsumexp(
            torch.stack(
                [self.log_1mpi + log_L_gg, self.log_pi + log_L_lin],
                dim=0,
            ),
            dim=0,
        )
