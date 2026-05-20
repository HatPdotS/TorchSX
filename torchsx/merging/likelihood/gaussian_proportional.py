"""Symmetric Gaussian observation likelihood with linear-in-mean variance.

Motivation
----------

The default :class:`~torchsx.merging.likelihood.Poisson` PMF is
intrinsically asymmetric (heavy mass at zero, no negative values) and
poorly conditioned for weak observations: its second derivative blows up
as ``λ → 0``, gradients are spiky at the boundary and LBFGS struggles.
The scale-noise prior — a symmetric GenGauss on the latent signal —
cannot fix that asymmetry by marginalisation alone, it just smears it.

This module replaces the inner ``log p(n | S, bg)`` with a *symmetric
Gaussian* whose variance grows linearly with the mean::

    n | S ~ Normal(S + bg, γ · (S + bg))

The scale-noise prior (GenGauss on ``S``) is unchanged.  Marginalising
``S = s·μ`` with ``s ~ GenGauss(1, α, β)`` gives the marginal moments::

    E[n | μ]   = μ + bg
    Var[n | μ] = γ·(μ + bg)            + Var(s) · μ²
                 └─ linear (this likelihood) ─┘  └─ quadratic (GenGauss prior) ─┘
                 dominates at weak μ              dominates at strong μ

So the GenGauss prior carries scale stochasticity for strong reflections
(its α, β fit as in the current model), and this likelihood carries
linear-in-mean overdispersion γ for weak reflections (where the Poisson
PMF is ill-conditioned).

``γ`` is a single dataset-level scalar:
  * γ = 1 ⇒ variance equals mean (Poisson moment-match) but with
    symmetric Gaussian shape and well-conditioned gradients everywhere.
  * γ > 1 ⇒ overdispersed; calibrate from the off-Bragg σ²-fit divided
    by the stream gain (the linear-coefficient of σ²_z vs bg under
    pure Poisson at the true detector gain).  Typical SFX values are
    in the range 2 – 5 due to detector excess-noise factor.

Triton fast path: shared with all other Gaussian-shape likelihoods
via :class:`~torchsx.merging.likelihood.gaussian_base.GaussianLikelihood`
and ``marginalize_signal_triton_gaussian``.  This class is therefore now
a thin subclass that just specifies the variance coefficients.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from torchsx.merging.likelihood.gaussian_base import GaussianLikelihood


class GaussianProportional(GaussianLikelihood):
    """``n ~ Normal(S + bg, γ · (S + bg))``.

    Parameters
    ----------
    gamma :
        Overdispersion factor (>0).  Multiplies the predicted variance.
    var_floor :
        Hard minimum on the predicted variance to keep ``log(var)`` and
        ``1/var`` finite when ``S + bg → 0``.  The combined floor on
        ``S + bg`` (1e-6, matching :class:`Poisson`) and ``γ > 0`` keeps
        the variance bounded away from zero in practice; this is
        belt-and-suspenders.
    """

    def __init__(self, gamma: float, var_floor: float = 1e-12) -> None:
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.gamma = float(gamma)
        self.var_floor = float(var_floor)

    def coefficients(self, bg: Tensor) -> Tuple[Tensor, float, float]:
        # σ² = gamma · μ → g₀ = 0, g₁ = gamma, g₂ = 0.
        return torch.zeros_like(bg), self.gamma, 0.0
