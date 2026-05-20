"""Symmetric Gaussian likelihood with polynomial variance.

Unified noise model
-------------------

This likelihood collapses the three noise contributions we've identified
on Jungfrau/CSPAD detectors into a single three-parameter polynomial in
the predicted mean ``μ = S + bg``::

    Var(n | μ) = g₀  +  g₁ · μ  +  g₂ · μ²

with the physical interpretation (all three are detector gain-chain
artefacts — hence "g"):

    g₀   constant (in photons²)   — detector read noise + bg-estimate
                                     finite-sample variance
    g₁   linear coefficient       — Poisson dispersion (g₁=1 at the
                                     stream-derived gain → pure Poisson)
    g₂   quadratic coefficient    — multiplicative gain-switching noise
                                     fraction (g₂ = γ_g², where γ_g is
                                     the RMS relative gain uncertainty)

Note: ``g₀, g₁, g₂`` are the *noise* polynomial coefficients and are
distinct from the per-HKL GenGauss-prior shape parameters ``log_alpha``,
``log_beta`` (which live in :class:`~torchsx.merging.scatter.gengauss.GenGaussLogScatter`
and parameterise the latent-signal distribution).

Equivalence with the other Gaussian-shape likelihoods:

    * g₀=0, g₁=1, g₂=0          → :class:`GaussianPoisson`
    * g₀=read_var + bg_var·bg   → :class:`GaussianPoissonExcess`
    * g₀=0, g₁=user, g₂=0       → :class:`GaussianProportional` (gamma=g₁)

The bg-estimate variance term ``(n_peak/n_bg)·bg`` from
``PoissonNoiseModel`` is folded into the *per-observation* g₀ via
``bg_var_factor·bg`` — it's bg-dependent but signal-independent, so it
naturally lives in the g₀ slot.

Trainable mode (``trainable=True``) wraps each of (g₀, g₁, g₂) in a
log-space ``nn.Parameter`` so LBFGS can co-optimise them with the rest
of the model.  Note: earlier experiments (T5/T6/T9 in
``experiments/weak_reflections/``) showed these are *not* identifiable
from the data NLL alone — they end up frozen at their init in the
cold-start case and converge to noise-inflating values in the
warm-start case that degrade R-free.  Keep ``trainable=False`` for
production and calibrate externally (off-Bragg σ²(bg) fit + detector
characterisation).

Closed-form note
----------------

The marginal ``∫ N(n; μ, σ²(μ)) · GenGauss(μ; μ_HKL, α_g, β_g) dμ`` is
NOT analytical for either non-quadratic σ² or non-Gaussian GenGauss
(β_g ≠ 2).  We rely on the existing ``FixedLogGrid`` quadrature on the
signal axis, which evaluates this likelihood at each (M, K) grid point
and handles heteroscedastic σ²(μ) without any extra machinery.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.likelihood.gaussian_base import GaussianLikelihood


class GaussianPolynomialVariance(GaussianLikelihood):
    """``n ~ Normal(μ, g₀ + g₁·μ + g₂·μ²)`` with ``μ = S + bg``.

    Any subset of (g₀, g₁, g₂) can be made trainable via ``trainable``.
    Log-space parameterisation keeps them positive.

    Public kwarg names retain the descriptive ``var_const``,
    ``var_linear``, ``var_quadratic`` form for clarity at the call site;
    these correspond to ``(g₀_const, g₁, g₂)`` in the math.

    Parameters
    ----------
    var_const :
        g₀_const (constant part) — read-noise floor in photons².  The
        bg-dependent part ``bg_var_factor·bg`` is added automatically to
        form the per-observation ``g₀_obs``.
    var_linear :
        g₁ — linear coefficient.  Default 1.0 (pure Poisson moment-match
        when the gain is correct).
    var_quadratic :
        g₂ — quadratic coefficient (γ_g², dimensionless).
    bg_var_factor :
        ``n_peak / n_bg``.  Adds ``bg_var_factor·bg`` to g₀ per observation
        — the bg-estimate finite-sample variance.  Default 0
        (caller responsible for absorbing it into ``var_const`` if desired).
    trainable :
        If True, all three coefficients become log-space ``nn.Parameter``s
        that LBFGS can optimise.  Floors at 1e-10 to avoid log(0).
    var_floor :
        Hard minimum on the predicted variance.  Overrides the base class
        default of 1e-12 if specified.
    """

    def __init__(
        self,
        var_const: float = 0.0,
        var_linear: float = 1.0,
        var_quadratic: float = 0.0,
        bg_var_factor: float = 0.0,
        trainable: bool = False,
        var_floor: float = 1e-12,
    ) -> None:
        super().__init__()
        self.bg_var_factor = float(bg_var_factor)
        self.var_floor = float(var_floor)
        self._trainable = bool(trainable)

        if trainable:
            log_a = math.log(max(var_const, 1e-10))
            log_b = math.log(max(var_linear, 1e-10))
            log_c = math.log(max(var_quadratic, 1e-10))
            self.log_var_const = nn.Parameter(torch.tensor(log_a, dtype=torch.float32))
            self.log_var_linear = nn.Parameter(torch.tensor(log_b, dtype=torch.float32))
            self.log_var_quadratic = nn.Parameter(torch.tensor(log_c, dtype=torch.float32))
            # placeholders for diagnostic helpers
            self.var_const = 0.0
            self.var_linear = 0.0
            self.var_quadratic = 0.0
        else:
            self.var_const = float(var_const)
            self.var_linear = float(var_linear)
            self.var_quadratic = float(var_quadratic)

    def _g0_const(self) -> Tensor | float:
        return self.log_var_const.exp() if self._trainable else self.var_const

    def _g1(self) -> Tensor | float:
        return self.log_var_linear.exp() if self._trainable else self.var_linear

    def _g2(self) -> Tensor | float:
        return self.log_var_quadratic.exp() if self._trainable else self.var_quadratic

    def coefficient_values(self) -> tuple[float, float, float]:
        """(g₀_const, g₁, g₂) — works for both fixed and trainable.

        Returns the *constant* part of g₀ (not the per-obs value that
        also folds in ``bg_var_factor·bg``).
        """
        out = []
        for f in (self._g0_const, self._g1, self._g2):
            v = f()
            out.append(float(v.item()) if isinstance(v, Tensor) else float(v))
        return tuple(out)  # type: ignore[return-value]

    def coefficients(self, bg: Tensor) -> Tuple[Tensor, Tensor | float, Tensor | float]:
        """Return ``(g0_obs, g1, g2)`` for the unified Gaussian likelihood.

        g₀ is folded into a per-observation tensor that includes the
        bg-estimate variance contribution ``bg_var_factor·bg``.  g₁ and
        g₂ are returned as scalars (or 0-d tensors when trainable).
        """
        g0_const = self._g0_const()
        # g0_const may be a scalar tensor (trainable) or Python float
        # (fixed).  The expression must broadcast against (M,) ``bg``.
        g0_obs = g0_const + self.bg_var_factor * bg                # (M,)
        return g0_obs, self._g1(), self._g2()
