"""Symmetric Gaussian likelihood with polynomial variance — per-observation σ².

Unified noise model
-------------------

We model the *per-observation* detector noise as a polynomial in the
data-side mean estimate ``μ_obs = max(n, bg)``::

    σ²_obs = g₀  +  g₁ · μ_obs  +  g₂ · μ_obs²

with the physical interpretation (all three are detector gain-chain
artefacts — hence "g"):

    g₀   constant (in photons²)   — detector read noise + bg-estimate
                                     finite-sample variance
    g₁   linear coefficient       — Poisson dispersion (g₁=1 at the
                                     stream-derived gain → pure Poisson)
    g₂   quadratic coefficient    — multiplicative gain-switching noise
                                     fraction (g₂ = γ_g², where γ_g is
                                     the RMS relative gain uncertainty)

Note: σ² is evaluated *at the data*, not per-(M, K) on the
S-integration grid. σ is a property of the observation (set by the
detector), not of the latent-signal hypothesis space. See
:mod:`gaussian_base` for the conceptual rationale.

The bg-estimate variance term ``(n_peak/n_bg)·bg`` from
``PoissonNoiseModel`` is folded into ``g₀`` per observation via
``bg_var_factor·bg`` — it's bg-dependent but signal-independent.

This is the only ``GaussianLikelihood`` subclass in the public surface;
its three coefficients subsume what used to be separate classes:

    * g₀=0, g₁=1, g₂=0          — pure Poisson moment-match in σ²
                                  (the production T7 default; ``--variance stream``)
    * g₀=read_var + bg_var·bg   — Poisson + signal-independent excess noise
    * g₀=0, g₁=γ, g₂=0          — symmetric Gaussian with γ-scaled variance

Trainable mode (``trainable=True``) wraps each of (g₀, g₁, g₂) in a
log-space ``nn.Parameter`` so LBFGS can co-optimise them with the rest
of the model. Autograd chains through ``sigma_squared_obs → var →
log_p`` via the Triton kernel's ``S2_var`` sufficient stat.

EXPERIMENTAL — the trainable branch is preserved as an opt-in research
knob (``--refine-detector-noise`` on the CLI). At production scale
(50k tubulin) it converges to a non-physical ``g₁ ≈ 14`` mode and is
NaN-prone in single precision; the data NLL alone does not pin the
detector-noise coefficients. **Keep ``trainable=False`` for production.**
The T9 run set in ``experiments/weak_reflections/README.md`` documents
the failure mode.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.likelihood.gaussian_base import GaussianLikelihood


class GaussianPolynomialVariance(GaussianLikelihood):
    """``n ~ Normal(S + bg, σ²_obs)`` with ``σ²_obs = g₀ + g₁·μ_obs + g₂·μ_obs²``.

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
        form the per-observation g₀.
    var_linear :
        g₁ — linear coefficient.  Default 1.0 (pure Poisson moment-match
        when the gain is correct).
    var_quadratic :
        g₂ — quadratic coefficient (γ_g², dimensionless).
    bg_var_factor :
        ``n_peak / n_bg``.  Adds ``bg_var_factor·bg`` to g₀ per observation
        — the bg-estimate finite-sample variance.
    trainable :
        If True, all three coefficients become log-space ``nn.Parameter``s
        that LBFGS can optimise.  Floors at 1e-10 to avoid log(0).
        **EXPERIMENTAL** — at production scale (50k tubulin) this branch
        converges to a non-physical ``g₁ ≈ 14`` mode and is NaN-prone in
        single precision. Keep ``trainable=False`` for production; see the
        T9 run set in ``experiments/weak_reflections/README.md``.
    var_floor :
        Hard minimum on σ².  Overrides the base class default of 1e-12.
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

    def sigma_squared_obs(self, n: Tensor, bg: Tensor) -> Tensor:
        """``σ²_obs = (g₀_const + bg_var·bg₊) + g₁·μ_obs + g₂·μ_obs²``.

        ``μ_obs = max(n, bg, 0)`` — Poisson MLE for μ, floored at 0.
        ``bg₊ = max(bg, 0)`` — bg-variance contribution uses only the
        non-negative part of bg (the (n_peak/n_bg)·bg variance estimator
        is only meaningful for non-negative bg).

        Without these floors, observations with strongly negative
        background-subtracted ``n`` or noise-dominated negative ``bg``
        can drive σ² below zero, and the var_floor clamp downstream
        blocks the gradient pull-back.
        """
        mu_obs = torch.maximum(n, bg).clamp(min=0.0)
        bg_pos = bg.clamp(min=0.0)
        g0_obs = self._g0_const() + self.bg_var_factor * bg_pos
        return g0_obs + self._g1() * mu_obs + self._g2() * mu_obs * mu_obs
