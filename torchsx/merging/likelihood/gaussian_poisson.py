"""Gaussian moment-match approximations of Poisson and Poisson+excess-noise.

Both are stand-ins for the existing :class:`Poisson` and
:class:`PoissonWithExcessNoise` likelihoods that route through the
generalised :class:`GaussianLikelihood` base — and therefore (once the
Triton kernel is in place) share its fast path.

The Gaussian approximation collapses two integrals:

* The Poisson PMF ``Poisson(n | λ)`` becomes ``N(n | λ, λ)``.  Exact in
  the moment-matched sense (mean λ, variance λ) and a good approximation
  for ``λ ≳ 5`` photons.  At very low counts (n ∈ {0, 1, 2}) the
  asymmetric Poisson PMF is more accurate, but for SFX Bragg peaks at
  corrected gain typical λ values are several photons; the empirical
  R-free match (T7 vs T2 on tubulin: 0.2302 vs 0.2296) confirms the
  approximation doesn't bite.

* The Gauss-Hermite axis used by :class:`PoissonWithExcessNoise` to
  marginalise the bg-estimate variance disappears — that variance just
  folds into ``σ²`` directly.  Saves the n_gh=5-fold tensor expansion.

The variance polynomial coefficients ``g₀, g₁, g₂`` here are the same
ones used by :class:`GaussianLikelihood` (detector gain-chain
contributions to ``σ²``); they are NOT the per-HKL GenGauss prior
shape parameters.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from torchsx.merging.likelihood.gaussian_base import GaussianLikelihood


class GaussianPoisson(GaussianLikelihood):
    """``n | S, bg ~ N(S + bg, S + bg)`` — Gaussian moment-match of pure Poisson.

    Variance polynomial: ``g₀ = 0, g₁ = 1, g₂ = 0``.

    Use as a drop-in replacement for :class:`Poisson` whenever the
    expected per-observation photon count is high enough that the
    Gaussian approximation is fine (≳ 5 photons).
    """

    def coefficients(self, bg: Tensor) -> Tuple[Tensor, float, float]:
        return torch.zeros_like(bg), 1.0, 0.0


class GaussianPoissonExcess(GaussianLikelihood):
    """Gaussian moment-match of Poisson plus signal-independent excess variance.

    Variance polynomial::

        σ²(μ, bg) = bg_var_factor·bg + read_var_photons   + 1·μ + 0·μ²
                 = g₀(bg)                              + g₁·μ + g₂·μ²

    Drop-in replacement for :class:`PoissonWithExcessNoise` that skips
    the Gauss-Hermite marginalisation by folding the excess directly
    into the variance.

    Parameters
    ----------
    bg_var_factor :
        Coefficient on ``bg`` in the bg-estimate variance: ``n_peak / n_bg``,
        set by ``PoissonNoiseModel`` from the integration ring geometry.
    read_var_photons :
        Detector read-noise variance per peak-pixel-sum, in photons²
        (``σ²_read_ADU · n_peak / gain²``).  Defaults to 0; calibrate
        from dark frames or off-Bragg σ²(bg) fit.
    """

    def __init__(
        self,
        bg_var_factor: float,
        read_var_photons: float = 0.0,
    ) -> None:
        super().__init__()
        self.bg_var_factor = float(bg_var_factor)
        self.read_var_photons = float(read_var_photons)

    def coefficients(self, bg: Tensor) -> Tuple[Tensor, float, float]:
        g0_obs = self.bg_var_factor * bg + self.read_var_photons
        return g0_obs, 1.0, 0.0
