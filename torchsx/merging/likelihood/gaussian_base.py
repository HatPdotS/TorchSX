"""Base class for Gaussian-noise detection likelihoods with polynomial-in-μ variance.

Every Gaussian-shape likelihood in the merging pipeline produces a
per-(observation, S-grid-point) ``(μ, σ²)`` pair where ``σ²`` is a
low-order polynomial in ``μ = S + bg``::

    σ²(μ, bg) = g₀(bg) + g₁ · μ + g₂ · μ²

The naming ``g₀, g₁, g₂`` is chosen deliberately: all three coefficients
are detector-gain-chain artefacts.  They are NOT the per-HKL GenGauss
scatter parameters ``log_alpha``, ``log_beta`` (which live in
:class:`~torchsx.merging.scatter.gengauss.GenGaussLogScatter`) — those
parameterise the latent-signal prior, not the noise.

Physical meaning:

- ``g₀(bg)``  — constant (signal-independent) variance: detector read
  noise per peak ring + bg-estimate finite-sample variance.  May depend
  on bg (the latter contribution is ``(n_peak/n_bg)·bg``).
- ``g₁``      — Poisson-like dispersion (g₁ = 1 at the correct stream
  gain is the photon-space moment-match of a Poisson likelihood).
- ``g₂``      — multiplicative variance fraction, e.g.
  gain-stage-switching on Jungfrau detectors (g₂ = γ_g² where γ_g is
  the RMS relative gain uncertainty).

Subclasses only have to implement ``coefficients(bg) → (g0_obs, g1, g2)``;
the eager ``log_prob_given_signal`` here is one line.  The Triton fast
path in :mod:`_fixed_log_grid_triton_gaussian` accepts the same three
inputs and runs 3× faster than the eager evaluation at production
scale.

See ``experiments/weak_reflections/README.md`` for the empirical
justification of the Gaussian approximation: on tubulin 50k it matches
exact Poisson + GH-marginalised excess on phenix R-free to within 0.1 pp
(0.230 vs 0.230), while removing the n_gh=5-fold tensor expansion.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood

_LOG_2PI = math.log(2.0 * math.pi)


class GaussianLikelihood(DetectionLikelihood):
    """Heteroscedastic Gaussian noise with polynomial-in-μ variance.

    Subclasses set the three coefficients via :meth:`coefficients`; the
    inner-grid likelihood computation here is shared.

    The "g₀ / g₁ / g₂" naming follows the variance polynomial
    ``σ² = g₀ + g₁·μ + g₂·μ²``.  These are *noise* polynomial
    coefficients — not the per-HKL GenGauss prior shape parameters
    ``log_alpha``, ``log_beta``.
    """

    # Hard floor on the per-cell variance so log(σ²) and 1/σ² stay finite.
    var_floor: float = 1e-12

    def coefficients(self, bg: Tensor) -> Tuple[Tensor, float, float]:
        """Return ``(g0_obs, g1, g2)``.

        - ``g0_obs``: ``(M,)`` tensor of per-observation constant variance
          contribution (read noise + bg-estimate variance, the latter
          being bg-dependent — hence the ``_obs`` suffix).
        - ``g1``: scalar (Python float or 0-d tensor).
        - ``g2``: scalar.

        The kernel evaluates ``σ²(μ) = g0_obs + g1·μ + g2·μ²`` per (M, K) pair.
        """
        raise NotImplementedError

    def log_prob_given_signal(
        self, n: Tensor, S: Tensor, bg: Tensor,
    ) -> Tensor:                                              # (M, K)
        g0_obs, g1, g2 = self.coefficients(bg)
        # Broadcast (M,) → (M, 1) for the constant term.
        g0 = g0_obs[:, None] if g0_obs.dim() == 1 else g0_obs
        mu = (S + bg[:, None]).clamp(min=1e-6)                 # (M, K)
        var = (g0 + g1 * mu + g2 * mu * mu).clamp(min=self.var_floor)
        diff = n[:, None] - mu
        return -0.5 * (_LOG_2PI + var.log() + diff * diff / var)
