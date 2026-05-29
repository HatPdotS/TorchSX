"""Base class for Gaussian-noise detection likelihoods with per-observation σ².

Every Gaussian-shape likelihood in the merging pipeline carries a single
σ² per observation — the detector noise on that specific measurement.
σ² is *not* a function of the integration variable S inside the
marginalization; it is set by the data (e.g. ``√(max(n,bg))`` style
classical estimators) or by external detector calibration, and treated
as constant while the prior on S is being marginalized.

Subclasses implement ``sigma_squared_obs(n, bg) → Tensor[M]``; the
shared eager ``log_prob_given_signal`` here pulls σ² once per obs and
broadcasts it across the K-grid (which is required only because the
rest of the integrand — ``μ = S + bg`` — is (M, K)).

Why per-observation, not per-(M, K)? In the analogous Poisson kernel
the variance equals λ = S+bg only because the Poisson PMF has a single
parameter; that is ``Var[Poisson(λ)] = λ``, not a noise model. A
Gaussian likelihood has σ² as an independent second parameter, and
physically it represents the noise on *this particular observation*
— a property of the detector reading, not of the hypothesis space.
Letting σ² vary with S during marginalization mixes the noise model
into the prior integral.

The Triton fast path in :mod:`_fixed_log_grid_triton_gaussian` accepts
the same per-obs ``var`` tensor directly.
"""
from __future__ import annotations

import math

from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood

_LOG_2PI = math.log(2.0 * math.pi)


class GaussianLikelihood(DetectionLikelihood):
    """Gaussian noise with per-observation σ²; subclasses define how σ² is computed.

    Concrete subclasses override :meth:`sigma_squared_obs` to return a
    ``(M,)`` tensor; the eager-path integrand and the Triton kernel both
    consume that tensor directly.
    """

    # Hard floor on σ² so log(σ²) and 1/σ² stay finite.
    var_floor: float = 1e-12

    def sigma_squared_obs(self, n: Tensor, bg: Tensor) -> Tensor:
        """Return ``(M,)`` per-observation σ².

        Subclass responsibility: evaluate the variance using whatever
        recipe is appropriate (e.g. ``max(n, bg)`` for a pure-Poisson
        moment match, a polynomial in ``μ_obs = max(n, bg)`` for the
        general gain-chain model, or a lookup from external
        calibration).
        """
        raise NotImplementedError

    def log_prob_given_signal(
        self, n: Tensor, S: Tensor, bg: Tensor,
    ) -> Tensor:                                                  # (M, K)
        var = self.sigma_squared_obs(n, bg).clamp(min=self.var_floor)  # (M,)
        mu = (S + bg[:, None]).clamp(min=1e-6)                          # (M, K)
        diff = n[:, None] - mu
        return -0.5 * (
            _LOG_2PI + var[:, None].log() + diff * diff / var[:, None]
        )
