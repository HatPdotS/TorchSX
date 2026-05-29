"""Detection likelihood: ``p(n | signal, bg)``.

Concrete subclasses own both the per-grid-point ``log_prob(n | S, bg)``
(used by the Quadrature to integrate the latent signal out) and any
post-marginalisation bg-mixture logic (``combine_with_marginal``).
"""

from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.likelihood.gaussian_base import GaussianLikelihood
from torchsx.merging.likelihood.gaussian_bg_mix import GaussianPolynomialWithBgMixture
from torchsx.merging.likelihood.gaussian_polynomial import GaussianPolynomialVariance
from torchsx.merging.likelihood.poisson import Poisson
from torchsx.merging.likelihood.poisson_bg_mix import PoissonWithBackgroundMixture

__all__ = [
    "DetectionLikelihood",
    "GaussianLikelihood",
    "GaussianPolynomialVariance",
    "GaussianPolynomialWithBgMixture",
    "Poisson",
    "PoissonWithBackgroundMixture",
]
