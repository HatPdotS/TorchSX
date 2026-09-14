"""Detection likelihood: ``p(n | signal, bg)``.

A concrete subclass owns the per-grid-point ``log_prob(n | S, bg)`` that the
quadrature integrates the latent signal against.
"""

from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.likelihood.poisson import Poisson

__all__ = ["DetectionLikelihood", "Poisson"]
