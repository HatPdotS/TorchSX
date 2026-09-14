"""Latent-intensity priors on ``log_I0``.

The Wilson prior in use is parameterised on a smooth reciprocal-space basis
rather than per-shell: :mod:`torchsx.merging.priors.reciprocal_basis`.
"""

from torchsx.merging.priors.reciprocal_basis import (
    ReciprocalBasis,
    fit_wilson_log_sigma,
    wilson_b_equivalent,
)

__all__ = ["ReciprocalBasis", "fit_wilson_log_sigma", "wilson_b_equivalent"]
