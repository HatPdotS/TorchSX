"""Gamma multiplicative-scatter density.

Alternative to ``GenGaussLogScatter`` that lets the latent signal cover
``S → 0`` smoothly.  The "scatter" is on the *linear* signal scale via a
Gamma distribution whose shape α controls dispersion; reparameterised
into ε = log(S/μ) the density is stationary (μ-independent).

If S ~ Gamma(shape=α, rate=α/μ) (mean μ, var μ²/α), then
ε = log(S/μ) has density::

    log p(ε; α) = α·log α − log Γ(α) + α·ε − α·exp(ε)

— peaks at ε=0 (S=μ), and has heavy lower tail down to ε → −∞ (S → 0).
β is unused (Gamma is single-parameter); we read α from ``params.log_alpha``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter.base import ScatterModel


class GammaLogScatter(ScatterModel):
    """Per-HKL Gamma scatter density on ε = log(S/μ).

    Notes
    -----
    * α is per-HKL (reuses ``params.log_alpha``).  ``params.log_beta`` is
      ignored — Gamma is single-parameter.
    * Smaller α ⇒ more dispersion (Var[S/μ] = 1/α).
    * Mode at S = μ; mean of S/μ is exactly 1.
    """

    def log_prob(
        self,
        eps: Tensor,           # (M, K)
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:               # (M, K)
        h = meta.hkl_idx
        log_alpha = params.log_alpha[h]                 # (M,)
        alpha = log_alpha.exp().clamp(min=1e-3)         # (M,)
        # log p(ε; α) = α·log α − log Γ(α) + α·ε − α·exp(ε)
        log_norm = alpha * log_alpha - torch.lgamma(alpha)              # (M,)
        return (
            log_norm[:, None]
            + alpha[:, None] * eps
            - alpha[:, None] * eps.exp()
        )
