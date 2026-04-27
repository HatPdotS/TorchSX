"""Generalised-Gaussian log-scatter density (per-HKL α, β)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter.base import ScatterModel


def _log_gg_norm(beta: Tensor) -> Tensor:
    """GenGauss log-normalisation: log β − log 2 − lgamma(1/β)."""
    return torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)


class GenGaussLogScatter(ScatterModel):
    """Generalised-Gaussian density on ``eps = log S − log_mean``.

    .. math::
        \\log p(\\epsilon; \\alpha, \\beta)
            = -(|\\epsilon|/\\alpha)^{\\beta} + \\log\\beta - \\log 2\\alpha
              - \\log\\Gamma(1/\\beta)

    α_h, β_h are per-HKL and live on ``Parameters`` as ``log_alpha``,
    ``log_beta`` (log-parameterised for positivity).
    """

    def log_prob(
        self,
        eps: Tensor,              # (M, K)
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:
        h = meta.hkl_idx
        log_alpha = params.log_alpha[h]           # (M,)
        log_beta = params.log_beta[h]             # (M,)
        beta = log_beta.exp()
        alpha = log_alpha.exp()

        z = eps.abs() / alpha[:, None].clamp(min=1e-8)
        return (
            -z.clamp(min=1e-12).pow(beta[:, None])
            + _log_gg_norm(beta)[:, None]
            - log_alpha[:, None]
        )
