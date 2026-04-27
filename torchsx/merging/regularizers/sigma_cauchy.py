"""Half-Cauchy prior on the GenGauss width σ = α · c(β)."""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.parameters import Parameters
from torchsx.merging.regularizers.base import Regularizer


class SigmaCauchy(Regularizer):
    """Half-Cauchy shrinkage on the per-HKL multiplicative-scatter width.

    σ_h = α_h · c(β_h),   c(β) = √(Γ(3/β) / Γ(1/β))

    -log p_σ = log(τ² + σ²)  (proper Cauchy up to constants)

    Parameters
    ----------
    tau : "innocent" width below which the penalty is flat.  The old
        loss expressed this as ``sigma_cauchy_scale``.
    """

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.register_buffer("tau2", torch.tensor(float(tau) ** 2))

    def forward(self, params: Parameters) -> Tensor:
        beta = params.beta
        log_c_beta = 0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
        log_sigma = params.log_alpha + log_c_beta
        sigma2 = torch.exp(2.0 * log_sigma)
        return torch.log(self.tau2 + sigma2).sum()
