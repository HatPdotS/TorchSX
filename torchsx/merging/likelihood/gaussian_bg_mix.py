"""Gaussian-polynomial foreground with per-obs background-hypothesis mixture.

For each observation we mix the signal-marginal likelihood
``log_p_sig_marginal`` returned by the quadrature with a pure-background
Gaussian, weighted by ``q(|r|) = sigmoid(q_a + q_b · |r| · 1e3)``:

    log p(n) = logaddexp(log(1 − q) + log_p_sig_marginal,
                          log( q ) + log_p_bg)

where ``log_p_bg`` is the same Gaussian likelihood evaluated at S = 0
(so the predicted mean is ``bg``). The scalars ``q_a, q_b`` live on
``Parameters`` and are co-fit by LBFGS together with ``log_I0`` and the
GenGauss exponents — gradients flow through ``logaddexp`` and the
``F.logsigmoid`` of the logistic.

|r| is taken from ``meta.ewald_offset`` (populated by
``prepare_observations``). The ``* 1e3`` rescaling brings it from native
``Å⁻¹`` into ``×10⁻³ Å⁻¹`` so ``q_b`` is an order-unity number — matches
the scale used elsewhere in the diagnostics.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torchsx.merging.likelihood.gaussian_polynomial import GaussianPolynomialVariance
from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters


class GaussianPolynomialWithBgMixture(GaussianPolynomialVariance):
    """Gaussian-polynomial likelihood with a learnable, per-obs |r|-driven
    background-hypothesis mixture.

    Inherits the polynomial σ² model from :class:`GaussianPolynomialVariance`
    (including the trainable mode); overrides only ``combine_with_marginal``.
    """

    def combine_with_marginal(
        self,
        n: Tensor,
        bg: Tensor,
        log_p_sig_marginal: Tensor,
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:
        if meta.ewald_offset is None:
            raise RuntimeError(
                "GaussianPolynomialWithBgMixture needs meta.ewald_offset; "
                "check DataNLLTarget plumbing (data.py:_obs_info_slice)."
            )
        # |r| in ×10⁻³ Å⁻¹ — keeps q_b order-unity.
        abs_r = meta.ewald_offset.abs() * 1e3
        z = params.q_a + params.q_b * abs_r
        log_q = F.logsigmoid(z)         # log q(|r|)
        log_1mq = F.logsigmoid(-z)      # log(1 − q(|r|))

        # σ² from the same polynomial used by the signal branch.
        var = self.sigma_squared_obs(n, bg).clamp(min=self.var_floor)
        # Background-only Gaussian: μ = bg → residual is n − bg.
        log_p_bg = -0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(var)
            + (n - bg) ** 2 / var
        )

        return torch.logaddexp(
            log_1mq + log_p_sig_marginal,
            log_q + log_p_bg,
        )
