"""GenGauss scatter recentered so E[S] = 1.

Standard GenGauss (``GenGaussLogScatter``) puts the prior on log S
symmetric around 0, so ``median(S) = 1`` but ``E[S] ≈ 1.36`` for
β=1.7, α=1.  As a result the marginal MLE for log_I0 targets the
latent *median* intensity — which is correct asymptotically on
strong HKLs but suffers a finite-sample bias on weak HKLs where the
data can't pin down the median.

This recentered version shifts the GenGauss center to
``μ_ε(α,β) = −log E[exp(ε)]`` so that ``E[S] = 1`` exactly.  The MLE
for log_I0 then targets the *per-obs mean intensity* — what the linear
(no-scatter) estimator already estimates unbiasedly on every HKL,
weak or strong.  See
``experiments/synthetic_gengauss/compare_linear_vs_gengauss.py`` for
the empirical demonstration of why this matters.

E[S] is computed via a global numerical quadrature over ε once at
init time, given a single (α_ref, β_ref) for the dataset.  A
per-HKL E[S(α_h, β_h)] is the natural extension once α/β estimation
is stable.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter.base import ScatterModel


def _log_gg_norm(beta: Tensor) -> Tensor:
    return torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)


def compute_log_e_s(alpha: float, beta: float, n_grid: int = 200_001) -> float:
    """log E[exp(ε)] for ε ~ GenGauss(0, α, β) — fixed-grid numerical."""
    xs_np = torch.linspace(-15.0, 15.0, n_grid).numpy()
    import numpy as np
    dx = float(xs_np[1] - xs_np[0])
    norm = beta / (2.0 * alpha * math.gamma(1.0 / beta))
    p = norm * np.exp(-(np.abs(xs_np) / alpha) ** beta)
    p = p / (p.sum() * dx)
    return float(np.log((np.exp(xs_np) * p).sum() * dx))


class GenGaussLogScatterRecentered(ScatterModel):
    """GenGauss density on ε with the center shifted so E[exp(ε)] = 1.

    ``log p(ε; α, β) = -(|ε - μ_ε(α,β)|/α)^β + log β - log 2α - lgamma(1/β)``

    where ``μ_ε`` is chosen so that S = exp(ε) has unit mean.  For the
    per-HKL learnable α, β we use a single global μ_ε computed from a
    reference (α_ref, β_ref).  On the synthetic harness this is the
    truth; in production it's the median of the GG-MLE α, β.

    Parameters
    ----------
    alpha_ref, beta_ref : reference (α, β) used to compute μ_ε.  In
        practice these should be the dataset median; on synthetic
        data they should match the generative truth.
    """

    def __init__(self, alpha_ref: float = 1.0, beta_ref: float = 1.7) -> None:
        super().__init__()
        log_e_s = compute_log_e_s(alpha_ref, beta_ref)
        self.register_buffer("mu_eps", torch.tensor(-log_e_s))    # = -log E[S]
        self.alpha_ref = float(alpha_ref)
        self.beta_ref = float(beta_ref)

    def log_prob(
        self,
        eps: Tensor,              # (M, K)
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:
        h = meta.hkl_idx
        log_alpha = params.log_alpha[h]
        log_beta = params.log_beta[h]
        beta = log_beta.exp()
        alpha = log_alpha.exp()

        # Shifted residual: under the prior centered at μ_eps,
        #   log S = log_mean + ε  ⇒  ε is centered at μ_eps  ⇒
        # the density evaluation uses (ε − μ_eps).
        eps_centered = eps - self.mu_eps
        z = eps_centered.abs() / alpha[:, None].clamp(min=1e-8)
        return (
            -z.clamp(min=1e-12).pow(beta[:, None])
            + _log_gg_norm(beta)[:, None]
            - log_alpha[:, None]
        )
