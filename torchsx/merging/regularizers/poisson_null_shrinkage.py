"""Per-HKL shrinkage on (log α, log β) toward population values.

Shrinks weak/uninformative HKLs (those well-explained by pure Poisson
noise) toward the population (α_pop, β_pop), while leaving HKLs that
decisively reject the no-scatter null near their MLE.

Per-HKL weight ``w_h`` is precomputed once from a likelihood-ratio test:

    LR_h = NLL_pois,h − NLL_gg,h(α_pop, β_pop)
    w_h  = exp(−LR_h / τ)

where NLL_pois,h is the no-scatter Poisson NLL and NLL_gg,h is the
GenGauss-marginal NLL evaluated at the population values, both
aggregated over the HKL's observations. ``w_h ≈ 1`` when the data are
indistinguishable from pure Poisson; ``w_h → 0`` when the data demand
real scatter.

Use case: bias-correction of per-HKL (α, β) MLEs in low-SNR regimes
where the per-HKL likelihood is too flat to identify scatter shape.
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.parameters import Parameters
from torchsx.merging.regularizers.base import Regularizer


class PoissonNullShrinkage(Regularizer):
    """Quadratic shrinkage on per-HKL (log α, log β) toward population values.

    The penalty is

        Σ_h w_h · [ (log α_h − log α_pop)² + (log β_h − log β_pop)² ]

    where ``w_h`` is a precomputed per-HKL informativeness score in [0, 1].
    """

    def __init__(
        self,
        w_per_hkl: Tensor,
        log_alpha_pop: float,
        log_beta_pop: float,
    ) -> None:
        super().__init__()
        self.register_buffer("w", w_per_hkl.detach().clone())
        self.register_buffer(
            "log_alpha_pop", torch.tensor(float(log_alpha_pop))
        )
        self.register_buffer(
            "log_beta_pop", torch.tensor(float(log_beta_pop))
        )

    def forward(self, params: Parameters) -> Tensor:
        d_a = params.log_alpha - self.log_alpha_pop
        d_b = params.log_beta - self.log_beta_pop
        return (self.w * (d_a * d_a + d_b * d_b)).sum()


@torch.no_grad()
def compute_poisson_null_score(
    data_target,
    n_hkl: int,
    log_alpha_pop: float,
    log_beta_pop: float,
    *,
    tau: float | None = None,
) -> tuple[Tensor, float]:
    """Per-HKL informativeness score ``w_h = exp(−LR_h / τ)``.

    The two NLL passes share log_mean = log_I0 + log_k (current params)
    and bg; the GenGauss pass evaluates at fixed (α_pop, β_pop) by
    temporarily filling ``params.log_alpha`` / ``params.log_beta``.

    Parameters
    ----------
    data_target : DataNLLTarget
        The merging-loss data target (provides chunking, scale,
        scatter, quadrature, likelihood, and per-obs buffers).
    n_hkl : int
        Number of valid HKLs (size of ``params.log_I0``).
    log_alpha_pop, log_beta_pop : float
        Population (α, β) at which to evaluate the GenGauss-marginal NLL.
    tau : float, optional
        Score scale. If None, defaults to ``median(LR_h[LR_h > 0])``.

    Returns
    -------
    w : Tensor, shape (n_hkl,)
    tau_used : float
    """
    params = data_target.params
    device = params.log_I0.device
    dtype = params.log_I0.dtype
    nll_pois = torch.zeros(n_hkl, device=device, dtype=dtype)
    nll_gg = torch.zeros_like(nll_pois)

    saved_log_alpha = params.log_alpha.detach().clone()
    saved_log_beta = params.log_beta.detach().clone()
    try:
        params.log_alpha.fill_(float(log_alpha_pop))
        params.log_beta.fill_(float(log_beta_pop))

        for s, e in data_target._iter_chunks():
            meta = data_target._obs_info_slice(s, e)
            h = data_target.hkl_id[s:e]
            log_I0 = params.log_I0[h]
            log_k = data_target.scale.log_k(params, meta, obs_range=(s, e))
            log_mean = log_I0 + log_k
            if data_target.has_partiality:
                log_mean = log_mean + torch.log(
                    data_target.partiality[s:e].clamp(min=1e-12)
                )

            n_ph_slice = data_target.n_ph[s:e]
            bg_slice = data_target.bg[s:e]

            # Pure Poisson NLL (ε = 0)
            S = log_mean.exp()
            lam = (S + bg_slice).clamp(min=1e-6)
            log_p_pois = (
                n_ph_slice * torch.log(lam)
                - lam
                - torch.lgamma(n_ph_slice + 1)
            )
            nll_pois.scatter_add_(0, h, -log_p_pois)

            # GenGauss-marginal NLL at (α_pop, β_pop)
            log_p_sig = data_target.quadrature.marginalize_signal(
                data_target.scatter,
                data_target.likelihood,
                log_mean,
                n_ph_slice,
                bg_slice,
                params,
                meta,
                obs_range=(s, e),
            )
            log_p_gg = data_target.likelihood.combine_with_marginal(
                n_ph_slice, bg_slice, log_p_sig, params, meta,
            )
            nll_gg.scatter_add_(0, h, -log_p_gg)
    finally:
        params.log_alpha.copy_(saved_log_alpha)
        params.log_beta.copy_(saved_log_beta)

    LR = (nll_pois - nll_gg).clamp(min=0.0)
    if tau is None:
        nz = LR[LR > 0]
        tau_used = float(nz.median().item()) if nz.numel() > 0 else 1.0
    else:
        tau_used = float(tau)
    w = torch.exp(-LR / max(tau_used, 1e-12))
    return w, tau_used
