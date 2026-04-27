"""Poisson foreground with a background-hypothesis mixture.

Replaces the ``bg_mode ∈ {global, per_hkl}`` branches in the old
``UnifiedMergeLoss.data_nll``.  The ``off`` branch corresponds to using
``Poisson`` without a mixture.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from torchsx.merging.likelihood.poisson import Poisson
from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters

BgMode = Literal["global", "global_learned", "per_hkl"]


class PoissonWithBackgroundMixture(Poisson):
    """Add a background hypothesis ``log p(n | bg)`` and mix with the signal
    marginal.

    Three mix modes:

    * ``"global"`` — 50/50 mixture (implicit weight 1 on each term), same
      as ``logaddexp(log_p_sig, log_p_bg)`` with no learned weights.
    * ``"global_learned"`` — single learnable scalar ``π = sigmoid(pi_logit_global)``
      shared across ALL observations.  LBFGS finds the global noise fraction
      consistent with the data.  Per-obs responsibility γ_obs = π·p_bg /
      (π·p_bg + (1−π)·p_sig) is computed analytically from the likelihood
      ratio — bright HKLs commit to signal regardless of π, dark to bg.
    * ``"per_hkl"`` — learn a per-HKL Bernoulli weight ``π_h`` from
      ``params.pi_logit``; p(obs) = (1−π_h)·p_sig_marginal + π_h·p_bg.
      Dark HKLs commit π_h → 1 (data gradient on I₀ vanishes); bright
      HKLs commit π_h → 0 (GenGauss scatter unperturbed).

    Also includes an opt-in Beta(a, b) prior on π_h that the Merger can
    register as a separate regulariser target.
    """

    def __init__(self, mode: BgMode = "global", pi_init: float = 0.1) -> None:
        super().__init__()
        if mode not in ("global", "global_learned", "per_hkl"):
            raise ValueError(f"Unknown bg mode: {mode!r}")
        self.mode = mode
        # Single global learnable π; only used when mode="global_learned".
        # Initialised at logit(pi_init) so the prior expects ~10% noise.
        if mode == "global_learned":
            init_logit = float(torch.logit(torch.tensor(pi_init)).item())
            self.pi_logit_global = torch.nn.Parameter(
                torch.tensor(init_logit, dtype=torch.float32)
            )

    def combine_with_marginal(
        self,
        n: Tensor,
        bg: Tensor,
        log_p_sig_marginal: Tensor,
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:
        bg_safe = bg.clamp(min=1e-12)
        log_p_bg = n * torch.log(bg_safe) - bg_safe - torch.lgamma(n + 1)

        if self.mode == "global":
            return torch.logaddexp(log_p_sig_marginal, log_p_bg)

        if self.mode == "global_learned":
            log_pi = F.logsigmoid(self.pi_logit_global)
            log_1mpi = F.logsigmoid(-self.pi_logit_global)
            return torch.logaddexp(
                log_1mpi + log_p_sig_marginal, log_pi + log_p_bg
            )

        # per_hkl
        pi_logit = params.pi_logit[meta.hkl_idx]
        log_pi = F.logsigmoid(pi_logit)
        log_1mpi = F.logsigmoid(-pi_logit)
        return torch.logaddexp(log_1mpi + log_p_sig_marginal, log_pi + log_p_bg)


class PiBetaPrior(torch.nn.Module):
    """Beta(a, b) prior on the learned per-HKL ``π_h`` weight.

    Useful only with ``PoissonWithBackgroundMixture(mode="per_hkl")``.
    Register it as a regulariser target (weight = lam_pi_beta).
    """

    def __init__(self, a: float = 1.0, b: float = 2.0) -> None:
        super().__init__()
        self.a_m1 = float(a) - 1.0
        self.b_m1 = float(b) - 1.0

    def forward(self, params: Parameters) -> Tensor:
        log_pi = F.logsigmoid(params.pi_logit)
        log_1mpi = F.logsigmoid(-params.pi_logit)
        return (-self.a_m1 * log_pi - self.b_m1 * log_1mpi).sum()
