"""Wilson log-prior on ``log_I0`` (MAP-style, per shell).

Ported from the legacy ``FrenchWilsonPrior`` in
``torchsx.merging_clean.priors`` (name simplified — the "French" part of
the classical French–Wilson name refers to the posterior I→F conversion,
which is a *separate* post-merging step, not this prior).
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.observations import PreparedObs
from torchsx.merging.priors.base import IntensityPrior


class WilsonPrior(IntensityPrior):
    """Wilson log-prior placed directly on ``φ = ½·log I0 = log F``.

    Starting from the standard Wilson distributions on F::

        acentric   F ~ Rayleigh(σ=√(Σ/2))       p_F(F) = (2F/Σ) exp(−F²/Σ)
        centric    F ~ half-Gaussian(σ=√Σ)      p_F(F) = √(2/πΣ) exp(−F²/2Σ)

    and doing the change of variables ``φ = log F`` once up front, the
    density on φ is ``p_φ(φ) = p_F(F)·F``.  Writing it on the model
    parameter ``log_I0 = 2φ`` and dropping constants::

        acentric:  −log p_φ = I/Σ + log Σ − log_I0
        centric:   −log p_φ = I/(2Σ) + ½·log Σ − ½·log_I0

    Both MAP at ``I = Σ`` — symmetric shrinkage toward the shell mean.
    Σ_h is recomputed every forward pass via ``scatter_add`` on
    ``exp(log_I0)``; the gradient back-coupling through Σ is O(1/N_h) ≪ 1
    and the prior behaves as EM in the many-HKL-per-shell limit.
    """

    def __init__(
        self,
        shell_id: Tensor,
        is_centric: Tensor,
        n_shells: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.register_buffer("shell_id", shell_id.long())
        self.register_buffer("is_centric", is_centric.bool())
        self.n_shells = int(n_shells)
        self.eps = float(eps)

    @classmethod
    def from_prep(cls, prep: PreparedObs, config: MergeConfig) -> "WilsonPrior":
        return cls(
            shell_id=prep.shell_id_fw,
            is_centric=prep.is_centric,
            n_shells=config.fw_n_shells,
            eps=config.fw_eps,
        )

    def shell_mean(self, log_I0: Tensor) -> Tensor:
        """Return per-HKL Σ_h = ⟨I⟩ over the HKL's shell, clamped to ``eps``."""
        I0 = torch.exp(log_I0)
        shell_sum = torch.zeros(
            self.n_shells, dtype=I0.dtype, device=I0.device
        ).scatter_add(0, self.shell_id, I0)
        shell_cnt = torch.bincount(self.shell_id, minlength=self.n_shells).to(I0.dtype)
        Sigma_per_shell = shell_sum / shell_cnt.clamp(min=1.0)
        return Sigma_per_shell[self.shell_id].clamp(min=self.eps)

    def forward(self, log_I0: Tensor) -> Tensor:
        """Per-HKL negative-log-prior vector of shape ``(n_valid,)``."""
        I0 = torch.exp(log_I0)
        Sigma = self.shell_mean(log_I0)
        J = I0 / Sigma
        log_Sigma = torch.log(Sigma)
        nlp_acen = J + log_Sigma - log_I0
        nlp_cent = 0.5 * J + 0.5 * log_Sigma - 0.5 * log_I0
        return torch.where(self.is_centric, nlp_cent, nlp_acen)
