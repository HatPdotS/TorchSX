"""Distribution-match prior: KL of empirical log(I/Σ) against Wilson."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.observations import PreparedObs
from torchsx.merging.priors.base import IntensityPrior

# Euler–Mascheroni constant (= -E[log X] for X~Exp(1), and -γ-log2 for χ²_1).
_EULER_GAMMA = 0.5772156649015329


class WilsonMatchPrior(IntensityPrior):
    """Penalises deviation of the empirical distribution of ``log J = log(I/Σ)``
    from Wilson log-moments, per shell × centricity group.

    The per-group empirical is approximated by a Gaussian (max-entropy fit
    given two moments), and we penalise

    .. math::
        KL[\\, \\mathcal{N}(m_{emp}, v_{emp}) \\;\\|\\; \\mathcal{N}(m_*, v_*) \\,]
            = \\tfrac{1}{2}\\!\\left(\\log\\tfrac{v_*}{v_{emp}}
                + \\tfrac{v_{emp}}{v_*}
                + \\tfrac{(m_{emp} - m_*)^2}{v_*} - 1\\right)

    with standard Wilson targets:

        acentric  (I/Σ ~ Exp(1)):    m* = −γ,          v* = π²/6
        centric   (I/Σ ~ χ²₁):       m* = −γ − log 2,  v* = π²/2

    Minimising this KL preserves the *shape* of the Wilson distribution
    rather than collapsing all HKLs toward the mode.
    """

    _M_TARGET_ACEN: float = -_EULER_GAMMA
    _V_TARGET_ACEN: float = math.pi ** 2 / 6.0
    _M_TARGET_CENT: float = -_EULER_GAMMA - math.log(2.0)
    _V_TARGET_CENT: float = math.pi ** 2 / 2.0

    def __init__(
        self,
        shell_id: Tensor,
        is_centric: Tensor,
        n_shells: int,
        eps: float = 1e-6,
        min_count: int = 20,
    ) -> None:
        super().__init__()
        self.register_buffer("shell_id", shell_id.long())
        self.register_buffer("is_centric", is_centric.bool())
        self.n_shells = int(n_shells)
        self.eps = float(eps)
        self.min_count = int(min_count)
        group_id = self.shell_id * 2 + self.is_centric.long()
        self.register_buffer("group_id", group_id)
        self.n_groups = 2 * self.n_shells

    @classmethod
    def from_prep(cls, prep: PreparedObs, config: MergeConfig) -> "WilsonMatchPrior":
        return cls(
            shell_id=prep.shell_id_fw,
            is_centric=prep.is_centric,
            n_shells=config.fw_n_shells,
            eps=config.fw_eps,
            min_count=config.wilson_match_min_count,
        )

    def shell_mean(self, log_I0: Tensor) -> Tensor:
        """Per-HKL Σ_h (arithmetic mean of I over the HKL's shell)."""
        I0 = torch.exp(log_I0)
        shell_sum = torch.zeros(
            self.n_shells, dtype=I0.dtype, device=I0.device
        ).scatter_add(0, self.shell_id, I0)
        shell_cnt = torch.bincount(self.shell_id, minlength=self.n_shells).to(I0.dtype)
        Sigma_per_shell = shell_sum / shell_cnt.clamp(min=1.0)
        return Sigma_per_shell[self.shell_id].clamp(min=self.eps)

    def forward(self, log_I0: Tensor) -> Tensor:
        """Scalar weighted KL loss across shell × centricity groups."""
        device, dtype = log_I0.device, log_I0.dtype
        log_Sigma = torch.log(self.shell_mean(log_I0))
        log_J = log_I0 - log_Sigma

        ones = torch.ones_like(log_J)
        count = torch.zeros(self.n_groups, dtype=dtype, device=device).scatter_add(
            0, self.group_id, ones
        )
        mean_sum = torch.zeros(self.n_groups, dtype=dtype, device=device).scatter_add(
            0, self.group_id, log_J
        )
        safe_count = count.clamp(min=1.0)
        mean_per_group = mean_sum / safe_count
        dev = log_J - mean_per_group[self.group_id]
        var_sum = torch.zeros(self.n_groups, dtype=dtype, device=device).scatter_add(
            0, self.group_id, dev * dev
        )
        var_per_group = (var_sum / safe_count).clamp(min=self.eps)

        idx = torch.arange(self.n_groups, device=device)
        is_cent_group = (idx % 2 == 1)
        target_m = torch.where(
            is_cent_group,
            torch.tensor(self._M_TARGET_CENT, dtype=dtype, device=device),
            torch.tensor(self._M_TARGET_ACEN, dtype=dtype, device=device),
        )
        target_v = torch.where(
            is_cent_group,
            torch.tensor(self._V_TARGET_CENT, dtype=dtype, device=device),
            torch.tensor(self._V_TARGET_ACEN, dtype=dtype, device=device),
        )

        kl_per_group = 0.5 * (
            torch.log(target_v / var_per_group)
            + var_per_group / target_v
            + (mean_per_group - target_m) ** 2 / target_v
            - 1.0
        )

        valid = (count >= float(self.min_count)).to(dtype)
        return (count * kl_per_group * valid).sum()
