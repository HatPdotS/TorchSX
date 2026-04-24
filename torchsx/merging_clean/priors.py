"""Prior modules for the unified merger.

Each prior is an ``nn.Module`` that owns its buffers, reads config, and
returns a scalar penalty (or per-HKL vector) from its ``forward`` call.
``UnifiedMergeLoss.regularizers_only`` composes them with the data-size
scale factor and the per-prior λ.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging_clean.config import MergeConfig
from torchsx.merging_clean.observations import PreparedObs

# Euler–Mascheroni constant (= -E[log X] for X~Exp(1), and -γ-log2 for χ²_1).
_EULER_GAMMA = 0.5772156649015329


class FrenchWilsonPrior(nn.Module):
    """Wilson log-prior placed directly on ``φ = ½·log I0 = log F``.

    Starting from the standard Wilson distributions on F,
        acentric   F ~ Rayleigh(σ=√(Σ/2))      p_F(F) = (2F/Σ) exp(−F²/Σ)
        centric    F ~ half-Gaussian(σ=√Σ)     p_F(F) = √(2/πΣ) exp(−F²/2Σ)
    and doing the change of variables ``φ = log F`` once up front,
    the corresponding density on φ is ``p_φ(φ) = p_F(F)·F``.  Writing it
    on the model parameter ``log_I0 = 2φ`` and dropping constants:

        acentric:  −log p_φ = I/Σ + log Σ − log_I0
        centric:   −log p_φ = I/(2Σ) + ½·log Σ − ½·log_I0

    Gradients w.r.t. ``log_I0``:
        acentric:  I/Σ − 1           MAP at I = Σ
        centric:   I/(2Σ) − ½        MAP at I = Σ

    Both types shrink toward the per-shell mean Σ: weak HKLs pulled *up*
    toward shell-typical, bright HKLs pulled *down* toward shell-typical.
    No unbounded I→0 pull (which was the I-space acentric failure mode) and
    no linear-in-I bright-end compression (which was the F-space-without-
    Jacobian failure mode).

    Σ_h is recomputed each forward pass via ``scatter_add`` on
    ``exp(log_I0)`` and gradients flow through Σ without detachment (with
    O(n_HKL_per_shell) ≫ 1 the self-coupling is negligible next to the
    direct I/Σ pull, so the prior behaves as EM in the large-shell limit).
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
    def from_prep(cls, prep: PreparedObs, config: MergeConfig) -> "FrenchWilsonPrior":
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
        """Per-HKL negative-log-prior vector, shape ``(n_valid,)``.

        Wilson priors expressed directly on ``φ = ½·log_I0`` (constants dropped):
            acentric:  −log p_φ = I/Σ + log Σ − log_I0
            centric:   −log p_φ = I/(2Σ) + ½·log Σ − ½·log_I0

        Both MAP at ``I = Σ`` — symmetric shrinkage toward the shell mean.
        """
        I0 = torch.exp(log_I0)
        Sigma = self.shell_mean(log_I0)
        J = I0 / Sigma
        log_Sigma = torch.log(Sigma)
        nlp_acen = J + log_Sigma - log_I0
        nlp_cent = 0.5 * J + 0.5 * log_Sigma - 0.5 * log_I0
        return torch.where(self.is_centric, nlp_cent, nlp_acen)


class WilsonMatchPrior(nn.Module):
    """Distribution-match prior: KL of empirical log(I/Σ) against Wilson.

    Instead of pulling every HKL toward a mode (as MAP-style priors do), this
    penalises deviation of the **empirical distribution** of ``log J = log(I/Σ)``
    from the known Wilson log-moments, per shell × centricity group.

    The per-group empirical is approximated by a Gaussian (maximum-entropy fit
    given two moments), and we penalise

        KL[ N(m_emp, v_emp)  ||  N(m_target, v_target) ]
            = ½·( log(v_target / v_emp) + v_emp / v_target
                   + (m_emp − m_target)² / v_target − 1 )

    where targets come from standard Wilson statistics:

        acentric  (I/Σ ~ Exp(1)):    m* = −γ,          v* = π²/6
        centric   (I/Σ ~ χ²₁):       m* = −γ − log 2, v* = π²/2

    Minimising this KL makes the shell's spread and mean of ``log I`` match
    Wilson — it preserves the *shape* of the distribution rather than
    collapsing all HKLs toward the mode.
    """

    # Cache targets as constants; torch conversion happens in forward().
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
        # Per-HKL group id: shell * 2 + centricity (0 acentric, 1 centric).
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
        """Return scalar weighted KL loss across shell × centricity groups."""
        device, dtype = log_I0.device, log_I0.dtype
        log_Sigma = torch.log(self.shell_mean(log_I0))
        log_J = log_I0 - log_Sigma  # (n_valid,)

        # Vectorised per-group mean/variance via scatter_add.
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

        # Target vectors, indexed by group (even=acentric, odd=centric).
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

        # KL[N(m_emp, v_emp) || N(m_t, v_t)].
        kl_per_group = 0.5 * (
            torch.log(target_v / var_per_group)
            + var_per_group / target_v
            + (mean_per_group - target_m) ** 2 / target_v
            - 1.0
        )

        # Weight by group size (treat KL as a rate, total ∝ samples) and
        # mask out groups with too few samples for a reliable estimate.
        valid = (count >= float(self.min_count)).to(dtype)
        return (count * kl_per_group * valid).sum()
