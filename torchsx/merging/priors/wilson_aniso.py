"""Wilson prior with continuous log_I_mean + anisotropic resolution decay.

Replaces the per-shell discrete Σ in :class:`WilsonPrior` with a single
``log_I_mean`` plus a 6-parameter ADP-like tensor ``U`` so that

    log Σ_h  =  log_I_mean  −  2π² · sᵀ · U · s

with ``s = hkl @ B_orth`` (Å⁻¹) and ``B_orth`` the reciprocal-basis matrix
of the dataset's median cell.  ``U`` is stored as a 6-vector
``[u11, u22, u33, u12, u13, u23]`` matching torchref's convention
(:func:`torchref.base.math_torch.U_to_matrix`); the same exponent clamp
and L2 ridge as in ``ScalerBase.anisotropy_correction`` keep the decay
physical without enforcing positive definiteness.

The Wilson NLP form (centric / acentric branches) is unchanged from the
shell-Wilson prior; only Σ_h is replaced.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from torchref.base.math_torch import U_to_matrix
from torchref.base.reciprocal.basis import reciprocal_basis_matrix

from torchsx.merging.config import MergeConfig
from torchsx.merging.observations import PreparedObs
from torchsx.merging.priors.base import IntensityPrior


class WilsonAnisoPrior(IntensityPrior):
    """Wilson MAP prior with one global log_I_mean and an anisotropic decay.

    Parameters
    ----------
    s_vec
        Reciprocal-space vectors per HKL, shape ``(n_valid, 3)``, in Å⁻¹.
        Computed from ``hkl_valid @ reciprocal_basis_matrix(median_cell)``.
    is_centric
        Boolean per-HKL centricity flag, shape ``(n_valid,)``.
    log_I_mean_init
        Initial log mean intensity (scalar).
    U_init
        Initial 6-vector for the anisotropic tensor in
        ``[u11, u22, u33, u12, u13, u23]`` order.
    lam_U_ridge
        Strength of the ``||U||²`` ridge added inside ``forward``.
    """

    def __init__(
        self,
        s_vec: Tensor,
        is_centric: Tensor,
        *,
        log_I_mean_init: float = 0.0,
        U_init: Tensor | None = None,
        lam_U_ridge: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("s_vec", s_vec.float())
        self.register_buffer("is_centric", is_centric.bool())
        self.log_I_mean = nn.Parameter(torch.tensor(float(log_I_mean_init)))
        if U_init is None:
            U_init = 0.001 * torch.randn(6)
        self.U = nn.Parameter(U_init.detach().clone().float())
        self.lam_U_ridge = float(lam_U_ridge)

    @classmethod
    def from_prep(cls, prep: PreparedObs, config: MergeConfig) -> "WilsonAnisoPrior":
        median_cell = prep.median_cell.detach().cpu().float()
        B_orth = reciprocal_basis_matrix(median_cell)
        s_vec = prep.hkl_valid.float().cpu() @ B_orth.float()

        log_I_mean_init = 0.0
        U_init = None

        if config.wilson_aniso_init_from_data:
            mu = prep.mu_init.detach().cpu().float()
            s2 = (s_vec * s_vec).sum(-1)
            # Wilson plot: mu ≈ a − 2·B_iso·s²; design [1, −2·s²].
            try:
                design = torch.stack([torch.ones_like(s2), -2.0 * s2], dim=-1)
                sol = torch.linalg.lstsq(design, mu.unsqueeze(-1)).solution.squeeze(-1)
                a, B_iso = float(sol[0].item()), float(sol[1].item())
                B_iso = max(B_iso, 0.0)
                log_I_mean_init = a
                u_iso = B_iso / (2.0 * math.pi ** 2)
                U_init = torch.tensor([u_iso, u_iso, u_iso, 0.0, 0.0, 0.0])
            except Exception:
                pass

        return cls(
            s_vec=s_vec,
            is_centric=prep.is_centric,
            log_I_mean_init=log_I_mean_init,
            U_init=U_init,
            lam_U_ridge=config.lam_wilson_aniso_U_ridge,
        )

    def log_sigma(self) -> Tensor:
        """Per-HKL log Σ_h = log_I_mean − 2π² · sᵀUs (clamped)."""
        U_mat = U_to_matrix(self.U)
        sU = self.s_vec @ U_mat  # (n_valid, 3)
        exp_arg = (-2.0 * math.pi ** 2 * (sU * self.s_vec).sum(-1)).clamp(
            min=-10.0, max=10.0
        )
        return self.log_I_mean + exp_arg

    def forward(self, log_I0: Tensor) -> Tensor:
        log_Sigma = self.log_sigma()
        J = torch.exp(log_I0 - log_Sigma)
        nlp_acen = J + log_Sigma - log_I0
        nlp_cent = 0.5 * J + 0.5 * log_Sigma - 0.5 * log_I0
        nlp = torch.where(self.is_centric, nlp_cent, nlp_acen)

        # ||U||² ridge spread over n_valid so it adds to the per-HKL sum at the
        # same scale as the data-side NLP. (IntensityPriorTarget calls .sum().)
        n_valid = log_I0.shape[0]
        ridge = self.lam_U_ridge * (self.U.pow(2).sum()) / max(n_valid, 1)
        return nlp + ridge
