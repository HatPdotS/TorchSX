"""Trapezoidal quadrature on a softplus-reparameterised signal grid.

Drop-in replacement for ``merging_clean.quadrature.build_softplus_grid`` +
the inline ``logsumexp`` in ``data_nll``: same grid construction, same
integrand, same numerics.  The grid is rebuilt per forward pass (per chunk
when called from ``DataNLLTarget._chunk_loss``) because it depends only on
``n_ph`` and ``bg``, not on any learned parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.scatter.base import ScatterModel


def _softplus_inv(y: Tensor) -> Tensor:
    return torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(min=1e-8))))


def _compute_sig_lo_per_obs(
    n_valid: int,
    hkl_id: Tensor,
    bg: Tensor,
    obs_counts: Tensor,
    override: float | None,
) -> Tensor:
    """Per-observation signal-floor.  If ``override`` is not None, return a
    constant tensor; otherwise 0.5·√(⟨bg⟩_h / N_h)."""
    if override is not None:
        return torch.full_like(bg, float(override))
    bg_sum = torch.zeros(n_valid, device=bg.device).scatter_add_(
        0, hkl_id, bg.clamp(min=0)
    )
    bg_per_hkl = (bg_sum / obs_counts.float().clamp(min=1)).clamp(min=1.0)
    sigma_floor = (bg_per_hkl / obs_counts.float().clamp(min=1)).sqrt()
    sig_lo_hkl = (sigma_floor * 0.5).clamp(min=0.01)
    return sig_lo_hkl[hkl_id]


class TrapezoidalLogGrid(Quadrature):
    """K-point log-space trapezoidal quadrature on softplus-reparameterised S.

    Per-observation range::

        signal_max = clamp(n_ph − bg + n_sigma · √n_ph, ≥ 1)
        signal_lo  = sig_lo_per_obs   (clamped ≤ signal_max/2 and ≥ signal_min)

    The softplus reparameterisation ``S = softplus(t)`` smooths the lower
    bound at ``S → 0`` (avoids a log-singularity); the Jacobian ``σ(t)/S``
    is added to each grid-point log-weight.

    Parameters
    ----------
    n_quad : number of quadrature points per obs (default 51).
    n_sigma : upper-bound coverage in √(n_ph) units (default 5).
    signal_min : absolute lower bound on the signal grid.
    sig_lo_per_obs : precomputed per-obs lower bound (registered as buffer);
        a ``None`` value means compute on the fly from ``n_ph`` + ``bg``.
    """

    def __init__(
        self,
        n_quad: int = 51,
        n_sigma: float = 5.0,
        signal_min: float = 0.01,
        sig_lo_per_obs: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_quad = int(n_quad)
        self.n_sigma = float(n_sigma)
        self.signal_min = float(signal_min)
        if sig_lo_per_obs is None:
            self.register_buffer("sig_lo_per_obs", torch.empty(0))
            self._has_sig_lo = False
        else:
            self.register_buffer("sig_lo_per_obs", sig_lo_per_obs.clone())
            self._has_sig_lo = True

    @classmethod
    def from_prep(cls, prep: PreparedObs, config: MergeConfig) -> "TrapezoidalLogGrid":
        sig_lo = _compute_sig_lo_per_obs(
            n_valid=prep.n_valid,
            hkl_id=prep.hkl_id,
            bg=prep.bg,
            obs_counts=prep.obs_counts,
            override=config.sig_lo_override,
        )
        return cls(
            n_quad=config.n_quad,
            n_sigma=config.n_sigma_quad,
            signal_min=0.01,
            sig_lo_per_obs=sig_lo,
        )

    def _sig_lo_slice(
        self, n_ph: Tensor, bg: Tensor, obs_range: tuple[int, int] | None
    ) -> Tensor:
        if self._has_sig_lo:
            if obs_range is None:
                return self.sig_lo_per_obs
            s, e = obs_range
            return self.sig_lo_per_obs[s:e]
        # No precomputed buffer: signal floor from n_ph/bg alone.
        return (n_ph - bg).clamp(min=self.signal_min)

    def _build_grid(
        self, n_ph: Tensor, bg: Tensor, sig_lo_per_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns ``(signal, log_jac_plus_dt)`` each of shape ``(M, K)``."""
        sigma = n_ph.clamp(min=1).sqrt()
        signal_max = (n_ph - bg + self.n_sigma * sigma).clamp(min=1.0)

        signal_lo = sig_lo_per_obs.clamp(min=self.signal_min)
        signal_lo = torch.min(signal_lo, signal_max * 0.5)

        t_hi = _softplus_inv(signal_max)
        t_lo = _softplus_inv(signal_lo)

        frac = torch.linspace(0.0, 1.0, self.n_quad, device=n_ph.device)
        dt = (t_hi - t_lo) / (self.n_quad - 1)
        t = t_lo[:, None] + (t_hi - t_lo)[:, None] * frac[None, :]

        signal = F.softplus(t)
        log_jac_dt = F.logsigmoid(t) - torch.log(signal.clamp(min=1e-12))
        log_dt = torch.log(dt)
        log_jac_plus_dt = log_jac_dt + log_dt[:, None]
        return signal, log_jac_plus_dt

    def marginalize_signal(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,
        n_ph: Tensor,
        bg: Tensor,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        sig_lo_per_obs = self._sig_lo_slice(n_ph, bg, obs_range)
        signal, log_jac_plus_dt = self._build_grid(n_ph, bg, sig_lo_per_obs)
        log_signal = torch.log(signal.clamp(min=1e-12))
        eps = log_signal - log_mean[:, None]

        log_scatter = scatter.log_prob(eps, params, meta)         # (M, K)
        log_lik = likelihood.log_prob_given_signal(n_ph, signal, bg)  # (M, K)
        return torch.logsumexp(log_scatter + log_lik + log_jac_plus_dt, dim=1)
