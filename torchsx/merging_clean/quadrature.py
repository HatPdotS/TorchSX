"""Softplus quadrature grid for the Poisson marginal likelihood.

Adapted from `torchsx.merge._dstep.build_softplus_grid`.  The grid is
built once from the raw observations and is independent of all fitted
parameters (I0, α, β, corrections) — so it can be reused across every
closure call.

Layout:
    signal     (N_obs, K)  — signal at each quadrature point
    log_poi    (N_obs, K)  — log Poisson PMF  (I₀-independent)
    log_jac_dt (N_obs, K)  — log Jacobian of ε → t mapping
    log_dt     (N_obs,)    — log uniform t-spacing

The closure only needs `signal` (to compute ε = log(signal / (I0·scale)))
and the pre-summed `log_poi + log_jac_dt + log_dt[:, None]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class QuadratureGrid:
    signal: Tensor           # (N_obs, K)
    log_poi_plus_jac: Tensor  # (N_obs, K) = log_poi + log_jac_dt + log_dt[:, None]

    def to(self, device) -> "QuadratureGrid":
        return QuadratureGrid(
            signal=self.signal.to(device),
            log_poi_plus_jac=self.log_poi_plus_jac.to(device),
        )


def _softplus_inv(y: Tensor) -> Tensor:
    return torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(min=1e-8))))


def build_softplus_grid(
    n_ph: Tensor,
    bg: Tensor | float,
    *,
    n_quad: int = 51,
    n_sigma: float = 5.0,
    signal_min: float = 0.01,
    signal_lo_override: Tensor | None = None,
) -> QuadratureGrid:
    """Build a per-observation softplus quadrature grid.

    Parameters
    ----------
    n_ph : (N,) peak photon counts ≡ I_obs + bg_total, clamped ≥ 0
    bg   : (N,) total background in peak region, or scalar
    n_quad : number of quadrature points per observation
    n_sigma : upper-bound coverage in Poisson σ units
    signal_min : absolute lower bound on signal grid
    signal_lo_override : (N,) per-obs lower bound (usually 0.5·√(bg/mult))
    """
    dev = n_ph.device

    sigma = n_ph.clamp(min=1).sqrt()
    if isinstance(bg, (int, float)):
        signal_max = (n_ph - bg + n_sigma * sigma).clamp(min=1.0)
    else:
        signal_max = (n_ph - bg + n_sigma * sigma).clamp(min=1.0)

    if signal_lo_override is not None:
        signal_lo = signal_lo_override.clamp(min=signal_min)
        signal_lo = torch.min(signal_lo, signal_max * 0.5)
    else:
        signal_lo = (signal_max * 0.01).clamp(min=signal_min)

    t_hi = _softplus_inv(signal_max)
    t_lo = _softplus_inv(signal_lo)

    frac = torch.linspace(0, 1, n_quad, device=dev)
    dt = (t_hi - t_lo) / (n_quad - 1)
    t = t_lo[:, None] + (t_hi - t_lo)[:, None] * frac[None, :]

    signal = F.softplus(t)
    if isinstance(bg, (int, float)):
        lam = signal + bg
    else:
        lam = signal + bg[:, None]
    lam = lam.clamp(min=1e-6)

    log_nf = torch.lgamma(n_ph + 1)
    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_jac_dt = F.logsigmoid(t) - torch.log(signal)
    log_dt = torch.log(dt)

    log_poi_plus_jac = log_poi + log_jac_dt + log_dt[:, None]

    return QuadratureGrid(signal=signal, log_poi_plus_jac=log_poi_plus_jac)
