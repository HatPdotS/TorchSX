"""Abstract base for latent-signal quadrature modules.

A ``Quadrature`` marginalises the latent signal ``S`` out of the joint
``p(n | S, bg) · p_scatter(log S | log_mean)`` per observation.  It owns
any grid-construction state (for example ``sig_lo_per_obs`` for a
softplus grid) and returns a scalar per observation from
:meth:`marginalize_signal`.
"""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter.base import ScatterModel


class Quadrature(nn.Module):
    """Signal-marginal integrator.

    Returns ``log ∫ p(n | S, bg) · p_scatter(log S − log_mean) · d(log S)``
    per observation, shape ``(n_obs,)``.  A bg-mixture likelihood wraps
    this result via ``DetectionLikelihood.combine_with_marginal``.
    """

    @abstractmethod
    def marginalize_signal(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,          # (n_obs,)
        n_ph: Tensor,              # (n_obs,) — peak photon counts (I_obs + bg)
        bg: Tensor,                # (n_obs,)
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:                   # (n_obs,)
        ...
