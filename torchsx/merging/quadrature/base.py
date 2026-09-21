"""Abstract base for latent-signal quadrature modules.

A ``Quadrature`` marginalises the latent signal ``S`` out of the joint
``p(n | S, bg) · p_scatter(log S | log_mean)`` per observation.  It owns
any grid-construction state (for example ``sig_lo_per_obs`` for a
softplus grid) and returns a scalar per observation from
:meth:`marginalize_signal`.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
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


    def marginalize_mixture(
        self, scatter, likelihood, log_mean, n_ph, bg, params, meta, obs_range=None,
    ) -> Tensor:
        """Weighted component marginals; quadratures may fuse this operation."""
        terms = [
            log_pi + self.marginalize_signal(
                view, likelihood, log_mean + offset, n_ph, bg, params, meta,
                obs_range=obs_range,
            )
            for log_pi, offset, view in scatter.components(meta)
        ]
        return torch.logsumexp(torch.stack(terms, dim=0), dim=0)

    def _reject_unmodelled_bg(
        self, likelihood: DetectionLikelihood, *, variance: bool = False,
    ) -> None:
        """Refuse a detection background this quadrature does not implement.

        For the handful of quadratures that reimplement the plain-Poisson
        integrand instead of calling ``log_prob_given_signal``: silently
        computing the uncorrected quantity is the failure mode that a merge
        would never surface, so they say so instead.
        """
        if float(getattr(likelihood, "bg_mean_offset", 0.0) or 0.0):
            raise NotImplementedError(
                f"{type(self).__name__} hardcodes the plain-Poisson integrand "
                "and would silently ignore the likelihood's bg_mean_offset")
        if variance and (float(getattr(likelihood, "bg_var_factor", 0.0) or 0.0)
                         or getattr(likelihood, "has_bg_var", False)):
            raise NotImplementedError(
                f"{type(self).__name__} hardcodes the plain-Poisson integrand "
                "and would silently ignore the likelihood's background variance")

    def log_miss(
        self, likelihood: DetectionLikelihood, n_ph: Tensor, bg: Tensor,
        meta: ObsInfo | None = None,
    ) -> Tensor:                   # (n_obs,)
        """``log p(n | S = 0)`` under the same detection model as the marginal.

        The miss component of :class:`MarginalNLL` needs the S = 0 limit of the
        very integrand this quadrature integrates, so it asks the quadrature
        rather than re-deriving it: a subclass that already computes that value
        (``SplitLogGrid``'s analytic tail carries it as ``log_L0``) returns the
        cached one, and the two can then never drift apart.
        """
        from torchsx.merging.likelihood.poisson import (
            poisson_log_prob_at_zero_signal,
        )
        bg_mean, sig_b2 = likelihood.detection_params(bg, meta)
        return poisson_log_prob_at_zero_signal(n_ph, bg_mean, sig_b2)

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
