"""Abstract base for detection likelihoods."""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters


class DetectionLikelihood(nn.Module):
    """``p(n | signal, bg)`` and (optionally) a bg-mixture wrapper.

    Two hooks:

    * ``log_prob_given_signal(n, S, bg)`` — per-grid-point log likelihood.
      The Quadrature calls this inside the signal integral.
    * ``combine_with_marginal(n, bg, log_p_sig_marginal, params, meta)`` —
      post-marginal combining (e.g. mixing in a bg hypothesis).  Default
      implementation just returns the signal marginal unchanged.

    Plus :meth:`detection_params`, describing how the *stored* ``bg`` — an
    annulus-derived ESTIMATE, not a measurement — enters the detection model.
    """

    @abstractmethod
    def log_prob_given_signal(
        self, n: Tensor, S: Tensor, bg: Tensor, meta: ObsInfo | None = None,
    ) -> Tensor:  # (M, K)
        ...

    #: True when the background variance is per-observation (gathered by ``meta.obs_id``).
    has_bg_var: bool = False

    def detection_params(self, bg: Tensor, meta: ObsInfo | None = None) -> tuple[Tensor, Tensor | None]:
        """``(bg_mean, sig_b2)`` for the raw stored estimate ``bg``.

        ``meta`` (the batch's :class:`ObsInfo`) lets a per-observation variance
        gather by ``obs_id``.

        ``bg_mean`` is the background the detection mean uses; ``sig_b2`` is the
        estimate's own variance, or ``None`` when it is not modelled.  Identity
        and ``None`` by default, so a likelihood that does neither is untouched.

        This is **one** call returning both on purpose.  The two fast
        quadratures (``SplitLogGrid``, ``FixedLogGrid``) and their Triton
        kernels reimplement the Poisson core instead of calling
        :meth:`log_prob_given_signal`, so they have to ask for this mapping —
        and with two separate accessors the plausible-looking
        ``bg_variance(model_bg(bg))`` silently computes the variance at the
        shifted background.  Returning the pair makes that unrepresentable.

        Non-likelihood consumers of ``bg`` — the pooled ``(n − bg)/K``
        estimator, the encoder features, the bg gate, the σ²(bg) calibration —
        deliberately keep the raw value and must NOT route through this.
        """
        return bg, None

    def combine_with_marginal(
        self,
        n: Tensor,
        bg: Tensor,
        log_p_sig_marginal: Tensor,     # (M,)
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:                         # (M,)
        return log_p_sig_marginal
