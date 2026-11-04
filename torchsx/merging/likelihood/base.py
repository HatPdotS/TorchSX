"""Abstract base for detection likelihoods."""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor


class DetectionLikelihood(nn.Module):
    """``p(n | signal, bg)``.

    :meth:`log_prob_given_signal` is the per-grid-point log likelihood the
    Quadrature calls inside the signal integral; :meth:`detection_params`
    describes how the *stored* ``bg`` — an annulus-derived ESTIMATE, not a
    measurement — enters the detection model.
    """

    @abstractmethod
    def log_prob_given_signal(
        self, n: Tensor, S: Tensor, bg: Tensor
    ) -> Tensor:  # (M, K)
        ...

    def detection_params(self, bg: Tensor) -> tuple[Tensor, Tensor | None]:
        """``(bg_mean, sig_b2)`` for the raw stored estimate ``bg``.

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
