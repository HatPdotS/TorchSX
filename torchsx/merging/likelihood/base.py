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
    """

    @abstractmethod
    def log_prob_given_signal(
        self, n: Tensor, S: Tensor, bg: Tensor
    ) -> Tensor:  # (M, K)
        ...

    def combine_with_marginal(
        self,
        n: Tensor,
        bg: Tensor,
        log_p_sig_marginal: Tensor,     # (M,)
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:                         # (M,)
        return log_p_sig_marginal
