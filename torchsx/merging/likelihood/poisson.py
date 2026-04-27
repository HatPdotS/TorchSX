"""Plain Poisson detection likelihood (no bg mixture)."""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood


class Poisson(DetectionLikelihood):
    """``n ~ Poisson(S + bg)``.

    Per-grid-point log-PMF is computed without the factorial term — the
    constant ``−log Γ(n+1)`` cancels in the logsumexp over the signal
    grid if and only if it's the same for every grid point (which it is,
    because ``n`` doesn't depend on the grid index); we keep it anyway
    because it costs nothing and keeps the output interpretable as a
    true log-probability.
    """

    def log_prob_given_signal(self, n: Tensor, S: Tensor, bg: Tensor) -> Tensor:
        lam = S + bg[:, None]
        lam = lam.clamp(min=1e-6)
        log_nf = torch.lgamma(n + 1)
        return n[:, None] * torch.log(lam) - lam - log_nf[:, None]
