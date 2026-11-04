"""Output heads: embedding → ``log K`` and a Gaussian scatter mixture per observation.

Per observation, every quantity is a linear map of the embedding:

=======  ==================================================================
log K    the scale
log w    the width of component 0, as the SD of the log-scatter
mixture  ``K`` Gaussian components in log space: a weight each, and for
         components ``1 .. K-1`` an offset and lower and upper widths
         relative to component 0
=======  ==================================================================

A component may be wider below its centre than above it: the scatter of
partial observations is one-sided, a long lower tail and a hard upper edge.
Gaussian components have a finite mean at any width, so the mixture is centred
on its mean (``log_e_s``), which is what the linear merge estimates.  A wider
scatter costs ``log`` of its width through the Gaussian normaliser, so no
explicit penalty on the width is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.scatter.gengauss import log_alpha_from_log_width
from torchsx.merging.scatter.mixture import mixture_log_e_s, mixture_log_sd

__all__ = ["ScalePrediction", "ScaleHeads"]


@dataclass
class ScalePrediction:
    """Per-observation network outputs."""

    log_K: Tensor        # (n,)
    log_w: Tensor        # (n,)  log SD of the mixture's log-scatter
    log_e_s: Tensor      # (n,)  log E[exp(eps)], the mixture's mean shift
    #: (n, 5K) mixture parameters: normalised log weights, offsets, lower and
    #: upper log widths and log shape (2 for every component) per component
    #: (see ``torchsx.merging.scatter.mixture.split_mix``)
    mix: Tensor

    def detach(self) -> "ScalePrediction":
        return replace(self, log_K=self.log_K.detach(), log_w=self.log_w.detach(),
                       log_e_s=self.log_e_s.detach(), mix=self.mix.detach())


class ScaleHeads(nn.Module):
    """Zero-initialised linear heads on the encoder embedding.

    Every weight starts at zero, so the first forward returns the biases for
    every observation: ``K == 1``; component 0 at width ``w_init`` carrying
    most of the weight; the other components at small but non-negligible
    weights with distinct offsets and widths, so that their weight logits
    receive usable gradients from the start.  The first merge is then a unity
    merge.

    Parameters
    ----------
    w_init
        Initial width of component 0, the SD of the log-scatter.
    n_components
        Mixture components ``K``.
    """

    #: initial offsets and lower widths of components 1, 2, ..., cycled; the
    #: upper widths start narrower than component 0's by ``_HI_NARROW``
    _OFFSET = (-1.0, 0.3, -2.0, 0.6, -3.0, 0.9)
    _LO_WIDE = (0.5, -0.5, 0.8, -0.8, 1.0, -1.0)
    _HI_NARROW = (0.5, 0.5, 0.8, 0.8, 1.0, 1.0)

    def __init__(self, d_model: int, *, w_init: float = 1.65, n_components: int = 4) -> None:
        super().__init__()
        if w_init <= 0:
            raise ValueError(f"w_init must be positive, got {w_init}")
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = K = int(n_components)
        self.head_log_K = self._zero_head(d_model, torch.zeros(1))
        self.head_log_w = self._zero_head(d_model, torch.full((1,), math.log(w_init)))
        #: weight logits (K), then offsets, lower and upper width deltas of
        #: components 1 .. K-1
        self.mixture = self._zero_head(d_model, self.init_bias(K))

    @classmethod
    def init_bias(cls, n_components: int) -> Tensor:
        K = n_components
        rest = [(k - 1) % len(cls._OFFSET) for k in range(1, K)]
        logits = torch.tensor([0.0] + [-3.0] * (K - 1))
        offset = torch.tensor([cls._OFFSET[i] for i in rest])
        d_lo = torch.tensor([cls._LO_WIDE[i] for i in rest])
        d_hi = torch.tensor([-cls._HI_NARROW[i] for i in rest])
        return torch.cat([logits, offset, d_lo, d_hi])

    @staticmethod
    def _zero_head(d_model: int, bias: Tensor) -> nn.Linear:
        head = nn.Linear(d_model, bias.numel())
        nn.init.zeros_(head.weight)
        with torch.no_grad():
            head.bias.copy_(bias)
        return head

    def forward(self, z: Tensor) -> ScalePrediction:
        K = self.n_components
        log_K = self.head_log_K(z).squeeze(-1)
        log_w0 = self.head_log_w(z).squeeze(-1)
        raw = self.mixture(z)
        log_pi = torch.log_softmax(raw[:, :K], dim=1)
        zero = raw.new_zeros(raw.shape[0], 1)
        offset = torch.cat([zero, raw[:, K:2 * K - 1]], dim=1)
        d_lo = torch.cat([zero, raw[:, 2 * K - 1:3 * K - 2]], dim=1)
        d_hi = torch.cat([zero, raw[:, 3 * K - 2:]], dim=1)
        beta = torch.full_like(log_pi, 2.0)
        base = log_alpha_from_log_width(log_w0, beta[:, 0])[:, None]
        la_lo, la_hi = base + d_lo, base + d_hi
        mix = torch.cat([log_pi, offset, la_lo, la_hi, beta.log()], dim=1)
        return ScalePrediction(
            log_K=log_K,
            log_w=mixture_log_sd(log_pi, offset, la_lo, la_hi, beta, gaussian=True),
            log_e_s=mixture_log_e_s(log_pi, offset, la_lo, la_hi, beta, gaussian=True),
            mix=mix,
        )
