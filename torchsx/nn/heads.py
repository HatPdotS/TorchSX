"""Output heads: embedding → (log K, log w, β) per observation.

Three quantities per observation:

======  ==========================================================
log K   the scale — this is the ``ScaleModel.log_k`` contract
log w   scatter **width**, as the actual SD of the log-scatter
β       scatter **shape**, via a bounded sigmoid
======  ==========================================================

Why (w, β) and not (α, β).  ``Var(GenGauss) = α²·Γ(3/β)/Γ(1/β)``, so α and β
*both* set the width and compete during fitting.  Reparameterising on the SD
makes them nearly orthogonal: ``w`` carries all the width and ``β`` is then
identified only by the *shape* of the residual distribution, which is a
genuinely different signal.  α is derived for the kernel by
:func:`~torchsx.merging.scatter.gengauss.log_alpha_from_log_width`.

No explicit penalty on ``w`` is needed or wanted.  The GenGauss normaliser
already contains ``−log α``, so widening by a factor ``c`` costs exactly
``log c`` per observation — automatic, correctly scaled, no hyperparameter.
That is what makes this safe where a *free global* noise coefficient was not:
``Sdfac`` running to 4.75 in abismal, and torchsx's own
``GaussianPolynomialLikelihood`` converging to a non-physical ``g1 ≈ 14``, are
both one global number being forced to describe every observation when scale
determinacy varies >20× across crystals.  Here every quantity is predicted
per observation from features by a shared network, which is regularisation by
construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.scatter.gengauss import (
    log_alpha_from_log_width,
    log_e_s_gengauss,
)

__all__ = ["ScalePrediction", "ScaleHeads"]


@dataclass
class ScalePrediction:
    """Per-observation network outputs.

    ``log_alpha`` is derived from ``(log_w, beta)`` and is what the GenGauss
    kernel actually consumes; ``log_w`` is what you should *report*, because it
    is the SD and therefore comparable across β.
    """

    log_K: Tensor        # (n,)
    log_w: Tensor        # (n,)  log SD of the log-scatter
    beta: Tensor         # (n,)  shape
    log_alpha: Tensor    # (n,)  GenGauss α, derived from (w, β); the LOWER width when asymmetric
    log_e_s: Tensor      # (n,)  log E[exp(ε)] — the mean/median correction
    log_alpha_hi: Tensor | None = None   # (n,) upper-side width (two-width family), None = symmetric

    @property
    def log_mean_shift(self) -> Tensor:
        """Subtract from ``log I0 + log K`` to get the log-scatter centre.

        The merge target is a **linear** estimator, so it estimates the *mean*
        intensity; ``GenGaussLogScatter`` centres its density on ``log_mean``,
        so ``log_mean`` is a *median*.  They differ by ``E[exp(ε)]``, which is
        3.9x at w = 1.65 and 22.8x at w = 2.5 — and because it grows with the
        width it closes a loop: bigger w -> more biased target -> worse fit ->
        bigger w.
        """
        return self.log_e_s

    def detach(self) -> "ScalePrediction":
        return ScalePrediction(
            self.log_K.detach(), self.log_w.detach(),
            self.beta.detach(), self.log_alpha.detach(), self.log_e_s.detach(),
            None if self.log_alpha_hi is None else self.log_alpha_hi.detach(),
        )


class ScaleHeads(nn.Module):
    """Zero-initialised linear heads on the encoder embedding.

    All three heads have zero weights at construction, so the first forward
    pass returns exactly the bias values: ``K == 1``, ``w == w_init``,
    ``β == beta_init``, regardless of the encoder.  The first in-batch merge is
    then a unity merge.

    Parameters
    ----------
    w_init
        Initial scatter SD.  Default 1.65 — the measured median α on this data
        under the **Poisson, all-observations** fit.  Deliberately *not* 1.25,
        which is the log-space positives-only number: 32.5 % of observations
        here are negative, and any fit that takes logs discards them and
        underestimates the spread.
    beta_init
        Initial shape.  Default 2.0 = Gaussian in log space, i.e. a log-normal
        scale — the right neutral starting point.
    beta_min, beta_max
        Sigmoid bounds.  β → 1 is Laplace, β → 2 Gaussian, β < 1 heavier tails.
    predict_width, predict_shape
        Turn a head off to replace the per-observation prediction with **one
        learnable global scalar**, initialised at ``w_init`` / ``beta_init``.
        This does *not* freeze the value: over a 300-epoch run a "beta_init=1"
        global beta drifted to 2.16.  To actually pin it, set
        ``requires_grad_(False)`` on ``global_log_w`` / ``global_beta_raw``.
        Ablation switches — with both off this is a pure scale model.
    """

    def __init__(
        self,
        d_model: int,
        *,
        w_init: float = 1.65,
        beta_init: float = 2.0,
        beta_min: float = 0.3,
        beta_max: float = 4.0,
        predict_width: bool = True,
        predict_shape: bool = True,
        predict_skew: bool = False,
        global_skew: bool = False,
        unity_k: bool = False,
    ) -> None:
        super().__init__()
        if not (beta_min < beta_init < beta_max):
            raise ValueError(
                f"beta_init={beta_init} must lie strictly inside "
                f"({beta_min}, {beta_max})"
            )
        if w_init <= 0:
            raise ValueError(f"w_init must be positive, got {w_init}")

        self.beta_min = float(beta_min)
        self.beta_span = float(beta_max - beta_min)
        self.predict_width = bool(predict_width)
        self.predict_shape = bool(predict_shape)
        #: Two-width (asymmetric) scatter: a head predicts s = log(alpha_hi / alpha_lo)
        #: and the two widths are alpha * exp(-s/2), alpha * exp(+s/2), so log w
        #: stays the mean log width.  Zero-initialised: the symmetric model is the
        #: starting point.  Partiality scatter is one-sided (long lower tail, hard
        #: upper edge), which the symmetric family absorbs by mislocating its centre.
        self.predict_skew = bool(predict_skew)
        #: One global skew scalar instead of a head — the two-width family with
        #: no per-observation content.  Exclusive with ``predict_skew``.
        self.global_skew = bool(global_skew)
        if self.predict_skew and self.global_skew:
            raise ValueError("predict_skew and global_skew are exclusive")
        #: ``log K = 0`` for every observation, not a head.  With the width,
        #: shape and skew global too, nothing reads the embedding and the
        #: encoder can be skipped entirely (see ``needs_embedding``).
        self.unity_k = bool(unity_k)

        frac = (beta_init - beta_min) / (beta_max - beta_min)
        beta_raw_init = math.log(frac / (1.0 - frac))       # logit
        log_w_init = math.log(w_init)

        if not self.unity_k:
            self.head_log_K = self._zero_head(d_model, 0.0)
        if predict_width:
            self.head_log_w = self._zero_head(d_model, log_w_init)
        else:
            self.register_parameter(
                "global_log_w", nn.Parameter(torch.tensor(log_w_init))
            )
        if predict_shape:
            self.head_beta = self._zero_head(d_model, beta_raw_init)
        else:
            self.register_parameter(
                "global_beta_raw", nn.Parameter(torch.tensor(beta_raw_init))
            )
        if predict_skew:
            self.head_log_skew = self._zero_head(d_model, 0.0)
        elif global_skew:
            self.register_parameter("global_log_skew", nn.Parameter(torch.zeros(())))

    @staticmethod
    def _zero_head(d_model: int, bias: float) -> nn.Linear:
        head = nn.Linear(d_model, 1)
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, bias)
        return head

    @property
    def needs_embedding(self) -> bool:
        """False when every quantity is a global scalar or a fixed constant.

        The caller may then skip the feature build and the encoder outright and
        pass an ``(n, 0)`` placeholder — running a 5-layer Deep Sets over
        hundreds of millions of observations to produce three constants is pure
        waste, and on the global-fit path it is the whole cost.
        """
        return not (
            self.unity_k
            and not self.predict_width
            and not self.predict_shape
            and not self.predict_skew
        )

    def forward(self, z: Tensor) -> ScalePrediction:
        log_K = (
            z.new_zeros(z.shape[0]) if self.unity_k
            else self.head_log_K(z).squeeze(-1)
        )

        if self.predict_width:
            log_w = self.head_log_w(z).squeeze(-1)
        else:
            log_w = self.global_log_w.expand(z.shape[0])

        if self.predict_shape:
            beta_raw = self.head_beta(z).squeeze(-1)
        else:
            beta_raw = self.global_beta_raw.expand(z.shape[0])
        beta = self.beta_min + self.beta_span * torch.sigmoid(beta_raw)

        log_alpha = log_alpha_from_log_width(log_w, beta)
        if self.predict_skew or self.global_skew:
            s = (self.head_log_skew(z).squeeze(-1) if self.predict_skew
                 else self.global_log_skew.expand(z.shape[0]))
            half_s = 0.5 * s
            log_alpha_lo, log_alpha_hi = log_alpha - half_s, log_alpha + half_s
            return ScalePrediction(
                log_K=log_K, log_w=log_w, beta=beta, log_alpha=log_alpha_lo,
                log_e_s=log_e_s_gengauss(log_alpha_lo, beta, log_alpha_hi=log_alpha_hi),
                log_alpha_hi=log_alpha_hi,
            )
        return ScalePrediction(
            log_K=log_K,
            log_w=log_w,
            beta=beta,
            log_alpha=log_alpha,
            log_e_s=log_e_s_gengauss(log_alpha, beta),
        )
