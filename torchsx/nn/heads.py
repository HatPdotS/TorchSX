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
from torchsx.merging.scatter.mixture import mixture_log_e_s, mixture_log_sd

__all__ = ["ScalePrediction", "ScaleHeads", "MixtureConditioner"]


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
    #: (n, 5K) mixture parameters when the scatter family is a mixture:
    #: normalised log weights, offsets, lower and upper log widths and log shape per component
    #: (see ``torchsx.merging.scatter.mixture.split_mix``); None = single family
    mix: Tensor | None = None

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
            None if self.mix is None else self.mix.detach(),
        )


class MixtureConditioner(nn.Module):
    """Mixture parameters from the predicted scatter scalars.

    A small network reads the per-observation ``(log w, beta, s)`` the heads
    already produce and returns, per component, a weight logit, an offset in
    log space and lower/upper log-width deltas relative to the base two-width
    component.  Component 0 is the base family itself (offset and deltas fixed
    at zero); the other components may be wider or narrower below the centre
    but only narrower above it, because the mean multiplicative factor of a
    heavy-tailed component that is wide on the upper side is not finite, and
    the mixture is centred on that mean.  The output layer is zero-initialised,
    so the first forward returns the biases: component 0 carries most of the
    weight and the others start at a small but non-negligible weight with
    distinct offsets and widths, large enough that their weight logits receive
    usable gradients from the start.
    """

    SHAPES = ("shared", "gaussian", "free", "partiality")

    def __init__(self, n_components: int, hidden: int = 8, *, log_w_ref: float = 0.5,
                 shape: str = "shared") -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if shape not in self.SHAPES:
            raise ValueError(f"shape must be one of {self.SHAPES}, got {shape!r}")
        self.n_components = int(n_components)
        self.hidden = int(hidden)
        #: Component shapes: ``"shared"`` -- every component takes the base
        #: beta; ``"gaussian"`` -- every component is Gaussian in log space
        #: (beta = 2), so the mixture is smooth and its mean is finite for any
        #: widths; ``"free"`` -- each component has its own beta, offset from
        #: the base beta on the logit scale; ``"partiality"`` -- eps = log p + eta
        #: with log p on a fixed grid of K support points from 0 downward (a hard
        #: edge at p = 1) whose extent is predicted, every component sharing the
        #: base width as the Gaussian residual eta, and the weights describing
        #: the distribution of the partial fraction.
        self.shape = shape
        self.register_buffer("_grid", torch.linspace(0.0, 1.0, self.n_components))
        self.register_buffer("_in_loc", torch.tensor([log_w_ref, 2.0, 0.0]))
        self.register_buffer("_in_scale", torch.tensor([0.5, 1.0, 0.5]))
        self.l1 = nn.Linear(3, self.hidden)
        self.l2 = nn.Linear(self.hidden, 5 * self.n_components)
        nn.init.zeros_(self.l2.weight)
        #: pre-softplus grid extent of the partiality support (softplus(raw) nats at init)
        self.lambda_init_raw = math.log(math.expm1(3.0))
        with torch.no_grad():
            self.l2.bias.copy_(self.init_bias(self.n_components, shape))

    @staticmethod
    def init_bias(n_components: int, shape: str = "shared") -> Tensor:
        """Raw biases: logits, offsets, lower deltas, the pre-softplus upper
        narrowing (``d_hi = -softplus(raw)``) and the shape-logit deltas per
        component.  The partiality mode starts with weights decreasing away from
        the edge and its offsets unused apart from the grid extent."""
        K = n_components
        logits = torch.zeros(K); offsets = torch.zeros(K); d_lo = torch.zeros(K); raw_hi = torch.zeros(K)
        d_beta = torch.zeros(K)
        if shape == "partiality":
            logits = -0.5 * torch.arange(K, dtype=torch.float32)
            return torch.cat([logits, offsets, d_lo, raw_hi, d_beta])
        off_pattern = [-1.0, 0.3, -2.0, 0.6, -3.0, 0.9]
        lo_pattern = [0.5, -0.5, 0.8, -0.8, 1.0, -1.0]
        hi_narrow = [0.5, 0.5, 0.8, 0.8, 1.0, 1.0]
        for k in range(1, K):
            logits[k] = -3.0
            offsets[k] = off_pattern[(k - 1) % len(off_pattern)]
            d_lo[k] = lo_pattern[(k - 1) % len(lo_pattern)]
            raw_hi[k] = math.log(math.expm1(hi_narrow[(k - 1) % len(hi_narrow)]))
        return torch.cat([logits, offsets, d_lo, raw_hi, d_beta])

    def forward(self, log_w: Tensor, beta: Tensor, s: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """``(logits, offset, d_lo, d_hi, d_beta)``, each ``(n, K)``; component 0 has
        zero offset and width deltas.  ``d_beta`` is the per-component shape-logit
        offset (used by the ``"free"`` shape mode; zero for component 0)."""
        x = (torch.stack([log_w, beta, s], dim=1) - self._in_loc) / self._in_scale
        raw = self.l2(torch.nn.functional.silu(self.l1(x)))
        K = self.n_components
        keep = torch.ones(K, device=raw.device, dtype=raw.dtype); keep[0] = 0.0
        logits = raw[:, :K]
        offset = raw[:, K:2 * K] * keep
        d_lo = raw[:, 2 * K:3 * K] * keep
        d_hi = -torch.nn.functional.softplus(raw[:, 3 * K:4 * K]) * keep
        d_beta = raw[:, 4 * K:] * keep
        return logits, offset, d_lo, d_hi, d_beta


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
        scatter_family: str = "gengauss",
        n_components: int = 3,
        conditioner_hidden: int = 8,
        component_shape: str = "shared",
    ) -> None:
        super().__init__()
        if scatter_family not in ("gengauss", "mixture"):
            raise ValueError(f"scatter_family must be 'gengauss' or 'mixture', got {scatter_family!r}")
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

        frac = (beta_init - beta_min) / (beta_max - beta_min)
        beta_raw_init = math.log(frac / (1.0 - frac))       # logit
        log_w_init = math.log(w_init)

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
        #: Scatter family.  ``"mixture"`` adds a conditioner that turns the
        #: predicted (log w, beta, s) into K two-width components; the family
        #: is centred on its mean and ``log_w`` reports the mixture's log-SD.
        self.scatter_family = scatter_family
        self.n_components = int(n_components) if scatter_family == "mixture" else 1
        if scatter_family == "mixture":
            self.mixture = MixtureConditioner(self.n_components, conditioner_hidden, log_w_ref=log_w_init,
                                              shape=component_shape)

    @staticmethod
    def _zero_head(d_model: int, bias: float) -> nn.Linear:
        head = nn.Linear(d_model, 1)
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, bias)
        return head

    def forward(self, z: Tensor) -> ScalePrediction:
        log_K = self.head_log_K(z).squeeze(-1)

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
        skew = self.predict_skew
        if skew:
            s = self.head_log_skew(z).squeeze(-1)
            half_s = 0.5 * s
            log_alpha_lo, log_alpha_hi = log_alpha - half_s, log_alpha + half_s
        else:
            s = torch.zeros_like(log_w)
            log_alpha_lo, log_alpha_hi = log_alpha, None
        if self.scatter_family == "mixture":
            return self._mixture_prediction(log_K, log_w, beta, beta_raw, s, log_alpha_lo,
                                            log_alpha if log_alpha_hi is None else log_alpha_hi)
        if skew:
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

    def _mixture_prediction(self, log_K, log_w, beta, beta_raw, s, log_alpha_lo, log_alpha_hi) -> ScalePrediction:
        logits, offset, d_lo, d_hi, d_beta = self.mixture(log_w, beta, s)
        log_pi = torch.log_softmax(logits, dim=1)
        K = self.n_components
        if self.mixture.shape == "partiality":
            # support points of log p: 0, -Lambda/(K-1), ..., -Lambda; Lambda from the
            # first offset output (component 0 stays at the edge); one Gaussian
            # residual width for all components, taken from the base width
            lam = torch.nn.functional.softplus(offset[:, 1:2] + self.mixture.lambda_init_raw) if K > 1 else offset[:, :1]
            offset = -lam * self.mixture._grid[None, :]
            beta_k = torch.full_like(log_pi, 2.0)
            la = log_alpha_from_log_width(log_w, beta_k[:, 0])[:, None].expand(-1, K)
            mix = torch.cat([log_pi, offset, la, la, beta_k.log()], dim=1)
            return ScalePrediction(
                log_K=log_K,
                log_w=mixture_log_sd(log_pi, offset, la, la, beta_k, gaussian=True),
                beta=beta_k[:, 0],
                log_alpha=la[:, 0],
                log_e_s=mixture_log_e_s(log_pi, offset, la, la, beta_k, gaussian=True),
                log_alpha_hi=la[:, 0],
                mix=mix,
            )
        if self.mixture.shape == "gaussian":
            beta_k = torch.full_like(log_pi, 2.0)
        elif self.mixture.shape == "free":
            beta_k = self.beta_min + self.beta_span * torch.sigmoid(beta_raw[:, None] + d_beta)
        else:
            beta_k = beta[:, None].expand(-1, K)
        # widths: the base component's alpha follows from (w, beta) of the base; a
        # component with a different beta keeps the same SD unless its deltas say otherwise
        base_lo = log_alpha_from_log_width(log_w[:, None], beta_k) + (log_alpha_lo - log_alpha_from_log_width(log_w, beta))[:, None]
        base_hi = log_alpha_from_log_width(log_w[:, None], beta_k) + (log_alpha_hi - log_alpha_from_log_width(log_w, beta))[:, None]
        la_lo = base_lo + d_lo
        la_hi = base_hi + d_hi
        mix = torch.cat([log_pi, offset, la_lo, la_hi, beta_k.log()], dim=1)
        gaussian = self.mixture.shape == "gaussian"
        return ScalePrediction(
            log_K=log_K,
            log_w=mixture_log_sd(log_pi, offset, la_lo, la_hi, beta_k, gaussian=gaussian),
            beta=beta_k[:, 0],
            log_alpha=la_lo[:, 0],
            log_e_s=mixture_log_e_s(log_pi, offset, la_lo, la_hi, beta_k, gaussian=gaussian),
            log_alpha_hi=la_hi[:, 0],
            mix=mix,
        )
