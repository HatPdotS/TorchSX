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

    Parameters
    ----------
    bg_var_factor
        ``n_peak / n_bg``; ``0`` (default) reproduces the plain Poisson
        exactly.  Non-zero accounts for the fact that ``bg`` is not measured
        but ESTIMATED, from the 9–11 px annulus, so it carries
        ``σ_b² = (n_peak/n_bg)·bg`` — 11.6 counts at this data's median bg of
        116, against a median signal of 7.3.

        Treating it as exact does not merely lose precision: the error
        *rectifies*, because an upward fluctuation makes the apparent signal
        negative and ``S = K·I₀·e^ε > 0`` cannot represent that, so it is
        absorbed by raising ``I₀``.  Measured on ground truth, that inflates
        weak reflections by 2.4–3.4×, and it does not average down with
        multiplicity.  The mechanism shows up directly as a width error: on a
        real batch the plain Poisson's marginal is **0.70×** as wide as the
        exact marginal over the annulus's Gamma posterior, and only 0.32× as
        wide in the weakest quartile.

    bg_var_mode
        How the background estimate's variance enters, when
        ``bg_var_factor > 0``.

        ``"deviance"`` (default) scales the Poisson deviance by the dispersion
        ``φ = 1 + σ_b²/λ`` — the saddlepoint density of the overdispersed law
        with ``Var = λ + σ_b²``:

        .. math::
            \log p = -\frac{D(n, \lambda)}{2\varphi}
                     - \tfrac12\log\varphi + \log p_{\text{peak}}(n)

        It keeps the mode at ``λ = n``, matches the variance, collapses to the
        exact Poisson at ``σ_b² = 0``, and is bounded, so it needs no clamp.

        ``"second_order"`` is the earlier expansion of
        ``∫ Poisson(n|S+b)·N(b|bg,σ_b²) db`` about ``b = bg``,
        ``+½σ_b²[(n/λ−1)² − n/λ²]``, clamped to ``±10``.  Kept only to
        reproduce those measurements: it is unbounded as ``λ`` leaves ``n``, the
        clamp fires on 16 % of the grid points that carry mass, and it inflates
        the weakest quartile's posterior mean 5.4× (15.9 photons against the
        exact marginal's 2.9).

        Median |error| against the exact marginal on a real batch, over grid
        points within 15 nats of the peak: **0.13 nats** for ``"deviance"``,
        0.89 for ``"second_order"``, 1.48 for the plain Poisson.

        Note ``d log p/d bg`` is no longer the plain ``S2`` statistic when
        either is on, so a learned background (``bg_mix``) must not be combined
        with them; the Triton path raises rather than return a wrong gradient.
    """

    def __init__(self, bg_var_factor: float = 0.0,
                 bg_var_mode: str = "deviance") -> None:
        super().__init__()
        self.bg_var_factor = float(bg_var_factor)
        if bg_var_mode not in ("deviance", "second_order"):
            raise ValueError(f"unknown bg_var_mode {bg_var_mode!r}")
        self.bg_var_mode = bg_var_mode

    def log_prob_given_signal(self, n: Tensor, S: Tensor, bg: Tensor) -> Tensor:
        lam = S + bg[:, None]
        lam = lam.clamp(min=1e-6)
        log_nf = torch.lgamma(n + 1)
        if self.bg_var_factor <= 0.0:
            return n[:, None] * torch.log(lam) - lam - log_nf[:, None]

        sig_b2 = (self.bg_var_factor * bg.clamp(min=0.0))[:, None]
        if self.bg_var_mode == "second_order":
            out = n[:, None] * torch.log(lam) - lam - log_nf[:, None]
            inv = 1.0 / lam
            r = n[:, None] * inv
            corr = 0.5 * sig_b2 * ((r - 1.0) ** 2 - r * inv)
            return out + corr.clamp(-10.0, 10.0)

        # Deviance scaling.  Formed about the Poisson's own peak, as the Triton
        # path does, so the three ~3e6 terms of `n log λ − λ − log n!` never
        # cancel against each other in float32.
        nn = n.clamp(min=1e-12)[:, None]
        neg_half_D = nn * torch.log(lam / nn) - (lam - nn)
        inv_phi = lam / (lam + sig_b2)
        # float64: n log n, n and log n! are each ~3e6 at the brightest
        # observation and cancel to O(1), so forming this in float32 costs
        # ~0.4 nats.  It is one vector of length M, not the (M, K) integrand.
        n64 = n.double()
        logpeak = torch.where(
            n > 0,
            (n64 * n64.clamp(min=1e-300).log() - n64 - log_nf.double()).to(n.dtype),
            torch.zeros_like(n),
        )[:, None]
        return neg_half_D * inv_phi + 0.5 * torch.log(inv_phi) + logpeak
