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
        exactly.  Non-zero marginalises the BACKGROUND ESTIMATE's own error
        to second order.

        ``bg`` is not known — CrystFEL estimates it from the 9–11 px annulus,
        so it carries ``σ_b² = (n_peak/n_bg)·bg``, which is 11.6 counts at
        this data's median bg of 116 against a median signal of 7.3.  Treating
        it as exact does not merely lose precision: the error *rectifies*,
        because an upward fluctuation makes the apparent signal negative and
        ``S = K·I₀·e^ε > 0`` cannot represent that, so it is absorbed by
        raising ``I₀``.  Measured on ground truth, that inflates weak
        reflections by 2.4–3.4×, and it does not average down with
        multiplicity.

        Expanding ``∫ Poisson(n|S+b)·N(b|bg,σ_b²) db`` about ``b = bg`` and
        keeping the exponential form (equal to second order, but never
        negative, unlike ``1 + ½σ_b²P''/P``):

        .. math::
            \log p \approx \log \mathrm{Pois}(n|\lambda)
                + \tfrac12 \sigma_b^2\left[(n/\lambda - 1)^2
                - n/\lambda^2\right]

        At the peak ``λ = n`` this is ``−σ_b²/2n`` (a slight flattening); in
        the tails it is positive (a broadening).  It costs one reciprocal and
        no transcendental, and unlike switching to a Gaussian likelihood it
        keeps the variance growing with the hypothesised signal — which
        measurement showed matters: on real data a Gaussian likelihood cost
        +0.034 in R_free even with ``bg_var_factor = 0``.

        The expansion needs ``σ_b²·|P''/P| ≪ 1``; the correction is clamped to
        ``±10`` so a far-tail grid point, where the Poisson term is already
        e^-100, cannot blow up.

        Note ``d log p/d bg`` is no longer the plain ``S2`` statistic when this
        is on, so a learned background (``bg_mix``) must not be combined with it.
    """

    def __init__(self, bg_var_factor: float = 0.0) -> None:
        super().__init__()
        self.bg_var_factor = float(bg_var_factor)

    def log_prob_given_signal(self, n: Tensor, S: Tensor, bg: Tensor) -> Tensor:
        lam = S + bg[:, None]
        lam = lam.clamp(min=1e-6)
        log_nf = torch.lgamma(n + 1)
        out = n[:, None] * torch.log(lam) - lam - log_nf[:, None]
        if self.bg_var_factor > 0.0:
            inv = 1.0 / lam
            r = n[:, None] * inv
            sig_b2 = (self.bg_var_factor * bg.clamp(min=0.0))[:, None]
            corr = 0.5 * sig_b2 * ((r - 1.0) ** 2 - r * inv)
            out = out + corr.clamp(-10.0, 10.0)
        return out
