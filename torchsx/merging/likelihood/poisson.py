"""Plain Poisson detection likelihood (no bg mixture)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood


def poisson_log_prob_at_zero_signal(
    n: Tensor, bg: Tensor, sig_b2: Tensor | None = None,
) -> Tensor:
    """``log p(n | S = 0, bg)`` on the kernel's scale, shape ``(M,)``.

    The deviance form about the Poisson's own peak, with the background
    estimate's variance folded in as the dispersion when ``sig_b2`` is given.
    ``bg`` is the background the *model* uses — pass
    ``likelihood.model_bg(bg)``, not the raw stored estimate.

    Three call sites need exactly this quantity and used to each write it out:
    the analytic tail of :meth:`SplitLogGrid.grid_params` (``log_L0``), the miss
    component of :class:`MarginalNLL`, and the Triton kernels.  It is kept here,
    next to the likelihood whose algebra it is.
    """
    n = n.to(bg.dtype)
    lam0 = bg.clamp(min=1e-6)
    s2 = sig_b2 if sig_b2 is not None else torch.zeros_like(lam0)
    n_pos = n > 0
    nn_ = n.clamp(min=1e-12)
    A0 = torch.where(n_pos, nn_ * torch.log(lam0 / nn_) - (lam0 - nn_), -lam0)
    iphi = lam0 / (lam0 + s2)
    return A0 * iphi + 0.5 * iphi.log() + poisson_logpeak(n)


def poisson_logpeak(n: Tensor) -> Tensor:
    """``n log n − n − log n!`` — the Poisson log-PMF at its own peak ``λ = n``.

    Formed in float64: at the brightest observation the three terms are each
    ~3e6 and cancel to O(1), so float32 costs ~0.4 nats.  It is one vector of
    length M, not the (M, K) integrand, so the cast is free.
    """
    n64 = n.double()
    return torch.where(
        n > 0,
        (n64 * n64.clamp(min=1e-300).log() - n64 - torch.lgamma(n64 + 1)).to(n.dtype),
        torch.zeros_like(n),
    )


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

    Note ``d log p/d bg`` is no longer the plain ``S2`` statistic when
    ``bg_var_factor > 0``, so a learned background (``bg_mix``) must not be
    combined with it; the Triton path raises rather than return a wrong
    gradient.

    bg_mean_offset
        Added to the background the detection mean uses: ``λ = S + bg + offset``.
        ``0`` (default) reproduces every earlier result exactly.

        ``bg_var_factor`` above models the *variance* of the annulus estimate
        but leaves its *mean* as if the estimate were exact, and it is not.
        With ``A`` annulus counts over ``n_bg`` pixels and ``c = n_peak/n_bg``,
        the stored ``bg`` is ``c·A``, and the true peak background has the
        flat-prior predictive ``B | A ~ Gamma(A+1, rate 1/c)``, i.e.

        .. math::
            E[B \mid A] = c(A + 1) = \text{bg} + c

        so the modelled mean is low by ``c`` on **every** observation.  That
        error does not average down: ``S = K·I₀·e^ε > 0`` cannot represent a
        negative excursion, so it rectifies into inflated weak intensities —
        the same mechanism the ``bg_var_factor`` docstring describes for the
        variance.  The bias therefore scales with ``c``, which is what the data
        show: ``c`` is 1.1641 for the small-molecule geometry (149/128), 0.2788
        for BR and 0.1477 for TD1, and the measured damage is ordered the same
        way.

        Measured at ``offset = c``: independent-annulus recovery RMS falls from
        4.54 to 1.37 SIGF over three seeds, and a matched ten-epoch retrain
        moves small-molecule raw posterior R_free from 11.8676 % to 8.7519 %
        with native intensity R_split holding (1.5567 % → 1.5468 %).

        It is an approximation, not the exact predictive, and it is not
        universally unbiased: synthetic recovery keeps a residual *negative*
        bias, a true zero background makes a universal ``+c`` shift wrong, and
        real integrated backgrounds are non-integer and pixel-masked, so the
        pure Poisson-annulus law is itself a model.  Note also that the
        variance stays ``c·bg`` rather than the self-consistent predictive
        ``Var[B | A] = c·(bg + c)`` — a ``c²`` ≈ 1.36-count difference against a
        median λ of ~116, kept so this reproduces the qualified experiment
        exactly.  See ``experiments/posterior_accuracy_20260910/REPORT_20260911.md``.

        Detection-only: the raw ``bg`` still feeds the encoder features, the
        pooled ``(n − bg)/K`` estimator (unbiased at fixed true background
        precisely because it averages over ``A`` rather than conditioning on
        it), the bg gate and the σ²(bg) calibration.

    bg_var
        Per-observation variance of the model background (probixi's ``bg_model``),
        ``(n_obs_total,)`` photons^2, gathered by ``meta.obs_id``.  The producer
        measured it, so it replaces the annulus ``bg_var_factor``/``bg_mean_offset``
        (mutually exclusive with both).
    """

    def __init__(self, bg_var_factor: float = 0.0, *,
                 bg_mean_offset: float = 0.0, bg_var: Tensor | None = None) -> None:
        super().__init__()
        self.bg_var_factor = float(bg_var_factor)
        if not math.isfinite(bg_mean_offset) or bg_mean_offset < 0.0:
            raise ValueError("bg_mean_offset must be finite and non-negative, "
                             f"got {bg_mean_offset!r}")
        if bg_var is not None:
            if self.bg_var_factor > 0.0 or bg_mean_offset > 0.0:
                raise ValueError("bg_var carries the estimate's own variance; "
                                 "bg_var_factor and bg_mean_offset describe the "
                                 "annulus estimate and must both be 0 with it")
            if bg_var.ndim != 1 or not bool(torch.isfinite(bg_var).all()) or bool((bg_var < 0).any()):
                raise ValueError("bg_var must be a finite non-negative (n_obs,) tensor")
            self.register_buffer("bg_var_obs", bg_var.detach().float().clone(), persistent=False)
            self.has_bg_var = True
        if bg_mean_offset > 0.0 and self.bg_var_factor <= 0.0:
            # Mean shifted but the estimate's variance unmodelled: the two are
            # the same annulus constant c = n_peak/n_bg, and correcting one
            # without the other is strictly worse than the uncorrected model.
            raise ValueError("bg_mean_offset > 0 needs bg_var_factor > 0; both "
                             "are n_peak/n_bg and describe the same estimate")
        self.bg_mean_offset = float(bg_mean_offset)

    def detection_params(self, bg: Tensor, meta=None) -> tuple[Tensor, Tensor | None]:
        if self.bg_mean_offset and bg.requires_grad:
            # `SplitLogGrid.grid_params` is `no_grad`, so a shifted background
            # derived inside it would be silently detached; and the kernels'
            # analytic d log p/d bg is the plain-Poisson S2 statistic, which the
            # shift invalidates.  Refuse rather than return a wrong gradient.
            raise RuntimeError("bg_mean_offset does not support a learned "
                               "background (bg.requires_grad)")
        if self.has_bg_var:
            obs_id = getattr(meta, "obs_id", None)
            if obs_id is None:
                raise ValueError("a per-observation background variance needs the "
                                 "batch's ObsInfo with obs_id")
            if obs_id.numel() != bg.numel():
                raise ValueError(f"obs_id has {obs_id.numel()} rows for {bg.numel()} backgrounds")
            var = self.bg_var_obs[obs_id.to(self.bg_var_obs.device)]
            return bg, var.to(device=bg.device, dtype=bg.dtype)
        return (bg + self.bg_mean_offset if self.bg_mean_offset else bg), (
            self.bg_var_factor * bg.clamp(min=0.0)
            if self.bg_var_factor > 0.0 else None)

    def log_prob_given_signal(self, n: Tensor, S: Tensor, bg: Tensor, meta=None) -> Tensor:
        bg_mean, sig_b2 = self.detection_params(bg, meta)
        lam = S + bg_mean[:, None]
        lam = lam.clamp(min=1e-6)
        log_nf = torch.lgamma(n + 1)
        if sig_b2 is None:
            return n[:, None] * torch.log(lam) - lam - log_nf[:, None]

        sig_b2 = sig_b2[:, None]
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
