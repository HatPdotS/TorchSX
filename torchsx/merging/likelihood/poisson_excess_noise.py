"""Poisson detection with a Gaussian excess-variance term on the background.

Motivation (see ``experiments/integration_importance/RESULTS.md``):

The vanilla :class:`~torchsx.merging.likelihood.Poisson` treats ``bg`` as a
known constant and the only noise as ``Var(n) = S + bg``.  Two real-world
effects are *not* in that model:

1. **Bg-estimate finite-sample variance.**  The bg level in each peak ring
   is a sample mean over ``n_bg`` pixels, so its variance is
   ``bg / n_bg`` per pixel, i.e. an extra ``bg · n_peak / n_bg`` photons²
   when summed over the ``n_peak`` peak pixels.
2. **Detector read noise / pedestal drift.**  Per-pixel Gaussian
   electronic noise contributes ``σ²_read · n_peak`` (plus an
   ``n_peak / n_bg`` factor when subtracting the bg estimate).

Both manifest as the *same* compound Poisson-Gaussian model:

    n | bg_true ~ Poisson(S + bg_true)
    bg_true ~ Normal(bg_obs, σ²_excess)

with ``σ²_excess`` packaging both effects.  This module marginalises
``bg_true`` out via ``n_gh``-point Gauss-Hermite quadrature, yielding

    log p(n | S, bg_obs) = logsumexp_i ( log w_i + log Poisson(n; S + bg_true_i) )

where ``bg_true_i = bg_obs + √2·z_i · σ_excess`` and ``w_i`` are the
GH weights re-normalised for a unit Gaussian.

The likelihood is **opt-in**: ``Poisson`` remains the default
``LogSpaceLoss`` likelihood.  Pass ``likelihood="poisson_excess_noise"`` to
``LogSpaceLoss.from_stream`` (and the corresponding ``--bg-noise on`` CLI
flag in ``torchsx-merge2``) to activate it.

The Triton fast path
(:func:`~torchsx.merging.quadrature.fixed_log_grid.marginalize_signal_triton`)
checks ``isinstance(likelihood, Poisson)`` exactly — this subclass
deliberately is one but the dispatch falls through to the eager path
whenever ``log_prob_given_signal`` is overridden.  TODO: a Triton kernel
that adds the ``n_gh`` axis is straightforward but not yet implemented.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood


def _gh_unit_normal_nodes(n: int, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """Gauss-Hermite nodes/weights re-normalised for ∫ N(z;0,1) f(z) dz.

    SciPy/numpy use the physicist convention ``∫ e^(-x²) f(x) dx``, so we
    substitute ``z = √2·x`` and divide weights by √π.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    z = torch.as_tensor(nodes * math.sqrt(2.0), dtype=dtype)
    w = torch.as_tensor(weights / math.sqrt(math.pi), dtype=dtype)
    return z, w


class PoissonWithExcessNoise(DetectionLikelihood):
    """Poisson with marginalised Gaussian excess variance on bg.

    Inherits ``DetectionLikelihood`` directly (NOT ``Poisson``) so that
    :meth:`~torchsx.merging.quadrature.fixed_log_grid.FixedLogGrid._can_use_triton`
    — which gates the Triton fast path on ``isinstance(likelihood, Poisson)``
    — falls through to the eager path that calls
    :meth:`log_prob_given_signal` properly.  Once a Triton kernel that
    handles the GH dimension is written, we can re-enable the fast path
    explicitly.

    Parameters
    ----------
    bg_var_factor :
        Coefficient on ``bg`` in the excess variance: ``n_peak / n_bg``.
        Set from the integration ring geometry by ``PoissonNoiseModel``.
    read_var_photons :
        Detector read-noise variance per peak-pixel-sum, **in
        photons²** (i.e. ``σ²_read_ADU · n_peak / gain²``).  Defaults to 0
        for backwards compatibility; calibrate from dark frames or fit
        jointly with gain.
    gain_switch_var_frac :
        γ² where γ = relative gain uncertainty per observation.  Models
        gain-stage-switching noise on dynamic-gain detectors
        (Jungfrau): a random multiplicative factor on the photon
        counts contributes variance ``γ² · (S + bg)²`` on top of the
        Poisson term.  Default 0 (no gain switching).  Typical
        Jungfrau values: 0.0025 (γ ≈ 5 %) for mode-boundary regions.
    n_gh :
        Gauss-Hermite quadrature order.  5 is usually enough for the
        bg-variance regime we care about; raise to 7–9 if the bg-noise
        share approaches 1 (then the Poisson term has heavy tails).
    """

    def __init__(
        self,
        bg_var_factor: float,
        read_var_photons: float = 0.0,
        gain_switch_var_frac: float = 0.0,
        n_gh: int = 5,
        trainable_excess: bool = False,
    ) -> None:
        super().__init__()
        self.bg_var_factor = float(bg_var_factor)
        self.n_gh = int(n_gh)
        self._trainable_excess = bool(trainable_excess)

        if trainable_excess:
            # Log-space trainable.  Tiny floor on init so log() doesn't
            # NaN at zero-start.  At log(1e-10) ≈ -23, the effective
            # value is ~1e-10 — numerically negligible but gradient still
            # flows from there.
            log_rv = math.log(max(read_var_photons, 1e-10))
            log_gsv = math.log(max(gain_switch_var_frac, 1e-10))
            self.log_read_var_photons = nn.Parameter(
                torch.tensor(log_rv, dtype=torch.float32)
            )
            self.log_gain_switch_var_frac = nn.Parameter(
                torch.tensor(log_gsv, dtype=torch.float32)
            )
            self.read_var_photons = 0.0      # kept for diagnostic prints; see property below
            self.gain_switch_var_frac = 0.0
        else:
            self.read_var_photons = float(read_var_photons)
            self.gain_switch_var_frac = float(gain_switch_var_frac)

        z, w = _gh_unit_normal_nodes(n_gh)
        # log w fed into the inner logsumexp; pre-log to avoid recomputing.
        self.register_buffer("gh_z", z)
        self.register_buffer("gh_logw", w.log())

    def _read_var(self) -> Tensor | float:
        if self._trainable_excess:
            return self.log_read_var_photons.exp()
        return self.read_var_photons

    def _gsv(self) -> Tensor | float:
        if self._trainable_excess:
            return self.log_gain_switch_var_frac.exp()
        return self.gain_switch_var_frac

    def excess_values(self) -> tuple[float, float]:
        """(read_var_photons, gain_switch_var_frac) — works for both fixed and trainable."""
        rv = self._read_var()
        gsv = self._gsv()
        rv = float(rv.item()) if isinstance(rv, Tensor) else float(rv)
        gsv = float(gsv.item()) if isinstance(gsv, Tensor) else float(gsv)
        return rv, gsv

    def log_prob_given_signal(
        self, n: Tensor, S: Tensor, bg: Tensor,
    ) -> Tensor:                                       # (M, K)
        # σ²_excess = bg_var_factor · bg  +  read_var_photons  [+ γ²·(S+bg)² if on]
        # The first two terms are signal-independent (per-obs, shape (M,));
        # the gain-switching term scales with the trial photon flux (M, K).
        read_var = self._read_var()
        gsv = self._gsv()
        var_add = self.bg_var_factor * bg + read_var                  # (M,)
        z = self.gh_z.to(bg.device)                                   # (n_gh,)

        # Take the (M, K) path whenever γ² could be > 0 (always, in
        # trainable mode; only when configured > 0 otherwise).
        use_full = self._trainable_excess or (
            isinstance(gsv, (int, float)) and gsv > 0
        )

        if use_full:
            # γ²·(S + bg)² captures gain-stage-switching variance.  Use
            # the trial S grid (M, K), broadcasting bg → (M, 1).
            var_mult = gsv * (S + bg[:, None]) ** 2                    # (M, K)
            var_excess = (var_add[:, None] + var_mult).clamp(min=1e-12)  # (M, K)
            sigma = var_excess.sqrt()                                    # (M, K)
            # bg_true is (M, K, n_gh) since σ depends on (M, K).
            bg_true = (
                bg[:, None, None] + sigma[:, :, None] * z[None, None, :]
            ).clamp(min=0.0)
            lam = (S[:, :, None] + bg_true).clamp(min=1e-6)              # (M, K, n_gh)
        else:
            # Original fast path: σ depends only on bg, shape (M, n_gh).
            sigma = var_add.clamp(min=1e-12).sqrt()                       # (M,)
            bg_true = (bg[:, None] + sigma[:, None] * z[None, :]).clamp(min=0.0)
            lam = (S[:, :, None] + bg_true[:, None, :]).clamp(min=1e-6)  # (M, K, n_gh)

        log_nf = torch.lgamma(n + 1.0)                                     # (M,)
        log_p = (
            n[:, None, None] * torch.log(lam)
            - lam
            - log_nf[:, None, None]
        )

        # Marginalise the n_gh axis with GH weights.
        return torch.logsumexp(
            log_p + self.gh_logw.to(bg.device)[None, None, :], dim=-1,
        )                                                                  # (M, K)
