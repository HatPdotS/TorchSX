"""Split-domain log_S quadrature: analytic scatter tail + short data-driven window.

The integrand is ``GenGauss(log S - log_mean) * L(S)`` with ``L`` the detection
likelihood (Poisson, optionally deviance-corrected for the background
estimate's error).  ``FixedLogGrid`` covers ``log S`` in ``[-34.5, 34.5]`` with
1355 uniform nodes because the two factors set opposite requirements: the
Poisson factor of a bright observation is ``1/sqrt(n)`` wide, while the
exponential-tailed GenGauss of a weak one carries mass 30 alpha below the
counting floor.  Roughly 95 % of those evaluations land where the integrand is
numerically zero, and the running cost is 40 such passes per training step.

This quadrature uses what ``L`` actually does along ``log S``:

* **Below** ``x_lo(n, bg)`` the Poisson factor is flat to a tolerance ``tau``
  (``S`` is so small that ``lam = S + bg`` has not moved), so the integrand is
  ``L(0) * GenGauss`` and its integral is ``L(0)`` times the GenGauss CDF at
  ``x_lo`` -- closed form (:func:`gengauss_log_cdf_from_obs`), exact in the
  extent, differentiable in ``log_mean`` and ``alpha``.  Nothing is truncated.
* **Above** ``x_hi(n, bg)`` the Poisson factor is at least ``lam_hi`` nats
  below its peak.  ``S_hi = max(n-bg,0) + sqrt(2 lam_hi (n + sigma_b^2)) +
  lam_hi (1 + sigma_b^2/bg)`` bounds that depth in both the Gaussian regime
  (bright peak) and the exponential regime (``n < bg``, where the factor decays
  like ``e^-S``), deviance scaling included.  The scatter density is bounded by
  its normaliser, so what is left above is below ``e^-lam_hi`` of the peak for
  every ``log_mean`` -- a floor on ``log p``, not a bias in its shape.
* **Between**, ``n_points`` trapezoid nodes.  Uniform in ``log S`` at spacing
  ``(x_hi - x_lo)/(K-1)``, except that a bright observation whose Poisson peak
  is narrower than that spacing gets a 24-node dense zone of half-width
  ``6 sigma_x`` around ``log(n - bg)`` at spacing ``sigma_x / 2``; the coarse
  nodes are then shared between the two sides in proportion to their lengths.

Everything about the grid is a function of ``(n_ph, bg)`` only.  That is the
property that makes it a legal quadrature for training: the node positions
never depend on ``log_mean``, ``alpha`` or ``beta``, so the integral is a
smooth function of the parameters and the analytic per-node gradients of the
Triton kernel remain exact.  The window is sized from the Poisson factor but
does not cut the scatter tail: the tail is integrated in closed form.

The window is typically 7-15 nats against the fixed grid's 69, so the same
0.1-nat resolution the narrowest predicted scatter needs costs several times
fewer integrand evaluations.

Grid parameters are cached on the identity of ``(n_ph, bg)`` because
``PosteriorMergeTarget.accumulate`` calls this 40 times per step on the same
tensors with different ``log_mean``.
"""
from __future__ import annotations

import math
import os

import torch
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.likelihood.poisson import Poisson, poisson_logpeak
from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.scatter.base import ScatterModel
from torchsx.merging.scatter.gengauss import (
    GenGaussLogScatter, gengauss_log_cdf_from_obs, gengauss_log_prob_from_obs,
)

try:
    from torchsx.merging.quadrature._split_log_grid_triton import (
        HAVE_TRITON, marginalize_signal_split_triton,
    )
    from torchsx.merging.quadrature._split_mixture_triton import marginalize_gaussian_mixture
except ImportError:                                  # pragma: no cover
    HAVE_TRITON = False
    marginalize_signal_split_triton = None  # type: ignore[assignment]

_DISABLE_TRITON = os.environ.get("TORCHSX_FIXEDLOGGRID_NO_TRITON", "") == "1"

__all__ = ["SplitLogGrid"]


def noise_variance(lam: Tensor, s2, noise) -> Tensor:
    """Counting variance at expected count ``lam``: ``lam + s2``, or with detector
    noise ``(A, B)`` (broadcast against ``lam``) ``A + B (lam + s2)``."""
    if noise is None:
        return lam + s2
    A, B = noise
    return A + B * (lam + s2)


def _l0_with_grad(p: dict, noise) -> Tensor:
    """``log L(0)`` of the grid, differentiable in the detector-noise terms.

    ``grid_params`` is ``no_grad``; ``log L(0) = N0 lam0 / V0 + 1/2 log(lam0 / V0) +
    logpeak`` is rebuilt here from its stored pieces with ``V0`` a function of
    ``(A, B)`` (same value, now with the gradient).
    """
    lam0 = p["lam0"]
    V0 = noise_variance(lam0, p["s2"], noise)
    return p["N0"] * lam0 / V0 + 0.5 * (lam0 / V0).log() + p["logpeak"]


class SplitLogGrid(Quadrature):
    """Analytic GenGauss tail below ``x_lo`` plus a per-observation trapezoid window.

    Parameters
    ----------
    n_points
        Trapezoid nodes per observation in the window (``K``).  192 gives a
        coarse spacing of at most ~0.1 nat for windows up to ~15 nats, a few
        nodes per alpha for the narrowest predicted scatter.
    tau
        Flatness tolerance for the analytic tail: ``x_lo`` is where
        ``|log L(S) - log L(0)|`` first reaches ``tau``.  The tail piece is then
        in error by at most ``tau`` relative, times the tail's share of the mass.
    delta_cap
        ``x_lo`` is never more than this far below the peak anchor
        ``log max(n-bg, sigma_S)``.  Bright observations would otherwise open a
        window tens of nats long to cover a tail carrying ``e^-500`` of their
        mass; capping costs at most ``e^-delta_cap * (n-bg)^2/bg`` relative on
        that tail.
    lam_hi
        Depth in nats below the Poisson peak at which the window ends, in every
        regime.  40 costs about one extra nat of window on the weak end.
    dense_points, dense_sigma
        Node count and half-width (in Poisson sigmas) of the dense zone, used
        only when ``sigma_x / 2`` is below the coarse spacing.
    """


    def __init__(
        self,
        n_points: int = 192,
        *,
        tau: float = 1e-4,
        delta_cap: float = 14.0,
        lam_hi: float = 40.0,
        dense_points: int = 24,
        dense_sigma: float = 6.0,
    ) -> None:
        super().__init__()
        if n_points < dense_points + 8:
            raise ValueError("n_points must exceed dense_points + 8")
        self.n_points = int(n_points)
        self.tau, self.delta_cap, self.lam_hi = float(tau), float(delta_cap), float(lam_hi)
        self.dense_points, self.dense_sigma = int(dense_points), float(dense_sigma)
        self._cache_key: tuple | None = None
        self._cache_val: dict | None = None
        self._last_bvf = 0.0

    # ── grid geometry: a function of the data only ────────────────────

    @torch.no_grad()
    def grid_params(self, n_ph: Tensor, bg: Tensor, sig_b2: Tensor | None,
                    bg_mean_offset: float = 0.0, noise=None) -> dict:
        """Per-observation window and dense-zone parameters, plus ``log L(0)``.

        Returns float ``(M,)`` tensors ``x_lo, h1, d_lo, h_d, d_hi, h2, log_L0``
        and int32 ``(M,)`` tensors ``n1, n1d`` (``n1d = n1 + n_dense``).  Node
        ``k`` sits at ``x_lo + k h1`` for ``k < n1``, at ``d_lo + (k-n1) h_d``
        for ``n1 <= k < n1d``, and at ``d_hi + (k-n1d+1) h2`` beyond.

        ``bg`` is the RAW stored estimate -- the tensor the cache keys on and
        pins.  ``bg_mean_offset`` shifts the background the detection model
        uses; the shifted tensor is derived here, returned as ``bg_model``, and
        held by the cache, so callers never allocate their own (which would
        both miss this cache and re-open the address-recycling window it is
        pinning ``bg`` against).

        ``noise`` is the learned detector noise, ``(A, B)`` per observation
        (:class:`~torchsx.merging.likelihood.detector_noise.DetectorNoise`): the
        counting variance becomes ``V = A + B (lam + sig_b2)``, so the window, the
        dense zone and ``log L(0)`` follow ``V``.  The terms enter detached -- node
        positions are data here, as they are for the other parameters.
        ``g0 = d log L(0) / d V0`` and the pieces of ``log L(0)`` (``N0``, ``lam0``,
        ``s2``) are returned for the gradients.
        """
        # sig_b2 is a fresh tensor on every call; key on its *coefficient*
        bvf = None if sig_b2 is None else float(self._last_bvf)
        key = (id(n_ph), id(bg), bvf, float(bg_mean_offset), n_ph.numel(),
               None if noise is None else tuple(id(t) for t in noise))
        if self._cache_key == key and self._cache_val is not None:
            return self._cache_val

        K = self.n_points
        n = n_ph.to(bg.dtype)
        # Everything below -- the window, the dense zone, log L(0) -- is the
        # geometry of the integrand, so it is built from the background the
        # detection model actually uses, not from the raw estimate.
        bg_model = bg + bg_mean_offset if bg_mean_offset else bg
        lam0 = bg_model.clamp(min=1e-6)
        s2 = sig_b2 if sig_b2 is not None else torch.zeros_like(lam0)
        s0 = (n - bg_model).clamp(min=0.0)
        nz = None if noise is None else tuple(t.detach().to(bg.dtype) for t in noise)
        V0 = noise_variance(lam0, s2, nz)                       # at S = 0
        dV0 = torch.ones_like(lam0) if nz is None else nz[1]   # dV/dS
        V_n = noise_variance(n.clamp(min=1.0), s2, nz)          # at the peak, lam ~ n
        sig_S = V_n.sqrt()

        # Poisson factor about its own peak, on the kernel's scale
        n_pos = n > 0
        nn_ = n.clamp(min=1e-12)
        A0 = torch.where(n_pos, nn_ * torch.log(lam0 / nn_) - (lam0 - nn_), -lam0)
        iphi = lam0 / V0
        logpeak = poisson_logpeak(n)
        log_L0 = A0 * iphi + 0.5 * iphi.log() + logpeak
        g0 = -A0 * lam0 / (V0 * V0) - 0.5 / V0                  # d log L(0) / d V0

        # d log L / dS at S = 0 (deviance form; s2 = 0 gives n/bg - 1)
        dA = n / lam0 - 1.0
        diphi = (V0 - lam0 * dV0) / (V0 * V0)
        dl = dA * iphi + A0 * diphi + 0.5 * (1.0 / lam0 - dV0 / V0)
        # first-order flatness bound, with the Poisson curvature n/bg^2 as the
        # guard where the slope vanishes (n ~ bg)
        S_lo = torch.minimum(self.tau / dl.abs().clamp(min=1e-30),
                             lam0 * (2.0 * self.tau / n.clamp(min=1.0)).sqrt())
        S_hi = (s0 + (2.0 * self.lam_hi * V_n).sqrt()
                + self.lam_hi * V0 / lam0)
        x_hi = S_hi.log()
        x_anchor = torch.maximum(s0, sig_S).log()
        x_lo = torch.maximum(S_lo.clamp(min=1e-30).log(), x_anchor - self.delta_cap)
        x_lo = torch.minimum(x_lo, x_hi - 1.0)

        # dense zone around a resolved Poisson peak
        Nd = self.dense_points
        sigma_x = torch.where(s0 > 0, sig_S / s0.clamp(min=1e-30),
                              torch.full_like(s0, float("inf")))
        h_est = (x_hi - x_lo) / (K - Nd - 1)
        dense = (0.5 * sigma_x < h_est) & (s0 > 0)
        x_pk = s0.clamp(min=1e-30).log()
        d_lo = torch.maximum(x_pk - self.dense_sigma * sigma_x, x_lo)
        d_hi = torch.minimum(x_pk + self.dense_sigma * sigma_x, x_hi)
        L1, L2 = d_lo - x_lo, x_hi - d_hi
        n1_dense = torch.round((K - Nd) * L1 / (L1 + L2).clamp(min=1e-30)).clamp(0, K - Nd)
        # no dense zone: one interior node, uniform spacing on both sides
        n1_flat = torch.full_like(n1_dense, float((K - 1) // 2))
        c_flat = x_lo + n1_flat * (x_hi - x_lo) / (K - 1)

        n1 = torch.where(dense, n1_dense, n1_flat)
        nd = torch.where(dense, torch.full_like(n1, float(Nd)), torch.ones_like(n1))
        d_lo = torch.where(dense, d_lo, c_flat)
        d_hi = torch.where(dense, d_hi, c_flat)
        n2 = K - n1 - nd
        h1 = (d_lo - x_lo) / n1.clamp(min=1.0)
        h_d = torch.where(dense, (d_hi - d_lo) / (nd - 1.0).clamp(min=1.0),
                          torch.zeros_like(d_lo))
        h2 = (x_hi - d_hi) / n2.clamp(min=1.0)

        out = dict(
            x_lo=x_lo.contiguous(), h1=h1.contiguous(), d_lo=d_lo.contiguous(),
            h_d=h_d.contiguous(), d_hi=d_hi.contiguous(), h2=h2.contiguous(),
            n1=n1.to(torch.int32).contiguous(),
            n1d=(n1 + nd).to(torch.int32).contiguous(),
            log_L0=log_L0.contiguous(), x_hi=x_hi, logpeak=logpeak, iphi0=iphi,
            bg_model=bg_model, g0=g0.contiguous(), N0=A0, lam0=lam0, s2=s2,
            noise=None if nz is None else tuple(t.contiguous() for t in nz),
        )
        self._cache_key, self._cache_val = key, out
        # keep the keyed tensors alive so their addresses cannot be recycled
        # under the cache
        self._cache_ref = (n_ph, bg, sig_b2, noise)
        return out

    @staticmethod
    def nodes(p: dict, K: int) -> tuple[Tensor, Tensor]:
        """``(x (M,K), log_w (M,K))`` -- the eager form of the kernel's grid."""
        k = torch.arange(K, device=p["x_lo"].device, dtype=p["x_lo"].dtype)[None, :]
        n1, n1d = p["n1"].float()[:, None], p["n1d"].float()[:, None]

        def pos(kk):
            return torch.where(
                kk < n1, p["x_lo"][:, None] + kk * p["h1"][:, None],
                torch.where(kk < n1d, p["d_lo"][:, None] + (kk - n1) * p["h_d"][:, None],
                            p["d_hi"][:, None] + (kk - n1d + 1.0) * p["h2"][:, None]))

        x = pos(k)
        w = 0.5 * (pos((k + 1).clamp(max=K - 1)) - pos((k - 1).clamp(min=0)))
        return x, w.clamp(min=1e-30).log()

    # ── marginal ──────────────────────────────────────────────────────

    def marginalize_mixture(
        self, scatter, likelihood, log_mean, n_ph, bg, params, meta, obs_range=None,
        noise=None,
    ) -> Tensor:
        if noise is not None:
            # the per-component Triton kernel has no detector noise: eager
            return super().marginalize_mixture(
                scatter, likelihood, log_mean, n_ph, bg, params, meta, obs_range, noise=noise,
            )
        if (HAVE_TRITON and not _DISABLE_TRITON and scatter.fixed_gaussian
                and 1 <= scatter.n_components <= 8
                and log_mean.is_cuda and log_mean.dtype == torch.float32
                and bg.dtype == torch.float32 and not bg.requires_grad
                and isinstance(likelihood, Poisson)):
            mix = scatter.mixture_obs(meta)
            if mix.dtype == torch.float32:
                sig_b2 = self._sig_b2(likelihood, bg)
                p = self.grid_params(n_ph, bg, sig_b2, self._bg_mean_offset(likelihood))
                return marginalize_gaussian_mixture(log_mean, n_ph, mix, p, self.n_points,
                                                     sig_b2=sig_b2)
        return super().marginalize_mixture(
            scatter, likelihood, log_mean, n_ph, bg, params, meta, obs_range,
        )

    def _sig_b2(self, likelihood: DetectionLikelihood, bg: Tensor) -> Tensor | None:
        bvf = float(getattr(likelihood, "bg_var_factor", 0.0) or 0.0)
        self._last_bvf = bvf
        return likelihood.detection_params(bg)[1]

    def _bg_mean_offset(self, likelihood: DetectionLikelihood) -> float:
        return float(getattr(likelihood, "bg_mean_offset", 0.0) or 0.0)

    def log_miss(self, likelihood: DetectionLikelihood, n_ph: Tensor,
                 bg: Tensor, noise=None) -> Tensor:
        """``log L(0)`` -- the same one the analytic tail already uses.

        Reuses the cached grid, so the miss component costs nothing on top of
        the marginal it is mixed with and cannot drift from it.  ``grid_params``
        is ``no_grad`` and would silently detach a learned background, so that
        case takes the differentiable base implementation instead (which in turn
        refuses if the likelihood also carries a background-mean offset).
        """
        if bg.requires_grad:
            if noise is not None:
                raise NotImplementedError("detector noise with a learned background")
            return super().log_miss(likelihood, n_ph, bg)
        p = self.grid_params(n_ph, bg, self._sig_b2(likelihood, bg),
                             self._bg_mean_offset(likelihood), noise)
        return p["log_L0"] if noise is None else _l0_with_grad(p, noise)

    def marginalize_signal(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,        # (M,)
        n_ph: Tensor,            # (M,)
        bg: Tensor,              # (M,)
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
        noise=None,  # (A, B) per observation: detector noise
    ) -> Tensor:                  # (M,)
        if not isinstance(scatter, GenGaussLogScatter):
            raise NotImplementedError("SplitLogGrid needs a GenGauss scatter (tail CDF)")
        if not isinstance(likelihood, Poisson):
            raise NotImplementedError("SplitLogGrid is written for the Poisson likelihood")
        K = self.n_points
        log_alpha_obs, log_beta_obs = scatter.log_alpha_beta_obs(params, meta)
        log_alpha_hi_obs = scatter.log_alpha_hi_obs(params, meta)   # None = symmetric
        sig_b2 = self._sig_b2(likelihood, bg)
        # The RAW bg goes down to grid_params -- it is what the cache keys on
        # and pins; the shifted background comes back as `p["bg_model"]` and is
        # what the integrand below (and the kernel) must use.
        p = self.grid_params(n_ph, bg, sig_b2, self._bg_mean_offset(likelihood), noise)
        bg_model = p["bg_model"]
        log_L0 = p["log_L0"] if noise is None else _l0_with_grad(p, noise)

        # analytic tail: L(0) * P(eps <= x_lo - log_mean)
        log_tail = log_L0 + gengauss_log_cdf_from_obs(
            p["x_lo"] - log_mean, log_alpha_obs, log_beta_obs,
            log_alpha_hi_obs=log_alpha_hi_obs)

        if (HAVE_TRITON and not _DISABLE_TRITON and log_mean.is_cuda
                and log_mean.dtype == torch.float32 and noise is None):
            log_win = marginalize_signal_split_triton(
                log_mean, n_ph, bg_model, log_alpha_obs, log_beta_obs, p, K,
                sig_b2=sig_b2, log_alpha_hi_obs=log_alpha_hi_obs,
            )
        else:
            x, log_w = self.nodes(p, K)
            eps = x - log_mean[:, None]
            log_scatter = gengauss_log_prob_from_obs(eps, log_alpha_obs, log_beta_obs,
                                                     log_alpha_hi_obs)
            # Poisson factor about its own peak -- the kernel's form.  The
            # likelihood's own `n log lam - lam - log n!` cancels three ~3e6
            # terms in float32 at bright observations (0.2 nats at n = 2e5).
            n = n_ph.to(x.dtype)
            nn_ = n.clamp(min=1e-12)[:, None]
            lam = (x.exp() + bg_model[:, None]).clamp(min=1e-6)
            neg_half_D = torch.where(n[:, None] > 0, nn_ * torch.log(lam / nn_) - (lam - nn_), -lam)
            half_log_iphi = 0.0
            if noise is not None:
                # detector noise: V = A + B (lam + sig_b2) at every node
                s2 = sig_b2[:, None] if sig_b2 is not None else 0.0
                inv_phi = lam / noise_variance(lam, s2, tuple(t[:, None] for t in noise))
                neg_half_D = neg_half_D * inv_phi
                half_log_iphi = 0.5 * inv_phi.log()
            elif sig_b2 is not None:
                inv_phi = lam / (lam + sig_b2[:, None])
                neg_half_D = neg_half_D * inv_phi
                half_log_iphi = 0.5 * inv_phi.log()
            log_lik = neg_half_D + half_log_iphi + p["logpeak"][:, None]
            log_win = torch.logsumexp(log_scatter + log_lik + log_w, dim=1)
        return torch.logaddexp(log_tail, log_win)
