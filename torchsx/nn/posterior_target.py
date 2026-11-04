"""Merge target from the I0 posterior — integrated, never fitted.

I0 is never a fitted point estimate: it is integrated out, and the loss is the
marginal likelihood of each reflection's observations (the evidence).  The
quantity that is scored is therefore the quantity that is merged, and no
detached target stands between the network and its objective.

Geometry
--------
On the log-intensity axis each observation is a bump: ``log K_j`` **slides** it
(a multiplicative scale is a translation in log space) and ``alpha_j``
**blurs** it.  Merging is sliding every observation of a reflection into a
common frame and asking how well the bumps stack; the marginal likelihood is
the *area under their product*.  Because the GenGauss is normalised, blurring
also flattens — so agreement cannot be bought by widening the scatter.  That is
the geometric form of the ``-log alpha`` normaliser.

Why this is affordable
----------------------
The naive cost is one inner quadrature per outer grid point per observation.
Two facts collapse it:

1. **Individual observation curves are broad; only their product is narrow.**
   ``f_j(mu) = log p(n'_j | mu)`` has the width of one observation's
   uncertainty, while the posterior is ~sqrt(n) narrower.  So evaluate ``f_j``
   on a **coarse** grid and linearly interpolate to a fine one.  ~25-33 coarse
   points suffice, hence the default ``k_coarse = 31``.
2. **Linear interpolation commutes with the sum over observations.**
   ``sum_j interp(f_j) = interp(sum_j f_j)``, so accumulate on the *coarse*
   grid first — ``(n_hkl, K_coarse)``, tiny — and interpolate that.  The
   ``(n_obs, M_fine)`` tensor never exists.

With the Triton inner kernel the remaining cost is small: it never materialises
the ``(M, K_inner)`` integrand (it saves six tensors of size M), so a pass
moves a few percent of the eager traffic, and 31 passes come to roughly one
eager pass — a few percent of a training step.  Use a **wide** FixedLogGrid so
the Triton path applies without truncating the scatter; the inner node count is
nearly free.

Grid geometry (centre, width) is **detached**: it only places quadrature nodes,
exactly like choosing integration limits, and the answer is insensitive to it
provided the mass is covered.  ``edge_mass`` reports whether it is — taken
as weight in the outer 10 % of the *range*, which is node-count independent
(the outermost single node's weight shrinks with n_grid and would measure
resolution, not truncation).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.nn.pooling import group_sum

try:
    from torchsx.merging.quadrature._fused_posterior_triton import (
        fused_posterior_logp,
    )
    _HAVE_FUSED = True
except ImportError:                                  # pragma: no cover
    _HAVE_FUSED = False

__all__ = ["PosteriorOut", "PosteriorMergeTarget"]

#: TORCHSX_NODE_LOOP=1 forces the node-by-node accumulation (A/B timing, tests).
_NODE_LOOP = os.environ.get("TORCHSX_NODE_LOOP", "") == "1"


def _all_nodes_kernel(objective, scatter, n_ph: Tensor, bg: Tensor) -> bool:
    """Whether ``accumulate`` can take every outer node in one fused pass.

    That kernel is the split-grid fixed-Gaussian mixture marginal of the
    production model (with its optional miss component); anything else keeps
    the node-by-node loop.
    """
    if _NODE_LOOP or not n_ph.is_cuda or bg.requires_grad or bg.dtype != torch.float32:
        return False
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature import split_log_grid as slg
    from torchsx.merging.scatter.mixture import PredictedMixtureScatter
    from torchsx.nn.objective import MarginalNLL
    return (slg.HAVE_TRITON and not slg._DISABLE_TRITON
            and isinstance(objective, MarginalNLL) and objective.chunk_size is None
            and isinstance(objective.quadrature, slg.SplitLogGrid)
            and isinstance(objective.likelihood, Poisson)
            and isinstance(scatter, PredictedMixtureScatter) and scatter.fixed_gaussian
            and 1 <= scatter.n_components <= 8)


@dataclass
class PosteriorOut:
    """Per-reflection posterior summary plus the per-observation broadcast."""

    log_Z: Tensor        # (n_hkl,) marginal likelihood; -inf where unseen
    I0_mean: Tensor      # (n_hkl,) posterior mean
    I0_sd: Tensor        # (n_hkl,) posterior sd -- sigI, from the same integral
    I0_obs: Tensor       # (n,)     I0_mean broadcast to observations
    seen: Tensor         # (n_hkl,) bool
    edge_mass: Tensor    # (n_hkl,) truncation check, ~0 is healthy
    n_floored: int


class PosteriorMergeTarget(nn.Module):
    """Differentiable posterior mean + marginal likelihood over I0.

    Carries no per-reflection state from one batch to the next: every batch
    places its own integration grid (:meth:`batch_centre`, :meth:`locate`), and
    the Wilson surface is a function of hkl with a handful of coefficients
    trained by the same evidence (:meth:`enable_evidence_prior`).

    Parameters
    ----------
    I_start
        ``(n_hkl,)`` the unity merge of the input observations: the starting
        point of a merge's locating scan and the initial Wilson fit.
    k_coarse
        Inner-quadrature evaluations per reflection.  The real cost knob.
    m_fine
        Interpolated integration points.  Nearly free — raise before ``k_coarse``.
    n_sigma
        Grid half-width in Laplace sigmas.  5 suffices for the median but the
        90th-percentile error is *width*-limited (flat in node count), so 8 is
        the safer default since width is free.
    prior
        ``"none"`` integrates ``dI0`` (weight ``exp(v)``), with no Wilson
        density; ``"flat_I0"`` is its legacy alias. ``"flat_log"``
        integrates ``d log I0``; ``"wilson"`` uses ``p(I) = exp(-I/Sigma)/Sigma``
        with ``Sigma(h)`` the smooth Wilson surface.  **Use wilson.**  Where the
        likelihood is uninformative — and most serial observations are
        noise-limited — the posterior is whatever the prior says, and a
        flat-in-I0 prior on a log grid drags those reflections toward the *top
        of the grid*: per-shell CC1/2 rises at low resolution and collapses at
        high resolution, because exactly the weak reflections are inflated.
    """

    def __init__(
        self,
        I_start: Tensor,
        *,
        k_coarse: int = 31,
        m_fine: int = 257,
        n_sigma: float = 8.0,
        prior: str = "wilson",
        floor: float = 1e-3,
        scan_points: int = 9,
        scan_halfwidth: float = 8.0,
        use_fused: bool = False,
        max_halfwidth: float = 12.0,
    ) -> None:
        super().__init__()
        if prior == "flat_I0":
            prior = "none"
        if prior not in ("none", "flat_log", "wilson"):
            raise ValueError(
                f"prior must be 'none' or 'wilson', got {prior!r} "
                "(legacy 'flat_I0' and 'flat_log' are also accepted)"
            )
        if k_coarse < 3 or m_fine < k_coarse:
            raise ValueError("need k_coarse >= 3 and m_fine >= k_coarse")
        self.n_hkl = int(I_start.shape[0])
        self.k_coarse, self.m_fine = int(k_coarse), int(m_fine)
        self.n_sigma, self.prior = float(n_sigma), prior
        self.floor = float(floor)
        self.scan_points, self.scan_halfwidth = int(scan_points), float(scan_halfwidth)
        # Cap on the outer grid HALF-WIDTH in nats.  ``n_sigma`` was raised to
        # 24 to stop truncating sharp posteriors whose Laplace width is
        # underestimated; applied to a BROAD width it does the opposite damage.
        # A weak reflection's curve is flat below and a cliff above, the
        # parabola reads a width of several nats, and 24 x 4 nats puts the 31
        # coarse nodes 6.4 nats apart -- the cliff is then linearly interpolated
        # over 6 nats and the posterior mean lands nats too high.  The Wilson
        # prior removes all mass more than ~12 nats from the centre, so nothing
        # beyond needs nodes.
        self.max_halfwidth = float(max_halfwidth)
        # Recompute each outer node in backward instead of holding all C graphs.
        # OFF by default: it roughly doubles the accumulate's compute, which is
        # not worth paying on a data set that already fits.  Turn it on when the
        # batch is memory-bound -- a large cell gets several times less in-batch
        # redundancy than a small one at the same batch size, and that is what
        # it buys back.
        self.checkpoint_nodes = False
        self.use_fused = bool(use_fused)
        #: Lower bound of a grid centre, in nats below the Wilson surface.
        self.centre_floor_nats = 4.0

        self.register_buffer("I_start", I_start.detach().clone().float().clamp(min=floor),
                             persistent=False)
        # Mean intensity Sigma(h) of the Wilson prior.  Starts at the unity
        # merge per reflection; :meth:`fit_wilson_surface` replaces it with the
        # smooth surface once a basis is installed.
        self.register_buffer("wilson_sigma", self.I_start.clone())
        # Centric reflections have a different Wilson distribution:
        #     acentric  p(I) = exp(-I/Sigma) / Sigma
        #     centric   p(I) = exp(-I/2Sigma) / sqrt(2 pi Sigma I)
        # so the prior rate is 1/Sigma against 1/(2 Sigma), and the centric form
        # carries an extra I^(-1/2) that puts much more weight near zero.  Using
        # the acentric form on a centric reflection under-shrinks it, and in
        # Laue class mmm the three principal zones are centric -- exactly the
        # strong low-resolution reflections.  Default all-False, so a triclinic
        # merge is unchanged bit-for-bit.
        self.register_buffer("is_centric",
                             torch.zeros_like(self.wilson_sigma, dtype=torch.bool))
        # Smooth Sigma(h): log Sigma = phi @ coef on the reciprocal-space basis
        # installed through set_basis().
        self.register_buffer("wilson_phi", torch.zeros(0, 0))
        self.register_buffer("wilson_coef", torch.zeros(0))
        self.wilson_fit_iters = 0
        #: Empirical Bayes: with :meth:`enable_evidence_prior` the smooth
        #: surface's coefficients are a parameter of the evidence itself
        #: (type-II maximum likelihood, trained with the network).
        #: ``wilson_coef`` / ``wilson_sigma`` mirror it
        #: (:meth:`sync_evidence_prior`) for every no-grad consumer and for the
        #: checkpoint.
        self.register_parameter("wilson_coef_param", None)

        self.set_grid(k_coarse, m_fine, n_sigma)

    def set_grid(self, k_coarse: int, m_fine: int, n_sigma: float) -> None:
        """Replace node buffers and interpolation tables together at readout."""
        if k_coarse < 3 or m_fine < k_coarse or not math.isfinite(n_sigma) or n_sigma <= 0:
            raise ValueError("need k_coarse >= 3, m_fine >= k_coarse and finite n_sigma > 0")
        self.k_coarse, self.m_fine, self.n_sigma = int(k_coarse), int(m_fine), float(n_sigma)
        device = self.I_start.device
        u_c = torch.linspace(-n_sigma, n_sigma, k_coarse).to(device)
        u_f = torch.linspace(-n_sigma, n_sigma, m_fine).to(device)
        self.register_buffer("u_coarse", u_c)
        self.register_buffer("u_fine", u_f)
        idx = torch.searchsorted(u_c.contiguous(), u_f.contiguous()).clamp(1, k_coarse - 1)
        t = ((u_f - u_c[idx - 1]) / (u_c[idx] - u_c[idx - 1])).clamp(0.0, 1.0)
        self.register_buffer("_lo", idx - 1)
        self.register_buffer("_hi", idx)
        self.register_buffer("_t", t)

    # ── grid geometry (detached) ───────────────────────────────────────

    @torch.no_grad()
    def locate(
        self, objective, scatter, params, pred,
        n_ph: Tensor, bg: Tensor, hkl_id: Tensor, meta, centre: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Find the peak and width of the log-likelihood curve by MEASURING it,
        scanning around ``centre`` (log intensity per reflection).

        A *formula* for the grid width derived from the Gaussian variance model
        fails in every direction -- too narrow (mass at the grid edge),
        oscillating (a follow-the-measurement loop), or too wide (sigma pinned
        at its bound, so the merge returns the grid centre and the data
        contribute nothing) -- because the posterior is not that shape.

        So: sweep a deliberately generous grid once, then read the peak and
        curvature off the curve itself.  A parabola through the maximum and its
        neighbours gives ``v*`` and ``s = 1/sqrt(-S'')`` with no distributional
        assumption and nothing that can pin at a bound.
        """
        v_scan = self.scan_grid(centre)
        S = self.accumulate(
            objective, scatter, params, pred, n_ph, bg, hkl_id, meta, v_scan
        )
        return self.peak_from_scan(S, centre)

    @torch.no_grad()
    def batch_centre(self, pred, n_signal: Tensor, sigma: Tensor, hkl_id: Tensor) -> Tensor:
        """``log`` of the batch's weighted mean intensity at the network's scale.

        The weighted least-squares estimate ``sum K n' / V`` over ``sum K^2 / V``
        of each reflection's observations in this batch, with
        ``V = sigma^2 + (K I)^2 (e^{w^2} - 1)`` evaluated at the counting-only
        estimate ``I``; floored at ``centre_floor_nats`` below the Wilson surface
        (reflections without observations sit at the floor).  Only places the
        locating scan, which spans ``+-scan_halfwidth`` around it.
        """
        K = pred.log_K.detach().exp().clamp(min=1e-8)
        var0 = (sigma ** 2).clamp(min=1e-12)
        num = group_sum(K * n_signal / var0, hkl_id, self.n_hkl)
        den = group_sum(K * K / var0, hkl_id, self.n_hkl)
        lo = self.wilson_sigma.clamp(min=self.floor) * math.exp(-self.centre_floor_nats)
        I1 = torch.where(den > 0, num / den.clamp(min=1e-30), lo).clamp(min=self.floor)
        w2 = (pred.log_w.detach().exp() ** 2).clamp(max=20.0)
        var = (sigma ** 2 + (K * I1[hkl_id]) ** 2 * torch.expm1(w2)).clamp(min=1e-12)
        num = group_sum(K * n_signal / var, hkl_id, self.n_hkl)
        den = group_sum(K * K / var, hkl_id, self.n_hkl)
        I2 = torch.where(den > 0, num / den.clamp(min=1e-30), lo)
        return torch.maximum(I2, lo).clamp(min=self.floor).log()

    @torch.no_grad()
    def start_centre(self) -> Tensor:
        """``log`` of the unity merge floored ``centre_floor_nats`` below the
        Wilson surface: where a merge's first locating scan is centred."""
        lo = self.wilson_sigma.clamp(min=self.floor) * math.exp(-self.centre_floor_nats)
        return torch.maximum(self.I_start, lo).clamp(min=self.floor).log()

    @torch.no_grad()
    def scan_grid(self, centre: Tensor) -> Tensor:
        """Generous grid for the locating sweep: ``(n_hkl, scan_points)``,
        around ``centre`` (log intensity)."""
        u = torch.linspace(
            -self.scan_halfwidth, self.scan_halfwidth, self.scan_points,
            device=centre.device,
        )
        return centre[:, None] + u[None, :]

    @torch.no_grad()
    def peak_from_scan(self, S: Tensor, centre: Tensor) -> tuple[Tensor, Tensor]:
        """Peak location and width read off an accumulated scan curve.

        Separate from :meth:`locate` so a caller that spans several batches can
        accumulate ``S`` itself first, as the merge does.  A merge grid placed by
        a closed-form width instead collapses onto the grid centre: both halves
        return the same shared centre rather than anything from the data, and
        per-shell CC1/2 reads a spurious 1.
        """
        u = torch.linspace(
            -self.scan_halfwidth, self.scan_halfwidth, self.scan_points,
            device=S.device,
        )
        h = float(u[1] - u[0])

        k = S.argmax(dim=1).clamp(1, self.scan_points - 2)
        rows = torch.arange(S.shape[0], device=S.device)
        y0, y1, y2 = S[rows, k - 1], S[rows, k], S[rows, k + 1]
        curv = (y0 - 2 * y1 + y2) / h**2                 # S'' < 0 at a maximum

        denom = (y0 - 2 * y1 + y2)
        shift = torch.where(
            denom.abs() > 1e-12, 0.5 * (y0 - y2) / denom.clamp(min=-1e30),
            torch.zeros_like(denom),
        ).clamp(-1.0, 1.0) * h
        v_meas = centre + u[k] + shift
        sd_meas = (1.0 / (-curv).clamp(min=1e-8)).sqrt()

        # Trust the measured peak only in proportion to how well the curve
        # actually locates it.  For an uninformative reflection the curve is
        # flat, argmax is essentially random over the whole +-scan_halfwidth,
        # and taking it displaces the grid centre by a large random amount --
        # independently in each half, which is exactly what destroys their
        # correlation in the weakest shells.
        #
        # The curve locates the peak to +-sd_meas, so compare that against the
        # scan width: w = 1 when the peak is sharp, 0 when the curve says
        # nothing, and the fallback is the prior centre with a wide grid.
        peaked = curv < -1e-8
        rel = (sd_meas / self.scan_halfwidth).clamp(0.0, 1.0)
        trust = torch.where(peaked, (1.0 - rel).clamp(0.0, 1.0),
                            torch.zeros_like(rel))
        v_star = trust * v_meas + (1.0 - trust) * centre
        sd = torch.where(peaked, sd_meas, torch.full_like(sd_meas, self.scan_halfwidth / 2))
        sd = trust * sd + (1.0 - trust) * (self.scan_halfwidth / 2)

        # One-sided curves.  A weak reflection's likelihood is flat below some
        # bound and a cliff above it: no peak, so the parabola width is
        # meaningless (several nats) and ``n_sigma`` times that is a grid of
        # +-96 nats whose 31 coarse nodes straddle the cliff 6.4 nats apart.
        # Linear interpolation across the cliff then cuts the posterior off
        # nats below the true bound and collapses the reflection to ~0.
        # Anchor these on the plateau EDGE instead: the last scan
        # node within 2 nats of the maximum is the knee, and the Wilson prior
        # (density ~ e^v below its mode) leaves no mass more than ~10 nats
        # below it, so the support is [knee - 0.75 H, knee + 0.25 H] with
        # H = max_halfwidth (the knee itself is known only to the 2-nat scan
        # spacing, hence the margin above it).  That is a centre shift plus a width, so the
        # shared u axis and the coarse->fine interpolation are unchanged.
        H = self.max_halfwidth
        sd_max = min(self.scan_halfwidth, H / self.n_sigma)
        plateau = S >= (S.max(dim=1, keepdim=True).values - 2.0)
        k_knee = (plateau * torch.arange(self.scan_points, device=S.device)).max(dim=1).values
        v_knee = centre + u[k_knee]
        # a plateau that reaches the top of the scan is uninformative on the
        # high side too: keep the blended fallback for those (rare)
        one_sided = (sd > sd_max) & (k_knee < self.scan_points - 1)
        v_star = torch.where(one_sided, v_knee - 0.25 * H, v_star)
        sd = torch.where(one_sided, torch.full_like(sd, 0.5 * H / self.n_sigma), sd)
        return v_star, sd.clamp(min=1e-3, max=sd_max)

    @property
    def has_basis(self) -> bool:
        return self.wilson_phi.numel() > 0

    @torch.no_grad()
    def set_basis(self, phi: Tensor) -> None:
        """Install the smooth basis ``phi`` (``n_hkl x p``); Sigma is fitted on it from now on."""
        if phi.shape[0] != self.n_hkl:
            raise ValueError(f"basis has {phi.shape[0]} rows for {self.n_hkl} reflections")
        self.wilson_phi = phi.detach().to(self.I_start.device, torch.float32).clone()
        self.wilson_coef = torch.zeros(phi.shape[1], device=self.I_start.device)
        self.wilson_fit_iters = 0

    def enable_evidence_prior(self) -> nn.Parameter:
        """Make ``log Sigma = phi @ coef`` a trainable parameter, started at the
        current fit.  Needs the smooth basis (a shells prior has no surface)."""
        if not self.has_basis:
            raise ValueError("an evidence-fitted Wilson prior needs the smooth reciprocal basis")
        if self.prior != "wilson":
            raise ValueError("an evidence-fitted Wilson prior needs prior='wilson'")
        self.wilson_coef_param = nn.Parameter(self.wilson_coef.detach().clone())
        return self.wilson_coef_param

    @torch.no_grad()
    def sync_evidence_prior(self) -> None:
        """Mirror the trained coefficients into ``wilson_coef`` and ``wilson_sigma``."""
        if self.wilson_coef_param is not None:
            self.set_wilson_coef(self.wilson_coef_param)

    @torch.no_grad()
    def set_wilson_coef(self, coef: Tensor) -> None:
        """Install surface coefficients: ``wilson_sigma = exp(phi @ coef)``."""
        if not self.has_basis or coef.shape != self.wilson_coef.shape:
            raise ValueError("coefficients need the installed basis and its width")
        self.wilson_coef.copy_(coef)
        self.wilson_sigma.copy_(
            (self.wilson_phi.double() @ self.wilson_coef.double()).exp().float().clamp(min=self.floor))

    @torch.no_grad()
    def fit_wilson_surface(self, reference: Tensor | None = None) -> None:
        """Fit ``log Sigma = phi @ coef`` to ``reference`` (default the unity
        merge) by Wilson maximum likelihood: the start of the evidence fit."""
        if not self.has_basis:
            raise ValueError("the Wilson surface needs the smooth reciprocal basis (set_basis)")
        from torchsx.merging.priors.reciprocal_basis import fit_wilson_log_sigma
        reference = self.I_start if reference is None else reference
        coef, used = fit_wilson_log_sigma(
            self.wilson_phi, reference.clamp(min=self.floor), self.is_centric)
        self.set_wilson_coef(coef)
        self.wilson_fit_iters = int(used)

    @torch.no_grad()
    def set_is_centric(self, is_centric: Tensor) -> None:
        """Mark which reflections take the centric Wilson prior."""
        self.is_centric.copy_(is_centric.to(self.is_centric.device).bool())

    def _prior_rate(self) -> Tensor:
        """``1/Sigma`` acentric, ``1/(2 Sigma)`` centric."""
        sig = self.wilson_sigma.clamp(min=self.floor)
        return torch.where(self.is_centric, 0.5 / sig, 1.0 / sig)

    def accumulate(
        self, objective, scatter, params, pred,
        n_ph: Tensor, bg: Tensor, hkl_id: Tensor, meta, v_nodes: Tensor,
    ) -> Tensor:
        """Coarse log-likelihood curve summed over observations: ``(n_hkl, K)``.

        The only inner-quadrature work.  Each column is accumulated separately,
        so nothing of size ``(n_obs, k_coarse)`` is ever allocated.

        **Additive over observations**, hence over batches — which is what lets
        a merge over an arbitrary crystal subset reuse this directly instead of
        reimplementing an estimator.  Accumulate ``S`` across batches, then call
        :meth:`integrate` once.
        """
        base = pred.log_K - pred.log_e_s
        # Learned detector noise: per-panel (A, B, C), gathered once per batch and
        # grad mode by the objective (the node loop below reads the same terms).
        noise = objective.noise_terms(meta) if hasattr(objective, "noise_terms") else None

        if _all_nodes_kernel(objective, scatter, n_ph, bg):
            from torchsx.merging.quadrature._posterior_nodes_triton import (
                posterior_node_sums,
            )
            q, lik = objective.quadrature, objective.likelihood
            sig_b2 = q._sig_b2(lik, bg)
            p = q.grid_params(n_ph, bg, sig_b2, q._bg_mean_offset(lik), noise)
            olog = getattr(objective, "outlier_logit", None)
            return posterior_node_sums(
                base, scatter.mixture_obs(meta), objective.miss_logit, v_nodes, hkl_id,
                n_ph, p, q.n_points, sig_b2=sig_b2, n_hkl=self.n_hkl, outlier_logit=olog,
                log_out=objective.log_outlier(n_ph, bg) if olog is not None else None,
                noise=noise,
            )

        # Fused path -- OFF by default, not deprecated.  One launch
        # for all C nodes instead of C, hoisting the Poisson factor out of the
        # node loop and replacing the online-logsumexp maximum with a
        # closed-form bound.  Against the float64 oracle it gives *better*
        # values than the per-column path and is ~2x faster at C = 32, but its
        # gradient DIRECTION is orders of magnitude worse (about 5 degrees on
        # d/d log_alpha).
        #
        # Cause: dropping the per-(m, c) running max costs dynamic range.  When
        # one observation's accumulator underflows at a node, that node is
        # floored, S[h, c] is annihilated, and with it the gradient
        # contribution of every OTHER observation at that node.  The values
        # survive because the node carried little weight; the direction does
        # not.  Flooring higher instead over-estimates such nodes by hundreds
        # of nats, which is worse.  A periodic rescale would fix it and
        # would give back part of the speed the running max was costing.
        #
        # Speed also depends sharply on C, because the tile is next_pow2(C):
        # it only pays at a power-of-two k_coarse.
        #
        # Falls back silently -- it is a GenGauss x Poisson x FixedLogGrid
        # specialisation with no d/d bg statistic, so a learned background must
        # use the eager loop.
        if self.use_fused and noise is None:
            from torchsx.merging.likelihood import Poisson
            from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid
            from torchsx.merging.scatter.gengauss import GenGaussLogScatter
            q = getattr(objective, "quadrature", None)
            if (
                _HAVE_FUSED and n_ph.is_cuda
                and isinstance(q, FixedLogGrid)
                and isinstance(getattr(objective, "likelihood", None), Poisson)
                and isinstance(scatter, GenGaussLogScatter)
                and not getattr(objective, "renormalize", False)
                and not bg.requires_grad
                # The fused kernel hardcodes the plain Poisson integrand, so a
                # likelihood carrying extra structure -- e.g. bg_var_factor,
                # which marginalises the background estimate's own error --
                # would be SILENTLY ignored here while the eager loop honours
                # it.  Refuse the fast path rather than compute a different
                # quantity from the one the loss asked for.
                and not float(getattr(objective.likelihood, "bg_var_factor",
                                      0.0) or 0.0)
                # Same argument for the background MEAN: the kernel's
                # `lam = S + bg` is the raw estimate, so a bg_mean_offset would
                # be dropped here while the loss honours it.
                and not float(getattr(objective.likelihood, "bg_mean_offset",
                                      0.0) or 0.0)
            ):
                la, lb = scatter.log_alpha_beta_obs(params, meta)
                lp = fused_posterior_logp(
                    base.contiguous(), n_ph, bg, la, lb, hkl_id, v_nodes,
                    q.log_S_grid, q._trap_log_w(q.log_S_grid, q.n_points),
                )
                out = base.new_zeros(self.n_hkl, v_nodes.shape[1])
                return out.index_add(0, hkl_id, lp)

        cols = []
        ckpt = self.checkpoint_nodes and torch.is_grad_enabled() and base.requires_grad
        for c in range(v_nodes.shape[1]):   # not k_coarse: locate() uses its own width
            if ckpt:
                # Each node runs the WHOLE inner marginal, and the graphs of all
                # C nodes are otherwise alive at once.  The encoder and the
                # marginal's observation chunks are already checkpointed; this
                # loop was the one stage that was not, and it is the natural
                # place for it -- the accumulation is additive over nodes, so a
                # node can be recomputed in isolation from (base, v_c) alone.
                # Pass the node INDEX, not v_c: v_c is (M,) and storing one
                # per node is 4 B/obs x C, which at C=31 is as large as the
                # aux statistics we are trying to avoid keeping.  v_nodes is
                # (n_hkl, C) and costs nothing to close over.
                f = torch.utils.checkpoint.checkpoint(
                    lambda b, _c=c: objective.log_prob(
                        scatter, params,
                        v_nodes[:, _c].index_select(0, hkl_id) + b,
                        n_ph, bg, meta),
                    base, use_reentrant=False,
                )
            else:
                v_c = v_nodes[:, c].index_select(0, hkl_id)
                f = objective.log_prob(scatter, params, v_c + base, n_ph, bg, meta)
            cols.append(group_sum(f, hkl_id, self.n_hkl))
        return torch.stack(cols, dim=1)

    def integrate(
        self, S: Tensor, loc: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Coarse curve -> ``(log_Z, E[I0], posterior weights, v, edge_mass)``.

        ``loc = (centre, width)`` is the grid ``S`` was accumulated on (default
        the last one placed, ``_loc``).  Shared by the loss and by the merge, so
        the quantity being scored is by construction the quantity being
        minimised.
        """
        centre, sig_h = self._loc if loc is None else loc
        S_fine = self._interp(S)
        v = centre[:, None] + sig_h[:, None] * self.u_fine[None, :]
        du = float(2 * self.n_sigma / (self.m_fine - 1))
        log_w = (sig_h * du).log()[:, None].expand_as(v).clone()
        if self.prior in ("none", "flat_I0"):
            log_w = log_w + v                       # dI0 = I0 dv
        elif self.prior == "wilson":
            # the trained parameter while training; every no-grad pass
            # (evaluation, merge, readout) reads the mirrored buffers
            if self.wilson_coef_param is not None and torch.is_grad_enabled():
                log_sig = (self.wilson_phi @ self.wilson_coef_param).clamp(
                    min=math.log(self.floor))[:, None]
                inv = (-log_sig).exp()
                rate = torch.where(self.is_centric[:, None], 0.5 * inv, inv)
            else:
                rate = self._prior_rate()[:, None]
                log_sig = self.wilson_sigma.clamp(min=self.floor).log()[:, None]
            # v carries the dI = I dv Jacobian; the centric I^(-1/2) eats
            # half of it, leaving 0.5 * v.
            log_w = torch.where(
                self.is_centric[:, None],
                log_w + 0.5 * v - v.exp() * rate
                - 0.5 * (math.log(2 * math.pi) + log_sig),
                log_w + v - v.exp() * rate - log_sig,
            )
        log_w[:, 0] -= math.log(2.0)
        log_w[:, -1] -= math.log(2.0)

        logits = S_fine + log_w
        log_Z = torch.logsumexp(logits, dim=1)
        w_post = torch.softmax(logits, dim=1)
        # Moments in LOG space.  With n_sigma = 24 and the flat-curve fallback
        # width (scan_halfwidth / 2 = 4) the top node sits ~96 nats above the
        # centre, where ``v.exp()`` overflows float32 to inf; its posterior
        # weight is exactly 0 there, and 0 * inf = NaN; one NaN is enough to
        # take CC1/2 with it.  E[I] = exp(logsumexp(logits + v) - log Z) never
        # forms the overflowing value.
        I0_mean = (torch.logsumexp(logits + v, dim=1) - log_Z).exp()
        # sigI falls out of the same integral -- no separate error model, and
        # it is the posterior width rather than a propagated Laplace guess.
        # Nodes carrying no weight are clamped to a finite value so the
        # (I - mean)^2 term stays finite; where the clamp bites, w_post is 0.
        v_safe = v.clamp(max=(I0_mean.clamp(min=1e-30).log() + 20.0)[:, None])
        I0_sd = (
            (w_post * (v_safe.exp() - I0_mean[:, None]) ** 2).sum(dim=1).clamp(min=0)
        ).sqrt()
        n_edge = max(1, int(round(0.05 * logits.shape[1])))
        edge_mass = (
            w_post[:, :n_edge].sum(1) + w_post[:, -n_edge:].sum(1)
        ).detach()
        return log_Z, I0_mean, I0_sd, w_post, v, edge_mass

    def _interp(self, S_coarse: Tensor) -> Tensor:
        """``(n_hkl, k_coarse)`` -> ``(n_hkl, m_fine)``, linear on the u axis."""
        lo = S_coarse.index_select(1, self._lo)
        hi = S_coarse.index_select(1, self._hi)
        return lo + (hi - lo) * self._t[None, :]

    # ── forward ────────────────────────────────────────────────────────

    def forward(
        self, objective, scatter, params, pred,
        n_ph: Tensor, bg: Tensor, sigma: Tensor, hkl_id: Tensor, meta,
    ) -> PosteriorOut:
        """Infer this batch on a grid placed by the batch itself."""
        # Measure the curve around the batch's own weighted mean, then place
        # the grid on it.  One extra sweep of scan_points instead of a
        # closed-form width.
        start = self.batch_centre(pred, n_ph - bg, sigma, hkl_id)
        centre, sig_h = self.locate(
            objective, scatter, params, pred, n_ph, bg, hkl_id, meta, start
        )
        self._loc = (centre, sig_h)
        v_nodes = centre[:, None] + sig_h[:, None] * self.u_coarse[None, :]
        S = self.accumulate(
            objective, scatter, params, pred, n_ph, bg, hkl_id, meta, v_nodes
        )

        log_Z, I0_mean, I0_sd, w_post, v, edge_mass = self.integrate(S)
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=n_ph.device)
        seen[hkl_id] = True
        log_Z = torch.where(
            seen, log_Z,
            torch.full((self.n_hkl,), float("-inf"),
                       device=log_Z.device, dtype=log_Z.dtype),
        )
        floored = I0_mean < self.floor
        I0_mean = torch.where(seen, I0_mean.clamp(min=self.floor), self.I_start)
        return PosteriorOut(
            log_Z=log_Z, I0_mean=I0_mean, I0_sd=I0_sd,
            I0_obs=I0_mean.index_select(0, hkl_id),
            seen=seen, edge_mass=edge_mass,
            n_floored=int((floored & seen).sum()),
        )
