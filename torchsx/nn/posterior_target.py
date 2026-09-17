"""Merge target from the I0 posterior — integrated, never fitted.

The leave-one-out estimator in :mod:`torchsx.nn.merge_target` is the argmin of a
**Gaussian weighted-least-squares surrogate**, while the objective it feeds is
the **GenGauss marginal**.  Different optima, and that mismatch is the only
reason detaching the target is approximate: were I0 the argmax of the *same*
objective, ``dL/dI0 = 0`` and the envelope theorem would make detaching exact.

This module removes the mismatch from the other end — it never fits I0 at all.

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
Two measured facts collapse it:

1. **Individual observation curves are broad; only their product is narrow.**
   ``f_j(mu) = log p(n'_j | mu)`` has the width of one observation's
   uncertainty, while the posterior is ~sqrt(n) narrower.  So evaluate ``f_j``
   on a **coarse** grid and linearly interpolate to a fine one.  Measured
   requirement: ~25-33 coarse points (rel. error 5.6e-3 at 25, 8.2e-4 at 33),
   which is why the writeout pipeline in :mod:`torchsx.merging.posterior` uses
   ``K_coarse = 31``.
2. **Linear interpolation commutes with the sum over observations.**
   ``sum_j interp(f_j) = interp(sum_j f_j)``, so accumulate on the *coarse*
   grid first — ``(n_hkl, K_coarse)``, tiny — and interpolate that.  The
   ``(n_obs, M_fine)`` tensor never exists.

With the Triton inner kernel the remaining cost is small: it never materialises
the ``(M, K_inner)`` integrand (it saves six tensors of size M), so a pass is
~60 MB of traffic rather than ~1.8 GB eager, and 31 passes come to roughly one
eager pass — a few percent of a training step.  Use a **wide** FixedLogGrid so
the Triton path applies without truncating the scatter; the inner node count is
nearly free (K 501 -> 2001 costs +4 %).

Grid geometry (centre, width) is **detached**: it only places quadrature nodes,
exactly like choosing integration limits, and the answer is insensitive to it
provided the mass is covered.  ``edge_mass`` reports whether it is — measured
as weight in the outer 10 % of the *range*, which is node-count independent
(the outermost single node's weight shrinks with n_grid and would measure
resolution, not truncation).
"""

from __future__ import annotations

import math
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

    Parameters
    ----------
    I0_init
        ``(n_hkl,)`` seed intensities; the grid centre until the EMA moves.
    k_coarse
        Inner-quadrature evaluations per reflection.  The real cost knob.
    m_fine
        Interpolated integration points.  Nearly free — raise before ``k_coarse``.
    n_sigma
        Grid half-width in Laplace sigmas.  Measured: 5 suffices for the median
        but the 90th-percentile error is *width*-limited (flat in node count),
        so 8 is the safer default now that width is free.
    prior
        ``"none"`` integrates ``dI0`` (weight ``exp(v)``), with no Wilson
        density; ``"flat_I0"`` is its legacy alias. ``"flat_log"``
        integrates ``d log I0``; ``"wilson"`` uses ``p(I) = exp(-I/Sigma)/Sigma``
        with ``Sigma`` the shell mean.  **Use wilson.**  Where the likelihood is
        uninformative — and 84 % of observations here are noise-limited — the
        posterior is whatever the prior says, and a flat-in-I0 prior on a log
        grid drags those reflections toward the *top of the grid*.  Measured
        consequence: per-shell CC1/2 improved at low resolution
        (0.776 -> 0.821) while collapsing at high resolution
        (0.359 -> 0.246, 0.159 -> 0.094), because exactly the weak reflections
        were being inflated.
    node_placement
        ``"laplace"`` spaces nodes uniformly in ``log I0`` over
        ``+- n_sigma * sigma``; ``"wilson"`` places them at quantiles of the
        **approximate posterior** (Wilson prior x Gaussian likelihood, which is
        a Gaussian truncated at zero — i.e. French-Wilson).

        **``"wilson"`` is currently BROKEN — do not use.**  It gives a 58-65 %
        systematic error against the Laplace reference and does *not* improve
        with node count (0.650 / 0.611 / 0.577 at 9 / 17 / 31), so the fault is
        in the importance weights ``du * p_W / q``, not the resolution.  Left in
        place because the idea is right and the bug is findable.

        The idea: nodes drawn through an inverse CDF live on ``u in (0, 1)``, so
        the grid **cannot truncate**.  That deletes the
        extent choice, the sigma estimate, and the ratchet that existed only to
        stop the extent oscillating.  It interpolates correctly between regimes
        — an uninformative reflection gets prior-like placement, a
        well-determined one likelihood-like — which pure prior placement would
        not (with 751 observations per reflection the posterior is far narrower
        than the Wilson prior, so uniform-in-prior nodes would waste almost all
        of themselves).
    """

    def __init__(
        self,
        I0_init: Tensor,
        *,
        k_coarse: int = 31,
        m_fine: int = 257,
        n_sigma: float = 8.0,
        prior: str = "wilson",
        node_placement: str = "laplace",
        ema_decay: float = 0.99,
        reference_estimator: str = "wls",
        floor: float = 1e-3,
        sigma_bounds: tuple[float, float] = (0.02, 4.0),
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
        if node_placement not in ("laplace", "wilson"):
            raise ValueError(
                f"node_placement must be 'laplace' or 'wilson', got {node_placement!r}"
            )
        if node_placement == "wilson" and prior != "wilson":
            raise ValueError("Wilson node placement requires prior='wilson'; "
                             "use Laplace placement for prior='none'")
        if k_coarse < 3 or m_fine < k_coarse:
            raise ValueError("need k_coarse >= 3 and m_fine >= k_coarse")
        if reference_estimator not in ("wls", "posterior"):
            raise ValueError("reference_estimator must be 'wls' or 'posterior'")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be finite and in [0, 1)")
        self.reference_estimator = reference_estimator
        self.n_hkl = int(I0_init.shape[0])
        self.k_coarse, self.m_fine = int(k_coarse), int(m_fine)
        self.n_sigma, self.prior = float(n_sigma), prior
        self.node_placement = node_placement
        self.ema_decay, self.floor = float(ema_decay), float(floor)
        self.sigma_bounds = sigma_bounds
        self.scan_points, self.scan_halfwidth = int(scan_points), float(scan_halfwidth)
        # Cap on the outer grid HALF-WIDTH in nats.  ``n_sigma`` was raised to
        # 24 to stop truncating sharp posteriors whose Laplace width is
        # underestimated; applied to a BROAD width it does the opposite damage.
        # A weak reflection's curve is flat below and a cliff above, the
        # parabola reads a width of several nats, and 24 x 4 nats puts the 31
        # coarse nodes 6.4 nats apart -- the cliff is then linearly interpolated
        # over 6 nats and the posterior mean lands nats too high.  Measured on
        # the SACLA dark merge: 27 % of reflections sat on such a grid; R_free
        # 0.351 against partialator's 0.322.  The Wilson prior removes all mass
        # more than ~12 nats from the centre, so nothing beyond needs nodes.
        self.max_halfwidth = float(max_halfwidth)
        # Recompute each outer node in backward instead of holding all C graphs.
        # OFF by default: it roughly doubles the accumulate's compute, which is
        # not worth paying on a data set that already fits.  Turn it on when the
        # batch is memory-bound -- on a 114 k-reflection cell the in-batch
        # redundancy that fits on a 40 GB card is ~4x below what a 14 k-reflection
        # cell gets at the same batch size, and that is what it buys back.
        self.checkpoint_nodes = False
        self.use_fused = bool(use_fused)
        self.lam = 0.0                      # API parity with LOOInBatchTarget

        self.register_buffer("ema_I0", I0_init.detach().clone().float().clamp(min=floor))
        self.register_buffer("ema_den", torch.zeros(self.n_hkl))
        self.register_buffer("_den_seeded", torch.zeros((), dtype=torch.bool))
        # Grid width, learned from the OBSERVED posterior rather than predicted
        # from the Laplace approximation.  The Laplace sigma comes from the
        # Gaussian variance model; for weak reflections the true posterior is
        # wider and skewed, and the grid then clips it -- measured edge_mass up
        # to 0.93 on a minority of reflections even at +-8 sigma, with the
        # error tracking it exactly.  Measuring the width and rolling it forward
        # makes the grid self-tuning, and edge_mass reports whether it worked.
        self.register_buffer("ema_log_sigma", torch.zeros(self.n_hkl))
        self.register_buffer("_sigma_seeded", torch.zeros((), dtype=torch.bool))
        # Shell mean intensity Sigma for the Wilson prior; set via
        # set_wilson_sigma().  Defaults to the seed intensities, which is the
        # per-reflection rather than per-shell mean — adequate to start.
        self.register_buffer("wilson_sigma", I0_init.detach().clone().float().clamp(min=floor))
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
        # Smooth Sigma(h): when a reciprocal-space basis is installed through
        # set_basis(), set_wilson_sigma() fits log Sigma = phi @ coef by Wilson
        # maximum likelihood instead of taking equal-count shell means.  Both
        # buffers stay empty in shells mode so old checkpoints load unchanged.
        self.register_buffer("wilson_phi", torch.zeros(0, 0))
        self.register_buffer("wilson_coef", torch.zeros(0))
        self.wilson_fit_iters = 0

        self.set_grid(k_coarse, m_fine, n_sigma)

    def set_grid(self, k_coarse: int, m_fine: int, n_sigma: float) -> None:
        """Replace node buffers and interpolation tables together at readout."""
        if k_coarse < 3 or m_fine < k_coarse or not math.isfinite(n_sigma) or n_sigma <= 0:
            raise ValueError("need k_coarse >= 3, m_fine >= k_coarse and finite n_sigma > 0")
        self.k_coarse, self.m_fine, self.n_sigma = int(k_coarse), int(m_fine), float(n_sigma)
        device = self.ema_I0.device
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
        n_ph: Tensor, bg: Tensor, hkl_id: Tensor, meta,
    ) -> tuple[Tensor, Tensor]:
        """Find the peak and width of the log-likelihood curve by MEASURING it.

        Three formulas for the grid width have now failed in three different
        directions -- too narrow (edge_mass 0.93), oscillating (a
        follow-the-measurement loop swinging 0.29 / 0.03 / 0.52), and too wide
        (sigma pinned at its bound, half-width 30 in log I0, so the merge
        returned the grid centre and the data contributed nothing).  Every one
        was a *formula* for the posterior width derived from the Gaussian
        variance model, and the posterior is not that shape.

        So: sweep a deliberately generous grid once, then read the peak and
        curvature off the curve itself.  A parabola through the maximum and its
        neighbours gives ``v*`` and ``s = 1/sqrt(-S'')`` with no distributional
        assumption and nothing that can pin at a bound.
        """
        v_scan = self.scan_grid()
        S = self.accumulate(
            objective, scatter, params, pred, n_ph, bg, hkl_id, meta, v_scan
        )
        return self.peak_from_scan(S)

    @torch.no_grad()
    def scan_grid(self) -> Tensor:
        """Generous grid for the locating sweep: ``(n_hkl, scan_points)``."""
        centre = self.ema_I0.clamp(min=self.floor).log()
        u = torch.linspace(
            -self.scan_halfwidth, self.scan_halfwidth, self.scan_points,
            device=centre.device,
        )
        return centre[:, None] + u[None, :]

    @torch.no_grad()
    def peak_from_scan(self, S: Tensor) -> tuple[Tensor, Tensor]:
        """Peak location and width read off an accumulated scan curve.

        Separate from :meth:`locate` so a caller that spans several batches can
        accumulate ``S`` itself first — the merge does, and previously called
        ``nodes()`` instead, which still used the old closed-form width.  That
        placed the merge grid by the very formula this replaces, and the merge
        collapsed back onto the grid centre: per-shell CC1/2 read exactly
        1.0000 in five shells, because both halves returned the same shared
        ``ema_I0`` rather than anything from the data.
        """
        centre = self.ema_I0.clamp(min=self.floor).log()
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
        # correlation.  Measured: the two weakest shells fell 0.82 -> 0.73 and
        # 0.69 -> 0.50 while shells 2-8 improved.
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
        # Linear interpolation across the cliff then cut the posterior off
        # nats below the true bound -- 27 % of the SACLA dark reflections came
        # out collapsed to ~0 and R_free was 0.351 against 0.322 for
        # partialator.  Anchor these on the plateau EDGE instead: the last scan
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

    @torch.no_grad()
    def _centre_sigma(self) -> tuple[Tensor, Tensor]:
        I0 = self.ema_I0.clamp(min=self.floor)
        if not bool(self._sigma_seeded):
            # Bootstrap from Laplace: Var(I0_hat) ~ 1/den => sigma ~ 1/(I0 sqrt(den))
            den = self.ema_den.clamp(min=1e-12)
            self.ema_log_sigma.copy_(
                (1.0 / (I0 * den.sqrt())).clamp(*self.sigma_bounds).log()
            )
            self._sigma_seeded.fill_(True)
        return I0.log(), self.ema_log_sigma.exp().clamp(*self.sigma_bounds)

    @property
    def has_basis(self) -> bool:
        return self.wilson_phi.numel() > 0

    @torch.no_grad()
    def set_basis(self, phi: Tensor) -> None:
        """Install the smooth basis ``phi`` (``n_hkl x p``); Sigma is fitted on it from now on."""
        if phi.shape[0] != self.n_hkl:
            raise ValueError(f"basis has {phi.shape[0]} rows for {self.n_hkl} reflections")
        self.wilson_phi = phi.detach().to(self.ema_I0.device, torch.float32).clone()
        self.wilson_coef = torch.zeros(phi.shape[1], device=self.ema_I0.device)
        self.wilson_fit_iters = 0

    @torch.no_grad()
    def set_wilson_sigma(self, shell_id: Tensor, n_shells: int, *,
                         reference: Tensor | None = None,
                         sample_weight: Tensor | None = None) -> None:
        """Refresh ``Sigma`` from the running merge.

        Shells mode: the equal-count shell mean of ``ema_I0`` (``shell_id`` /
        ``n_shells``).  With a basis installed the arguments are ignored and
        ``log Sigma = phi @ coef`` is refitted by Wilson maximum likelihood,
        warm-started from the previous coefficients.
        """
        reference = self.ema_I0 if reference is None else reference
        if self.has_basis:
            from torchsx.merging.priors.reciprocal_basis import fit_wilson_log_sigma
            coef0 = self.wilson_coef if bool(self.wilson_coef.abs().sum() > 0) else None
            coef, used = fit_wilson_log_sigma(
                self.wilson_phi, reference.clamp(min=self.floor), self.is_centric, coef0,
                sample_weight=sample_weight,
            )
            self.wilson_coef.copy_(coef)
            self.wilson_fit_iters = int(used)
            self.wilson_sigma.copy_(
                (self.wilson_phi.double() @ coef.double()).exp().float().clamp(min=self.floor)
            )
            return
        m = torch.zeros(n_shells, device=self.ema_I0.device)
        c = torch.zeros(n_shells, device=self.ema_I0.device)
        weights = torch.ones_like(reference) if sample_weight is None else sample_weight
        m.index_add_(0, shell_id, reference * weights)
        c.index_add_(0, shell_id, weights)
        fitted = (m / c.clamp(min=1e-30))[shell_id]
        self.wilson_sigma.copy_(torch.where(c[shell_id] > 0, fitted, self.wilson_sigma).clamp(min=self.floor))

    @torch.no_grad()
    def set_is_centric(self, is_centric: Tensor) -> None:
        """Mark which reflections take the centric Wilson prior."""
        self.is_centric.copy_(is_centric.to(self.is_centric.device).bool())

    def _prior_rate(self) -> Tensor:
        """``1/Sigma`` acentric, ``1/(2 Sigma)`` centric."""
        sig = self.wilson_sigma.clamp(min=self.floor)
        return torch.where(self.is_centric, 0.5 / sig, 1.0 / sig)

    @torch.no_grad()
    def _wilson_nodes(self) -> tuple[Tensor, Tensor]:
        """Nodes at quantiles of the French-Wilson approximate posterior.

        Wilson prior ``exp(-I/Sigma)`` times a Gaussian likelihood ``N(Ihat, s)``
        is a Gaussian truncated at zero with mean ``Ihat - s^2/Sigma`` and the
        same ``s``.  Drawing nodes through its inverse CDF means the domain is
        ``u in (0,1)``: no extent to choose, and no truncation possible.
        """
        Ihat = self.ema_I0.clamp(min=self.floor)
        sd = (1.0 / self.ema_den.clamp(min=1e-12).sqrt()).clamp(min=1e-6)
        mu = Ihat - sd**2 * self._prior_rate()

        nrm = torch.distributions.Normal(0.0, 1.0)
        a = (-mu / sd)                                   # lower truncation in z
        Fa = nrm.cdf(a)
        # stratified midpoints avoid the u->0/1 endpoints where Phi^-1 diverges
        u = (torch.arange(self.k_coarse, device=Ihat.device) + 0.5) / self.k_coarse
        z = nrm.icdf((Fa[:, None] + u[None, :] * (1.0 - Fa[:, None])).clamp(1e-6, 1 - 1e-6))
        I_nodes = (mu[:, None] + sd[:, None] * z).clamp(min=self.floor)
        v = I_nodes.log()

        # importance weights: dI-integral of L*p_W with proposal q
        #   w_k = du * p_W(I_k) / q(I_k)
        rate = self._prior_rate()[:, None]
        log_sig = self.wilson_sigma.clamp(min=self.floor).log()[:, None]
        log_pw = torch.where(
            self.is_centric[:, None],
            -I_nodes * rate - 0.5 * (math.log(2 * math.pi) + log_sig
                                     + I_nodes.clamp(min=1e-30).log()),
            -I_nodes * rate - log_sig,
        )
        log_q = (
            nrm.log_prob((I_nodes - mu[:, None]) / sd[:, None])
            - sd.log()[:, None] - torch.log1p(-Fa)[:, None]
        )
        log_w = math.log(1.0 / self.k_coarse) + log_pw - log_q
        return v, log_w

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

        # Fused path -- OFF by default; measured, not deprecated.  One launch
        # for all C nodes instead of C, hoisting the Poisson factor out of the
        # node loop and replacing the online-logsumexp maximum with a
        # closed-form bound.  Against the float64 oracle it gives *better*
        # values than the per-column path (I0_mean 3.1e-4 vs 6.0e-4 relative,
        # I0_sd 6.9e-4 vs 1.1e-3) and is 2.08x faster at C = 32, but its
        # gradient DIRECTION is 46-880x worse (1-cos on d/d log_alpha 3.3e-3 vs
        # 5.0e-6, about 5 degrees).
        #
        # Cause: dropping the per-(m, c) running max costs dynamic range.  When
        # one observation's accumulator underflows at a node, that node is
        # floored, S[h, c] is annihilated, and with it the gradient
        # contribution of every OTHER observation at that node.  The values
        # survive because the node carried little weight; the direction does
        # not.  Flooring higher instead over-estimates such nodes by up to 233
        # nats (measured), which is worse.  A periodic rescale would fix it and
        # would give back part of the speed the running max was costing.
        #
        # Speed also depends sharply on C: 2.08x at 32 and 2.10x at 64, but
        # 1.03x at 21 and 0.61x at 33, because the tile is next_pow2(C).  So it
        # only pays at a power-of-two k_coarse.
        #
        # Falls back silently -- it is a GenGauss x Poisson x FixedLogGrid
        # specialisation with no d/d bg statistic, so a learned background must
        # use the eager loop.
        if self.use_fused:
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
        self, S: Tensor, v_nodes: Tensor, log_w_nodes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Coarse curve -> ``(log_Z, E[I0], posterior weights, v, edge_mass)``.

        Shared by the loss and by the merge, so the quantity being scored is by
        construction the quantity being minimised.
        """
        if log_w_nodes is None:
            # uniform-in-u placement: interpolate to the fine grid, then weight
            centre, sig_h = getattr(self, "_loc", None) or self._centre_sigma()
            S_fine = self._interp(S)
            v = centre[:, None] + sig_h[:, None] * self.u_fine[None, :]
            du = float(2 * self.n_sigma / (self.m_fine - 1))
            log_w = (sig_h * du).log()[:, None].expand_as(v).clone()
            if self.prior in ("none", "flat_I0"):
                log_w = log_w + v                       # dI0 = I0 dv
            elif self.prior == "wilson":
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
        else:
            S_fine, v, log_w = S, v_nodes, log_w_nodes

        logits = S_fine + log_w
        log_Z = torch.logsumexp(logits, dim=1)
        w_post = torch.softmax(logits, dim=1)
        # Moments in LOG space.  With n_sigma = 24 and the flat-curve fallback
        # width (scan_halfwidth / 2 = 4) the top node sits ~96 nats above the
        # centre, where ``v.exp()`` overflows float32 to inf; its posterior
        # weight is exactly 0 there, and 0 * inf = NaN.  Measured on the SACLA
        # light merge: 28 k of 67 k reflections NaN, and one NaN is enough to
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

    def nodes(self) -> tuple[Tensor, Tensor | None]:
        """Integration nodes ``v`` and their log-weights (``None`` -> uniform)."""
        if self.node_placement == "wilson":
            return self._wilson_nodes()
        centre, sig_h = self._centre_sigma()
        return centre[:, None] + sig_h[:, None] * self.u_coarse[None, :], None

    def _interp(self, S_coarse: Tensor) -> Tensor:
        """``(n_hkl, k_coarse)`` -> ``(n_hkl, m_fine)``, linear on the u axis."""
        lo = S_coarse.index_select(1, self._lo)
        hi = S_coarse.index_select(1, self._hi)
        return lo + (hi - lo) * self._t[None, :]

    # ── forward ────────────────────────────────────────────────────────

    def forward(
        self, objective, scatter, params, pred,
        n_ph: Tensor, bg: Tensor, sigma: Tensor, hkl_id: Tensor, meta,
        *, update_state: bool = True,
    ) -> PosteriorOut:
        """Infer this batch; advance running references only for training batches."""
        if self.node_placement == "wilson":
            # Importance placement: nodes already sit where the mass is, so the
            # coarse->fine interpolation that compensates for a badly placed
            # uniform grid is unnecessary.  Integrate on the nodes directly.
            v_nodes, log_w_nodes = self._wilson_nodes()
            interpolate = False
        else:
            # Measure the curve, then place the grid on it.  One extra sweep of
            # scan_points, against three failed attempts at a closed-form width.
            centre, sig_h = self.locate(
                objective, scatter, params, pred, n_ph, bg, hkl_id, meta
            )
            self._loc = (centre, sig_h)
            v_nodes = centre[:, None] + sig_h[:, None] * self.u_coarse[None, :]
            log_w_nodes = None
            interpolate = True

        S = self.accumulate(
            objective, scatter, params, pred, n_ph, bg, hkl_id, meta, v_nodes
        )

        log_Z, I0_mean, I0_sd, w_post, v, edge_mass = self.integrate(
            S, v_nodes, log_w_nodes
        )
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=n_ph.device)
        seen[hkl_id] = True
        log_Z = torch.where(
            seen, log_Z,
            torch.full((self.n_hkl,), float("-inf"),
                       device=log_Z.device, dtype=log_Z.dtype),
        )

        floored = I0_mean < self.floor
        I0_mean = torch.where(seen, I0_mean.clamp(min=self.floor), self.ema_I0)
        if update_state:
            self._update_ema(pred, n_ph - bg, sigma, hkl_id,
                             posterior_mean=I0_mean, posterior_seen=seen)
        if update_state and self.node_placement == "laplace":
            # Only the uniform-grid path has an extent to maintain.  Wilson
            # placement draws nodes through an inverse CDF over u in (0,1), so
            # there is nothing to widen or shrink and nothing to truncate.
            self._update_width(w_post, v, seen, edge_mass)

        return PosteriorOut(
            log_Z=log_Z, I0_mean=I0_mean, I0_sd=I0_sd,
            I0_obs=I0_mean.index_select(0, hkl_id),
            seen=seen, edge_mass=edge_mass,
            n_floored=int((floored & seen).sum()),
        )

    @torch.no_grad()
    def _update_width(self, w_post, v, seen, edge_mass) -> None:
        """Roll the grid width forward from the posterior we just measured.

        Uses the observed posterior sd in ``log I0``, inflated where the grid
        clipped: if ``edge_mass`` is large the measured sd is itself an
        underestimate (the tails were cut off), so widen rather than trust it.
        """
        mean = (w_post * v).sum(1)
        sd = ((w_post * (v - mean[:, None]) ** 2).sum(1)).clamp(min=1e-12).sqrt()
        cur = self.ema_log_sigma.exp()

        # Ratchet, not a symmetric update.  A naive "measure and follow" loop
        # oscillates (measured edge_mass 0.29 -> 0.03 -> 0.52 -> 0.01 -> 0.53):
        # a clipped grid under-reports its own width, the correction overshoots,
        # the now-wide grid measures a small sd, and it clips again.  So widen
        # hard when clipping is detected and shrink only slowly and only when it
        # is not, which is monotone enough to settle.
        clipped = edge_mass > 0.01
        widened = cur * (1.0 + 2.0 * edge_mass)
        shrunk = torch.maximum(0.98 * cur, sd)          # never below the measurement
        target = torch.where(clipped, widened, shrunk).clamp(*self.sigma_bounds)
        self.ema_log_sigma[seen] = target[seen].log()

    @torch.no_grad()
    def _update_ema(self, pred, n_signal, sigma, hkl_id, *,
                    posterior_mean=None, posterior_seen=None) -> None:
        """Update a detached running mean, once per observed training batch.

        Posterior mode reuses this batch's integrated mean. It averages estimates;
        it does not multiply old posteriors into the next batch's prior. Missing
        reflections retain their reference. Decay 0.99 gives each update 1% weight.
        The WLS denominator remains an auxiliary grid-placement statistic in both
        modes, so changing the mean estimator does not change its definition.
        """
        K = pred.log_K.exp().clamp(min=1e-8)
        I0_ref = self.ema_I0[hkl_id].clamp(min=self.floor)
        w2 = (pred.log_w.exp() ** 2).clamp(max=20.0)
        # Preserve the historical auxiliary weights in both modes. They omit
        # background-estimate variance. Posterior means, in contrast, integrate
        # the full likelihood. In WLS mode these weights also define the mean
        # used by network context and Wilson fitting.
        var = (sigma**2 + (K * I0_ref) ** 2 * torch.expm1(w2)).clamp(min=1e-12)
        den = group_sum(K * K / var, hkl_id, self.n_hkl)
        if self.reference_estimator == "posterior":
            if posterior_mean is None or posterior_seen is None:
                raise ValueError("posterior reference requires integrated means and seen mask")
            seen = posterior_seen
            batch = posterior_mean.detach().clamp(min=self.floor)
        else:
            num = group_sum(K * n_signal / var, hkl_id, self.n_hkl)
            seen = den > 0
            batch = torch.zeros_like(self.ema_I0)
            batch[seen] = (num[seen] / den[seen]).clamp(min=self.floor)
        if not bool(self._den_seeded):
            self.ema_den.copy_(den)
            self._den_seeded.fill_(True)
        if not bool(seen.any()):
            return
        d = self.ema_decay
        self.ema_I0[seen] = d * self.ema_I0[seen] + (1 - d) * batch[seen]
        self.ema_den[seen] = d * self.ema_den[seen] + (1 - d) * den[seen]
