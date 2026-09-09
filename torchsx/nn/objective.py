"""Training objectives for the amortised scale model.

Two implementations behind one seam.

:class:`MarginalNLL` — **primary**.  The proper marginal likelihood,
integrating the multiplicative scatter out of the detection likelihood::

    log p_j = log INT p_det(n_j | S, bg_j) . GenGauss(log S - log_mean_j; a_j, b_j) d log S

evaluated by the existing quadrature.  On CUDA this dispatches to the Triton
kernel, which takes **per-observation** ``log_alpha_obs`` / ``log_beta_obs``,
never materialises the ``(M, K)`` integrand (it saves four aux statistics,
~5 floats per observation) and returns analytic gradients for both.  So the
marginal is the *cheap* option here — the quadrature costs ~30 MB at 1.46 M
observations against several GB of network activations — and it is what makes
``beta`` identifiable, which a Gaussian closed form cannot be.

:class:`AgreementNLL` — reference.  Moment-matched Gaussian overlap between the
observations of a reflection, in closed form, no grid::

    -0.5 * sum_j (y_j - ybar)^2 / tau_j^2      agreement (chi^2)
    -0.5 * sum_j log tau_j^2                   no false precision
    -0.5 * log sum_j (1/tau_j^2)               Cox-Reid correction

Grid-free and exactly differentiable, so it is a fast dev loop and a
cross-check: at ``beta = 2`` and high counts the two must agree.  Its third
term is worth understanding even though the primary objective does not use it —
there is one nuisance parameter per reflection (~19,030, growing with the ASU),
so this is the Neyman-Scott incidental-parameter problem, and profiling I0 out
by maximisation biases exactly the quantity we intend to judge the model by,
the scatter width.  In :class:`MarginalNLL` leave-one-out plays that role:
every observation is scored against a target it did not contribute to.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.scatter.mixture import PredictedMixtureScatter
from torchsx.nn.pooling import group_sum

__all__ = [
    "MarginalNLL", "AgreementNLL", "gengauss_grid_log_mass", "wide_log_grid",
]

_LOG_2PI = math.log(2.0 * math.pi)


def wide_log_grid(n_points: int = 2001):
    """DEPRECATED — widening a *fixed* grid is not the fix.  Use
    :class:`~torchsx.merging.quadrature.wide_log_grid.WideLogGrid`, whose grid
    is per observation (``log_S = log_mean + alpha*u``) so it follows the
    integrand.

    Widening a fixed grid pads it with nodes where the detector likelihood and
    the scatter prior have no overlap at all, wasting resolution for nothing.

    It also removes an accidental brake.  Measured on the real dark: with the
    default narrow grid the width head ran to w = 6.75; with this wide fixed
    grid it ran to **26.96**, with beta pinned at the 4.0 ceiling.  Truncation
    was penalising ``alpha`` above ~2.5 and capping the runaway.

    **The runaway is not a quadrature artefact.**  A side-by-side sweep of
    NLL vs w over Trapezoidal, Wide and Fixed grids agrees to <0.01 nat and
    puts the minimum in the same place for all three, with a clean rise on
    either side.  The integral is fine; what runs away is the *joint* training
    dynamics of K, the leave-one-out target and w, which the static sweep
    cannot see because it holds ``log_mean`` at the converged value.

    Prefer :class:`~torchsx.merging.quadrature.trapezoidal.TrapezoidalLogGrid`,
    the pipeline default: its grid is per observation and data-driven, so nodes
    land where the detector likelihood has mass.

    A ``FixedLogGrid`` wide enough that the scatter never falls off it.

    The default grid spans ``log S`` in [-6.9, 13.8] — only 20.7 wide — so a
    GenGauss with ``alpha`` above ~2.5 already loses mass off the ends and the
    marginal stops being a density (measured: 81.4 % of the mass in-grid at
    w = 6.75, 54.7 % at w = 12).

    This spans [-34.5, 34.5], which holds +-5 sigma for widths up to w ~ 5 —
    comfortably beyond anything physical, since w = 5 already means the scale
    varies by a factor of e^5 = 148.  At ``n_points=2001`` the spacing is 0.034,
    slightly *finer* than the default grid's 0.041, and the runtime sweep showed
    the node count barely matters (K 501 -> 10001 cost 26 % wall clock).
    """
    from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid
    import math as _m
    return FixedLogGrid(S_min=_m.exp(-34.5), S_max=_m.exp(34.5), n_points=n_points)


def gengauss_grid_log_mass(
    log_mean: Tensor, log_alpha: Tensor, beta: Tensor, lo: float, hi: float
) -> Tensor:
    """``log`` of the GenGauss mass lying inside ``[lo, hi]`` in ``log S``.

    The quadrature integrates the scatter over a **finite** grid, so whatever
    mass falls off the ends is silently discarded and the "likelihood" stops
    being a density.  Measured on the fixed grid (``log S`` in [-6.9, 13.8]):
    ``Sum_n p(n)`` is 1.000 at ``w = 0.2`` but 0.941 at ``w = 1.65``, 0.586 at
    ``w = 6.75`` and 0.390 at ``w = 12``.  At ``w = 6.75, beta = 3.5`` the
    density is near-uniform across ``2*alpha = 22.8``, wider than the whole
    grid.

    **Use this as a diagnostic, not as a correction.**  Dividing ``p`` by this
    mass looks like it restores a density — ``p(n | S in grid)`` — but the
    conditioning event *moves with the parameters*, so it is not a valid
    likelihood across them.  Worse, it is unbounded in the wrong direction:
    ``log p - log mass`` goes to ``+inf`` as the mass goes to zero, so the
    width head is *rewarded* for growing the scatter past the grid.  Tried on
    the real dark, it drove the loss to 0 and NaN-ed the network within one
    epoch.

    The honest reading of a mass below 1 is "the model is putting probability
    where we are not integrating", and the fix is a wider grid (see
    ``WIDE_LOG_GRID``), not a correction factor.  The GenGauss CDF is closed form:

    .. math::
        F(t) = \tfrac12 + \tfrac12 \,\mathrm{sign}(t)\, P(1/\beta, |t|^\beta)

    with ``P`` the regularised lower incomplete gamma, so this costs two
    ``gammainc`` calls and no extra grid work.  Validated against dense
    numerical integration of the density over the grid: agreement to ~1e-6
    relative for w in {1, 1.65, 6.75, 12} x beta in {1, 2, 3.5}.

    **Gradient caveat.**  ``torch.special.gammainc`` implements no derivative
    with respect to its shape argument, so ``beta`` is detached *in that slot
    only*.  Gradients still flow through ``alpha`` (via ``t``) and through
    ``beta`` in the exponent of ``|t|^beta``.  The omitted path is ``beta``'s
    effect on the tail shape of the normaliser, which is second order next to
    the width dependence that this term exists to penalise.  Monitor
    ``grid_mass_median``: while it stays near 1 the whole correction — and
    therefore the approximation — is immaterial.
    """
    alpha = log_alpha.exp().clamp(min=1e-8)
    inv_b = (1.0 / beta).detach()          # gammainc has no d/d(shape)

    def _cdf(bound: float) -> Tensor:
        t = (bound - log_mean) / alpha
        # |t|**beta overflows float32 whenever alpha is small -- t reaches ~1e9
        # and beta up to 4 takes it past 3.4e38, producing inf and then NaN
        # gradients.  That is exactly the regime where the scatter sits well
        # inside the grid and the correction is a no-op, so the NaNs were
        # arriving where the term does nothing.  Clamp in the exponent:
        # P(a, x) is 1.0 to float32 precision for x beyond ~1e2, so capping
        # x at e**20 is numerically identical and cannot overflow.
        log_t = t.abs().clamp(min=1e-30).log()
        x = (beta * log_t).clamp(max=20.0).exp()
        return 0.5 + 0.5 * torch.sign(t) * torch.special.gammainc(inv_b, x)

    mass = (_cdf(hi) - _cdf(lo)).clamp(min=1e-12)
    return mass.log()


class MarginalNLL(nn.Module):
    """Per-observation ``-log p`` via the existing quadrature stack.

    Parameters
    ----------
    quadrature
        Any :class:`~torchsx.merging.quadrature.base.Quadrature`.  Use
        ``FixedLogGrid`` on CUDA to get the Triton fast path.
    likelihood
        Detection likelihood.  ``Poisson`` for counts; a
        ``GaussianLikelihood`` subclass to consume supplied CrystFEL sigmas.
    chunk_size
        Observations per quadrature call, or ``None`` for one call.  Chunk
        boundaries must not split a crystal if the caller is also pooling — but
        the pooling happens upstream in the encoder, so any split is safe here.
    checkpoint_chunks
        Recompute each chunk's ``(M, K)`` integrand in the backward pass rather
        than storing it.  **Chunking alone saves nothing**: concatenating the
        chunk outputs keeps every chunk's graph alive until backward, so peak
        memory is the same as one call.  That OOM'd a 40 GB A100 at 1.5 M
        observations with a per-observation grid.  Mirrors the strategy of
        ``merging.targets.data._ChunkedDataNLL``.
    """

    def __init__(
        self, quadrature, likelihood, *,
        chunk_size: int | None = None, renormalize: bool = False,
        checkpoint_chunks: bool = True, miss_frac: float = 0.0,
    ) -> None:
        super().__init__()
        self.quadrature = quadrature
        self.likelihood = likelihood
        self.chunk_size = chunk_size
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.renormalize = bool(renormalize)
        self.last_grid_mass: Tensor | None = None
        #: Miss component.  A predicted reflection that recorded no signal at
        #: all -- masked pixels, a mis-prediction, a spot that never formed --
        #: is not a partial observation, but the scatter has to stretch its
        #: lower tail to cover it, and on strong predictions that costs the
        #: width its calibration (measured: 7 % of rhodopsin's >200-photon
        #: predictions had n <= bg).  With ``miss_frac > 0`` the likelihood is
        #: the mixture ``(1 - pi) p(n | S, scatter) + pi p(n | S = 0)``, with
        #: ``pi`` a learnable global scalar started at ``miss_frac``.  The
        #: second term does not depend on the reflection intensity, so on the
        #: posterior grid it is an additive per-observation constant and needs
        #: no quadrature.  0 disables it exactly.
        self.miss_logit: nn.Parameter | None = None
        if miss_frac > 0.0:
            if not (0.0 < miss_frac < 1.0):
                raise ValueError(f"miss_frac must be in (0, 1), got {miss_frac}")
            self.miss_logit = nn.Parameter(torch.tensor(math.log(miss_frac / (1.0 - miss_frac))))

    def _log_miss(self, n_ph: Tensor, bg: Tensor) -> Tensor:
        """``log p(n | S = 0)`` on the kernel's scale (Poisson about its peak,
        with the background-variance term when the likelihood carries one)."""
        n = n_ph.to(bg.dtype)
        lam0 = bg.clamp(min=1e-6)
        bvf = float(getattr(self.likelihood, "bg_var_factor", 0.0) or 0.0)
        s2 = bvf * bg.clamp(min=0.0)
        nn_ = n.clamp(min=1e-12)
        A0 = torch.where(n > 0, nn_ * torch.log(lam0 / nn_) - (lam0 - nn_), -lam0)
        iphi = lam0 / (lam0 + s2)
        n64 = n.double()
        logpeak = torch.where(n > 0, (n64 * n64.clamp(min=1e-300).log() - n64
                                      - torch.lgamma(n64 + 1)).to(n.dtype), torch.zeros_like(n))
        return A0 * iphi + 0.5 * iphi.log() + logpeak

    def _mix_miss(self, lp: Tensor, n_ph: Tensor, bg: Tensor) -> Tensor:
        if self.miss_logit is None:
            return lp
        log_pi = torch.nn.functional.logsigmoid(self.miss_logit)
        log_1m = torch.nn.functional.logsigmoid(-self.miss_logit)
        return torch.logaddexp(log_1m + lp, log_pi + self._log_miss(n_ph, bg))

    def _grid_bounds(self) -> tuple[float, float] | None:
        g = getattr(self.quadrature, "log_S_grid", None)
        if g is None:
            return None
        return float(g[0].item()), float(g[-1].item())

    def _renorm(self, log_p: Tensor, scatter, params, meta, log_mean: Tensor) -> Tensor:
        """Divide out the scatter mass that fell off the grid."""
        bounds = self._grid_bounds()
        if bounds is None:
            return log_p
        log_alpha, log_beta = scatter.log_alpha_beta_obs(params, meta)
        log_mass = gengauss_grid_log_mass(
            log_mean, log_alpha, log_beta.exp(), *bounds
        )
        self.last_grid_mass = log_mass.detach().exp()
        return log_p - log_mass

    def _marginal(self, scatter, params, log_mean, n_ph, bg, meta, obs_range) -> Tensor:
        """``log p(n | log_mean)`` with the scatter integrated out.

        A mixture scatter is assembled from per-component calls on its child
        GenGauss views, each centred at ``log_mean + offset_k``, combined with
        the component weights; the quadrature itself never sees the mixture.
        """
        if isinstance(scatter, PredictedMixtureScatter):
            terms = [
                log_pi_k + self.quadrature.marginalize_signal(
                    view_k, self.likelihood, log_mean + offset_k, n_ph, bg, params, meta,
                    obs_range=obs_range,
                )
                for log_pi_k, offset_k, view_k in scatter.components(meta)
            ]
            return torch.logsumexp(torch.stack(terms, dim=0), dim=0)
        lp = self.quadrature.marginalize_signal(
            scatter, self.likelihood, log_mean, n_ph, bg, params, meta, obs_range=obs_range,
        )
        return self._renorm(lp, scatter, params, meta, log_mean) if self.renormalize else lp

    def log_prob(
        self, scatter, params, log_mean: Tensor, n_ph: Tensor, bg: Tensor, meta
    ) -> Tensor:
        n = log_mean.shape[0]
        cs = self.chunk_size or n
        if cs >= n:
            lp = self._marginal(scatter, params, log_mean, n_ph, bg, meta, (0, n))
            return self._mix_miss(lp, n_ph, bg)
        parts = []
        for s in range(0, n, cs):
            e = min(s + cs, n)
            sub = meta.slice(s, e)

            def _chunk(lm, nn, bb, _sub=sub, _s=s, _e=e):
                lp = self._marginal(scatter, params, lm, nn, bb, _sub, (_s, _e))
                return self._mix_miss(lp, nn, bb)

            args = (log_mean[s:e], n_ph[s:e], bg[s:e])
            if self.checkpoint_chunks and torch.is_grad_enabled():
                lp = torch.utils.checkpoint.checkpoint(
                    _chunk, *args, use_reentrant=False
                )
            else:
                lp = _chunk(*args)
            parts.append(lp)
        return torch.cat(parts)

    def forward(
        self, scatter, params, log_mean: Tensor, n_ph: Tensor, bg: Tensor, meta,
        *, mask: Tensor | None = None,
    ) -> Tensor:
        lp = self.log_prob(scatter, params, log_mean, n_ph, bg, meta)
        if mask is not None:
            lp = lp[mask]
        return -lp.sum()


class AgreementNLL(nn.Module):
    """Closed-form moment-matched agreement between a reflection's observations.

    Grid-free reference objective; see the module docstring.  ``beta`` is not
    identifiable here — a Gaussian overlap has no shape parameter, so
    ``(w, beta)`` collapse to one effective variance.  That is the price of the
    closed form and the reason it is not the primary objective.
    """

    def __init__(self, n_hkl: int, *, cox_reid: bool = True) -> None:
        super().__init__()
        self.n_hkl = int(n_hkl)
        self.cox_reid = bool(cox_reid)

    def forward(
        self,
        y: Tensor,          # (n,) per-observation estimate of I0
        tau2: Tensor,       # (n,) its variance
        ybar: Tensor,       # (n,) leave-one-out target
        hkl_id: Tensor,     # (n,)
        *, mask: Tensor | None = None,
    ) -> Tensor:
        w = 1.0 / tau2.clamp(min=1e-12)
        resid2 = (y - ybar) ** 2
        per_obs = 0.5 * (resid2 * w + tau2.clamp(min=1e-12).log() + _LOG_2PI)
        if mask is not None:
            per_obs = per_obs * mask
            w = w * mask
        nll = per_obs.sum()
        if self.cox_reid:
            fisher = group_sum(w, hkl_id, self.n_hkl)
            nll = nll + 0.5 * fisher.clamp(min=1e-12).log()[fisher > 0].sum()
        return nll
