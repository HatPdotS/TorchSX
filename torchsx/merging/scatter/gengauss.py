"""Generalised-Gaussian log-scatter density (per-HKL α, β)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter.base import ScatterModel


def _log_gg_norm(beta: Tensor) -> Tensor:
    """GenGauss log-normalisation: log β − log 2 − lgamma(1/β)."""
    return torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)


def log_alpha_from_log_width(log_w: Tensor, beta: Tensor) -> Tensor:
    r"""Width/shape reparameterisation: ``(w, β) → α``.

    ``Var(GenGauss) = α²·Γ(3/β)/Γ(1/β)``, so α and β *both* set the width and
    compete during fitting.  Reparameterising on the actual SD ``w`` makes them
    nearly orthogonal — ``w`` carries all the width, and ``β`` is then
    identified only by the *shape* of the residual distribution, which is a
    genuinely different signal:

    .. math::
        \log α = \log w - \tfrac12 [\log Γ(3/β) - \log Γ(1/β)]

    Sanity points: β=2 (Gaussian) gives α = w·√2; β=1 (Laplace) gives
    α = w/√2.
    """
    return log_w - 0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))


_LOG_E_S_U = None


def log_e_s_gengauss(
    log_alpha: Tensor, beta: Tensor, *, n_u: int = 257, u_max: float = 12.0,
    log_alpha_hi: Tensor | None = None,
) -> Tensor:
    """``log E[exp(eps)]`` for ``eps ~ GenGauss(0, alpha, beta)``, per observation.

    With ``log_alpha_hi`` the density is the two-width (asymmetric) GenGauss:
    width ``alpha`` below the centre, ``alpha_hi`` above, one ``beta``, mass
    fractions ``alpha/(alpha+alpha_hi)`` and ``alpha_hi/(alpha+alpha_hi)``.  The
    centre is then the mode, not the median.  ``None`` is the symmetric case.

    This is the mean/median correction, and getting it wrong closes a runaway
    loop.  ``GenGaussLogScatter`` centres the density on ``log_mean``, so
    ``median(S) = exp(log_mean)`` but ``E[S] = exp(log_mean) E[exp(eps)]``.  A
    **linear** merge estimator — which is what you must use when a large share
    of observations is negative — estimates the *mean*.  So unless ``log_mean``
    is shifted by this term, the target is biased high by ``E[exp(eps)]``,
    which at ``w = 1.65, beta = 2`` is a factor of 3.9 and **grows with the
    width**: bigger w -> more bias -> worse fit -> bigger w.

    Substituting ``eps = alpha * u`` makes the standardised grid shared across
    observations, so this is one ``(M, n_u)`` reduction with a per-observation
    ``alpha``, and it is differentiable in both arguments.

    Sanity: for ``beta = 2`` the closed form is ``alpha^2 / 4`` (equivalently
    ``w^2 / 2``, since ``alpha = w*sqrt(2)``).

    ``GenGaussLogScatterRecentered`` applies the same shift from a single
    global ``(alpha_ref, beta_ref)`` fixed at init.  That is an approximation,
    and not a safe one: the correction is a function of the width, so a global
    constant is only right while the width is uniform and static.  Here alpha
    and beta are predicted per observation and move every step, so the
    per-observation form is **required for correctness**, not a refinement --
    a fixed shift would leave exactly the width-dependent bias that drives the
    runaway.
    """
    global _LOG_E_S_U
    if _LOG_E_S_U is None or _LOG_E_S_U[0] != (n_u, u_max):
        u = torch.linspace(-u_max, u_max, n_u)
        _LOG_E_S_U = ((n_u, u_max), u, float(2 * u_max / (n_u - 1)))
    _, u, du = _LOG_E_S_U
    u = u.to(log_alpha.device, log_alpha.dtype)

    b = beta[:, None]
    log_p = _log_gg_norm(beta)[:, None] - u.abs()[None, :].pow(b)   # alpha=1
    log_w = torch.full_like(u, math.log(du))
    log_w[0] -= math.log(2.0)
    log_w[-1] -= math.log(2.0)
    if log_alpha_hi is None:
        integrand = log_p + log_alpha.exp()[:, None] * u[None, :] + log_w[None, :]
        return torch.logsumexp(integrand, dim=1)
    # eps = alpha_side * u: the Jacobian alpha_side and the shared normaliser
    # 1/(alpha_lo + alpha_hi) replace the symmetric 1/alpha (2 * mean width)
    # the Jacobian alpha_side jumps at u = 0; the node ON the kink carries the
    # mean of the two sides, or the trapezoid rule is off by du (alpha_hi -
    # alpha_lo) / 2 there
    la_mid = torch.logaddexp(log_alpha, log_alpha_hi) - math.log(2.0)
    la_side = torch.where(u[None, :] > 0, log_alpha_hi[:, None],
                          torch.where(u[None, :] < 0, log_alpha[:, None], la_mid[:, None]))
    log_norm_asym = _log_asym_norm_shift(log_alpha, log_alpha_hi)
    integrand = (log_p + la_side + log_norm_asym[:, None]
                 + la_side.exp() * u[None, :] + log_w[None, :])
    return torch.logsumexp(integrand, dim=1)


def _log_asym_norm_shift(log_alpha: Tensor, log_alpha_hi: Tensor) -> Tensor:
    """``-log((alpha_lo + alpha_hi) / 2)``: the two-width normaliser relative to
    ``_log_gg_norm`` (which carries the ``-log 2``).  Zero when the widths match
    -- then ``_log_gg_norm - log_alpha`` is recovered exactly."""
    return -(torch.logaddexp(log_alpha, log_alpha_hi) - math.log(2.0))


def _log_alpha_side(eps: Tensor, log_alpha: Tensor, log_alpha_hi: Tensor | None) -> Tensor:
    """Per-node log width: ``alpha_hi`` above the centre, ``alpha`` below."""
    if log_alpha_hi is None:
        return log_alpha[:, None].expand_as(eps)
    return torch.where(eps > 0, log_alpha_hi[:, None], log_alpha[:, None])


def gengauss_log_prob_from_obs(
    eps: Tensor,              # (M, K)
    log_alpha_obs: Tensor,    # (M,) per-observation (already indexed by hkl_idx)
    log_beta_obs: Tensor,     # (M,) per-observation
    log_alpha_hi_obs: Tensor | None = None,   # (M,) upper-side width; None = symmetric
) -> Tensor:                  # (M, K)
    """GenGauss log-density evaluated from per-observation (log α, log β).

    Bypasses the (Parameters, ObsInfo) indirection so callers that need
    elementwise gradient gating on (α, β) — e.g. the bg-gate weak/strong
    split — can pass `torch.where(mask, x, x.detach())` directly.

    ``log_alpha_hi_obs`` selects the two-width family: ``alpha`` below the
    centre, ``alpha_hi`` above, normaliser ``beta / ((alpha + alpha_hi)
    Gamma(1/beta))``.  Partiality scatter is one-sided (a long tail of partial
    observations below, a hard edge near the full reflection above), which a
    symmetric width cannot represent without mislocating its centre.
    """
    beta = log_beta_obs.exp()
    if log_alpha_hi_obs is None:
        alpha = log_alpha_obs.exp()
        z = eps.abs() / alpha[:, None].clamp(min=1e-8)
        return (
            -z.clamp(min=1e-12).pow(beta[:, None])
            + _log_gg_norm(beta)[:, None]
            - log_alpha_obs[:, None]
        )
    la_side = _log_alpha_side(eps, log_alpha_obs, log_alpha_hi_obs)
    z = eps.abs() / la_side.exp().clamp(min=1e-8)
    return (
        -z.clamp(min=1e-12).pow(beta[:, None])
        + _log_gg_norm(beta)[:, None]
        + _log_asym_norm_shift(log_alpha_obs, log_alpha_hi_obs)[:, None]
    )


class GenGaussLogScatter(ScatterModel):
    """Generalised-Gaussian density on ``eps = log S − log_mean``.

    .. math::
        \\log p(\\epsilon; \\alpha, \\beta)
            = -(|\\epsilon|/\\alpha)^{\\beta} + \\log\\beta - \\log 2\\alpha
              - \\log\\Gamma(1/\\beta)

    α_h, β_h are per-HKL and live on ``Parameters`` as ``log_alpha``,
    ``log_beta`` (log-parameterised for positivity).
    """

    def log_prob(
        self,
        eps: Tensor,              # (M, K)
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:
        log_alpha_obs, log_beta_obs = self.log_alpha_beta_obs(params, meta)
        return gengauss_log_prob_from_obs(eps, log_alpha_obs, log_beta_obs,
                                          self.log_alpha_hi_obs(params, meta))


def _log_gammaincc(a: Tensor, x: Tensor, x_switch: float) -> Tensor:
    """``log Q(a, x)``: direct below ``x_switch``, 3-term asymptotic series above.

    Both branches are evaluated on clamped arguments so the unused side stays
    finite and cannot poison the gradient through ``torch.where``.
    """
    q_direct = torch.special.gammaincc(a, x.clamp(max=x_switch)).clamp(min=1e-38)
    xa = x.clamp(min=x_switch)
    series = (1.0 + (a - 1.0) / xa + (a - 1.0) * (a - 2.0) / xa**2
              + (a - 1.0) * (a - 2.0) * (a - 3.0) / xa**3)
    log_q_asym = ((a - 1.0) * xa.log() - xa - torch.lgamma(a)
                  + series.clamp(min=1e-30).log())
    return torch.where(x > x_switch, log_q_asym, q_direct.log())


def _log_cdf_value(t: Tensor, log_alpha: Tensor, log_beta: Tensor, x_switch: float,
                   log_alpha_hi: Tensor | None = None) -> Tensor:
    """``log F(t)`` with no autograd bookkeeping (see gengauss_log_cdf_from_obs).

    Two-width form: ``F(t <= 0) = w_lo Q(1/beta, (|t|/alpha_lo)^beta)`` and
    ``F(t > 0) = 1 - w_hi Q(1/beta, (t/alpha_hi)^beta)`` with
    ``w_side = alpha_side / (alpha_lo + alpha_hi)``; ``w = 1/2`` when symmetric.
    """
    beta = log_beta.exp()
    a = 1.0 / beta
    if log_alpha_hi is None:
        la_side = log_alpha
        log_w_lo = log_w_hi = torch.full_like(log_alpha, math.log(0.5))
    else:
        la_side = torch.where(t > 0, log_alpha_hi, log_alpha)
        log_sum = torch.logaddexp(log_alpha, log_alpha_hi)
        log_w_lo, log_w_hi = log_alpha - log_sum, log_alpha_hi - log_sum
    log_x = beta * (t.abs().clamp(min=1e-30).log() - la_side)
    x = log_x.clamp(max=60.0).exp()
    log_q = _log_gammaincc(a, x, x_switch)
    lower = log_w_lo + log_q                                   # t <= 0: F = w_lo Q
    upper = torch.log1p(-(log_w_hi + log_q).exp().clamp(max=1.0 - 1e-7))   # t > 0: F = 1 - w_hi Q
    return torch.where(t > 0, upper, lower)


class _GenGaussLogCDF(torch.autograd.Function):
    """``log F`` with closed-form gradients and three saved ``(M,)`` tensors.

    Plain autograd through the incomplete gamma keeps ~10 intermediates per
    call; at 31 outer nodes x ~1e7 observations that is over 10 GB and forces
    node checkpointing.  Here

        d log F / d t         = f(t) / F(t)
        d log F / d log alpha = -t f(t) / F(t)           (F depends on t/alpha)
        d log F / d log beta  = central difference in log beta

    with ``f`` the GenGauss density.  The beta derivative is a finite
    difference of the whole ``log F`` because torch has no derivative of the
    incomplete gamma in its shape argument; ``shape_fd`` is the relative step.
    """

    @staticmethod
    def forward(ctx, t, log_alpha, log_beta, x_switch, shape_fd, need_beta, log_alpha_hi):
        asym = log_alpha_hi is not None
        log_F = _log_cdf_value(t, log_alpha, log_beta, x_switch, log_alpha_hi)
        beta = log_beta.exp()
        la_side = torch.where(t > 0, log_alpha_hi, log_alpha) if asym else log_alpha
        norm_shift = _log_asym_norm_shift(log_alpha, log_alpha_hi) if asym else -log_alpha
        log_f = (-(t.abs().clamp(min=1e-30).log() - la_side).mul(beta).clamp(max=60.0).exp()
                 + _log_gg_norm(beta) + norm_shift)
        r = (log_f - log_F).exp()                       # f / F
        if need_beta:
            d = shape_fd
            g_lb = (_log_cdf_value(t, log_alpha, log_beta + d, x_switch, log_alpha_hi)
                    - _log_cdf_value(t, log_alpha, log_beta - d, x_switch, log_alpha_hi)) / (2.0 * d)
        else:
            g_lb = None
        if asym:
            # d log F / d log alpha_side, two-width form (see gengauss_log_cdf_from_obs):
            #   t <= 0:  lo: w_hi - t f/F            hi: -w_hi
            #   t >  0:  lo: w_lo (1-F)/F           hi: -(w_lo (1-F) + t f)/F
            log_sum = torch.logaddexp(log_alpha, log_alpha_hi)
            w_lo, w_hi = (log_alpha - log_sum).exp(), (log_alpha_hi - log_sum).exp()
            G_over_F = (-log_F).exp() - 1.0                 # (1 - F) / F
            pos = t > 0
            g_lo = torch.where(pos, w_lo * G_over_F, w_hi - t * r)
            g_hi = torch.where(pos, -(w_lo * G_over_F + t * r), -w_hi)
            ctx.save_for_backward(r, t, g_lb if g_lb is not None else r.new_zeros(1), g_lo, g_hi)
        else:
            ctx.save_for_backward(r, t, g_lb if g_lb is not None else r.new_zeros(1))
        ctx.need_beta = need_beta
        ctx.asym = asym
        return log_F

    @staticmethod
    def backward(ctx, grad):
        if ctx.asym:
            r, t, g_lb, g_lo, g_hi = ctx.saved_tensors
            g_t = grad * r
            g_lb = grad * g_lb if ctx.need_beta else None
            return g_t, grad * g_lo, g_lb, None, None, None, grad * g_hi
        r, t, g_lb = ctx.saved_tensors
        g_t = grad * r
        g_la = -grad * t * r
        g_lb = grad * g_lb if ctx.need_beta else None
        return g_t, g_la, g_lb, None, None, None, None


def gengauss_log_cdf_from_obs(
    t: Tensor,                # (M,) or (M, C): eps at which the CDF is taken
    log_alpha_obs: Tensor,    # (M,)
    log_beta_obs: Tensor,     # (M,)
    *, x_switch: float = 25.0, shape_fd: float = 1e-3,
    log_alpha_hi_obs: Tensor | None = None,   # upper-side width; None = symmetric
) -> Tensor:
    """``log P(eps <= t)`` for ``eps ~ GenGauss(0, alpha, beta)``, elementwise.

    Closed form: ``F(t) = 1/2 + sign(t)/2 * P(1/beta, |t/alpha|^beta)`` with
    ``P`` the regularised lower incomplete gamma.  Written so that the lower
    tail is never formed as ``1 - P``: for ``t <= 0`` it is ``Q/2`` directly,
    and beyond ``x_switch`` the upper incomplete gamma is taken from its
    asymptotic series, which stays accurate where the float32 ``gammaincc``
    has underflowed.  This is what makes an analytic scatter tail usable down
    to ``e^-1e3`` of the mass, which a grid could only truncate.

    Gradients are closed form in ``t`` and ``log alpha`` and a central finite
    difference in ``log beta`` (torch has no shape derivative of the incomplete
    gamma; a detached shape slot biased the beta gradient direction by 1-cos
    4.8e-3 against the oracle).  Only three ``(M,)`` tensors are saved.
    """
    if log_alpha_obs.dim() < t.dim():
        log_alpha_obs = log_alpha_obs.unsqueeze(-1).expand_as(t)
        log_beta_obs = log_beta_obs.unsqueeze(-1).expand_as(t)
        if log_alpha_hi_obs is not None:
            log_alpha_hi_obs = log_alpha_hi_obs.unsqueeze(-1).expand_as(t)
    needs = [t, log_alpha_obs, log_beta_obs] + ([log_alpha_hi_obs] if log_alpha_hi_obs is not None else [])
    if not (torch.is_grad_enabled() and any(x.requires_grad for x in needs)):
        return _log_cdf_value(t, log_alpha_obs, log_beta_obs, x_switch, log_alpha_hi_obs)
    return _GenGaussLogCDF.apply(t, log_alpha_obs, log_beta_obs, float(x_switch),
                                 float(shape_fd),
                                 bool(log_beta_obs.requires_grad), log_alpha_hi_obs)
