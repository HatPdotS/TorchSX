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
    """Width/shape reparameterisation: ``(w, β) → α``.

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
    log_alpha: Tensor, beta: Tensor, *, n_u: int = 257, u_max: float = 12.0
) -> Tensor:
    """``log E[exp(eps)]`` for ``eps ~ GenGauss(0, alpha, beta)``, per observation.

    This is the mean/median correction, and getting it wrong closes a runaway
    loop.  ``GenGaussLogScatter`` centres the density on ``log_mean``, so
    ``median(S) = exp(log_mean)`` but ``E[S] = exp(log_mean) E[exp(eps)]``.  A
    **linear** merge estimator — which is what you must use when 32 % of
    observations are negative — estimates the *mean*.  So unless ``log_mean``
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
    integrand = log_p + log_alpha.exp()[:, None] * u[None, :] + log_w[None, :]
    return torch.logsumexp(integrand, dim=1)


def gengauss_log_prob_from_obs(
    eps: Tensor,              # (M, K)
    log_alpha_obs: Tensor,    # (M,) per-observation (already indexed by hkl_idx)
    log_beta_obs: Tensor,     # (M,) per-observation
) -> Tensor:                  # (M, K)
    """GenGauss log-density evaluated from per-observation (log α, log β).

    Bypasses the (Parameters, ObsInfo) indirection so callers that need
    elementwise gradient gating on (α, β) — e.g. the bg-gate weak/strong
    split — can pass `torch.where(mask, x, x.detach())` directly.
    """
    beta = log_beta_obs.exp()
    alpha = log_alpha_obs.exp()
    z = eps.abs() / alpha[:, None].clamp(min=1e-8)
    return (
        -z.clamp(min=1e-12).pow(beta[:, None])
        + _log_gg_norm(beta)[:, None]
        - log_alpha_obs[:, None]
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
        return gengauss_log_prob_from_obs(eps, log_alpha_obs, log_beta_obs)
