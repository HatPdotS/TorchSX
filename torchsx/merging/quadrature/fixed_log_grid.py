"""Fixed-range uniform log_S grid: no centering on log_mean, n, or bg.

Both ``TrapezoidalLogGrid`` (centered on the Poisson MLE ``n - bg``) and
``WideLogGrid`` (centered on the gengauss prior mode ``log_mean``) bake
in an assumption about where the integrand has mass.  This quadrature
makes no such assumption — the grid is the same tensor for every
observation and every µ-coarse posterior offset:

    log_S ∈ [log(S_min), log(S_max)],  uniform, K nodes

with trapezoidal weights ``Δlog_S`` (half on the endpoints).  At the
default ``S_min=1e-3, S_max=1e6, n_points=10001`` the spacing is
``Δlog_S ≈ 2.07e-3`` over a 9-decade window — wide enough that the
likelihood × scatter integrand is numerically zero at both ends for any
realistic ``log_mean``, and dense enough that the discretisation error
sits below MC noise.  Used as a reference to test whether the
centering choice in the other grids biases the merged intensities.
"""
from __future__ import annotations

import contextlib
import contextvars
import math
import os

import torch
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.likelihood.poisson import Poisson
from torchsx.merging.likelihood.poisson_excess_noise import PoissonWithExcessNoise
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.trapezoidal import (
    TrapezoidalLogGrid, _compute_sig_lo_per_obs,
)
from torchsx.merging.scatter.base import ScatterModel
from torchsx.merging.scatter.gengauss import GenGaussLogScatter

# Triton fast path is optional — falls back to eager when unavailable.
try:
    from torchsx.merging.quadrature._fixed_log_grid_triton import (
        HAVE_TRITON, marginalize_signal_triton,
    )
    from torchsx.merging.quadrature._fixed_log_grid_triton_excess import (
        marginalize_signal_triton_excess,
    )
    from torchsx.merging.quadrature._fixed_log_grid_triton_gaussian import (
        marginalize_signal_triton_gaussian,
    )
except ImportError:
    HAVE_TRITON = False
    marginalize_signal_triton = None  # type: ignore[assignment]
    marginalize_signal_triton_excess = None  # type: ignore[assignment]
    marginalize_signal_triton_gaussian = None  # type: ignore[assignment]

# Set ``TORCHSX_FIXEDLOGGRID_NO_TRITON=1`` to force the eager fallback
# (e.g. for debugging numerical drift between paths).
_DISABLE_TRITON = os.environ.get("TORCHSX_FIXEDLOGGRID_NO_TRITON", "") == "1"


# Scoped switch used by callers that need double-backward (Hessian /
# σ_Laplace). The Triton kernel's saved tensors S1-S4 are detached
# from the input graph, so re-differentiating its first-order grad
# yields None. Wrap the Hessian computation in ``with eager_only():``
# to route through the eager PyTorch path, which supports double-back
# natively. Outside the scope, the Triton fast path is used as before.
_eager_only: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "torchsx_fixed_log_grid_eager_only", default=False,
)


@contextlib.contextmanager
def eager_only():
    """Within this scope, ``FixedLogGrid.marginalize_signal`` skips the
    Triton fast path and uses the eager PyTorch fallback. Use this when
    computing second-order quantities (Hessian, σ_Laplace) — the eager
    path supports ``create_graph=True``."""
    token = _eager_only.set(True)
    try:
        yield
    finally:
        _eager_only.reset(token)


class FixedLogGrid(Quadrature):
    """Uniform log_S grid in absolute coordinates ``[log(S_min), log(S_max)]``.

    The grid is independent of ``log_mean``, ``n_ph``, ``bg``, and any
    iterate — same K nodes for every observation, every µ-coarse offset.
    ``requires_per_call_posterior = True`` so the merger routes posterior
    moment computation through ``compute_posterior_moments_per_call``,
    which keeps the MAP marginal and every posterior offset on this same
    fixed grid (no fallback to the softplus inner grid).

    Parameters
    ----------
    S_min, S_max
        Linear-S range of the integration window (default ``1e-3`` to
        ``1e6`` — 9 decades, ample for BR-scale data).
    n_points
        Number of nodes (default 10001 → ``Δlog_S ≈ 2.07e-3``).
    sig_lo_per_obs
        Optional precomputed buffer; only forwarded to an inner
        ``TrapezoidalLogGrid`` so legacy ``_sig_lo_slice`` /
        ``_build_grid`` callers keep working.  Never touched by
        ``marginalize_signal``.
    """

    requires_per_call_posterior: bool = True

    def __init__(
        self,
        S_min: float = 1e-3,
        S_max: float = 1e6,
        n_points: int = 10001,
        sig_lo_per_obs: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not (S_min > 0.0 and S_max > S_min):
            raise ValueError(f"need 0 < S_min < S_max, got {S_min=} {S_max=}")
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.n_points = int(n_points)
        log_S = torch.linspace(
            math.log(self.S_min), math.log(self.S_max), self.n_points,
        )
        self.register_buffer("log_S_grid", log_S)
        self.register_buffer("S_grid", log_S.exp())
        self.dlog_S = float(
            (math.log(self.S_max) - math.log(self.S_min)) / (self.n_points - 1)
        )
        self.s_grid = TrapezoidalLogGrid(sig_lo_per_obs=sig_lo_per_obs)

    @classmethod
    def from_prep(
        cls, prep: PreparedObs, config: MergeConfig,
    ) -> "FixedLogGrid":
        sig_lo = _compute_sig_lo_per_obs(
            n_valid=prep.n_valid, hkl_id=prep.hkl_id, bg=prep.bg,
            obs_counts=prep.obs_counts, override=config.sig_lo_override,
        )
        return cls(
            S_min=getattr(config, "fixed_log_S_min", 1e-3),
            S_max=getattr(config, "fixed_log_S_max", 1e6),
            n_points=getattr(config, "fixed_log_n_points", 10001),
            sig_lo_per_obs=sig_lo,
        )

    def _sig_lo_slice(self, n_ph, bg, obs_range):
        return self.s_grid._sig_lo_slice(n_ph, bg, obs_range)

    def _build_grid(self, n_ph, bg, sig_lo_per_obs):
        return self.s_grid._build_grid(n_ph, bg, sig_lo_per_obs)

    def _trap_log_w(self, ref: Tensor, K: int) -> Tensor:
        """Trapezoidal log-weights with half-weight endpoints."""
        log_w = ref.new_full((K,), math.log(self.dlog_S))
        log_w[0] = log_w[0] - math.log(2.0)
        log_w[-1] = log_w[-1] - math.log(2.0)
        return log_w

    def _can_use_triton(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,
    ) -> bool:
        """Triton path is valid for (GenGauss × {Poisson, PoissonWithExcessNoise,
        GaussianLikelihood}) on a CUDA tensor with Triton installed.

        For ``GaussianLikelihood`` subclasses the kernel takes ``g1`` and
        ``g2`` (noise polynomial coefficients) as ``constexpr`` scalars,
        so trainable-coefficient variants (where those become 0-d
        tensors) fall back to the eager path.
        """
        from torchsx.merging.likelihood import GaussianLikelihood
        if not (
            HAVE_TRITON
            and not _DISABLE_TRITON
            and not _eager_only.get()
            and log_mean.is_cuda
            and isinstance(scatter, GenGaussLogScatter)
        ):
            return False
        if isinstance(likelihood, (Poisson, PoissonWithExcessNoise)):
            return True
        if isinstance(likelihood, GaussianLikelihood):
            # Probe the coefficients to ensure g1, g2 are plain floats —
            # we can't pass tensor scalars as Triton constexpr.
            try:
                _, g1, g2 = likelihood.coefficients(log_mean.new_zeros(1))
            except Exception:
                return False
            return not (
                isinstance(g1, Tensor) and g1.requires_grad
                or isinstance(g2, Tensor) and g2.requires_grad
            )
        return False

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
    ) -> Tensor:                  # (M,)
        """Marginal ``log p(n | log_mean, α, β, bg)`` on the fixed log_S grid."""
        K = self.n_points

        log_alpha_obs = params.log_alpha[meta.hkl_idx]
        log_beta_obs = params.log_beta[meta.hkl_idx]

        if self._can_use_triton(scatter, likelihood, log_mean):
            log_w = self._trap_log_w(self.log_S_grid, K)
            from torchsx.merging.likelihood import GaussianLikelihood
            if isinstance(likelihood, GaussianLikelihood):
                g0_obs, g1, g2 = likelihood.coefficients(bg)
                # g0_obs may be a 0-d or (M,) tensor; expand to (M,).
                if g0_obs.dim() == 0:
                    g0_obs = g0_obs.expand_as(bg)
                return marginalize_signal_triton_gaussian(
                    log_mean, n_ph, bg, g0_obs.contiguous(),
                    log_alpha_obs, log_beta_obs,
                    self.log_S_grid, log_w,
                    float(g1), float(g2),
                    var_floor=getattr(likelihood, "var_floor", 1e-12),
                )
            if isinstance(likelihood, PoissonWithExcessNoise):
                sigma_excess = (
                    likelihood.bg_var_factor * bg + likelihood.read_var_photons
                ).clamp(min=1e-12).sqrt()
                return marginalize_signal_triton_excess(
                    log_mean, n_ph, bg, sigma_excess,
                    log_alpha_obs, log_beta_obs,
                    self.log_S_grid, log_w,
                    likelihood.gh_z.to(log_mean.device),
                    likelihood.gh_logw.to(log_mean.device),
                    likelihood.bg_var_factor,
                )
            return marginalize_signal_triton(
                log_mean, n_ph, bg, log_alpha_obs, log_beta_obs,
                self.log_S_grid, log_w,
            )

        # Eager fallback — bit-exact w.r.t. the original implementation.
        M = log_mean.shape[0]
        log_S = self.log_S_grid[None, :].expand(M, K)        # (M, K)
        S     = self.S_grid[None, :].expand(M, K)            # (M, K)
        eps   = log_S - log_mean[:, None]                    # (M, K)

        if isinstance(scatter, GenGaussLogScatter):
            from torchsx.merging.scatter.gengauss import gengauss_log_prob_from_obs
            log_scatter = gengauss_log_prob_from_obs(eps, log_alpha_obs, log_beta_obs)
        else:
            log_scatter = scatter.log_prob(eps, params, meta)        # (M, K)
        log_lik = likelihood.log_prob_given_signal(n_ph, S, bg)      # (M, K)

        log_w = self._trap_log_w(log_scatter, K)
        return torch.logsumexp(log_scatter + log_lik + log_w[None, :], dim=1)

    def marginalize_signal_with_reml_aux(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,        # (M,)
        n_ph: Tensor,            # (M,)
        bg: Tensor,              # (M,)
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Marginal ``log p`` plus the three per-obs aux stats needed for the
        Cox–Reid modified-profile-likelihood (REML) correction on µ_h.

        Returns ``(log_p, A, B, C)`` all of shape ``(M,)``.

        Derivation. The integral is parameterised on the log_S grid:

            p_i = ∫ exp(g(µ, log_S)) d(log_S),
            g(µ, log_S) = log_scatter(log_S - µ; α, β)
                        + log_pois(n | exp(log_S) + bg).

        On this grid the Poisson factor is *grid-fixed* (independent of µ);
        all µ-dependence sits in the GG (scatter) factor through
        ``ε = log_S - µ``. Hence with the GG log-density
        ``log_scatter = -|ε|^β / α^β + const``,

            ∂g/∂µ = β · sign(ε) · |ε|^{β-1} / α^β
            ∂²g/∂µ² = -β(β-1) · |ε|^{β-2} / α^β       (β > 1)

        and the per-obs Fisher info for µ follows from
        ``∂²log_p/∂µ² = E_q[∂²g/∂µ²] + Var_q[∂g/∂µ]`` so that

            j_µµ = -∂²log_p/∂µ² = C - (B - A²)
            A = E_q[∂g/∂µ],   B = E_q[(∂g/∂µ)²],   C = -E_q[∂²g/∂µ²]

        with ``q(log_S) ∝ exp(g(µ, log_S))``.

        Eager-only: this method does *not* dispatch to a Triton kernel —
        the prototype validates the math before any kernel work. GenGauss
        × Poisson is required (the only combination we currently fit).
        """
        if not isinstance(scatter, GenGaussLogScatter):
            raise NotImplementedError(
                "REML aux-stat path currently requires GenGaussLogScatter."
            )
        if not isinstance(likelihood, Poisson):
            raise NotImplementedError(
                "REML aux-stat path currently requires Poisson likelihood."
            )

        from torchsx.merging.scatter.gengauss import gengauss_log_prob_from_obs

        K = self.n_points
        M = log_mean.shape[0]

        log_alpha_obs = params.log_alpha[meta.hkl_idx]
        log_beta_obs = params.log_beta[meta.hkl_idx]

        log_S = self.log_S_grid[None, :].expand(M, K)            # (M, K)
        S     = self.S_grid[None, :].expand(M, K)                # (M, K)
        eps   = log_S - log_mean[:, None]                        # (M, K)

        log_scatter = gengauss_log_prob_from_obs(
            eps, log_alpha_obs, log_beta_obs,
        )                                                        # (M, K)

        lam = (S + bg[:, None]).clamp(min=1e-12)                 # (M, K)
        n = n_ph[:, None]                                        # (M, 1)
        log_pois = n * lam.log() - lam - torch.lgamma(n_ph[:, None] + 1)

        log_w = self._trap_log_w(log_scatter, K)                 # (K,)
        log_int = log_scatter + log_pois + log_w[None, :]        # (M, K)
        log_p = torch.logsumexp(log_int, dim=1)                  # (M,)

        q = (log_int - log_p[:, None]).exp()                     # (M, K)

        # GG-based ∂g/∂µ and ∂²g/∂µ² (the Pois factor is grid-fixed in
        # log_S and contributes nothing). ``alpha_pow_negbeta = α^{-β}``
        # is broadcast over (M, 1).
        beta = log_beta_obs.exp()                                # (M,)
        alpha = log_alpha_obs.exp()                              # (M,)
        alpha_pow_negbeta = (-beta * log_alpha_obs).exp()        # = α^{-β}, (M,)

        eps_abs_safe = eps.abs().clamp(min=1e-12)                # (M, K)

        # ∂g/∂µ = β · sign(ε) · |ε|^{β-1} · α^{-β}
        score_grid = (
            beta[:, None]
            * eps.sign()
            * eps_abs_safe.pow(beta[:, None] - 1.0)
            * alpha_pow_negbeta[:, None]
        )                                                         # (M, K)
        # -∂²g/∂µ² = β(β-1) · |ε|^{β-2} · α^{-β}.
        # For β just above 1 the |ε|^{β-2} factor diverges at ε = 0; the
        # 1e-12 floor on |ε| is a numerical safety, not a model statement
        # — the integrand × q is still finite because q vanishes there.
        c_grid = (
            beta[:, None]
            * (beta[:, None] - 1.0)
            * eps_abs_safe.pow(beta[:, None] - 2.0)
            * alpha_pow_negbeta[:, None]
        )                                                         # (M, K)

        A = (q * score_grid).sum(dim=1)                          # (M,)
        B = (q * score_grid * score_grid).sum(dim=1)             # (M,)
        C = (q * c_grid).sum(dim=1)                              # (M,)

        return log_p, A, B, C

