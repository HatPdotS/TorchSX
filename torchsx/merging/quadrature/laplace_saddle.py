"""Coarse-scan + Newton-saddle + trapezoid-in-u quadrature.

For each observation, locate the joint integrand peak

    ℓ(log_S) = log p_GG(log_S − log_mean; α_h, β_h)
             + log Poisson(n | exp(log_S) + bg)

by (1) a coarse scan around log_mean, (2) damped Newton refinement of the
analytic ∂ℓ/∂log_S, then place K trapezoidal nodes in
``u ∈ [-n_sigma_joint, +n_sigma_joint]`` scaled by the local width
``σ_joint = 1/√(-ℓ″)``.  Grid positions and σ are detached from autograd;
gradients flow only through the integrand evaluation at the nodes.

Designed to fix two failure modes of `HybridLogGrid`:

* upward MLE bias at high bg (where the data-driven softplus grid covers
  the Poisson peak but misses the GG mode at log_mean), and
* discrete band collapse at low SNR (where uniform fixed-grid spacing
  under-resolves the joint peak, and the marginal develops spurious dips
  periodic in (log_S_hi − log_S_lo)/K_log).
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.scatter.base import ScatterModel


# Numerical floors used in derivatives near the GG cusp at eps=0.
# Only matters for β < 2; for β ≥ 2 the second derivative is regular.
_EPS_DERIV_FLOOR = 1e-3   # |eps| floor in d²(GG)/dlog_S²
_HESS_NEG_FLOOR = 1e-6    # |-h| floor in σ_joint = 1/√(-h)

# Hard caps to keep LBFGS line search numerically safe even when the
# optimiser temporarily proposes extreme α / β / log_I that would otherwise
# blow up exp(log_S) or pow(z, β).
_SIGMA_JOINT_MAX = 8.0    # bounds the fine-grid window in log_S
_LOG_S_ABS_MAX = 50.0     # bounds log_S nodes; exp(50) ≈ 5e21 (well below FP64)
_ALPHA_MAX_FOR_GRID = 8.0 # cap on α used as a *width* (not on the prior itself)


def _gg_first_deriv(eps: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """∂/∂log_S  log p_GG(eps; α, β)  =  −β·sign(eps)·|eps/α|^(β−1)/α."""
    abs_eps = eps.abs().clamp(min=1e-12)
    z = abs_eps / alpha
    return -beta * torch.sign(eps) * z.pow(beta - 1.0) / alpha


def _gg_second_deriv(eps: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """∂²/∂log_S²  log p_GG(eps; α, β)  =  −β(β−1)·|eps/α|^(β−2)/α²."""
    abs_eps = eps.abs().clamp(min=_EPS_DERIV_FLOOR)
    z = abs_eps / alpha
    return -beta * (beta - 1.0) * z.pow(beta - 2.0) / (alpha * alpha)


def _poisson_first_deriv(n: Tensor, S: Tensor, bg: Tensor) -> Tensor:
    """∂/∂log_S  log Poisson(n | S + bg)  =  (n/(S+bg) − 1) · S."""
    lam = (S + bg).clamp(min=1e-12)
    return (n / lam - 1.0) * S


def _poisson_second_deriv(n: Tensor, S: Tensor, bg: Tensor) -> Tensor:
    """∂²/∂log_S²  log Poisson(n | S + bg)  =  n·S·bg/(S+bg)² − S."""
    lam = (S + bg).clamp(min=1e-12)
    return n * S * bg / (lam * lam) - S


class LaplaceSaddleGrid(Quadrature):
    """Per-obs joint-saddle Laplace quadrature.

    Parameters
    ----------
    n_coarse
        (legacy) coarse-scan node count — no longer used in the
        deterministic-seed implementation.  Kept for config-compat.
    coarse_n_sigma_alpha
        (legacy) coarse-scan span — see above.
    n_starts
        (legacy) — replaced by deterministic 2-saddle seeding:
        one at log_mean (GG center), one at log((n−bg)⁺) (Poisson
        center).  Always 2 starts; the param is retained but ignored.
        This replaces the old top-K-by-coarse-value scheme which
        introduced discrete saddle-switching artifacts as log_I varied
        during LBFGS line search.
    n_newton_iter
        Newton refinement iterations from each of the 2 deterministic
        seeds (default 5 — enough to converge to ~1e-4 from a far seed).
    newton_step_cap
        Maximum ``|Δlog_S|`` per Newton step, in absolute log-S units
        (default 5.0).
    n_quad
        Fine integration node count (uniform trapezoid in u-coords;
        default 21).
    n_sigma_joint
        Fine-grid span = ``± this · σ_joint`` around ``log_S*``
        (default 5.0).
    signal_min
        Floor on ``exp(log_S)`` for numerical safety (default 0.01).
    detach_grid
        If True (default), the saddle position and width are detached
        from autograd; gradients flow only through the integrand
        evaluation at fixed nodes.  This is the correct approximation
        for a moving grid that's rebuilt every forward pass.
    """

    def __init__(
        self,
        n_coarse: int = 11,
        coarse_n_sigma_alpha: float = 8.0,
        n_starts: int = 3,
        n_newton_iter: int = 5,
        newton_step_cap: float = 5.0,
        n_quad: int = 21,
        n_sigma_joint: float = 5.0,
        signal_min: float = 0.01,
        detach_grid: bool = True,
        pois_seed_kind: str = "clamp",
    ) -> None:
        super().__init__()
        self.n_coarse = int(n_coarse)
        self.coarse_n_sigma_alpha = float(coarse_n_sigma_alpha)
        self.n_starts = int(n_starts)
        self.n_newton_iter = int(n_newton_iter)
        self.newton_step_cap = float(newton_step_cap)
        self.n_quad = int(n_quad)
        self.n_sigma_joint = float(n_sigma_joint)
        self.signal_min = float(signal_min)
        self.detach_grid = bool(detach_grid)
        if pois_seed_kind not in ("clamp", "softplus", "gated"):
            raise ValueError(f"pois_seed_kind must be 'clamp', 'softplus', or "
                             f"'gated', got {pois_seed_kind!r}")
        self.pois_seed_kind = str(pois_seed_kind)
        # Threshold (in photons above bg) for activating seed 2 when
        # pois_seed_kind == "gated".  Obs with (n − bg) ≤ this threshold
        # have seed 2 collapsed to log_mean (= seed 1) so the sort+union
        # grid integrates the single GG-centered window only.  Removes the
        # spurious shared attractor at fixed log_S that pulls all weak
        # HKLs (n ≈ bg) toward a common log_I.
        self.pois_seed_gate_thresh = 1.0

        # Cached u-axes (registered as buffers so .to(device/dtype) propagates).
        self.register_buffer(
            "u_coarse",
            torch.linspace(-self.coarse_n_sigma_alpha,
                           self.coarse_n_sigma_alpha,
                           self.n_coarse),
        )
        self.register_buffer(
            "u_fine",
            torch.linspace(-self.n_sigma_joint,
                           self.n_sigma_joint,
                           self.n_quad),
        )

    @classmethod
    def from_prep(cls, prep: PreparedObs, config: MergeConfig) -> "LaplaceSaddleGrid":
        return cls(
            n_coarse=getattr(config, "laplace_n_coarse", 11),
            coarse_n_sigma_alpha=getattr(config, "laplace_coarse_n_sigma_alpha", 8.0),
            n_starts=getattr(config, "laplace_n_starts", 3),
            n_newton_iter=getattr(config, "laplace_n_newton_iter", 5),
            newton_step_cap=getattr(config, "laplace_newton_step_cap", 5.0),
            n_quad=getattr(config, "n_quad", 21),
            n_sigma_joint=getattr(config, "laplace_n_sigma_joint", 5.0),
            signal_min=0.01,
            detach_grid=getattr(config, "laplace_detach_grid", True),
            pois_seed_kind=getattr(config, "laplace_pois_seed_kind", "clamp"),
        )

    # ── Saddle finder ──────────────────────────────────────────────────
    def _locate_saddles(
        self,
        log_mean: Tensor,        # (M,)
        n_ph: Tensor,            # (M,)
        bg: Tensor,              # (M,)
        alpha: Tensor,           # (M,)
        beta: Tensor,            # (M,)
    ) -> tuple[Tensor, Tensor]:
        """Return (log_S_saddles, sigma_saddles) each of shape (M, K_starts).

        K_starts coarse seeds (top-K by integrand) → damped Newton
        refinement → returns ALL converged saddles (not just the winner)
        with their local σ from the Hessian.  marginalize_signal then
        integrates over the union of saddle windows so that bimodal
        integrands contribute both peaks.
        """
        # Clamp α and β to safe ranges for the saddle locator.  LBFGS line
        # search occasionally proposes extreme params (e.g. α≫1 or β≪1)
        # that make z.pow(β) overflow inside Newton; clamp protects the
        # *grid construction* without affecting the prior used in the
        # final integrand evaluation (which still sees the un-clamped
        # values via scatter.log_prob).
        alpha = alpha.clamp(min=1e-3, max=_ALPHA_MAX_FOR_GRID)
        beta = beta.clamp(min=0.5, max=4.0)

        # ── Deterministic 2-saddle seeding ─────────────────────────────
        # Seed 1: GG center at log_mean (favoured for weak obs)
        # Seed 2: Poisson center at f(n−bg) (favoured for strong obs)
        # Both seeds are ALWAYS present.  Newton refines each independently;
        # for unimodal integrands they converge to the same saddle (collapse
        # to single peak after sort+union); for bimodal they stay in their
        # basins.  Crucially, the seed positions are CONTINUOUS in (n, bg,
        # log_mean), so the saddle set is smooth in trial log_I — no
        # top-K-by-value flipping that would cause discrete jumps in the
        # marginal during LBFGS line search.
        #
        # pois_seed_kind:
        #   "clamp"    : log_S_pois = log( max(n − bg, signal_min) )
        #                — every weak obs (n ≈ bg) clamps to log(signal_min),
        #                  creating a SHARED attractor at log_S = log(signal_min)
        #                  whose marginal contribution peaks at the same log_I
        #                  for every weak HKL → "shrinkage" failure mode.
        #   "softplus" : log_S_pois = log( softplus(n − bg) )
        #                — smooth: ≈ log(n − bg) for n ≫ bg, ≈ log(log 2) ≈ −0.37
        #                  at n = bg, ≈ (n − bg) for n ≪ bg.  Each obs gets a
        #                  data-dependent seed; no shared attractor.
        x = n_ph - bg
        if self.pois_seed_kind == "clamp":
            log_S_pois_seed = x.clamp(min=self.signal_min).log()        # (M,)
        elif self.pois_seed_kind == "softplus":
            log_S_pois_seed = torch.nn.functional.softplus(x) \
                              .clamp(min=1e-30).log()                    # (M,)
        else:  # "gated"
            # Real Poisson mode exists only when n > bg.  When (n − bg) ≤
            # threshold there is no interior maximum to seed, so collapse
            # seed 2 to seed 1 (= log_mean).  Sort+union dedupes, leaving
            # only the GG-centered integration window.
            real_pois = x.clamp(min=self.signal_min).log()
            has_mode = x > self.pois_seed_gate_thresh
            log_S_pois_seed = torch.where(has_mode, real_pois, log_mean)
        log_S = torch.stack([log_mean, log_S_pois_seed], dim=1)         # (M, 2)
        K_starts = 2

        n_b = n_ph[:, None]
        bg_b = bg[:, None]
        alpha_b = alpha[:, None]
        beta_b = beta[:, None]
        log_mean_b = log_mean[:, None]

        # ── Step 2: damped Newton refinement (vectorised over obs × seeds) ──
        # No signal_min clamp here — the saddle-finder must be free to
        # converge to arbitrary log_S (including very negative), as
        # required by low-trial-I observations.  Only clamp from above
        # to avoid exp() overflow.
        for _ in range(self.n_newton_iter):
            eps = log_S - log_mean_b
            S = log_S.exp().clamp(max=1e30)
            g = _gg_first_deriv(eps, alpha_b, beta_b) \
                + _poisson_first_deriv(n_b, S, bg_b)
            h = _gg_second_deriv(eps, alpha_b, beta_b) \
                + _poisson_second_deriv(n_b, S, bg_b)
            # Replace any non-finite g/h with 0 / safe-negative so the
            # step degenerates to "stay put" rather than producing NaN.
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            h = torch.nan_to_num(h, nan=-1.0, posinf=-1.0, neginf=-1.0)
            safe = h < -_HESS_NEG_FLOOR
            step = torch.where(safe, -g / h.clamp(max=-_HESS_NEG_FLOOR),
                               torch.zeros_like(g))
            step = step.clamp(min=-self.newton_step_cap,
                              max=self.newton_step_cap)
            log_S = log_S + step

        # ── Step 3: σ at each saddle from final Hessian ─────────────────
        eps_final = log_S - log_mean_b
        S_final = log_S.exp().clamp(min=self.signal_min)
        h_final = _gg_second_deriv(eps_final, alpha_b, beta_b) \
            + _poisson_second_deriv(n_b, S_final, bg_b)
        neg_h = (-h_final).clamp(min=_HESS_NEG_FLOOR)
        sigma_concave = neg_h.rsqrt()
        # For non-converged saddles (h>=0), fall back to GG width α — this
        # widens the integration window enough to still capture mass.
        # Cap α used here so that LBFGS line-search transients (α swinging
        # to exp(50) etc.) don't make the integration window blow up.
        alpha_capped = alpha_b.clamp(max=_ALPHA_MAX_FOR_GRID).expand_as(log_S)
        sigma_saddles = torch.where(h_final < -_HESS_NEG_FLOOR,
                                     sigma_concave, alpha_capped)
        # Cap σ_joint itself: huge widths give |log_S| → very large which
        # overflows exp(log_S) inside the integrand evaluation.
        sigma_saddles = sigma_saddles.clamp(max=_SIGMA_JOINT_MAX)

        return log_S, sigma_saddles

    # ── Quadrature interface ───────────────────────────────────────────
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
        # Per-obs α_h, β_h via meta.hkl_idx.
        h = meta.hkl_idx
        alpha = params.log_alpha[h].exp().clamp(min=1e-8)
        beta = params.log_beta[h].exp().clamp(min=1e-6)

        # Locate top-K saddles (positions + per-saddle σ).
        if self.detach_grid:
            with torch.no_grad():
                log_S_saddles, sigma_saddles = self._locate_saddles(
                    log_mean.detach(), n_ph, bg,
                    alpha.detach(), beta.detach(),
                )
        else:
            log_S_saddles, sigma_saddles = self._locate_saddles(
                log_mean, n_ph, bg, alpha, beta
            )

        # ── Multi-saddle sub-grids + sorted-union trap weights ──────────
        # Each saddle k gets the FULL n_quad nodes at log_S_k + σ_k·u_sub
        # (u_sub spans ±n_sigma_joint).  Concatenate, sort, and compute
        # trapezoid weights from neighbour distances (HybridLogGrid
        # pattern).
        #
        # Unimodal: all K_starts saddles converge to the same point so
        # K_starts·n_quad nodes cluster at the same window — sort+
        # neighbour-distance trapezoid gives effective n_quad resolution
        # (duplicates contribute 0 weight).  Multimodal: each peak gets
        # full n_quad resolution; the inter-peak gap contributes
        # sparse-weight nodes which is fine because the integrand is
        # exponentially small there.
        K_starts = log_S_saddles.shape[1]
        K_sub = self.n_quad   # per-saddle node count; total = K_starts·K_sub

        u_sub = torch.linspace(-self.n_sigma_joint, self.n_sigma_joint, K_sub,
                                dtype=log_mean.dtype, device=log_mean.device)
        # (M, K_starts, K_sub)
        log_S_per = log_S_saddles[:, :, None] \
                    + sigma_saddles[:, :, None] * u_sub[None, None, :]
        log_S_all = log_S_per.reshape(log_mean.shape[0], K_starts * K_sub)
        # Hard cap to prevent exp(log_S) overflow during LBFGS line search.
        log_S_all = log_S_all.clamp(min=-_LOG_S_ABS_MAX, max=_LOG_S_ABS_MAX)

        log_S_sorted, _ = log_S_all.sort(dim=1)                          # (M, K_total)
        diffs = log_S_sorted[:, 1:] - log_S_sorted[:, :-1]               # (M, K-1)
        w = torch.zeros_like(log_S_sorted)
        w[:, 0]    = diffs[:, 0] * 0.5
        w[:, -1]   = diffs[:, -1] * 0.5
        w[:, 1:-1] = (diffs[:, :-1] + diffs[:, 1:]) * 0.5
        log_w = torch.log(w.clamp(min=1e-30))                            # (M, K_total)

        # ── Integrand at nodes (gradients flow through eps via log_mean) ──
        eps = log_S_sorted - log_mean[:, None]
        S = log_S_sorted.exp()
        log_scatter = scatter.log_prob(eps, params, meta)                # (M, K_total)
        log_lik = likelihood.log_prob_given_signal(n_ph, S, bg)          # (M, K_total)

        return torch.logsumexp(log_scatter + log_lik + log_w, dim=1)
