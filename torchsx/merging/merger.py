"""Central orchestration class: stream → ``MergedResult`` via one LBFGS.

The Merger builds a ``torchref.refinement.loss_state.LossState`` from the
component modules (scale, scatter, quadrature, likelihood, priors,
regularisers) with per-target weights, then runs LBFGS on
``loss = state.aggregate(); loss.backward()``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor
from torchref.refinement.loss_state import LossState

from torchsx.merging.config import MergeConfig
from torchsx.merging.indexing import PoissonNoiseModel
from torchsx.merging.likelihood import (
    DetectionLikelihood,
    Poisson,
    PoissonWithBackgroundMixture,
)
from torchsx.merging.likelihood.poisson_bg_mix import PiBetaPrior
from torchsx.merging.observations import PreparedObs, prepare_observations
from torchsx.merging.parameters import Parameters
from torchsx.merging.priors import (
    FlatPrior,
    WilsonAnisoPrior,
    WilsonMatchPrior,
    WilsonPrior,
)
from torchsx.merging.quadrature import TrapezoidalLogGrid
from torchsx.merging.regularizers import (
    GaussianAnchor,
    MillerGraphSmoothness,
    ScaleBMeanRegularizer,
    SigmaCauchy,
)
from torchsx.merging.posterior import (
    PosteriorMoments,
    compute_posterior_moments,
    compute_posterior_moments_global_grid,
    compute_posterior_moments_per_call,
    make_flat_prior,
    make_wilson_prior,
    shell_mean_from_log_I0,
)
from torchsx.merging.result import MergedResult
from torchsx.merging.scale import CompositeScale, ScaleBPerCrystal, ScaleModel
from torchsx.merging.scatter import GammaLogScatter, GenGaussLogScatter, ScatterModel
from torchsx.merging.targets import (
    DataNLLTarget,
    IntensityPriorTarget,
    RegularizerTarget,
    ScaleRegTarget,
)


def _q_moments(
    alpha: Tensor, beta: Tensor, *, n_grid: int = 201, u_max: float = 10.0
) -> tuple[Tensor, Tensor]:
    """E[q] and Var[q] for q = exp(ε), ε ~ GenGauss(0, α, β) per HKL.

    Numerical integration on standardised u = ε/α over [−u_max, u_max].
    Kept for diagnostics even though not all callers need it.
    """
    device = alpha.device
    u = torch.linspace(-u_max, u_max, n_grid, device=device)
    du = (2.0 * u_max) / (n_grid - 1)
    log_norm = torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)
    abs_u_pow_beta = u.abs().unsqueeze(0).pow(beta.unsqueeze(1))
    log_p = log_norm.unsqueeze(1) - abs_u_pow_beta
    p = log_p.exp()
    q = torch.exp(alpha.unsqueeze(1) * u.unsqueeze(0))
    E_q = (q * p).sum(dim=1) * du
    E_q2 = (q * q * p).sum(dim=1) * du
    Var_q = (E_q2 - E_q * E_q).clamp(min=0.0)
    return E_q, Var_q


def _find_scale_b_per_crystal(scale: ScaleModel) -> ScaleBPerCrystal | None:
    """Locate the ScaleBPerCrystal inside a (possibly composite) scale module."""
    if isinstance(scale, ScaleBPerCrystal):
        return scale
    if isinstance(scale, CompositeScale):
        for sub in scale.submodules_list:
            if isinstance(sub, ScaleBPerCrystal):
                return sub
    return None


class Merger:
    """Merge unmerged SFX observations via one joint LBFGS minimisation.

    Parameters
    ----------
    stream : torchsx SFXStream.
    config : MergeConfig (defaults applied if None).
    scale_models : list of ScaleModel classes or instances, or None.
        - ``None`` → defaults to ``[ScaleBPerCrystal]``
        - class → built from ``obs_info`` via ``from_obs_info``
        - instance → used as-is
    partiality : (N_obs,) tensor in stream order, optional.
    device : 'cpu' or 'cuda'.
    poisson_noise : PoissonNoiseModel, optional.
    init_from : optional ``MergedResult`` to seed log_I0 / α / β.
    """

    def __init__(
        self,
        stream,
        *,
        config: MergeConfig | None = None,
        scale_models: Sequence[ScaleModel] | None = None,
        partiality: Tensor | None = None,
        device: str | torch.device = "cpu",
        poisson_noise: PoissonNoiseModel | None = None,
        init_from: MergedResult | None = None,
    ) -> None:
        self.stream = stream
        self.config = config or MergeConfig()
        self.device = torch.device(device)
        self.partiality = partiality
        self.poisson_noise = poisson_noise or PoissonNoiseModel()
        self._user_scale_models = scale_models
        self.init_from = init_from

        # Filled by merge()
        self.prep: PreparedObs | None = None
        self.params: Parameters | None = None
        self.scale: CompositeScale | None = None
        self.scatter: GenGaussLogScatter | None = None
        self.quadrature: TrapezoidalLogGrid | None = None
        self.likelihood: DetectionLikelihood | None = None
        self.data_target: DataNLLTarget | None = None
        self.state: LossState | None = None
        self.diagnostics: dict = {}

    # ── public API ─────────────────────────────────────────────────

    def merge(self) -> MergedResult:
        self._prepare()
        self._build_loss()
        self._run_lbfgs()
        return self._package_result()

    # ── internals ──────────────────────────────────────────────────

    def _prepare(self) -> None:
        prep = prepare_observations(
            self.stream,
            self.config,
            partiality=self.partiality,
            poisson_noise=self.poisson_noise,
        )
        if self.init_from is not None:
            self._seed_init_from(prep, self.init_from)
        if self.device.type != "cpu":
            prep = prep.to(self.device)
        self.prep = prep

    @staticmethod
    def _seed_init_from(prep: PreparedObs, source: MergedResult) -> None:
        src_hkl = source.hkl.long()
        lookup: dict[tuple[int, int, int], int] = {}
        for i in range(src_hkl.shape[0]):
            key = (int(src_hkl[i, 0]), int(src_hkl[i, 1]), int(src_hkl[i, 2]))
            lookup[key] = i

        tgt_hkl = prep.hkl_valid.long()
        match = torch.full((prep.n_valid,), -1, dtype=torch.long)
        for i in range(prep.n_valid):
            key = (int(tgt_hkl[i, 0]), int(tgt_hkl[i, 1]), int(tgt_hkl[i, 2]))
            j = lookup.get(key)
            if j is not None:
                match[i] = j
        has = match >= 0
        n_match = int(has.sum())
        if n_match == 0:
            return
        idx = match[has]
        # Guard against non-finite / overflowed source values (e.g. posterior
        # blow-up on underconstrained HKLs that missed the fallback).  Bad
        # values are replaced with the existing shell-fit init so the halves'
        # LBFGS cannot inherit inf / NaN mu_init.
        src_I0 = source.I0[idx].clone()
        src_alpha = source.alpha[idx].clone()
        src_beta = source.beta[idx].clone()
        bad_I0 = ~torch.isfinite(src_I0) | (src_I0 <= 0)
        bad_alpha = ~torch.isfinite(src_alpha) | (src_alpha <= 0)
        bad_beta = ~torch.isfinite(src_beta) | (src_beta <= 0)
        if bool(bad_I0.any() | bad_alpha.any() | bad_beta.any()):
            print(
                f"  init_from: discarding {int(bad_I0.sum())} bad I0 / "
                f"{int(bad_alpha.sum())} bad α / {int(bad_beta.sum())} bad β "
                "from source (using shell-fit init for those HKLs)",
                flush=True,
            )
        where_idx = has.nonzero(as_tuple=True)[0]
        ok = ~(bad_I0 | bad_alpha | bad_beta)
        good = where_idx[ok]
        src_ok_idx = idx[ok]
        prep.mu_init[good] = torch.log(
            source.I0[src_ok_idx].clamp(min=1e-6, max=1e20)
        )
        prep.alpha_init[good] = source.alpha[src_ok_idx].clamp(min=0.01, max=100.0)
        prep.beta_perhkl[good] = source.beta[src_ok_idx].clamp(min=0.5, max=4.0)
        print(
            f"  init_from: matched {n_match}/{prep.n_valid} HKLs "
            f"({int(ok.sum())} used as-is, {int((~ok).sum())} fell back to shell init)",
            flush=True,
        )

    def _build_scale(self) -> CompositeScale:
        """Resolve user-supplied scale models into a ``CompositeScale``."""
        user = self._user_scale_models
        if user is None:
            user = [ScaleBPerCrystal]
        subs: list[ScaleModel] = []
        for item in user:
            if isinstance(item, type) and issubclass(item, ScaleModel):
                subs.append(item.from_obs_info(self.prep.obs_info))
            elif isinstance(item, ScaleModel):
                subs.append(item)
            else:
                raise TypeError(
                    f"scale_models entry must be a ScaleModel class or instance, got {type(item)!r}"
                )
        comp = CompositeScale(subs)
        comp.to(self.device)
        return comp

    def _build_scatter(self) -> ScatterModel:
        kind = getattr(self.config, "scatter", "gengauss")
        if kind == "gengauss":
            return GenGaussLogScatter()
        if kind == "gamma":
            return GammaLogScatter()
        raise ValueError(f"Unknown scatter: {kind!r}")

    def _build_likelihood(self) -> DetectionLikelihood:
        mode = getattr(self.config, "bg_mode", "off")
        if mode == "off":
            return Poisson()
        if mode == "global":
            return PoissonWithBackgroundMixture(mode="global")
        if mode == "global_learned":
            return PoissonWithBackgroundMixture(
                mode="global_learned",
                pi_init=getattr(self.config, "pi_global_init", 0.1),
            )
        if mode == "per_hkl":
            return PoissonWithBackgroundMixture(mode="per_hkl")
        raise ValueError(f"Unknown bg_mode: {mode!r}")

    def _build_loss(self) -> None:
        prep = self.prep
        cfg = self.config
        device = self.device

        # ── Shared learnable state ─────────────────────────────────
        params = Parameters.from_prep(prep)
        # Seed pi_logit at logit(0.1) ≈ -2.2 (most HKLs bright) when per_hkl.
        if getattr(cfg, "bg_mode", "off") == "per_hkl":
            with torch.no_grad():
                params.pi_logit.fill_(-2.2)
        params.to(device)

        # ── Component modules ──────────────────────────────────────
        scale = self._build_scale()
        scatter = self._build_scatter().to(device)
        quadrature = TrapezoidalLogGrid.from_prep(prep, cfg).to(device)
        likelihood = self._build_likelihood().to(device)

        # ── Data target ────────────────────────────────────────────
        data_target = DataNLLTarget(
            params=params,
            scale=scale,
            scatter=scatter,
            quadrature=quadrature,
            likelihood=likelihood,
            prep=prep,
            chunk_size=cfg.obs_chunk_size,
        ).to(device)

        # ── LossState wiring ───────────────────────────────────────
        state = LossState(device=device)
        state.register_target("merging/data", data_target)
        state.set_weight("merging/data", 1.0)

        # Data-size scaling factor — DEPRECATED.  The historical
        # ``s = n_obs / reg_scale_ref_obs`` multiplier kept the
        # regulariser/data NLL ratio invariant to dataset size, but as a
        # side-effect tied the *absolute* prior weight to N_obs and broke
        # standard Bayesian asymptotics (prior should shrink in relative
        # influence as data grows).  Fixed at 1.0; absolute weights are
        # now taken straight from ``cfg.lam_*``.  ``reg_scale_ref_obs``
        # is retained on ``MergeConfig`` only for backwards compatibility
        # and is no longer consulted.
        s = 1.0

        def _reg_target(key: str, module, weight: float) -> None:
            if weight <= 0:
                return
            if isinstance(module, ScaleModel):
                tgt = ScaleRegTarget(params, module).to(device)
            elif hasattr(module, "forward") and "Parameters" in str(
                getattr(module.forward, "__annotations__", {}).get("params", "")
            ):
                tgt = RegularizerTarget(params, module).to(device)
            else:
                tgt = RegularizerTarget(params, module).to(device)
            state.register_target(key, tgt)
            state.set_weight(key, weight)

        # Scale regulariser (ridge on log_G, B etc. — not data-size-scaled:
        # matches the old pipeline, where correction regularisers lived
        # outside the ``s`` multiplier).
        scale_reg = ScaleRegTarget(params, scale).to(device)
        state.register_target("merging/scale_reg", scale_reg)
        state.set_weight("merging/scale_reg", 1.0)

        # Miller-graph smoothing on α, β, I₀.
        if cfg.lam_alpha_smooth > 0:
            state.register_target(
                "merging/reg/alpha_smooth",
                RegularizerTarget(
                    params,
                    MillerGraphSmoothness(prep.adjacency, target="log_alpha").to(device),
                ).to(device),
            )
            state.set_weight("merging/reg/alpha_smooth", s * cfg.lam_alpha_smooth)
        if cfg.lam_beta_smooth > 0:
            state.register_target(
                "merging/reg/beta_smooth",
                RegularizerTarget(
                    params,
                    MillerGraphSmoothness(prep.adjacency, target="log_beta").to(device),
                ).to(device),
            )
            state.set_weight("merging/reg/beta_smooth", s * cfg.lam_beta_smooth)
        if cfg.lam_I0_smooth > 0:
            state.register_target(
                "merging/reg/I0_smooth",
                RegularizerTarget(
                    params,
                    MillerGraphSmoothness(prep.adjacency, target="log_I0").to(device),
                ).to(device),
            )
            state.set_weight("merging/reg/I0_smooth", s * cfg.lam_I0_smooth)

        # Gaussian anchors toward shell-fit init values.
        if cfg.lam_alpha_anchor > 0:
            anchor = GaussianAnchor("log_alpha", params.log_alpha.detach().clone()).to(device)
            state.register_target(
                "merging/reg/alpha_anchor",
                RegularizerTarget(params, anchor).to(device),
            )
            state.set_weight("merging/reg/alpha_anchor", s * cfg.lam_alpha_anchor)
        if cfg.lam_beta_anchor > 0:
            anchor = GaussianAnchor("log_beta", params.log_beta.detach().clone()).to(device)
            state.register_target(
                "merging/reg/beta_anchor",
                RegularizerTarget(params, anchor).to(device),
            )
            state.set_weight("merging/reg/beta_anchor", s * cfg.lam_beta_anchor)

        # Weak Gaussian floor pull on log_I0.
        if cfg.lam_I0_floor > 0:
            floor_anchor = GaussianAnchor(
                "log_I0", float(cfg.log_I0_floor_target)
            ).to(device)
            state.register_target(
                "merging/reg/I0_floor",
                RegularizerTarget(params, floor_anchor).to(device),
            )
            state.set_weight("merging/reg/I0_floor", s * cfg.lam_I0_floor)

        # Half-Cauchy prior on σ = α · c(β).
        if cfg.lam_sigma_cauchy > 0:
            sc = SigmaCauchy(tau=cfg.sigma_cauchy_scale).to(device)
            state.register_target(
                "merging/reg/sigma_cauchy",
                RegularizerTarget(params, sc).to(device),
            )
            state.set_weight("merging/reg/sigma_cauchy", s * cfg.lam_sigma_cauchy)

        # Wilson MAP prior — shell and anisotropic forms are alternative
        # parameterizations of the same prior; require at most one.
        if cfg.lam_frac_wilson > 0 and cfg.lam_wilson_aniso > 0:
            raise ValueError(
                "lam_frac_wilson and lam_wilson_aniso are alternative "
                "parameterizations of the Wilson prior; set at most one > 0."
            )
        if cfg.lam_frac_wilson > 0:
            wilson = WilsonPrior.from_prep(prep, cfg).to(device)
            state.register_target(
                "merging/prior/wilson",
                IntensityPriorTarget(params, wilson).to(device),
            )
            state.set_weight("merging/prior/wilson", s * cfg.lam_frac_wilson)

        # Wilson anisotropic prior (continuous log_I_mean + ADP-like U).
        if cfg.lam_wilson_aniso > 0:
            wilson_aniso = WilsonAnisoPrior.from_prep(prep, cfg).to(device)
            state.register_target(
                "merging/prior/wilson_aniso",
                IntensityPriorTarget(params, wilson_aniso).to(device),
            )
            state.set_weight("merging/prior/wilson_aniso", s * cfg.lam_wilson_aniso)

        # Pin mean(per-crystal B) → 0 (independent of Wilson aniso prior).
        # When lam_wilson_aniso > 0, the global decay flows into the prior's U
        # so B-mean must be centred; otherwise centring B-mean prevents the
        # per-crystal B from absorbing the global resolution decay and
        # collapsing per-HKL dynamic range at high resolution.
        if cfg.lam_wilson_aniso_b_center > 0:
            scale_b = _find_scale_b_per_crystal(scale)
            if scale_b is not None:
                state.register_target(
                    "merging/reg/scale_b_center",
                    RegularizerTarget(params, ScaleBMeanRegularizer(scale_b)).to(device),
                )
                state.set_weight(
                    "merging/reg/scale_b_center",
                    s * cfg.lam_wilson_aniso_b_center,
                )

        # Wilson distribution-match prior.
        if cfg.lam_wilson_match > 0:
            wm = WilsonMatchPrior.from_prep(prep, cfg).to(device)
            state.register_target(
                "merging/prior/wilson_match",
                IntensityPriorTarget(params, wm).to(device),
            )
            state.set_weight("merging/prior/wilson_match", s * cfg.lam_wilson_match)

        # Beta(a, b) prior on π_h (per-HKL bg mode only).
        if getattr(cfg, "bg_mode", "off") == "per_hkl" and cfg.lam_pi_beta > 0:
            pb = PiBetaPrior(a=cfg.pi_beta_a, b=cfg.pi_beta_b).to(device)
            state.register_target(
                "merging/reg/pi_beta",
                RegularizerTarget(params, pb).to(device),
            )
            # NOTE: legacy pipeline did NOT apply the data-size factor `s`
            # to lam_pi_beta.  Match that for equivalence.
            state.set_weight("merging/reg/pi_beta", cfg.lam_pi_beta)

        # Keep references alive (LossState holds targets; we hold components
        # so the caller can inspect them post-merge).
        self.params = params
        self.scale = scale
        self.scatter = scatter
        self.quadrature = quadrature
        self.likelihood = likelihood
        self.data_target = data_target
        self.state = state

    def _collect_parameters(self) -> list[Tensor]:
        """Every learnable parameter reachable from the LossState targets."""
        seen: set[int] = set()
        ps: list[Tensor] = []
        for target in self.state.targets.values():
            for p in target.parameters():
                if id(p) in seen or not p.requires_grad:
                    continue
                seen.add(id(p))
                ps.append(p)
        return ps

    def _run_lbfgs(self) -> None:
        cfg = self.config
        state = self.state

        params_all = self._collect_parameters()
        opt = torch.optim.LBFGS(
            params_all,
            lr=cfg.lr,
            max_iter=cfg.max_iter,
            history_size=cfg.history_size,
            tolerance_grad=cfg.tolerance_grad,
            tolerance_change=cfg.tolerance_change,
            line_search_fn="strong_wolfe",
        )

        n_calls = {"n": 0}

        def closure() -> Tensor:
            opt.zero_grad(set_to_none=False)
            total = state.aggregate()
            total.backward()
            n_calls["n"] += 1
            return total

        # Initial-gradient baseline (so we can report grad-reduction).
        opt.zero_grad(set_to_none=False)
        state.aggregate().backward()
        initial_max_grad = max(
            (p.grad.detach().abs().max().item() for p in params_all if p.grad is not None),
            default=0.0,
        )

        final_loss = None
        grad_history: list[float] = []
        for outer in range(max(1, cfg.n_outer_iters)):
            final_loss = opt.step(closure)
            rnd_max_grad = max(
                (p.grad.detach().abs().max().item() for p in params_all if p.grad is not None),
                default=0.0,
            )
            grad_history.append(rnd_max_grad)
            if cfg.n_outer_iters > 1:
                print(
                    f"    LBFGS outer {outer+1}/{cfg.n_outer_iters}: "
                    f"max|grad|={rnd_max_grad:.3e}",
                    flush=True,
                )
            if rnd_max_grad < cfg.tolerance_grad:
                break

        # Final breakdown of data vs reg for the diagnostic dict.
        with torch.no_grad():
            state.aggregate(log_values=True)
        final_data = state.get_loss("merging/data")
        final_reg = sum(
            (v.detach() for k, v in state._losses.items() if k != "merging/data"),
            torch.zeros((), device=self.device),
        )

        max_grad = max(
            (p.grad.detach().abs().max().item() for p in params_all if p.grad is not None),
            default=0.0,
        )
        self.diagnostics.update(
            final_loss=float(final_loss.detach()) if final_loss is not None else None,
            final_data_nll=float(final_data.item() if final_data is not None else 0.0),
            final_reg=float(final_reg.item()) if isinstance(final_reg, Tensor) else float(final_reg),
            n_closure_calls=n_calls["n"],
            initial_max_grad=initial_max_grad,
            max_grad=max_grad,
            grad_reduction=initial_max_grad / max(max_grad, 1e-30),
            grad_history=grad_history,
            n_outer_iters_used=len(grad_history),
            max_iter=cfg.max_iter,
            tolerance_grad=cfg.tolerance_grad,
        )

    def _package_result(self) -> MergedResult:
        prep = self.prep
        cfg = self.config
        params = self.params
        data_target = self.data_target

        # ── Hessian-diagonal σ(log_I0) from the FULL LBFGS objective ──
        # All targets (data + scale_reg + intensity priors + Miller-graph
        # smoothness + ...) are evaluated together so the curvature reflects
        # exactly what LBFGS minimised.  Without this, the data Hessian alone
        # can be negative at the LBFGS-MAP whenever a prior pulled log_I0
        # away from the data-only minimum (saddle of the data NLL); only the
        # prior's curvature keeps the *full* objective convex.
        #
        # Two-pass diagonal: vJp with v = ones gives the row-sum of the
        # Hessian; for terms whose log_I0 dependence is per-HKL (data NLL,
        # WilsonPrior, GaussianAnchor, I0_floor) this equals the diagonal.
        # Miller-graph smoothness adds off-diagonal coupling — so its row-sum
        # is an *upper bound* on the diagonal (over-estimates σ slightly for
        # HKLs with active smoothing).  Acceptable for σ reporting.
        H_diag = torch.zeros_like(params.log_I0)
        # We need to chunk only the data target; priors fit in one shot.
        # Easiest implementation: re-aggregate the full state but split the
        # data target's autograd into chunks.
        for s, e in data_target._iter_chunks():
            nll_chunk = data_target._chunk_loss(s, e)
            # ``allow_unused=True`` covers piecewise-affine quadratures
            # (e.g. PrecomputedGridQuadrature) whose 2nd derivative wrt
            # log_I0 is identically zero; the data Hessian is then 0 and
            # σ comes from priors + posterior moments.
            grad_logI0 = torch.autograd.grad(
                nll_chunk, params.log_I0,
                create_graph=True, allow_unused=True,
            )[0]
            if grad_logI0 is None:
                continue
            H_chunk = torch.autograd.grad(
                grad_logI0, params.log_I0,
                grad_outputs=torch.ones_like(params.log_I0),
                retain_graph=False, allow_unused=True,
            )[0]
            if H_chunk is not None:
                H_diag = H_diag + H_chunk.detach()

        # All non-data targets, in one shot.  The state's losses dict was
        # populated by the last LBFGS step; we need fresh values with grad
        # to take second derivatives.
        for name, target in self.state.targets.items():
            if name == data_target.name:
                continue
            with torch.enable_grad():
                w = float(self.state.get_effective_weight(name))
                if w == 0.0:
                    continue
                loss = w * target()
                if not loss.requires_grad:
                    continue
                grad_logI0 = torch.autograd.grad(
                    loss, params.log_I0, create_graph=True, allow_unused=True,
                )[0]
                if grad_logI0 is None:
                    continue
                H_chunk = torch.autograd.grad(
                    grad_logI0, params.log_I0,
                    grad_outputs=torch.ones_like(params.log_I0),
                    retain_graph=False, allow_unused=True,
                )[0]
                if H_chunk is not None:
                    H_diag = H_diag + H_chunk.detach()

        with torch.no_grad():
            I0_map = torch.exp(params.log_I0)
            alpha = torch.exp(params.log_alpha)
            beta = torch.exp(params.log_beta)

            sigma_logI0 = (1.0 / H_diag.clamp(min=1e-6)).sqrt()
            sigma_I0_map = sigma_logI0 * I0_map

            # Truncation-bias correction (carried over from the legacy merger).
            # Applied to the MAP column only; the posterior-moment pass below
            # does not need it because the integral already accounts for the
            # quadrature-grid lower bound via ``sig_lo_per_obs``.
            if cfg.apply_bias_correction:
                bg_sum = torch.zeros(prep.n_valid, device=self.device).scatter_add_(
                    0, data_target.hkl_id, prep.bg.clamp(min=0)
                )
                bg_per_hkl = (bg_sum / prep.obs_counts.float().clamp(min=1)).clamp(min=1.0)
                sigma_floor = (bg_per_hkl / prep.obs_counts.float().clamp(min=1)).sqrt()
                sig_lo_hkl = (sigma_floor * 0.5).clamp(min=0.01)

                log_I0_raw = torch.log(I0_map.clamp(min=1e-8))
                eps_min = torch.log(sig_lo_hkl.clamp(min=1e-8)) - log_I0_raw
                z_min = eps_min / alpha.clamp(min=1e-8)
                phi = torch.exp(-0.5 * z_min * z_min) / math.sqrt(2 * math.pi)
                Phi_c = (0.5 * torch.erfc(z_min / math.sqrt(2))).clamp(min=1e-8)
                bias = alpha * phi / Phi_c
                I0_map = torch.exp(log_I0_raw - bias)

            mu_map = torch.log(I0_map.clamp(min=1e-8))
            sigma_mu_map = sigma_I0_map / I0_map.clamp(min=1e-8)

            pi_bg = torch.sigmoid(params.pi_logit).detach().cpu()

        # ── Posterior moments (optional) ─────────────────────────────
        # Both flat and Wilson priors are evaluated by default.  Σ̂ for
        # the Wilson prior is the shell-arithmetic-mean of exp(log_I0_MAP);
        # with ~hundreds-to-thousands of HKLs per shell, std(Σ̂) scales
        # like std(I)/√N_shell, so Σ̂ is far more tightly determined than
        # any individual I₀ and the plug-in empirical-Bayes approximation
        # is well-behaved.  The primary writeout column is picked by
        # ``cfg.posterior_primary`` (default: "flat"); the other variant
        # is stored in diagnostics for comparison.
        posteriors: dict[str, PosteriorMoments] = {}
        if cfg.posterior_enabled:
            prior_specs = {
                "flat": make_flat_prior(),
                "wilson": make_wilson_prior(
                    Sigma_shell=shell_mean_from_log_I0(
                        params.log_I0.detach(),
                        prep.shell_id_fw.to(self.device),
                        n_shells=cfg.fw_n_shells,
                        eps=cfg.fw_eps,
                    ),
                    is_centric=prep.is_centric.to(self.device),
                ),
            }
            # Quadratures whose grid moves with log_mean (e.g. WideLogGrid,
            # LaplaceSaddleGrid, HierarchicalImportanceQuadrature) need the
            # per-call posterior path so the µ-coarse scan and the MAP
            # marginal use the *same* grid; otherwise the scan silently
            # falls back to the softplus inner grid (via _sig_lo_slice /
            # _build_grid) and the posterior MAP inflates.
            _per_call = getattr(
                self.quadrature, "requires_per_call_posterior", False
            )
            _grid_mode = getattr(cfg, "posterior_grid_mode", "laplace")
            if _grid_mode == "global":
                # Use the inner quadrature's log_S grid as the candidate
                # log_I0 grid when available (FixedLogGrid); otherwise
                # build one from config knobs.
                qgrid = getattr(self.quadrature, "log_S_grid", None)
                if qgrid is not None:
                    log_I_grid = qgrid.detach().to(self.device)
                else:
                    log_I_grid = torch.linspace(
                        cfg.posterior_global_log_I_min,
                        cfg.posterior_global_log_I_max,
                        int(cfg.posterior_global_n_points),
                        device=self.device,
                    )
                posteriors = compute_posterior_moments_global_grid(
                    prep=prep, params=params,
                    scale=self.scale, scatter=self.scatter,
                    quadrature=self.quadrature, likelihood=self.likelihood,
                    sigma_laplace=sigma_logI0, priors=prior_specs,
                    log_I_grid=log_I_grid,
                    obs_chunk_size=cfg.posterior_obs_chunk_size,
                    sigma_max=cfg.posterior_sigma_max,
                    map_fallback_log_I0=params.log_I0.detach(),
                    summary=getattr(cfg, "posterior_summary", "linear_mean"),
                )
            else:
                _post_fn = (
                    compute_posterior_moments_per_call
                    if _per_call else compute_posterior_moments
                )
                posteriors = _post_fn(
                    prep=prep,
                    params=params,
                    scale=self.scale,
                    scatter=self.scatter,
                    quadrature=self.quadrature,
                    likelihood=self.likelihood,
                    sigma_laplace=sigma_logI0,
                    priors=prior_specs,
                    n_grid_coarse=cfg.posterior_n_grid_coarse,
                    n_grid_fine=cfg.posterior_n_grid_fine,
                    n_sigma=cfg.posterior_n_sigma,
                    obs_chunk_size=cfg.posterior_obs_chunk_size,
                    sigma_max=cfg.posterior_sigma_max,
                    map_fallback_log_I0=params.log_I0.detach(),
                    summary=getattr(cfg, "posterior_summary", "linear_mean"),
                )

        diag = dict(self.diagnostics)
        diag["pi_bg"] = pi_bg
        diag["bg_mode"] = getattr(cfg, "bg_mode", "off")
        diag["loss_history"] = self.state.history
        # Surface global learned π if active.
        if hasattr(self.likelihood, "pi_logit_global"):
            pi_g = torch.sigmoid(self.likelihood.pi_logit_global).detach()
            diag["pi_global_learned"] = float(pi_g.item())

        # ── Per-HKL + per-shell quality diagnostics ──────────────
        with torch.no_grad():
            n_obs_per_hkl = prep.obs_counts.cpu()
            shell_id_fw = prep.shell_id_fw.cpu()
            # Per-shell aggregates (count, median I0_MAP, med α, med β).
            n_shells = int(shell_id_fw.max().item()) + 1 if prep.n_valid > 0 else 0
            d_cpu = prep.d_spacing.cpu()
            I0_map_cpu = I0_map.cpu()
            alpha_cpu = alpha.cpu()
            beta_cpu = beta.cpu()
            per_shell_count = torch.zeros(n_shells, dtype=torch.int64)
            per_shell_med_I0 = torch.zeros(n_shells)
            per_shell_med_alpha = torch.zeros(n_shells)
            per_shell_med_beta = torch.zeros(n_shells)
            per_shell_d_min = torch.full((n_shells,), float("nan"))
            per_shell_d_max = torch.full((n_shells,), float("nan"))
            for sid in range(n_shells):
                mask = shell_id_fw == sid
                if not mask.any():
                    continue
                per_shell_count[sid] = int(mask.sum())
                per_shell_med_I0[sid] = I0_map_cpu[mask].median()
                per_shell_med_alpha[sid] = alpha_cpu[mask].median()
                per_shell_med_beta[sid] = beta_cpu[mask].median()
                per_shell_d_min[sid] = d_cpu[mask].min()
                per_shell_d_max[sid] = d_cpu[mask].max()

        diag["per_hkl"] = {
            "n_obs": n_obs_per_hkl,
            "shell_id": shell_id_fw,
            "is_centric": prep.is_centric.cpu(),
            "d_spacing": d_cpu,
        }
        diag["per_shell"] = {
            "count": per_shell_count,
            "d_min": per_shell_d_min,
            "d_max": per_shell_d_max,
            "med_I0_map": per_shell_med_I0,
            "med_alpha": per_shell_med_alpha,
            "med_beta": per_shell_med_beta,
        }

        # Keep MAP + Laplace column alongside the posterior mean.
        diag["I0_map"] = I0_map.cpu()
        diag["sigma_I0_map"] = sigma_I0_map.cpu()
        diag["mu_map"] = mu_map.cpu()
        diag["sigma_mu_map"] = sigma_mu_map.cpu()

        # LossState breakdown — each registered target's final value + weight.
        final_breakdown: dict[str, dict] = {}
        for name, loss_t in self.state._losses.items():
            w = self.state.get_effective_weight(name)
            lv = float(loss_t.detach().item()) if loss_t is not None else 0.0
            final_breakdown[name] = {"loss": lv, "weight": w, "weighted": w * lv}
        diag["loss_breakdown"] = final_breakdown

        if posteriors:
            diag["posteriors"] = {
                k: {
                    "I0": v.I0.cpu(),
                    "sigma_I0": v.sigma_I0.cpu(),
                    "mu": v.mu.cpu(),
                    "sigma_mu": v.sigma_mu.cpu(),
                    "log_evidence": v.log_evidence.cpu(),
                    "fallback_mask": v.fallback_mask.cpu(),
                    "grid_mass_frac": v.grid_mass_frac.cpu(),
                    "mode_shift": v.mode_shift.cpu(),
                    "sigma_laplace_clamped": v.sigma_laplace_clamped.cpu(),
                    # Aggregate summary that's cheap to print:
                    "n_fallback": int(v.fallback_mask.sum().item()),
                    "grid_mass_frac_q05": float(
                        v.grid_mass_frac.kthvalue(
                            max(1, int(0.05 * v.grid_mass_frac.numel()))
                        ).values.item()
                    ),
                }
                for k, v in posteriors.items()
            }

        # Pick the primary (I0, σ_I0) column per config.
        primary_key = cfg.posterior_primary if posteriors else None
        if primary_key and primary_key in posteriors:
            pm = posteriors[primary_key]
            I0_out, sigma_I0_out = pm.I0, pm.sigma_I0
            mu_out, sigma_mu_out = pm.mu, pm.sigma_mu
            diag["primary_posterior"] = primary_key
            diag["primary_fallback_mask"] = pm.fallback_mask.cpu()
            # Log one-line summary for the user (visible in fit.out).
            n_fb = int(pm.fallback_mask.sum().item())
            med_shift = pm.mode_shift.abs().median().item()
            q05_mass = diag["posteriors"][primary_key]["grid_mass_frac_q05"]
            print(
                f"  [posterior/{primary_key}] n_fallback={n_fb}/{prep.n_valid} "
                f"median|mode_shift|={med_shift:.3f}σ  "
                f"grid_mass_frac q05={q05_mass:.3f}",
                flush=True,
            )
        else:
            I0_out, sigma_I0_out = I0_map, sigma_I0_map
            mu_out, sigma_mu_out = mu_map, sigma_mu_map
            diag["primary_posterior"] = "map"

        return MergedResult(
            hkl=prep.hkl_valid.cpu(),
            I0=I0_out.cpu(),
            alpha=alpha.cpu(),
            beta=beta.cpu(),
            mu=mu_out.cpu(),
            d_spacing=prep.d_spacing.cpu(),
            n_obs=prep.obs_counts.cpu(),
            sigma_I0=sigma_I0_out.cpu(),
            sigma_mu=sigma_mu_out.cpu(),
            diagnostics=diag,
        )
