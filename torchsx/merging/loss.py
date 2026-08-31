"""Streamlined log-space merging loss.

Composes (does not subclass) :class:`torchref.refinement.loss_state.LossState`
with the merging components — observations, parameters, scale, scatter,
quadrature, likelihood, data target — into one object that the user
constructs from a stream and refines with LBFGS.

The chain reads top-to-bottom:

    stream    = SFXStream.load(path)
    loss      = stream.make_loss(n_points=513, gain="auto")
    loss.refine(max_iter=150)
    posterior = loss.posterior(prior="flat")
    data      = posterior.integrate()
    data.write_mtz("out.mtz")

This is the canonical path for the no-prior, log-space-quadrature, global
log-space-posterior workflow.  See the legacy :class:`Merger` for the
configurable multi-prior / multi-quadrature variant.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor
from torchref.refinement.loss_state import LossState

from torchsx.merging.config import MergeConfig
from torchsx.merging.hkl import PoissonNoiseModel
from torchsx.merging.likelihood import Poisson
from torchsx.merging.observations import PreparedObs, prepare_observations
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature import FixedLogGrid
from torchsx.merging.scale import (
    CompositeScale,
    PanelScale,
    ScaleBPerCrystal,
    ScaleModel,
)
from torchsx.merging.regularizers import (
    MillerGraphSmoothness,
    PoissonNullShrinkage,
    ScaleBMeanRegularizer,
    compute_poisson_null_score,
)
from torchsx.merging.scale.per_crystal import ScaleBPerCrystal
from torchsx.merging.scatter import GenGaussLogScatter
from torchsx.merging.scatter_shape import (
    AnisoQuadShape,
    FreePerHKLShape,
    GlobalShape,
    ScatterShape,
)
from torchsx.merging.targets import (
    DataNLLTarget,
    IntensityPriorTarget,
    RegularizerTarget,
    ScaleRegTarget,
)


# DEPRECATED: ``s = n_obs / _REG_SCALE_REF_OBS`` regularizer scaling
# kept the (reg / data) ratio invariant to dataset size but broke
# standard Bayesian asymptotics (it tied the *absolute* prior weight to
# N_obs, so the prior never relinquished relative influence as data
# grew).  ``s`` is now hardcoded to 1.0 below; ``cfg.lam_*`` weights are
# used as absolute losses.  This constant is retained only to keep
# import-by-name compatibility for old call sites.
_REG_SCALE_REF_OBS = 2_200_000


def _find_scale_b_per_crystal(scale: ScaleModel) -> ScaleBPerCrystal | None:
    """Locate the ScaleBPerCrystal inside a (possibly composite) scale module."""
    if isinstance(scale, ScaleBPerCrystal):
        return scale
    if isinstance(scale, CompositeScale):
        for sub in scale.submodules_list:
            if isinstance(sub, ScaleBPerCrystal):
                return sub
    return None


def _resolve_scatter_shape(
    spec: str | ScatterShape,
    prep: PreparedObs,
) -> ScatterShape:
    """Materialize a scatter-shape spec into a :class:`ScatterShape`.

    String specs read seeds from ``prep.alpha_init`` / ``prep.beta_perhkl``
    (the same MoM seeds used by the legacy ``Parameters.from_prep``).
    """
    if isinstance(spec, ScatterShape):
        return spec
    log_alpha_init = torch.log(prep.alpha_init.clamp(min=1e-3))
    log_beta_init = torch.log(prep.beta_perhkl.clamp(min=1e-3))
    if spec == "free":
        return FreePerHKLShape(log_alpha_init, log_beta_init)
    if spec == "global":
        return GlobalShape(
            n_hkl=int(prep.n_valid),
            log_alpha_init=float(log_alpha_init.median()),
            log_beta_init=float(log_beta_init.median()),
        )
    if spec == "aniso_quad":
        from torchref.base.reciprocal.basis import reciprocal_basis_matrix
        median_cell = prep.median_cell.detach().cpu().float()
        B_orth = reciprocal_basis_matrix(median_cell).float()
        s_vec = prep.hkl_valid.float().cpu() @ B_orth
        return AnisoQuadShape(
            s_vec=s_vec,
            log_alpha_init=float(log_alpha_init.median()),
            log_beta_init=float(log_beta_init.median()),
        )
    raise ValueError(
        f"Unknown scatter_shape {spec!r}; expected 'free', 'global', "
        "'aniso_quad', or a ScatterShape instance"
    )


def _resolve_scale(
    spec: str | type[ScaleModel] | ScaleModel | Iterable,
    prep: PreparedObs,
    device: torch.device,
) -> CompositeScale:
    """Materialize a scale spec into a :class:`CompositeScale` on ``device``."""
    if isinstance(spec, str):
        if spec == "b_per_crystal":
            specs: list = [ScaleBPerCrystal]
        elif spec == "panel":
            specs = [PanelScale]
        else:
            raise ValueError(
                f"Unknown scale {spec!r}; expected 'b_per_crystal' or 'panel'"
            )
    elif isinstance(spec, ScaleModel):
        specs = [spec]
    elif isinstance(spec, type) and issubclass(spec, ScaleModel):
        specs = [spec]
    else:
        specs = list(spec)

    subs: list[ScaleModel] = []
    for item in specs:
        if isinstance(item, type) and issubclass(item, ScaleModel):
            subs.append(item.from_obs_info(prep.obs_info))
        elif isinstance(item, ScaleModel):
            subs.append(item)
        else:
            raise TypeError(
                f"Scale entry must be a ScaleModel class or instance, got {type(item)!r}"
            )
    comp = CompositeScale(subs)
    comp.to(device)
    return comp


class LogSpaceLoss:
    """Refinable log-space merging loss for a single :class:`SFXStream`.

    Build via :meth:`SFXStream.make_loss`.  Use :meth:`refine` to drive
    LBFGS in place, then :meth:`posterior` to integrate the per-HKL
    posterior on the same fixed log-S grid.
    """

    def __init__(
        self,
        stream,
        *,
        prep: PreparedObs,
        params: Parameters,
        scale: CompositeScale,
        scatter: GenGaussLogScatter,
        quadrature: FixedLogGrid,
        likelihood: Poisson,
        data_target: DataNLLTarget,
        scale_reg_target: ScaleRegTarget,
        state: LossState,
        device: torch.device,
        gain: float,
        gain_bias_sigma: float | None,
        obs_chunk_size: int,
        wilson_aniso_prior=None,
    ) -> None:
        self._stream = stream
        self._prep = prep
        self._params = params
        self._scale = scale
        self._scatter = scatter
        self._quadrature = quadrature
        self._likelihood = likelihood
        self._data_target = data_target
        self._scale_reg_target = scale_reg_target
        self._state = state
        self._device = device
        self._gain = gain
        self._gain_bias_sigma = gain_bias_sigma
        self._obs_chunk_size = obs_chunk_size
        self._wilson_aniso_prior = wilson_aniso_prior
        self._final_nll: float | None = None
        self._lbfgs_history: list[dict] = []

    # ── construction ──────────────────────────────────────────────

    @classmethod
    def from_stream(
        cls,
        stream,
        *,
        n_points: int = 513,
        S_min: float = 1e-3,
        S_max: float = 1e6,
        gain: float | str = "auto",
        min_obs: int = 5,
        scale: str | type[ScaleModel] | ScaleModel | Iterable = "b_per_crystal",
        scale_reg_weight: float = 1.0,
        scatter_shape: str | ScatterShape = "free",
        smooth: float = 500.0,
        wilson_aniso: float = 4.0,
        wilson_aniso_U_ridge: float = 1.0,
        wilson_aniso_b_center: float = 1.0,
        poisson_null_shrinkage: float = 0.0,
        poisson_null_tau: float | None = None,
        reml: bool = False,
        obs_chunk_size: int = 0,
        device: str | torch.device = "cpu",
        r_peak: float = 7.0,
        r_gap: float = 9.0,
        r_bg: float = 11.0,
        likelihood: str = "gaussian_polynomial",
        var_const: float = 0.0,
        var_linear: float = 1.0,
        var_quadratic: float = 0.0,
        trainable_polynomial_var: bool = False,
        crystal_weights: torch.Tensor | None = None,
    ) -> "LogSpaceLoss":
        """Construct from a stream.  See :meth:`SFXStream.make_loss`.

        ``r_peak / r_gap / r_bg`` are the CrystFEL integration ring radii
        in pixels.  They feed ``n_peak = π·r_peak²`` and
        ``n_bg = π·(r_bg² − r_gap²)`` into both the truncnorm gain
        calibration and the bg-in-peak-ring computation; the right values
        must match the radii used by the indexamajig run that produced
        the stream (otherwise gain comes out badly biased on
        non-default-radii datasets — Rhodopsin uses 2/4/6, tubulin 2/3/6).
        """
        device = torch.device(device)

        # The legacy MergeConfig is the single source of defaults consumed by
        # `prepare_observations` for shell β-fit, FW shell binning, etc.  All
        # priors are off; only the gain / min_obs / obs_chunk_size knobs are
        # routed through.
        cfg = MergeConfig(
            min_obs=min_obs,
            gain=gain,
            obs_chunk_size=obs_chunk_size,
            bg_mode="off",
            posterior_enabled=False,  # streamlined path uses Posterior class
            lam_wilson_aniso_U_ridge=wilson_aniso_U_ridge,
        )

        poisson_noise = PoissonNoiseModel(r_peak=r_peak, r_gap=r_gap, r_bg=r_bg)
        prep = prepare_observations(
            stream, cfg, poisson_noise=poisson_noise,
            crystal_weights=crystal_weights,
        )
        gain_value = float(poisson_noise.gain)
        if device.type != "cpu":
            prep = prep.to(device)

        shape_module = _resolve_scatter_shape(scatter_shape, prep)
        params = Parameters.from_prep(prep, scatter_shape=shape_module).to(device)

        scale_module = _resolve_scale(scale, prep, device)
        scatter = GenGaussLogScatter().to(device)
        quadrature = FixedLogGrid(
            S_min=S_min, S_max=S_max, n_points=n_points,
        ).to(device)
        if likelihood == "poisson":
            likelihood_mod = Poisson().to(device)
            print("  [likelihood] poisson", flush=True)
        elif likelihood == "gaussian_polynomial":
            from torchsx.merging.likelihood import GaussianPolynomialVariance
            bg_var_factor = poisson_noise.n_peak / poisson_noise.n_bg
            likelihood_mod = GaussianPolynomialVariance(
                var_const=var_const,
                var_linear=var_linear,
                var_quadratic=var_quadratic,
                bg_var_factor=bg_var_factor,
                trainable=trainable_polynomial_var,
            ).to(device)
            print(
                f"  [likelihood] gaussian_polynomial  "
                f"g0={var_const:.4g}  g1={var_linear:.4g}  g2={var_quadratic:.4g}  "
                f"bg_var_factor={bg_var_factor:.4f} (geom)  "
                f"trainable={trainable_polynomial_var}",
                flush=True,
            )
        elif likelihood == "gaussian_polynomial_bg_mixture":
            from torchsx.merging.likelihood import GaussianPolynomialWithBgMixture
            bg_var_factor = poisson_noise.n_peak / poisson_noise.n_bg
            likelihood_mod = GaussianPolynomialWithBgMixture(
                var_const=var_const,
                var_linear=var_linear,
                var_quadratic=var_quadratic,
                bg_var_factor=bg_var_factor,
                trainable=trainable_polynomial_var,
            ).to(device)
            print(
                f"  [likelihood] gaussian_polynomial_bg_mixture  "
                f"g0={var_const:.4g}  g1={var_linear:.4g}  g2={var_quadratic:.4g}  "
                f"bg_var_factor={bg_var_factor:.4f} (geom)  "
                f"trainable_var={trainable_polynomial_var}  "
                f"q_a_init={params.q_a.item():.3f}  q_b_init={params.q_b.item():.3f}",
                flush=True,
            )
        else:
            raise ValueError(
                f"Unknown likelihood: {likelihood!r}; expected one of "
                "'gaussian_polynomial', 'gaussian_polynomial_bg_mixture', 'poisson'"
            )

        data_target = DataNLLTarget(
            params=params,
            scale=scale_module,
            scatter=scatter,
            quadrature=quadrature,
            likelihood=likelihood_mod,
            prep=prep,
            chunk_size=obs_chunk_size,
            reml=reml,
        ).to(device)
        scale_reg_target = ScaleRegTarget(params, scale_module).to(device)

        state = LossState(device=device)
        state.register_target("merging/data", data_target)
        state.set_weight("merging/data", 1.0)
        state.register_target("merging/scale_reg", scale_reg_target)
        state.set_weight("merging/scale_reg", scale_reg_weight)

        # DEPRECATED data-size scaling — fixed at 1.0; see module-level
        # ``_REG_SCALE_REF_OBS`` deprecation note.
        s = 1.0

        # Miller-graph Dirichlet smoothing on per-HKL log_α and log_β.
        # Without it, weak HKLs can drive α or β to NaN under LBFGS in
        # float32, which then poisons every Triton-kernel evaluation
        # downstream.  Weight is scaled by data size to keep (reg / data)
        # ratio invariant — same convention as the legacy MergeConfig.
        if smooth > 0:
            for tgt_name, attr in (("alpha_smooth", "log_alpha"),
                                   ("beta_smooth",  "log_beta")):
                module = MillerGraphSmoothness(
                    prep.adjacency, target=attr,
                ).to(device)
                state.register_target(
                    f"merging/reg/{tgt_name}",
                    RegularizerTarget(params, module).to(device),
                )
                state.set_weight(f"merging/reg/{tgt_name}", s * smooth)

        # Anisotropic Wilson MAP prior — active during LBFGS.  When on, also
        # pin mean(per-crystal B) ≈ 0 so the global resolution decay flows
        # into the prior's U tensor instead of being absorbed by the scale.
        wilson_aniso_module = None
        if wilson_aniso > 0:
            from torchsx.merging.priors import WilsonAnisoPrior
            wilson_aniso_module = WilsonAnisoPrior.from_prep(prep, cfg).to(device)
            state.register_target(
                "merging/prior/wilson_aniso",
                IntensityPriorTarget(params, wilson_aniso_module).to(device),
            )
            state.set_weight("merging/prior/wilson_aniso", s * wilson_aniso)

            if wilson_aniso_b_center > 0:
                scale_b = _find_scale_b_per_crystal(scale_module)
                if scale_b is not None:
                    state.register_target(
                        "merging/reg/scale_b_center",
                        RegularizerTarget(
                            params, ScaleBMeanRegularizer(scale_b),
                        ).to(device),
                    )
                    state.set_weight(
                        "merging/reg/scale_b_center",
                        s * wilson_aniso_b_center,
                    )

        # Per-HKL Poisson-null shrinkage on (log α, log β).  Pulls weak
        # HKLs (those well-explained by pure Poisson noise) toward
        # population values while leaving strongly-identified HKLs free.
        if poisson_null_shrinkage > 0:
            log_alpha_pop = float(params.log_alpha.detach().median().item())
            log_beta_pop = float(params.log_beta.detach().median().item())
            w_h, tau_used = compute_poisson_null_score(
                data_target,
                n_hkl=int(params.log_I0.shape[0]),
                log_alpha_pop=log_alpha_pop,
                log_beta_pop=log_beta_pop,
                tau=poisson_null_tau,
            )
            print(
                f"[poisson_null_shrinkage] τ={tau_used:.4g}  "
                f"w mean={float(w_h.mean()):.4f}  "
                f"w median={float(w_h.median()):.4f}  "
                f"frac(w>0.5)={float((w_h > 0.5).float().mean()):.4f}  "
                f"log_α_pop={log_alpha_pop:.4f}  "
                f"log_β_pop={log_beta_pop:.4f}",
                flush=True,
            )
            pn_module = PoissonNullShrinkage(
                w_h, log_alpha_pop, log_beta_pop,
            ).to(device)
            state.register_target(
                "merging/reg/poisson_null",
                RegularizerTarget(params, pn_module).to(device),
            )
            state.set_weight(
                "merging/reg/poisson_null", s * poisson_null_shrinkage,
            )

        return cls(
            stream=stream,
            prep=prep,
            params=params,
            scale=scale_module,
            scatter=scatter,
            quadrature=quadrature,
            likelihood=likelihood_mod,
            data_target=data_target,
            scale_reg_target=scale_reg_target,
            state=state,
            device=device,
            gain=gain_value,
            gain_bias_sigma=None,
            obs_chunk_size=obs_chunk_size,
            wilson_aniso_prior=wilson_aniso_module,
        )

    # ── parametrization swap ──────────────────────────────────────

    def swap_scatter_shape(self, spec: "str | ScatterShape") -> "LogSpaceLoss":
        """Replace the (α, β) parametrization in place, projecting the current
        per-HKL ``log_α``, ``log_β`` onto the new basis.

        Designed for two-pass workflows: do a free per-HKL fit to get the
        gold-standard log_I0 / scale / per-HKL α, β; then swap to a
        lower-rank parametrization (e.g. ``"aniso_quad"``) and refine
        further.  ``log_I0``, scale, ``pi_logit`` are untouched — only the
        ``scatter_shape`` submodule is rebound.  Existing regularizers
        (Miller-graph smoothness, etc.) automatically follow the new
        parametrization on the next ``refine()`` because they read
        ``params.log_alpha`` via the property.

        ``self.final_nll`` is invalidated; call ``refine()`` again to
        re-establish.
        """
        from torchsx.merging.scatter_shape import project_to_shape

        old_log_alpha = self._params.log_alpha.detach().clone()
        old_log_beta = self._params.log_beta.detach().clone()

        if isinstance(spec, ScatterShape):
            new_shape = spec.to(self._device)
        else:
            new_shape = _resolve_scatter_shape(spec, self._prep).to(self._device)

        project_to_shape(new_shape, old_log_alpha, old_log_beta)

        self._params.scatter_shape = new_shape
        self._final_nll = None
        return self

    # ── refinement ────────────────────────────────────────────────

    def _collect_parameters(self) -> list[Tensor]:
        seen: set[int] = set()
        ps: list[Tensor] = []
        for target in self._state.targets.values():
            for p in target.parameters():
                if id(p) in seen or not p.requires_grad:
                    continue
                seen.add(id(p))
                ps.append(p)
        return ps

    def refine(
        self,
        *,
        max_iter: int = 150,
        history_size: int = 10,
        lr: float = 1.0,
        tolerance_grad: float = 1e-6,
        tolerance_change: float = 1e-9,
    ) -> "LogSpaceLoss":
        """Run LBFGS in place on the aggregated state. Returns ``self``."""
        params_all = self._collect_parameters()
        opt = torch.optim.LBFGS(
            params_all,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )

        n_calls = {"n": 0}

        def closure() -> Tensor:
            opt.zero_grad(set_to_none=False)
            total = self._state.aggregate()
            total.backward()
            n_calls["n"] += 1
            return total

        opt.zero_grad(set_to_none=False)
        self._state.aggregate().backward()
        initial_max_grad = max(
            (p.grad.detach().abs().max().item()
             for p in params_all if p.grad is not None),
            default=0.0,
        )

        final_loss = opt.step(closure)
        max_grad = max(
            (p.grad.detach().abs().max().item()
             for p in params_all if p.grad is not None),
            default=0.0,
        )

        with torch.no_grad():
            self._state.aggregate(log_values=True)
        final_data = self._state.get_loss("merging/data")
        self._final_nll = float(final_data.detach().item()) if final_data is not None else None

        self._lbfgs_history.append({
            "n_closure_calls": n_calls["n"],
            "initial_max_grad": initial_max_grad,
            "max_grad": max_grad,
            "final_loss": float(final_loss.detach()) if final_loss is not None else None,
            "final_data_nll": self._final_nll,
        })
        return self

    # ── posterior ─────────────────────────────────────────────────

    def posterior(
        self,
        prior: str = "flat",
        *,
        hessian_filter: str | None = None,
    ):
        """Build a :class:`Posterior` on the loss's fixed log-S grid.

        See :meth:`Posterior.from_loss` for the ``hessian_filter`` argument.
        """
        from torchsx.merging.posterior import Posterior
        return Posterior.from_loss(
            self, prior=prior, hessian_filter=hessian_filter,
        )

    # ── diagnostics ───────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Snapshot of per-target stats, group totals, and gain.

        Pulls from torchref's existing stats infrastructure
        (``target.stats()`` + ``LossState.get_group_totals()``) plus the
        few merging-specific scalars not on a target.
        """
        out: dict = {
            "gain": self._gain,
            "n_valid": self._prep.n_valid,
            "n_obs_total": self._prep.n_obs_total,
        }
        if self._gain_bias_sigma is not None:
            out["gain_bias_sigma"] = self._gain_bias_sigma
        if self._final_nll is not None:
            out["final_data_nll"] = self._final_nll

        per_target: dict = {}
        for name, target in self._state.targets.items():
            if hasattr(target, "stats"):
                try:
                    s = target.stats()
                except Exception:  # noqa: BLE001 — stats are optional
                    s = None
                if s:
                    per_target[name] = s
        if per_target:
            out["targets"] = per_target

        try:
            out["group_totals"] = self._state.get_group_totals()
        except Exception:  # noqa: BLE001
            pass

        return out

    @property
    def history(self) -> list[dict]:
        """Per-iteration history.  Combines the LossState log with the
        per-LBFGS-call summaries recorded by :meth:`refine`."""
        return list(self._state.history) + list(self._lbfgs_history)

    @property
    def final_nll(self) -> float:
        if self._final_nll is None:
            raise RuntimeError("LogSpaceLoss.refine() has not been called yet.")
        return self._final_nll

    @property
    def n_valid(self) -> int:
        return self._prep.n_valid

    @property
    def state(self) -> LossState:
        """Underlying torchref LossState (escape hatch for advanced use)."""
        return self._state

    @property
    def params(self) -> Parameters:
        return self._params

    # ── internals reused by Posterior ──────────────────────────────

    def _hessian_diag_log_I0(
        self,
        *,
        targets_filter: str | None = None,
        as_hessian: bool = False,
    ) -> Tensor:
        """Hessian-diagonal σ on log_I0 from the LBFGS objective.

        Mirrors the legacy `Merger._package_result` curvature pass: data
        target is chunked (allow_unused=True so piecewise-affine quadratures
        contribute 0), all other targets are evaluated in one shot.  Used
        for the posterior's ``mode_shift`` diagnostic.

        Parameters
        ----------
        targets_filter
            If set, include only registered targets whose name starts
            with this prefix (e.g. ``"merging/data"`` for the data-only
            Hessian, ``"merging/prior"`` for prior-only).  Default
            ``None`` walks every target — preserves σ_Laplace behaviour.
        as_hessian
            If True, return the raw H_diag tensor (Fisher info per HKL).
            Default False preserves the legacy σ form
            ``1/sqrt(H.clamp(min=1e-6))``.
        """
        from torchsx.merging.quadrature.fixed_log_grid import eager_only

        params = self._params
        data_target = self._data_target
        data_name = data_target.name
        H_diag = torch.zeros_like(params.log_I0)

        include_data = (
            targets_filter is None or data_name.startswith(targets_filter)
        )
        if include_data:
            # Triton kernel only emits first-order gradients (its saved S1-S4
            # are detached); route the data target through the eager
            # PyTorch path so create_graph=True yields a real second-back.
            with torch.enable_grad(), eager_only():
                for s, e in data_target._iter_chunks():
                    nll_chunk = data_target._chunk_loss(s, e)
                    if not nll_chunk.requires_grad:
                        continue
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

        for name, target in self._state.targets.items():
            if name == data_name:
                continue
            if targets_filter is not None and not name.startswith(targets_filter):
                continue
            with torch.enable_grad():
                w = float(self._state.get_effective_weight(name))
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

        if as_hessian:
            return H_diag
        return (1.0 / H_diag.clamp(min=1e-6)).sqrt()

    def __repr__(self) -> str:
        return (
            f"LogSpaceLoss(n_valid={self._prep.n_valid}, "
            f"n_obs={self._prep.n_obs_total}, "
            f"gain={self._gain:.4g}, "
            f"n_points={self._quadrature.n_points}, "
            f"device={self._device})"
        )
