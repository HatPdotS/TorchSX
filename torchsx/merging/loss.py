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
from torchsx.merging.indexing import PoissonNoiseModel
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
from torchsx.merging.regularizers import MillerGraphSmoothness, ScaleBMeanRegularizer
from torchsx.merging.scale.per_crystal import ScaleBPerCrystal
from torchsx.merging.scatter import GenGaussLogScatter
from torchsx.merging.targets import (
    DataNLLTarget,
    IntensityPriorTarget,
    RegularizerTarget,
    ScaleRegTarget,
)


# Reference observation count used to scale the Miller-graph smoothing
# weight so the (reg / data) ratio is invariant across dataset sizes.
# Matches the legacy ``MergeConfig.reg_scale_ref_obs`` default.
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
        smooth: float = 500.0,
        wilson_aniso: float = 0.0,
        wilson_aniso_U_ridge: float = 1.0,
        wilson_aniso_b_center: float = 1.0,
        obs_chunk_size: int = 0,
        device: str | torch.device = "cpu",
    ) -> "LogSpaceLoss":
        """Construct from a stream.  See :meth:`SFXStream.make_loss`."""
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

        poisson_noise = PoissonNoiseModel()
        prep = prepare_observations(stream, cfg, poisson_noise=poisson_noise)
        gain_value = float(poisson_noise.gain)
        if device.type != "cpu":
            prep = prep.to(device)

        params = Parameters.from_prep(prep).to(device)

        scale_module = _resolve_scale(scale, prep, device)
        scatter = GenGaussLogScatter().to(device)
        quadrature = FixedLogGrid(
            S_min=S_min, S_max=S_max, n_points=n_points,
        ).to(device)
        likelihood = Poisson().to(device)

        data_target = DataNLLTarget(
            params=params,
            scale=scale_module,
            scatter=scatter,
            quadrature=quadrature,
            likelihood=likelihood,
            prep=prep,
            chunk_size=obs_chunk_size,
        ).to(device)
        scale_reg_target = ScaleRegTarget(params, scale_module).to(device)

        state = LossState(device=device)
        state.register_target("merging/data", data_target)
        state.set_weight("merging/data", 1.0)
        state.register_target("merging/scale_reg", scale_reg_target)
        state.set_weight("merging/scale_reg", 1.0)

        # Dataset-size-invariant scaling factor for all reg/prior terms.
        s = float(prep.n_obs_total) / float(_REG_SCALE_REF_OBS)

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

        return cls(
            stream=stream,
            prep=prep,
            params=params,
            scale=scale_module,
            scatter=scatter,
            quadrature=quadrature,
            likelihood=likelihood,
            data_target=data_target,
            scale_reg_target=scale_reg_target,
            state=state,
            device=device,
            gain=gain_value,
            gain_bias_sigma=None,
            obs_chunk_size=obs_chunk_size,
            wilson_aniso_prior=wilson_aniso_module,
        )

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

    def posterior(self, prior: str = "flat"):
        """Build a :class:`Posterior` on the loss's fixed log-S grid."""
        from torchsx.merging.posterior import Posterior
        return Posterior.from_loss(self, prior=prior)

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

    def _hessian_diag_log_I0(self) -> Tensor:
        """Hessian-diagonal σ on log_I0 from the full LBFGS objective.

        Mirrors the legacy `Merger._package_result` curvature pass: data
        target is chunked (allow_unused=True so piecewise-affine quadratures
        contribute 0), all other targets are evaluated in one shot.  Used
        for the posterior's ``mode_shift`` diagnostic.
        """
        params = self._params
        data_target = self._data_target
        H_diag = torch.zeros_like(params.log_I0)

        for s, e in data_target._iter_chunks():
            nll_chunk = data_target._chunk_loss(s, e)
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
            if name == data_target.name:
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

        return (1.0 / H_diag.clamp(min=1e-6)).sqrt()

    def __repr__(self) -> str:
        return (
            f"LogSpaceLoss(n_valid={self._prep.n_valid}, "
            f"n_obs={self._prep.n_obs_total}, "
            f"gain={self._gain:.4g}, "
            f"n_points={self._quadrature.n_points}, "
            f"device={self._device})"
        )
