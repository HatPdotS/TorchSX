"""Kinetic-aware photon-level merging loss.

Subclasses :class:`torchsx.merging.loss.LogSpaceLoss` and swaps the
standard :class:`DataNLLTarget` for the kinetic
:class:`KineticDataNLLTarget`.  ``params.log_I0`` is frozen
(``requires_grad=False``) — the per-observation forward mean comes
from the kinetic model now, and refining ``log_I0`` on top would
introduce a degenerate scale axis between ``log_I0`` and
``F_dark``.  The kinetic ``log_F_dark`` parameter takes over.

Public surface
--------------

* :meth:`KineticLogSpaceLoss.from_stream` — constructor mirroring
  :meth:`LogSpaceLoss.from_stream` with the extra ``kinetic_params``
  / ``kinetic_mixing`` / ``phase_restraint_lambda`` keyword arguments.
* :attr:`KineticLogSpaceLoss.kinetic_params` — handle on the per-state
  amplitude tensors (read for output writers).
* :attr:`KineticLogSpaceLoss.kinetic_mixing` — handle on the kinetic
  model (rate constants, IRF, ``populations()`` for predicted-bin MTZ
  writers).

The rest of the API (``refine``, ``stats``, ``history``,
``final_nll``, ``params``) is inherited unchanged.

``posterior()`` is *not* exposed here — the standard log-I0 posterior
is meaningless when ``log_I0`` is frozen and the per-obs mean comes
from ``|F_total|²``.  Output is via
:mod:`torchsx.merging.kinetics.output` instead (per-state + predicted
bin MTZs derived analytically from the fitted parameters).
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from torchsx.merging.kinetics.data_target import KineticDataNLLTarget
from torchsx.merging.kinetics.mixing import KineticMixing
from torchsx.merging.kinetics.parameters import KineticEnsembleParameters
from torchsx.merging.kinetics.phase_restraint import PhaseRestraint
from torchsx.merging.kinetics.regularizers import ExcitedAmplitudeAnchor
from torchsx.merging.loss import (
    LogSpaceLoss,
    _resolve_scale,
    _resolve_scatter_shape,
)


class KineticLogSpaceLoss(LogSpaceLoss):
    """Photon-level kinetic merge loss.

    Constructed via :meth:`from_stream`.  Holds the same standard
    machinery as :class:`LogSpaceLoss` plus the kinetic forward model;
    ``refine()`` drives LBFGS over the union of standard and kinetic
    learnable tensors.
    """

    def __init__(
        self,
        *,
        kinetic_params: KineticEnsembleParameters,
        kinetic_mixing: KineticMixing,
        phase_restraint: PhaseRestraint | None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._kinetic_params = kinetic_params
        self._kinetic_mixing = kinetic_mixing
        self._phase_restraint = phase_restraint

    @classmethod
    def from_stream(
        cls,
        stream,
        *,
        kinetic_params: KineticEnsembleParameters,
        kinetic_mixing: KineticMixing,
        phase_restraint_lambda: float = 0.0,
        f_anchor_lambda: float = 0.0,
        freeze_phase: bool = False,
        freeze_log_F_dark: bool = False,
        # ── Standard LogSpaceLoss knobs (same defaults) ──────────────
        n_points: int = 513,
        S_min: float = 1e-3,
        S_max: float = 1e6,
        gain: float | str = "auto",
        min_obs: int = 5,
        scale: str = "b_per_crystal",
        scale_reg_weight: float = 1.0,
        scatter_shape: str = "free",
        smooth: float = 500.0,
        wilson_aniso: float = 0.0,
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
    ) -> "KineticLogSpaceLoss":
        """Build a kinetic-aware photon-level loss.

        ``kinetic_params`` and ``kinetic_mixing`` should already be on
        ``device`` (typically warm-started via the amplitude-level
        :func:`fit_initial_kinetic_ensemble`).  Wilson_aniso defaults
        to 0 — the kinetic model already breaks the LBFGS scale
        degeneracy via the shared rate constants, so the Wilson prior
        isn't needed to stabilise per-bin scales.
        """
        # Re-use the standard ``LogSpaceLoss.from_stream`` to build prep
        # / params / scale / scatter / quadrature / likelihood and the
        # baseline LossState (regularizers, scale reg, etc.).  Then swap
        # the data target for the kinetic variant in-place.
        base = LogSpaceLoss.from_stream(
            stream,
            n_points=n_points, S_min=S_min, S_max=S_max,
            gain=gain, min_obs=min_obs,
            scale=scale, scale_reg_weight=scale_reg_weight,
            scatter_shape=scatter_shape,
            smooth=smooth,
            wilson_aniso=wilson_aniso,
            wilson_aniso_U_ridge=wilson_aniso_U_ridge,
            wilson_aniso_b_center=wilson_aniso_b_center,
            poisson_null_shrinkage=poisson_null_shrinkage,
            poisson_null_tau=poisson_null_tau,
            reml=reml,
            obs_chunk_size=obs_chunk_size,
            device=device,
            r_peak=r_peak, r_gap=r_gap, r_bg=r_bg,
            likelihood=likelihood,
            var_const=var_const,
            var_linear=var_linear,
            var_quadratic=var_quadratic,
            trainable_polynomial_var=trainable_polynomial_var,
            crystal_weights=crystal_weights,
        )

        device_t = torch.device(device)
        kinetic_params = kinetic_params.to(device_t)
        kinetic_mixing = kinetic_mixing.to(device_t)

        # Optional phase freeze — keep F_im at its initialised value
        # (usually 0).  With ``freeze_phase=True`` the phase-restraint
        # target and Adam both skip F_im; only F_re and log_F_dark
        # move per HKL.  Useful when the photocycle differential is
        # small and dominated by amplitude change, not phase rotation.
        if freeze_phase and kinetic_params.n_excited > 0:
            kinetic_params.F_im.requires_grad_(False)
            with torch.no_grad():
                kinetic_params.F_im.data.zero_()

        # Optional F_dark freeze — pin ``log_F_dark`` at its warm-start
        # value (typically Stage-0's dark seed).  With
        # ``activation_level`` small, most crystals are photoinactive
        # and F_A ≡ F_dark by construction; no photon evidence can
        # discriminate F_A from F_dark, so leaving it trainable just
        # opens a shallow "shrink everything" direction that pulls
        # F_re, F_im down too.  Freezing removes that direction.
        if freeze_log_F_dark:
            kinetic_params.log_F_dark.requires_grad_(False)

        # ── Sanity: HKL alignment between standard prep and kinetic params
        if kinetic_params.n_hkl != base._prep.n_valid:
            raise ValueError(
                "kinetic_params.n_hkl "
                f"({kinetic_params.n_hkl}) != prep.n_valid "
                f"({base._prep.n_valid}).  Build kinetic_params on the "
                "same HKL grid as the loss (use the same stream / pre-fit)."
            )

        # ── Freeze log_I0 — kinetic forward model provides per-obs mean
        # The kinetic ``log_F_dark`` is the per-HKL absolute scale knob;
        # leaving the standard ``log_I0`` trainable on top creates a
        # degenerate scale axis.  Freezing both keeps gradients flowing
        # only to (log_F_dark, F_re, F_im, log_G, B, rate constants).
        base._params.log_I0.requires_grad_(False)

        # ── Swap the data target ────────────────────────────────────
        kinetic_data_target = KineticDataNLLTarget(
            params=base._params,
            scale=base._scale,
            scatter=base._scatter,
            quadrature=base._quadrature,
            likelihood=base._likelihood,
            prep=base._prep,
            kinetic_params=kinetic_params,
            kinetic_mixing=kinetic_mixing,
            chunk_size=obs_chunk_size,
            reml=reml,
        ).to(device_t)

        # The standard LogSpaceLoss registered ``merging/data`` →
        # DataNLLTarget; drop it and register the kinetic variant.
        # LossState has no public ``unregister`` — pop the entry from the
        # state's `targets` dict and invalidate any compiled-aggregate
        # closure so the next ``aggregate()`` rebuilds.
        base._state.targets.pop("merging/data", None)
        base._state._compiled_aggregate = None
        base._state.register_target(
            kinetic_data_target.name, kinetic_data_target,
        )
        base._state.set_weight(kinetic_data_target.name, 1.0)
        base._data_target = kinetic_data_target

        # ── F-anchor: |F_k| ≈ F_dark prior ─────────────────────────
        if f_anchor_lambda > 0:
            f_anchor = ExcitedAmplitudeAnchor(lam=f_anchor_lambda).to(device_t)
            from torchref.refinement.targets import Target

            class _FAnchorTarget(Target):
                name = "merging/kinetic_f_anchor"

                def __init__(self, kinetic_params, anchor):
                    super().__init__()
                    self.kp = kinetic_params
                    self.a = anchor

                def forward(self):
                    return self.a(self.kp)

            fat = _FAnchorTarget(kinetic_params, f_anchor).to(device_t)
            base._state.register_target(fat.name, fat)
            base._state.set_weight(fat.name, 1.0)

        # ── Phase restraint ─────────────────────────────────────────
        phase_restraint: PhaseRestraint | None = None
        if phase_restraint_lambda > 0 and not freeze_phase:
            phase_restraint = PhaseRestraint(lam=phase_restraint_lambda).to(device_t)
            # Wrap as a RegularizerTarget-style scalar target so it
            # participates in the aggregate.  Inline a tiny target — the
            # generic RegularizerTarget needs a ``Parameters`` instance
            # and ``params.X`` lookup, which doesn't fit our F_re/F_im
            # layout.
            from torchref.refinement.targets import Target

            class _PhaseRestraintTarget(Target):
                name = "merging/kinetic_phase_restraint"

                def __init__(self, kinetic_params, restraint):
                    super().__init__()
                    self.kp = kinetic_params
                    self.r = restraint

                def forward(self):
                    return self.r(self.kp)

            prt = _PhaseRestraintTarget(kinetic_params, phase_restraint).to(device_t)
            base._state.register_target(prt.name, prt)
            base._state.set_weight(prt.name, 1.0)

        # ── Repackage into the kinetic subclass ─────────────────────
        return cls(
            stream=base._stream,
            prep=base._prep,
            params=base._params,
            scale=base._scale,
            scatter=base._scatter,
            quadrature=base._quadrature,
            likelihood=base._likelihood,
            data_target=kinetic_data_target,
            scale_reg_target=base._scale_reg_target,
            state=base._state,
            device=base._device,
            gain=base._gain,
            gain_bias_sigma=base._gain_bias_sigma,
            obs_chunk_size=base._obs_chunk_size,
            wilson_aniso_prior=base._wilson_aniso_prior,
            kinetic_params=kinetic_params,
            kinetic_mixing=kinetic_mixing,
            phase_restraint=phase_restraint,
        )

    # ── Accessors used by the output writers ─────────────────────────

    @property
    def kinetic_params(self) -> KineticEnsembleParameters:
        return self._kinetic_params

    @property
    def kinetic_mixing(self) -> KineticMixing:
        return self._kinetic_mixing

    @property
    def phase_restraint(self) -> PhaseRestraint | None:
        return self._phase_restraint

    # ── Posterior is intentionally not supported here ───────────────

    def posterior(self, *args, **kwargs):  # noqa: D401 — short reason
        raise NotImplementedError(
            "Use torchsx.merging.kinetics.output.write_per_state_mtzs() and "
            "write_predicted_bin_mtzs() instead — the standard log_I0 "
            "posterior is meaningless when log_I0 is frozen and the per-obs "
            "mean comes from |F_total|²."
        )

    # ── Adam refine with per-step logging + centric F_im maintenance ──

    def refine(
        self,
        *,
        max_iter: int = 300,
        lr: float = 1e-2,
        warmup_steps: int = 20,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        log_every: int = 10,
        tolerance_change: float = 0.0,
        **_ignored,
    ) -> "KineticLogSpaceLoss":
        """Adam refine with linear warmup.  Replaces the parent's LBFGS.

        Adam is the natural optimizer for this problem: per-parameter
        adaptive step from ``m/v`` moments handles the wildly
        different scales of ``log_F_dark`` (per-HKL, ~1),
        ``F_re/F_im`` (per-HKL, ~0), ``log_G`` (per-crystal, ~0),
        ``B`` (per-crystal, ~0), rate constants (~-6 to -10 in log),
        and ``log_irf`` (~-8) uniformly, without a line search that
        strong-wolfe LBFGS chokes on at the F_re=F_im=0 saddle.

        Linear warmup ramps ``lr`` from 0 → target over
        ``warmup_steps`` epochs to keep Adam from overshooting on the
        first few steps (Adam's ``m/v`` bias correction is under-damped
        at step 1, and the huge initial gradient at F_re=F_im=0 makes
        a full-lr step wildly diverge otherwise).

        Parameters
        ----------
        max_iter : Adam epochs (each = one full aggregate + backward + step).
        lr : Peak learning rate after warmup.
        warmup_steps : Linear ramp from 0 to ``lr`` over this many
            epochs.  After warmup, lr is held constant.  Set to 0 to
            disable warmup.
        log_every : Print loss + τ evolution every N steps.
        tolerance_change : Early-stop if loss decrease over the last 10
             steps falls below this fraction of the initial loss.
             Default 0 → run all ``max_iter`` steps.
        """
        import time

        params_all = self._collect_parameters()
        opt = torch.optim.Adam(
            params_all, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        # Linear warmup: lr multiplier goes 0 → 1 over warmup_steps.
        # After warmup, held at 1.0.  scheduler.step() runs AFTER
        # opt.step(), so on Adam step k (1-indexed) the effective lr is
        # lr · min(k / max(1, warmup_steps), 1.0).
        def _warmup_lambda(step_zero_indexed: int) -> float:
            if warmup_steps <= 0:
                return 1.0
            step_one_indexed = step_zero_indexed + 1
            return min(step_one_indexed / warmup_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _warmup_lambda)

        t0 = time.time()

        # Initial pass — no step yet; report starting point.
        opt.zero_grad(set_to_none=False)
        initial_total = self._state.aggregate()
        initial_total.backward()
        self._kinetic_params.zero_centric_imag()
        initial_max_grad = max(
            (p.grad.detach().abs().max().item()
             for p in params_all if p.grad is not None),
            default=0.0,
        )
        initial_loss = float(initial_total.detach())
        print(
            f"[kinetic-refine] optimizer=Adam  lr={lr}  warmup_steps={warmup_steps}  "
            f"max_iter={max_iter}  n_params={len(params_all)}  "
            f"n_trainable_scalars={sum(p.numel() for p in params_all)}",
            flush=True,
        )
        print(
            f"[kinetic-refine] initial loss={initial_loss:.4e}  "
            f"initial max_grad={initial_max_grad:.2e}",
            flush=True,
        )

        loss_history: list[float] = [initial_loss]
        max_grad_history: list[float] = [initial_max_grad]
        best_loss = initial_loss
        final_loss = initial_loss
        final_max_grad = initial_max_grad
        actual_steps = 0

        for step in range(1, max_iter + 1):
            opt.step()
            scheduler.step()
            # Enforce centric F_im = 0 after each step.
            self._kinetic_params.zero_centric_imag()

            opt.zero_grad(set_to_none=False)
            total = self._state.aggregate()
            total.backward()
            max_grad = max(
                (p.grad.detach().abs().max().item()
                 for p in params_all if p.grad is not None),
                default=0.0,
            )
            loss_value = float(total.detach())
            loss_history.append(loss_value)
            max_grad_history.append(max_grad)
            final_loss = loss_value
            final_max_grad = max_grad
            actual_steps = step
            if loss_value < best_loss:
                best_loss = loss_value

            if log_every > 0 and (step % log_every == 0 or step == 1):
                rate_constants = self._kinetic_mixing.kinetic.get_rate_constants()
                tau_summary = ", ".join(
                    f"τ_{k}={1.0 / max(v, 1e-30):.2e}ps"
                    for k, v in rate_constants.items()
                )
                cur_lr = opt.param_groups[0]["lr"]
                dt = time.time() - t0
                print(
                    f"  [kinetic-refine] step {step:4d}  "
                    f"loss={loss_value:.4e}  "
                    f"Δ={loss_value - initial_loss:+.3e}  "
                    f"max_grad={max_grad:.2e}  "
                    f"lr={cur_lr:.2e}  "
                    f"[{tau_summary}]  "
                    f"elapsed={dt:6.1f}s",
                    flush=True,
                )

            # Optional early stop: loss stopped changing.
            if tolerance_change > 0 and step >= 20:
                recent = loss_history[-11:]
                rel_change = (max(recent) - min(recent)) / max(1.0, abs(initial_loss))
                if rel_change < tolerance_change:
                    print(
                        f"  [kinetic-refine] early stop at step {step}: "
                        f"loss stagnant (rel_change={rel_change:.2e} < "
                        f"tolerance_change={tolerance_change})",
                        flush=True,
                    )
                    break

        # Final housekeeping — log per-target losses.
        with torch.no_grad():
            self._state.aggregate(log_values=True)
        final_data = self._state.get_loss(self._data_target.name)
        self._final_nll = float(final_data.detach().item()) if final_data is not None else None

        dt = time.time() - t0
        print(
            f"[kinetic-refine] done.  steps={actual_steps}  "
            f"initial_loss={initial_loss:.4e}  "
            f"final_loss={final_loss:.4e}  "
            f"best_loss={best_loss:.4e}  "
            f"final_data_nll={self._final_nll}  "
            f"final max_grad={final_max_grad:.2e}  "
            f"elapsed={dt:.1f}s",
            flush=True,
        )

        self._lbfgs_history.append({
            "optimizer": "adam",
            "steps": actual_steps,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "initial_max_grad": initial_max_grad,
            "final_max_grad": final_max_grad,
            "final_data_nll": self._final_nll,
            "elapsed_s": dt,
        })
        return self
