"""Photon-level data NLL with kinetic forward model.

Subclasses :class:`torchsx.merging.targets.data.DataNLLTarget` so all
the chunked autograd machinery, REML path, scatter / scale / likelihood
plumbing is reused unchanged.  The only override is the source of the
per-observation log-intensity:

* Standard target:  ``log_I0 = self.params.log_I0[hkl_idx]``
* Kinetic target:   ``log_I0 = log(|Σ_s p_s(t_c) · F_s(h)|²)``

where ``F_s(h)`` is the per-HKL complex amplitude of state ``s``
(stored in :class:`KineticEnsembleParameters`) and ``p_s(t_c)`` is the
IRF-convolved population of state ``s`` at the per-observation pump
delay ``t_c`` (from :class:`KineticMixing`).

Populations caching
-------------------
The naive path would call ``KineticMixing.populations()`` once per
chunk — 3560 chunks × ~50 ms populations() eval ≈ 3 min per pass of
each aggregate.  Since populations depends only on rate constants +
IRF (not on any chunk data), we compute it **once per** ``forward()``
call and cache on ``self._pops_cache``.  Every ``_chunk_loss`` reads
the cached tensor via ``mix_with_populations``.

Autograd through the cache
--------------------------
The stock :class:`_ChunkedDataNLL` calls
``torch.autograd.grad(loss_c, live_params, retain_graph=False)`` per
chunk — the first chunk's ``retain_graph=False`` would drop the
shared populations graph and subsequent chunks would see no gradient
path to the rate constants.  This module ships its own
:class:`_KineticChunkedDataNLL` that keeps ``retain_graph=True`` for
every chunk so all chunks' contributions to the rate-constant
gradient accumulate through the same populations graph.
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.kinetics.mixing import KineticMixing
from torchsx.merging.kinetics.parameters import KineticEnsembleParameters
from torchsx.merging.observations import PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.targets.data import DataNLLTarget


class _KineticChunkedDataNLL(torch.autograd.Function):
    """Chunked autograd Function that separates chunk-level and rate-constant paths.

    Rationale
    ---------
    The naïve ``retain_graph=True`` variant kept the populations graph
    alive across all chunks in backward, which means the ODE gradient
    (matrix-exp × 134 timepoints) is re-traversed once per chunk —
    3560 traversals per aggregate on the sfx_scan run.  Fatal.

    This variant takes ``pops`` as a **separate input tensor** to the
    autograd Function.  Inside, chunks see ``pops`` as a fresh leaf
    (no graph), so each chunk's backward is cheap:

    * ``∂L_chunk/∂{F_dark, F_re, F_im, log_G, B, log_α, log_β, pi_logit}``
      — per-obs autograd, standard cost.
    * ``∂L_chunk/∂pops`` — direct partial, no ODE.

    After all chunks, this Function's backward returns
    ``∑_chunks ∂L_chunk/∂pops`` alongside the param grads.  The outer
    autograd machinery then propagates that gradient **once** through
    the populations graph → rate constants + IRF.  Result:
    populations backward runs exactly once per aggregate.
    """

    @staticmethod
    def forward(ctx, target, pops, *params):
        ctx.target = target
        ctx.save_for_backward(pops)
        # Chunks read pops (detached) via ``target._pops_cache``.
        prev_cache = target._pops_cache
        target._pops_cache = pops.detach()
        try:
            with torch.no_grad():
                total = torch.zeros((), dtype=params[0].dtype, device=params[0].device)
                for s, e in target._iter_chunks():
                    total = total + target._chunk_loss(s, e).detach()
        finally:
            target._pops_cache = prev_cache
        return total

    @staticmethod
    def backward(ctx, grad_output):
        target = ctx.target
        (pops_val,) = ctx.saved_tensors
        params = list(target._param_tuple())
        grads = [torch.zeros_like(p) for p in params]
        live_idx = [i for i, p in enumerate(params) if p.requires_grad]

        # Fresh pops leaf so autograd can compute ∂L_chunk/∂pops per
        # chunk with a cheap direct partial (no populations ODE).
        pops_leaf = pops_val.detach().clone().requires_grad_(True)
        pops_grad = torch.zeros_like(pops_val)

        live_params = [params[i] for i in live_idx]

        prev_cache = target._pops_cache
        target._pops_cache = pops_leaf
        try:
            for s, e in target._iter_chunks():
                with torch.enable_grad():
                    loss_c = target._chunk_loss(s, e)
                if not live_params:
                    partials = ()
                    pops_partial = torch.autograd.grad(
                        loss_c, pops_leaf, retain_graph=False, allow_unused=True,
                    )[0]
                else:
                    partials = torch.autograd.grad(
                        loss_c, live_params + [pops_leaf],
                        retain_graph=False, allow_unused=True,
                    )
                    pops_partial = partials[-1]
                    for slot, p in zip(live_idx, partials[:-1]):
                        if p is not None:
                            grads[slot].add_(p)
                if pops_partial is not None:
                    pops_grad.add_(pops_partial)
        finally:
            target._pops_cache = prev_cache

        # Return grad_target=None, grad_pops (chains via outer graph to
        # rate constants + IRF), then per-param grads.
        scaled_params = tuple(grad_output * g for g in grads)
        return (None, grad_output * pops_grad) + scaled_params


class KineticDataNLLTarget(DataNLLTarget):
    """Photon-level data NLL with a kinetic-ensemble forward model.

    ``params`` is still the standard :class:`Parameters` instance — its
    ``log_I0`` is frozen (``requires_grad=False`` in
    :meth:`KineticLogSpaceLoss.from_stream`) but its ``log_alpha`` /
    ``log_beta`` (the GenGauss scatter shape) and ``pi_logit`` continue
    to drive ``ScatterModel`` and the marginalization grid.

    The kinetic forward model contributes:

    * ``kinetic_params``  — per-HKL ``log_F_dark``, ``F_re``, ``F_im``
    * ``kinetic_mixing``  — population ODE + IRF (rate constants live
      inside ``kinetic_mixing.kinetic`` as nn.Parameter tensors)
    """

    name: str = "merging/data_kinetic"

    def __init__(
        self,
        params: Parameters,
        scale,
        scatter,
        quadrature,
        likelihood,
        prep: PreparedObs,
        *,
        kinetic_params: KineticEnsembleParameters,
        kinetic_mixing: KineticMixing,
        chunk_size: int | None = 1_000_000,
        reml: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            scale=scale,
            scatter=scatter,
            quadrature=quadrature,
            likelihood=likelihood,
            prep=prep,
            chunk_size=chunk_size,
            reml=reml,
        )
        self.add_module("kinetic_params", kinetic_params)
        self.add_module("kinetic_mixing", kinetic_mixing)

        oi = prep.obs_info
        if oi.delay_bin_idx is None:
            raise ValueError(
                "KineticDataNLLTarget requires a timepoint-labelled stream — "
                "ObsInfo.delay_bin_idx is None.  Call stream.assign_timepoints()"
                " or load timepoint-labelled streams first."
            )
        self.register_buffer("_oi_delay_bin_idx", oi.delay_bin_idx)
        if (oi.delay_bin_idx < 0).any():
            n_bad = int((oi.delay_bin_idx < 0).sum())
            raise ValueError(
                f"{n_bad} crystals have an unlabelled timepoint "
                f"(delay_bin_idx == -1) — drop them from the stream before "
                "building the kinetic loss."
            )

        # Set by forward() at the start of each aggregate call.  Cleared
        # once forward returns.  Every _chunk_loss during that aggregate
        # reads from this cache.
        self._pops_cache: Tensor | None = None

    # ── Per-obs forward through the cached populations ────────────────

    def _kinetic_log_I0_per_obs(self, s: int, e: int, h: Tensor) -> Tensor:
        """``log(|F_total(h, t_c)|²)`` for the observation slice ``[s, e)``."""
        if self._pops_cache is None:
            # Fallback path: no ambient populations cache — compute per call.
            # Reachable from the eager (non-chunked) branch of
            # DataNLLTarget.forward, or when someone calls _chunk_loss
            # directly.  Correct but slow if hit inside a chunked loop.
            F_re, F_im = self.kinetic_mixing.mix(
                self.kinetic_params,
                hkl_idx=h,
                crystal_id=self._oi_crystal_id[s:e],
                crystal_delay_bin=self._oi_delay_bin_idx,
            )
        else:
            F_re, F_im = self.kinetic_mixing.mix_with_populations(
                self._pops_cache,
                self.kinetic_params,
                hkl_idx=h,
                crystal_id=self._oi_crystal_id[s:e],
                crystal_delay_bin=self._oi_delay_bin_idx,
            )
        I_pred = F_re * F_re + F_im * F_im
        return torch.log(I_pred.clamp(min=1e-30))

    # ── Overrides ─────────────────────────────────────────────────────

    def _chunk_loss(self, s: int, e: int) -> Tensor:
        meta = self._obs_info_slice(s, e)
        h = self.hkl_id[s:e]

        log_I0 = self._kinetic_log_I0_per_obs(s, e, h)

        log_k = self.scale.log_k(self.params, meta, obs_range=(s, e))
        log_mean = log_I0 + log_k
        if self.has_partiality:
            log_mean = log_mean + torch.log(self.partiality[s:e].clamp(min=1e-12))

        n_ph_slice = self.n_ph[s:e]
        bg_slice = self.bg[s:e]

        log_p_sig = self.quadrature.marginalize_signal(
            self.scatter, self.likelihood,
            log_mean, n_ph_slice, bg_slice,
            self.params, meta, obs_range=(s, e),
        )
        log_p = self.likelihood.combine_with_marginal(
            n_ph_slice, bg_slice, log_p_sig, self.params, meta,
        )
        if self.has_obs_weights:
            return -(log_p * self.obs_weights[s:e]).sum()
        return -log_p.sum()

    def _chunk_log_p_and_aux(
        self, s: int, e: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        meta = self._obs_info_slice(s, e)
        h = self.hkl_id[s:e]

        log_I0 = self._kinetic_log_I0_per_obs(s, e, h)

        log_k = self.scale.log_k(self.params, meta, obs_range=(s, e))
        log_mean = log_I0 + log_k
        if self.has_partiality:
            log_mean = log_mean + torch.log(self.partiality[s:e].clamp(min=1e-12))

        n_ph_slice = self.n_ph[s:e]
        bg_slice = self.bg[s:e]

        log_p, A, B, C = self.quadrature.marginalize_signal_with_reml_aux(
            self.scatter, self.likelihood,
            log_mean, n_ph_slice, bg_slice,
            self.params, meta, obs_range=(s, e),
        )
        return log_p, A, B, C, meta.hkl_idx

    def _param_tuple(self) -> tuple[Tensor, ...]:
        """Params that receive direct per-chunk gradients.

        Rate constants + IRF (``kinetic_mixing.parameters()``) are
        deliberately excluded — they get gradients through the shared
        populations graph via the ``pops`` input to
        :class:`_KineticChunkedDataNLL`, not via each chunk's autograd.
        """
        base = list(super()._param_tuple())
        base += list(self.kinetic_params.parameters())
        return tuple(base)

    # ── Target.forward — precompute populations, dispatch ─────────────

    def forward(self) -> Tensor:
        """Precompute populations once, then run chunked NLL.

        Populations depend only on rate constants + IRF, not on any
        chunk data.  Computing them once per aggregate (rather than
        once per chunk) reduces populations() calls from ``N_chunks``
        to 1 per aggregate — 100–1000× speedup on the sfx_scan run
        with ``obs_chunk_size=50000`` and 178 M observations.
        """
        if self.reml:
            # REML path: the standard chunked REML machinery
            # ``_ChunkedDataNLL_REML`` also indexes ``params.log_I0``,
            # which is frozen in the kinetic loss.  Fall through to the
            # parent's forward() with pops cached; the parent will
            # dispatch REML, and each chunk still reads the cache.
            self._pops_cache = self.kinetic_mixing.populations()
            try:
                return super().forward()
            finally:
                self._pops_cache = None

        # Non-REML: eager if unchunked, else our own kinetic chunked FF.
        if (self.chunk_size is None
                or self.chunk_size <= 0
                or self.chunk_size >= self.n_obs_total):
            # Eager (full-batch) — populations computed inside
            # _kinetic_log_I0_per_obs' fallback; single call, no chunks.
            return self._chunk_loss(0, self.n_obs_total)

        # Chunked: compute populations with grad (single node in the
        # outer autograd graph feeding rate constants + IRF).  Pass it
        # explicitly to the chunked Function so its backward can chain
        # ``∑_chunks ∂L/∂pops`` back through populations exactly once.
        pops = self.kinetic_mixing.populations()
        params = self._param_tuple()
        return _KineticChunkedDataNLL.apply(self, pops, *params)
