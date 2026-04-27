"""Data-likelihood target — the full marginal Poisson loss.

Composes ``ScaleModel × ScatterModel × Quadrature × DetectionLikelihood``
into a scalar ``−Σ log p(n | …)`` over all observations.

Chunking is hidden behind ``_ChunkedDataNLL``, a custom
``torch.autograd.Function`` that runs the forward chunk-by-chunk under
``no_grad`` (constant memory) and re-computes each chunk with grad in
the backward pass, accumulating partial grads onto each ``nn.Parameter``.
This is gradient checkpointing on the observation axis, scoped to this
single target so ``LossState`` stays the stock torchref class.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor

from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.scale.base import ScaleModel
from torchsx.merging.scatter.base import ScatterModel
from torchsx.merging.targets.base import MergingTarget


class _ChunkedDataNLL(torch.autograd.Function):
    """Chunked forward + backward with gradient accumulation.

    Forward runs each chunk under ``no_grad`` (saves ~K× memory on the
    quadrature grid tensor).  Backward re-runs each chunk with grad
    enabled, computes partial grads via ``torch.autograd.grad``, and
    accumulates them onto the parameter tensors.
    """

    @staticmethod
    def forward(ctx, target, *params):
        ctx.target = target
        with torch.no_grad():
            total = torch.zeros((), dtype=params[0].dtype, device=params[0].device)
            for s, e in target._iter_chunks():
                total = total + target._chunk_loss(s, e).detach()
        return total

    @staticmethod
    def backward(ctx, grad_output):
        target = ctx.target
        params = list(target._param_tuple())
        grads = [torch.zeros_like(p) for p in params]
        for s, e in target._iter_chunks():
            with torch.enable_grad():
                loss_c = target._chunk_loss(s, e)
            partials = torch.autograd.grad(
                loss_c, params, retain_graph=False, allow_unused=True,
            )
            for i, p in enumerate(partials):
                if p is not None:
                    grads[i].add_(p)
        scaled = tuple(grad_output * g for g in grads)
        # None for the non-tensor ``target`` argument.
        return (None,) + scaled


class DataNLLTarget(MergingTarget):
    """``-Σ log p(n | scale × scatter × likelihood × quadrature)`` over obs."""

    name: str = "merging/data"

    def __init__(
        self,
        params: Parameters,
        scale: ScaleModel,
        scatter: ScatterModel,
        quadrature: Quadrature,
        likelihood: DetectionLikelihood,
        prep: PreparedObs,
        *,
        chunk_size: int | None = 1_000_000,
    ) -> None:
        super().__init__()
        self.params = params
        self.scale = scale
        self.scatter = scatter
        self.quadrature = quadrature
        self.likelihood = likelihood

        # Per-observation buffers.  We keep I_obs separate from n_ph so
        # future likelihoods (e.g. Gaussian) can use I_obs rather than
        # peak counts.
        self.register_buffer("I_obs", prep.I_obs)
        self.register_buffer("bg", prep.bg)
        self.register_buffer("n_ph", (prep.I_obs + prep.bg).clamp(min=0))
        self.register_buffer("hkl_id", prep.hkl_id)

        # ObsInfo fields — registered as buffers so .to(device) carries them.
        # Optional fields are registered only when present; ``_slice_opt``
        # uses ``getattr(self, name, None)`` so missing attributes read as None.
        oi = prep.obs_info
        self.register_buffer("_oi_crystal_id", oi.crystal_id)
        self.register_buffer("_oi_s_sq", oi.s_sq)
        self.register_buffer("_oi_hkl_idx", oi.hkl_idx)
        self._oi_n_crystals = oi.n_crystals
        for name in ("panel_id", "fs", "ss", "peak", "background"):
            t = getattr(oi, name)
            if t is not None:
                self.register_buffer(f"_oi_{name}", t)

        # Optional external partiality (sorted).
        if prep.partiality is not None:
            self.register_buffer("partiality", prep.partiality)
            self.has_partiality = True
        else:
            self.register_buffer("partiality", torch.empty(0))
            self.has_partiality = False

        self.n_obs_total = int(prep.n_obs_total)
        self.chunk_size = chunk_size

    # ── Internals ──────────────────────────────────────────────────

    def _slice_opt(self, attr: str, s: int, e: int) -> Tensor | None:
        t = getattr(self, attr, None)
        return None if t is None else t[s:e]

    def _obs_info_slice(self, s: int, e: int) -> ObsInfo:
        return ObsInfo(
            crystal_id=self._oi_crystal_id[s:e],
            s_sq=self._oi_s_sq[s:e],
            hkl_idx=self._oi_hkl_idx[s:e],
            n_crystals=self._oi_n_crystals,
            panel_id=self._slice_opt("_oi_panel_id", s, e),
            fs=self._slice_opt("_oi_fs", s, e),
            ss=self._slice_opt("_oi_ss", s, e),
            peak=self._slice_opt("_oi_peak", s, e),
            background=self._slice_opt("_oi_background", s, e),
        )

    def _iter_chunks(self) -> Iterator[tuple[int, int]]:
        n = self.n_obs_total
        if self.chunk_size is None or self.chunk_size <= 0 or self.chunk_size >= n:
            yield (0, n)
            return
        for s in range(0, n, self.chunk_size):
            yield (s, min(s + self.chunk_size, n))

    def _param_tuple(self) -> tuple[Tensor, ...]:
        """All learnable tensors touched by the data likelihood.

        Includes ``params`` (log_I0, log_alpha, log_beta, pi_logit) and
        whatever ``scale`` / ``scatter`` / ``likelihood`` expose as
        ``nn.Parameter``s.  The order is stable per Python iteration
        order, which is what the custom autograd.Function relies on.
        """
        return tuple(
            list(self.params.parameters())
            + list(self.scale.parameters())
            + list(self.scatter.parameters())
            + list(self.likelihood.parameters())
        )

    def _chunk_loss(self, s: int, e: int) -> Tensor:
        """``-Σ log p(n_j | ...)`` summed over obs[s:e], all params live."""
        meta = self._obs_info_slice(s, e)
        h = self.hkl_id[s:e]
        log_I0 = self.params.log_I0[h]

        log_k = self.scale.log_k(self.params, meta, obs_range=(s, e))
        log_mean = log_I0 + log_k
        if self.has_partiality:
            log_mean = log_mean + torch.log(self.partiality[s:e].clamp(min=1e-12))

        n_ph_slice = self.n_ph[s:e]
        bg_slice = self.bg[s:e]

        log_p_sig = self.quadrature.marginalize_signal(
            self.scatter,
            self.likelihood,
            log_mean,
            n_ph_slice,
            bg_slice,
            self.params,
            meta,
            obs_range=(s, e),
        )
        log_p = self.likelihood.combine_with_marginal(
            n_ph_slice, bg_slice, log_p_sig, self.params, meta
        )
        return -log_p.sum()

    # ── Target.forward ─────────────────────────────────────────────

    def forward(self) -> Tensor:
        if self.chunk_size is None or self.chunk_size <= 0 or self.chunk_size >= self.n_obs_total:
            # Full-batch: no need for the custom autograd.Function — one
            # forward, one graph, one backward.
            return self._chunk_loss(0, self.n_obs_total)
        params = self._param_tuple()
        return _ChunkedDataNLL.apply(self, *params)
