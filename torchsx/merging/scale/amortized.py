"""Amortised scale model: one network is the only source of scale.

One forward pass produces each observation's ``log K`` and its scatter
parameters, which :meth:`AmortizedScale.prime` installs into the predicted
scatter the quadrature integrates over.

Usage::

    scale = AmortizedScale.from_prep(prep)
    pred  = scale.prime(meta)        # one forward: pred.log_K, scatter installed
    ...                              # integrate over scale.scatter
    scale.clear()
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.scatter.mixture import PredictedMixtureScatter
from torchsx.nn.fold_transformer import FoldTransformerEncoder, scatter_positions
from torchsx.nn.heads import ScaleHeads, ScalePrediction
from torchsx.nn.obs_features import ObsFeatureSpec

__all__ = ["AmortizedScale"]


class AmortizedScale(nn.Module):
    """Network-predicted ``log K`` (and scatter) per observation.

    Parameters
    ----------
    features
        Tiered feature builder; enforces the own-intensity and
        condition-blindness contracts.
    encoder
        :class:`~torchsx.nn.fold_transformer.FoldTransformerEncoder`.
    heads
        :class:`~torchsx.nn.heads.ScaleHeads`.
    n_crystals
        Number of crystals, the encoder's group count.
    anchor_log_k
        Pin the gauge by subtracting a batch location statistic from
        ``log K``.  ``K → cK, I0 → I0/c`` leaves the likelihood exactly
        invariant, so without this the scale drifts freely.  On by default,
        from step 0.
    anchor_stat
        ``"median"`` (default) or ``"mean"``.  The gauge is arbitrary, so any
        location statistic is valid — but the mean is **not robust**, and that
        matters here: a few percent of observations can run away to very large
        ``log K`` while the bulk stays put, and under a mean anchor they shift
        the mean and drag every other observation with it.  A small
        runaway subset thereby corrupts the scale of every other observation.
        The median cannot be hijacked that way.  ``"basis"`` subtracts the
        least-squares projection of ``log K`` onto a smooth reciprocal-space
        basis (see :meth:`set_gauge_basis`): the scalar gauge and every smooth
        resolution-dependent common mode go at once, so the merged intensities
        keep the average crystal's fall-off instead of an arbitrary one.
    """

    def __init__(
        self,
        features: ObsFeatureSpec,
        encoder: FoldTransformerEncoder,
        heads: ScaleHeads,
        n_crystals: int,
        *,
        anchor_log_k: bool = True,
        anchor_stat: str = "median",
    ) -> None:
        super().__init__()
        self.features = features
        self.encoder = encoder
        self.heads = heads
        self.n_crystals = int(n_crystals)
        # Basis gauge pin (anchor_stat == "basis"): per-reflection rows of a
        # smooth reciprocal-space basis, installed by set_gauge_basis().  The
        # ensemble-mean component of log K in its span is projected out of every
        # prediction -- the rank-p generalisation of the scalar anchor, removing
        # the K(h) -> K(h) g(h), I0 -> I0 / g(h) freedom for every smooth g.
        self.register_buffer("gauge_phi", torch.zeros(0, 0), persistent=False)
        #: in training the blocks' outputs enter the graph as leaves and
        #: :meth:`encoder_backward` back-propagates them through the blocks
        #: after the loss's backward pass (:meth:`predict`)
        self.defer_encoder_backward = False
        self._deferred = None
        self.anchor_log_k = bool(anchor_log_k)
        if anchor_stat not in ("median", "mean", "basis"):
            raise ValueError(
                f"anchor_stat must be 'median', 'mean' or 'basis', got {anchor_stat!r}"
            )
        self.anchor_stat = anchor_stat
        self.scatter = PredictedMixtureScatter(heads.n_components, fixed_gaussian=True)

    # ── construction ───────────────────────────────────────────────────

    @classmethod
    def from_prep(
        cls,
        prep: PreparedObs,
        *,
        d_model: int = 128,
        d_hidden: int | None = None,
        context_folds: int = 5,
        ctx_layers: int = 2,
        query_layers: int = 1,
        n_heads: int = 8,
        refine_geometry: bool = False,
        w_init: float = 1.65,
        n_components: int = 4,
        anchor_log_k: bool = True,
        anchor_stat: str = "median",
        self_features: tuple[str, ...] | None = None,
        context_features: tuple[str, ...] | None = None,
        crystal_features: tuple[str, ...] | None = None,
        phase_classes=None,
    ) -> "AmortizedScale":
        """The fold transformer
        (:class:`~torchsx.nn.fold_transformer.FoldTransformerEncoder`: ``context_folds``
        runs per crystal, ``ctx_layers`` / ``query_layers`` blocks of ``n_heads``
        heads, ``refine_geometry``) and the heads, on ``prep``'s features.
        ``phase_classes``
        (:class:`~torchsx.merging.scale.pseudo_translation.PhaseClasses`) adds the
        reflections' phase cosines to the self features."""
        # None keeps each tier's default; a checkpoint passes its recorded tiers
        features = ObsFeatureSpec(
            prep, context_features=context_features,
            crystal_features=crystal_features, phase_classes=phase_classes,
            **({} if self_features is None else {"self_features": tuple(self_features)}),
        )
        # the measurement: what a visible token carries beyond the geometry
        geometry = set(features.self_features)
        meas = [i for i, f in enumerate(features.context_features) if f not in geometry]
        enc = FoldTransformerEncoder(
            features.n_self, features.n_context, meas,
            d_model=d_model, n_heads=n_heads, folds=context_folds,
            ctx_layers=ctx_layers, query_layers=query_layers,
            refine_geometry=refine_geometry,
            geometry=features.geometry_columns() if refine_geometry else None,
            d_hidden=d_hidden,
            n_crystal_features=features.n_crystal,
        )
        heads = ScaleHeads(d_model, w_init=w_init, n_components=n_components)
        return cls(
            features, enc, heads,
            n_crystals=prep.obs_info.n_crystals,
            anchor_log_k=anchor_log_k,
            anchor_stat=anchor_stat,
        )

    # ── forward ────────────────────────────────────────────────────────

    def _prediction_names(self) -> list[str]:
        """The :class:`ScalePrediction` fields the heads fill."""
        return ["log_K", "log_w", "log_e_s", "mix"]

    def predict(self, meta: ObsInfo) -> ScalePrediction:
        """One forward pass: the fold transformer and the heads, one packed block at a time.

        The encoder packs the batch's crystals into blocks once
        (:meth:`~torchsx.nn.fold_transformer.FoldTransformerEncoder.layout`).
        Each block gathers its rows, runs the network (compiled on CUDA,
        ``encoder.compile``, after a dry run on the first block,
        :meth:`~torchsx.nn.fold_transformer.FoldTransformerEncoder.dry_run`)
        and the heads, in training under one checkpoint,
        so only the batch tables and the block's outputs are stored.  One
        gather puts the outputs back in row order.

        Without autograd (evaluation, the merge) each block runs the same way,
        under a checkpoint with autograd on, its outputs detached: flex
        attention runs only in autograd graphs, and the masked kernel of a
        forward-only graph expands every run's key mask to its score matrix.

        With ``defer_encoder_backward`` (training) the outputs are detached and
        enter the rest of the graph as leaves, in row order: the loss's
        backward pass then holds no encoder block and no feature table, and
        :meth:`encoder_backward` builds the features again and runs each block
        again, alone, from the gradient its rows received.  The gradient is the
        same; the encoder's activations and the batch-wide graph of the
        likelihood are never in memory together.
        """
        x_self, x_ctx, x_crys = self.features.build(meta)
        enc = self.encoder
        enc.fit_device(x_self.device)
        run = self._fold_block
        ckpt = torch.is_grad_enabled() and self.training
        defer = ckpt and self.defer_encoder_backward
        flex_eval = (not torch.is_grad_enabled() and x_self.is_cuda and enc.flex
                     and enc.flex_device and any(p.requires_grad for p in enc.parameters()))
        layout = enc.layout(meta.crystal_id)
        if layout:
            enc.dry_run(x_self, x_ctx, x_crys, meta.crystal_id, *layout[0],
                        autograd=torch.is_grad_enabled() or flex_eval, backward=ckpt)
        flex_only = flex_eval and enc.compile
        n = x_self.shape[0]
        parts, dest, corrections, blocks = [], [], [], []
        for idx, valid in layout:
            args = (x_self, x_ctx, x_crys, meta.crystal_id, idx, valid)
            if flex_only:
                with torch.enable_grad():
                    out = torch.utils.checkpoint.checkpoint(run, *args, use_reentrant=False)
            else:
                out = (torch.utils.checkpoint.checkpoint(run, *args, use_reentrant=False)
                       if ckpt else run(*args))
            if defer or flex_only:
                out = tuple(t.detach() for t in out)
            if defer:
                blocks.append((idx, valid))
            parts.append(out[:-1])
            corrections.append(out[-1].detach())
            dest.append(torch.where(valid, idx, n).flatten())
        if enc.refine_geometry:
            enc.last_correction = torch.cat(corrections)
        pos = scatter_positions(dest, n)
        rows = [torch.cat(t).index_select(0, pos) for t in zip(*parts)]
        del parts
        if defer:
            rows = [r.requires_grad_() for r in rows]
            self._deferred = (run, meta, blocks, rows)
        pred = ScalePrediction(**dict(zip(self._prediction_names(), rows)))
        return self._anchor(pred, meta)

    def encoder_backward(self) -> None:
        """The encoder's share of a deferred backward pass (:meth:`predict`):
        each block again, with autograd, back-propagated from the gradient its
        outputs received; one block's activations at a time.  The batch's
        predictions are spent by then, and released first (:meth:`clear`)."""
        if self._deferred is None:
            return
        run, meta, blocks, rows = self._deferred
        grads = [r.grad for r in rows]
        del rows
        self.clear()
        x_self, x_ctx, x_crys = self.features.build(meta)
        n = x_self.shape[0]
        for idx, valid in blocks:
            out = run(x_self, x_ctx, x_crys, meta.crystal_id, idx, valid)[:-1]
            # each slot's gradient from its row; padding slots get none
            at, live = idx.flatten().clamp(max=n - 1), valid.flatten()
            pairs = [(o, g.index_select(0, at) * live.view(-1, *([1] * (g.dim() - 1))))
                     for o, g in zip(out, grads) if g is not None]
            if pairs:
                torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])

    def _fold_block(self, x_self: Tensor, x_ctx: Tensor, x_crys: Tensor | None, group_id: Tensor,
                    idx: Tensor, valid: Tensor) -> tuple[Tensor, ...]:
        """Heads on every slot of one packed block (padding included), flattened, and
        the block's geometry correction last."""
        z, correction = self.encoder.encode(x_self, x_ctx, x_crys, group_id, idx, valid)
        p = self.heads(z.flatten(0, 1))
        return tuple(getattr(p, name) for name in self._prediction_names()) + (correction,)

    @torch.no_grad()
    def set_gauge_basis(self, phi: Tensor) -> None:
        """Install the per-reflection basis rows (``n_hkl x p``) for the ``basis`` anchor."""
        self.gauge_phi = phi.detach().to(torch.float32).clone()

    def _anchor(self, pred: ScalePrediction, meta: ObsInfo) -> ScalePrediction:
        """Pin the gauge of ``log K`` for one batch.

        ``K -> cK, I0 -> I0/c`` (scalar) and ``K(h) -> K(h) g(h), I0 -> I0/g(h)``
        (any function of reciprocal space the encoder can express through its
        resolution inputs) both leave the likelihood exactly invariant.  The
        scalar modes subtract a location statistic.

        ``basis`` fits the two-way additive model

            log K_kh  ~  c_k  +  phi(h) @ beta

        (a constant per crystal plus a smooth function of the reflection) by a
        few alternating least-squares sweeps and subtracts only the smooth
        part, then pins the scalar gauge with the median as before.  Fitting
        ``phi @ beta`` to ``log K`` directly would be wrong: the observations
        of a given direction or resolution are not an orientation-neutral
        sample of crystals -- reflections near c* of a plate crystal are seen
        only edge-on, by the thick, strong crystals; high-resolution
        reflections come from the well-diffracting ones -- so the mean of a
        legitimate per-crystal scale over the observations of ``h`` is itself
        a smooth function of ``h``.  Removing it would push crystal scale into
        the merged intensities and distort their anisotropy.  The crystal term
        absorbs exactly that, and what
        is left in ``span(phi)`` is the common mode no crystal owns.  The
        projection is linear and stays in the graph, so the network receives no
        gradient along the removed direction.
        """
        if not self.anchor_log_k:
            return pred
        if self.anchor_stat == "basis":
            if self.gauge_phi.numel() == 0:
                raise RuntimeError(
                    "anchor_stat='basis' needs set_gauge_basis() before predicting"
                )
            log_K = pred.log_K - self._smooth_common_mode(pred.log_K, meta)
            log_K = log_K - log_K.median()
        else:
            loc = pred.log_K.median() if self.anchor_stat == "median" else pred.log_K.mean()
            log_K = pred.log_K - loc
        return dataclasses.replace(pred, log_K=log_K)

    def _smooth_common_mode(self, log_K: Tensor, meta: ObsInfo, sweeps: int = 3) -> Tensor:
        """``phi(h) @ beta`` from the two-way fit ``log K ~ c_crystal + phi(h) @ beta``.

        Alternating least squares: crystal means of the current residual, then
        the basis fit of ``log K - c``.  Three sweeps are enough because the
        crystal effects and the smooth term are nearly orthogonal once the
        crystals sample reciprocal space broadly; the fit is exact for a pure
        per-crystal scale (``beta = 0``) and for a pure common mode observed by
        every crystal alike.
        """
        phi = self.gauge_phi.index_select(0, meta.hkl_idx).to(log_K.dtype)
        cid = meta.crystal_id.long()
        n_cry = int(meta.n_crystals) if getattr(meta, "n_crystals", None) else int(cid.max()) + 1
        counts = torch.zeros(n_cry, device=log_K.device, dtype=log_K.dtype).index_add_(
            0, cid, torch.ones_like(log_K)
        ).clamp(min=1.0)
        gram = phi.T @ phi
        gram = gram + 1e-6 * gram.diagonal().mean() * torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        smooth = torch.zeros_like(log_K)
        for _ in range(sweeps):
            resid = log_K - smooth
            c = torch.zeros(n_cry, device=log_K.device, dtype=log_K.dtype).index_add_(
                0, cid, resid
            ) / counts
            beta = torch.linalg.solve(gram, phi.T @ (log_K - c.index_select(0, cid)))
            smooth = phi @ beta
        return smooth

    def prime(self, meta: ObsInfo) -> ScalePrediction:
        """Predict and install the scatter parameters; call once per batch or chunk."""
        pred = self.predict(meta)
        self.scatter.set_prediction(meta.obs_id, pred.mix)
        return pred

    def clear(self) -> None:
        self._deferred = None
        self.scatter.clear_prediction()
