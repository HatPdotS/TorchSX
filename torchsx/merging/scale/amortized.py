"""Amortised scale model — one network is the sole scale source.

Drops into the existing pipeline as a :class:`ScaleModel`, so the production
merge, the posterior moments and the residual diagnostics all consume it
unchanged.  It additionally owns a
:class:`~torchsx.merging.scatter.predicted.PredictedGenGaussScatter`, because
the same forward pass produces the scale *and* the scatter shape, and
recomputing the network for each would double the cost.

``ScaleBPerCrystal`` is deliberately **not** composed with this.  Two competing
scale sources — one parametric, one flexible — is an identifiability mess; the
amortised model must be the only place a scale can come from.

Usage::

    scale = AmortizedScale.from_prep(prep)
    pred  = scale.prime(meta)                 # one forward; installs the scatter
    log_K = pred.log_K                        # or scale.log_k(params, meta)
    nll   = quadrature.marginalize_signal(scale.scatter, likelihood, ...)
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.scale.base import ScaleModel
from torchsx.merging.scatter.mixture import PredictedMixtureScatter
from torchsx.merging.scatter.predicted import PredictedGenGaussScatter
from torchsx.nn.deepsets import DeepSetsEncoder, ScaleEncoder
from torchsx.nn.heads import ScaleHeads, ScalePrediction
from torchsx.nn.obs_features import ObsFeatureSpec

__all__ = ["AmortizedScale"]


class AmortizedScale(ScaleModel):
    """Network-predicted ``log K`` (and scatter) per observation.

    Parameters
    ----------
    features
        Tiered feature builder; enforces the own-intensity and
        condition-blindness contracts.
    encoder
        Anything satisfying :class:`~torchsx.nn.deepsets.ScaleEncoder`.  The
        Deep Sets default is abismal's architecture; a transformer over padded
        crystal batches is a drop-in replacement.
    heads
        :class:`~torchsx.nn.heads.ScaleHeads`.
    n_crystals
        Group count for the leave-one-out pool.
    anchor_log_k
        Pin the gauge by subtracting a batch location statistic from
        ``log K``.  ``K → cK, I0 → I0/c`` leaves the likelihood exactly
        invariant, so without this the scale drifts freely.  On by default,
        from step 0.
    anchor_stat
        ``"median"`` (default) or ``"mean"``.  The gauge is arbitrary, so any
        location statistic is valid — but the mean is **not robust**, and that
        matters here.  Measured on the real dark: ~1.5 % of observations ran
        away to ``log K ≈ +68`` while the bulk sat in ``[-5.5, +0.2]``.  Under a
        mean anchor those outliers contribute ≈ +1.0 to the mean and drag the
        other 98.5 % down by ≈ 1.0 — the bulk median came out at −2.4.  A small
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
        encoder: ScaleEncoder,
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
        #: Crystals per encoder chunk; 0 runs the whole batch in one shot.
        #: See :meth:`predict` -- the encoder is separable across crystals, so
        #: chunking is exact and moves the checkpoint boundary off the blocks.
        self.chunk_crystals = 0
        self.anchor_log_k = bool(anchor_log_k)
        if anchor_stat not in ("median", "mean", "basis"):
            raise ValueError(
                f"anchor_stat must be 'median', 'mean' or 'basis', got {anchor_stat!r}"
            )
        self.anchor_stat = anchor_stat
        fam = getattr(heads, "scatter_family", "gengauss")
        self.scatter = (PredictedMixtureScatter(heads.n_components) if fam == "mixture"
                        else PredictedGenGaussScatter())
        self._cached_obs_id: Tensor | None = None
        self._cached_pred: ScalePrediction | None = None

    # ── construction ───────────────────────────────────────────────────

    #: hkl defaults OFF.  The design doc said "hkl goes in for now -- but see
    #: the monitor"; the monitor has since spoken.  On the real dark stream with
    #: hkl enabled, ``hkl_permutation_sensitivity`` reached **0.586** (abismal:
    #: 0.003) -- 59 % of the predicted scale's variance moved when hkl labels
    #: were permuted, ``sd(log K)`` blew up to 14.5, and 80 % of that variance
    #: sat *within* crystal, i.e. per reflection.  That is the scale absorbing
    #: I0, which is degenerate by construction and is the one channel that can
    #: manufacture a difference peak.
    #:
    #: The failure has a signature worth remembering: training converged
    #: normally (var_z -> 1.02, held-out NLL 5.10 at epoch 18) and only *then*
    #: collapsed (held-out 6.33, var_z 12.5 by epoch 28).  The network learns
    #: the legitimate scale first and discovers the shortcut second, so a short
    #: run looks healthy.
    #:
    #: ``s_sq`` stays in: resolution dependence of the scale is a real physical
    #: effect (a B-factor), and the permutation monitor tests the h/k/l channel
    #: specifically, leaving s_sq untouched.
    @classmethod
    def from_prep(
        cls,
        prep: PreparedObs,
        *,
        d_model: int = 128,
        n_layers: int = 5,
        d_hidden: int | None = None,
        encoder: str = "deepsets",
        n_heads: int = 8,
        max_tokens: int | None = 512,
        include_hkl: bool = False,
        context_folds: int = 0,
        obs_context_layers: int | None = None,
        obs_self_layers: int | None = None,
        w_init: float = 1.65,
        beta_init: float = 2.0,
        predict_width: bool = True,
        predict_shape: bool = True,
        predict_skew: bool = False,
        global_skew: bool = False,
        unity_k: bool = False,
        anchor_log_k: bool = True,
        anchor_stat: str = "median",
        dropout: float = 0.0,
        scatter_family: str = "gengauss",
        n_components: int = 3,
        conditioner_hidden: int = 8,
    ) -> "AmortizedScale":
        if encoder not in ("deepsets", "attention"):
            raise ValueError(f"encoder must be 'deepsets' or 'attention', got {encoder!r}")
        features = ObsFeatureSpec(
            prep, include_hkl=include_hkl, token_mode=(encoder == "attention")
        )
        if encoder == "attention":
            from torchsx.nn.attention import PeerAttentionEncoder
            enc = PeerAttentionEncoder(
                features.n_self, features.n_context,
                d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                dropout=dropout, max_tokens=max_tokens,
            )
        else:
            enc = DeepSetsEncoder(
                features.n_self, features.n_context,
                d_model=d_model, n_layers=n_layers, d_hidden=d_hidden,
                dropout=dropout,
                n_crystal_features=features.n_crystal,
                context_folds=context_folds,
                obs_context_layers=obs_context_layers,
                obs_self_layers=obs_self_layers,
            )
        heads = ScaleHeads(
            d_model,
            w_init=w_init,
            beta_init=beta_init,
            predict_width=predict_width,
            predict_shape=predict_shape,
            predict_skew=predict_skew,
            global_skew=global_skew,
            unity_k=unity_k,
            scatter_family=scatter_family,
            n_components=n_components,
            conditioner_hidden=conditioner_hidden,
        )
        if not heads.needs_embedding:
            # Nothing reads the embedding, so the encoder is inert: freeze it
            # rather than letting an optimiser carry two dozen dead parameters
            # that can never receive a gradient.  It is still constructed so the
            # module tree, and therefore a checkpoint, keeps one shape.
            enc.requires_grad_(False)
        return cls(
            features, enc, heads,
            n_crystals=prep.obs_info.n_crystals,
            anchor_log_k=anchor_log_k,
            anchor_stat=anchor_stat,
        )

    @classmethod
    def from_obs_info(cls, obs_info: ObsInfo, **kwargs) -> "AmortizedScale":
        raise TypeError(
            "AmortizedScale needs the full PreparedObs (it reads I, bg and the "
            "Miller table for its context features). Use "
            "AmortizedScale.from_prep(prep) and pass the instance to Merger."
        )

    # ── forward ────────────────────────────────────────────────────────

    def set_encoder_chunking(self, crystals_per_chunk: int) -> None:
        """Chunk the encoder over crystals, moving the checkpoint boundary.

        Sets both halves of the change together: the chunk size, and turning
        OFF the encoder's per-block checkpointing, which the chunk-level
        checkpoint replaces.  Leaving both on would nest a recompute inside a
        recompute and keep the per-block boundaries we are trying to drop.
        """
        n = int(crystals_per_chunk)
        self.chunk_crystals = max(n, 0)
        self.encoder.checkpoint = not (n > 0)

    @staticmethod
    def _crystal_chunks(crystal_id: Tensor, per_chunk: int):
        """Slice bounds that never cut a crystal in half.

        ``leave_one_out_mean`` pools WITHIN a crystal, so a chunk edge that
        falls inside one silently gives that crystal a context built from part
        of itself.  Rows arrive grouped by crystal (``CrystalIndex.rows``
        concatenates per-crystal blocks), so the boundaries are just the points
        where ``crystal_id`` changes.
        """
        n = crystal_id.numel()
        change = (crystal_id[1:] != crystal_id[:-1]).nonzero().squeeze(1) + 1
        # Chunking is only exact if each crystal occupies ONE contiguous run.
        # `PreparedObs` is sorted by hkl, not by crystal, so passing it whole
        # would split crystals across chunks and give each a context built from
        # part of itself -- wrong, and silent.  `CrystalIndex.rows()` returns
        # crystal-grouped rows, which is what the trainer feeds.
        if change.numel() + 1 != int(torch.unique(crystal_id).numel()):
            raise ValueError(
                "encoder chunking needs crystal-grouped observations: "
                f"{change.numel() + 1} contiguous runs for "
                f"{int(torch.unique(crystal_id).numel())} crystals. "
                "Index through CrystalIndex.rows() rather than passing "
                "hkl-sorted PreparedObs directly."
            )
        edges = torch.cat([
            crystal_id.new_zeros(1), change, crystal_id.new_tensor([n])
        ]).tolist()
        out, lo = [], 0
        for i in range(1, len(edges)):
            if i - lo >= per_chunk or i == len(edges) - 1:
                out.append((edges[lo], edges[i]))
                lo = i
        return out

    def predict(self, meta: ObsInfo) -> ScalePrediction:
        """One forward pass: ``meta`` → per-observation (log K, log w, β, log α).

        With ``chunk_crystals > 0`` the encoder runs in crystal-aligned chunks,
        each wrapped in a SINGLE checkpoint.  That moves the checkpoint boundary
        from the block to the chunk: measured at 1.67 M observations the encoder
        held 14.0 GB, which is 16 retained ``(M, d_model)`` tensors at 512 B per
        observation each, and every one of them is interior -- only ``log_K``,
        ``log_w`` and ``beta`` leave.  Storing the chunk's inputs and outputs
        instead is ~52 B per observation, and the 16 are rematerialised one
        chunk at a time in backward.
        """
        if not self.heads.needs_embedding:
            # Nothing reads the embedding (unity K, global width/shape/skew),
            # so the features and the encoder are dead weight -- and on the
            # global-fit path they would be the entire cost.  The heads only
            # need the row count, which an (n, 0) placeholder carries.
            return self.heads(
                torch.zeros((meta.obs_id.numel(), 0), device=meta.obs_id.device)
            )
        n_chunk = int(getattr(self, "chunk_crystals", 0) or 0)
        if n_chunk > 0:
            # Chunk in EVERY mode, not only when building a graph.  The
            # no-grad paths -- the locate sweep, evaluation, the merge -- store
            # nothing, but a single SiLU on a whole batch is still one
            # allocation: at R=200 that is 13.3 M rows and 12.7 GB, which OOMs
            # a 40 GB card before any checkpointing question arises.  The
            # checkpoint is what is conditional on grad, not the chunking.
            return self._predict_chunked(meta, n_chunk)
        x_self, x_ctx, x_crys = self.features.build(meta)
        z = self.encoder(x_self, x_ctx, meta.crystal_id, self.n_crystals, x_crys=x_crys)
        pred = self.heads(z)
        return self._anchor(pred, meta)

    def _predict_chunked(self, meta: ObsInfo, per_chunk: int) -> ScalePrediction:
        # Features are built once and persist: they are (M, ~10), i.e. 40 B per
        # observation, against 8 KB for the interior tensors they let us drop.
        x_self, x_ctx, x_crys = self.features.build(meta)
        cid = meta.crystal_id
        spans = self._crystal_chunks(cid, per_chunk)

        # The encoder's per-block checkpointing must be OFF for the whole
        # lifetime of this call, not just the forward: toggling it back before
        # backward makes recomputation take a different path from the original
        # forward, and non-reentrant checkpoint detects that as a differing
        # number of saved tensors and raises.  `set_encoder_chunking` turns it
        # off permanently, so this only asserts it.
        if self.encoder.checkpoint and torch.is_grad_enabled() and self.training:
            raise RuntimeError(
                "encoder.checkpoint must be False when chunk_crystals > 0; "
                "use AmortizedScale.set_encoder_chunking() to set both"
            )

        skew = self.heads.predict_skew or self.heads.global_skew
        mixture = getattr(self.heads, "scatter_family", "gengauss") == "mixture"
        names = ["log_K", "log_w", "beta", "log_alpha", "log_e_s"]
        if skew or mixture:
            names.append("log_alpha_hi")
        if mixture:
            names.append("mix")

        def run(xs, xc, ci, xcr):
            z = self.encoder(xs, xc, ci, self.n_crystals, x_crys=xcr)
            p = self.heads(z)
            return tuple(getattr(p, name) for name in names)

        ckpt = torch.is_grad_enabled() and self.training
        parts = [
            torch.utils.checkpoint.checkpoint(
                run, x_self[a:b], x_ctx[a:b], cid[a:b], x_crys, use_reentrant=False
            )
            if ckpt else run(x_self[a:b], x_ctx[a:b], cid[a:b], x_crys)
            for a, b in spans
        ]

        pred = ScalePrediction(**{name: torch.cat(t) for name, t in zip(names, zip(*parts))})
        return self._anchor(pred, meta)

    def predict_split(
        self, meta_self: ObsInfo, meta_ctx: ObsInfo
    ) -> ScalePrediction:
        """Predict with the self branch and context branch fed *different* metas.

        Used by the hkl-permutation monitor to isolate the degeneracy that
        actually matters: whether ``log_K[j]`` depends on ``j``'s own
        reflection identity.  A whole-meta permutation cannot answer that once
        the context carries ``obs_scale = n'/I0``, because permuting hkl also
        permutes the *peers'* I0 lookup — which is legitimate use, not leakage.
        """
        if not self.heads.needs_embedding:
            return self.predict(meta_self)      # no embedding, nothing to split
        x_self, _, _ = self.features.build(meta_self)
        _, x_ctx, x_crys = self.features.build(meta_ctx)
        z = self.encoder(x_self, x_ctx, meta_ctx.crystal_id, self.n_crystals, x_crys=x_crys)
        pred = self.heads(z)
        return self._anchor(pred, meta_self)

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
        the merged intensities (measured on BR: the c*/ab intensity ratio went
        from 0.49 to 0.79).  The crystal term absorbs exactly that, and what
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
        return ScalePrediction(
            log_K=log_K, log_w=pred.log_w, beta=pred.beta,
            log_alpha=pred.log_alpha, log_e_s=pred.log_e_s,
            log_alpha_hi=pred.log_alpha_hi, mix=pred.mix,
        )

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
        """Predict, install into the scatter, and cache for :meth:`log_k`.

        Call once per batch/chunk.  The cache is keyed on the ``obs_id`` tensor
        so a mismatched later call raises rather than silently returning a
        stale prediction.
        """
        pred = self.predict(meta)
        if pred.mix is not None:
            self.scatter.set_prediction(meta.obs_id, torch.log(pred.beta), pred.mix)
        else:
            self.scatter.set_prediction(
                meta.obs_id, pred.log_alpha, torch.log(pred.beta), pred.log_alpha_hi
            )
        self._cached_obs_id = meta.obs_id
        self._cached_pred = pred
        return pred

    def clear(self) -> None:
        self._cached_obs_id = None
        self._cached_pred = None
        self.scatter.clear_prediction()

    def log_k(
        self,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        del params, obs_range          # addressed by obs_id, not position
        if (
            self._cached_pred is not None
            and self._cached_obs_id is not None
            and self._cached_obs_id.shape == meta.obs_id.shape
            and torch.equal(self._cached_obs_id, meta.obs_id)
        ):
            return self._cached_pred.log_K
        return self.predict(meta).log_K
