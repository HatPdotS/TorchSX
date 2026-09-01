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
        The median cannot be hijacked that way.
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
        self.anchor_log_k = bool(anchor_log_k)
        if anchor_stat not in ("median", "mean"):
            raise ValueError(f"anchor_stat must be 'median' or 'mean', got {anchor_stat!r}")
        self.anchor_stat = anchor_stat
        self.scatter = PredictedGenGaussScatter()
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
        w_init: float = 1.65,
        beta_init: float = 2.0,
        predict_width: bool = True,
        predict_shape: bool = True,
        anchor_log_k: bool = True,
        anchor_stat: str = "median",
        dropout: float = 0.0,
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
            )
        heads = ScaleHeads(
            d_model,
            w_init=w_init,
            beta_init=beta_init,
            predict_width=predict_width,
            predict_shape=predict_shape,
        )
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

    def predict(self, meta: ObsInfo) -> ScalePrediction:
        """One forward pass: ``meta`` → per-observation (log K, log w, β, log α)."""
        x_self, x_ctx = self.features.build(meta)
        z = self.encoder(x_self, x_ctx, meta.crystal_id, self.n_crystals)
        pred = self.heads(z)
        if self.anchor_log_k:
            loc = (
                pred.log_K.median() if self.anchor_stat == "median"
                else pred.log_K.mean()
            )
            pred = ScalePrediction(
                log_K=pred.log_K - loc,
                log_w=pred.log_w,
                beta=pred.beta,
                log_alpha=pred.log_alpha,
                log_e_s=pred.log_e_s,
            )
        return pred

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
        x_self, _ = self.features.build(meta_self)
        _, x_ctx = self.features.build(meta_ctx)
        z = self.encoder(x_self, x_ctx, meta_ctx.crystal_id, self.n_crystals)
        pred = self.heads(z)
        if self.anchor_log_k:
            loc = (
                pred.log_K.median() if self.anchor_stat == "median"
                else pred.log_K.mean()
            )
            pred = ScalePrediction(
                log_K=pred.log_K - loc, log_w=pred.log_w, beta=pred.beta,
                log_alpha=pred.log_alpha, log_e_s=pred.log_e_s,
            )
        return pred

    def prime(self, meta: ObsInfo) -> ScalePrediction:
        """Predict, install into the scatter, and cache for :meth:`log_k`.

        Call once per batch/chunk.  The cache is keyed on the ``obs_id`` tensor
        so a mismatched later call raises rather than silently returning a
        stale prediction.
        """
        pred = self.predict(meta)
        self.scatter.set_prediction(
            meta.obs_id, pred.log_alpha, torch.log(pred.beta)
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
