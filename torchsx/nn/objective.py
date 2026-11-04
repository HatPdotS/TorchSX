"""Training objective of the amortised scale model.

:class:`MarginalNLL` is the marginal likelihood of each observation, with the
multiplicative scatter integrated out of the detection likelihood::

    log p_j = log INT p_det(n_j | S, bg_j) . p_scatter(log S - log_mean_j) d log S

evaluated by the quadrature stack.  On CUDA the quadrature dispatches to Triton
kernels that take the per-observation scatter parameters, never materialise the
``(M, K)`` integrand and return analytic gradients, so the marginal costs tens
of MB per million observations against several GB of network activations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.likelihood.poisson import poisson_log_prob_at_zero_signal
from torchsx.merging.scatter.mixture import PredictedMixtureScatter

__all__ = ["MarginalNLL"]


class MarginalNLL(nn.Module):
    """Per-observation ``-log p`` via the existing quadrature stack.

    Parameters
    ----------
    quadrature
        Any :class:`~torchsx.merging.quadrature.base.Quadrature`.
    likelihood
        Detection likelihood; ``Poisson`` for counts.
    chunk_size
        Observations per quadrature call, or ``None`` for one call.  Chunk
        boundaries must not split a crystal if the caller is also pooling — but
        the pooling happens upstream in the encoder, so any split is safe here.
    checkpoint_chunks
        Recompute each chunk's ``(M, K)`` integrand in the backward pass rather
        than storing it.  **Chunking alone saves nothing**: concatenating the
        chunk outputs keeps every chunk's graph alive until backward, so peak
        memory is the same as one call, which exceeds a 40 GB card at ~1e6
        observations with a per-observation grid.
    """

    def __init__(
        self, quadrature, likelihood, *,
        chunk_size: int | None = None,
        checkpoint_chunks: bool = True, miss_frac: float = 0.0,
        outlier_frac: float = 0.0, outlier_scale: float = 3.0,
        detector_noise=None,
    ) -> None:
        super().__init__()
        self.quadrature = quadrature
        self.likelihood = likelihood
        self.chunk_size = chunk_size
        self.checkpoint_chunks = bool(checkpoint_chunks)
        #: Miss component.  A predicted reflection that recorded no signal at
        #: all -- masked pixels, a mis-prediction, a spot that never formed --
        #: is not a partial observation, but the scatter has to stretch its
        #: lower tail to cover it, and on strong predictions that costs the
        #: width its calibration.  With ``miss_frac > 0`` the likelihood is
        #: the mixture ``(1 - pi) p(n | S, scatter) + pi p(n | S = 0)``, with
        #: ``pi`` a learnable global scalar started at ``miss_frac``.  The
        #: second term does not depend on the reflection intensity, so on the
        #: posterior grid it is an additive per-observation constant and needs
        #: no quadrature.  0 disables it exactly.
        self.miss_logit: nn.Parameter | None = None
        if miss_frac > 0.0:
            if not (0.0 < miss_frac < 1.0):
                raise ValueError(f"miss_frac must be in (0, 1), got {miss_frac}")
            self.miss_logit = nn.Parameter(torch.tensor(math.log(miss_frac / (1.0 - miss_frac))))
        #: Detector-outlier component.  A small fraction of observations carry
        #: count errors no intensity can explain -- a zinger or a neighbouring
        #: spot in the peak, a contaminated annulus, a broken pedestal: at zero
        #: predicted signal they lie beyond 5 robust SDs of the detector law at
        #: orders of magnitude above a Gaussian rate.
        #: Without this term only the scatter can absorb them, and it does, with
        #: a heavy upper tail that can carry much of E[S].  With
        #: ``outlier_frac > 0`` the likelihood is
        #: ``(1 - rho) p_miss(n | S) + rho p_out(n)``, ``p_out`` a Cauchy about
        #: the background mean with ``outlier_scale`` times the detector SD at
        #: zero signal, ``rho`` learnable.  Independent of the intensity, so like
        #: the miss term it is an additive per-observation constant on the
        #: posterior grid.  0 disables it exactly.
        self.outlier_logit: nn.Parameter | None = None
        self.outlier_scale = float(outlier_scale)
        if outlier_frac > 0.0:
            if not (0.0 < outlier_frac < 1.0):
                raise ValueError(f"outlier_frac must be in (0, 1), got {outlier_frac}")
            self.outlier_logit = nn.Parameter(torch.tensor(math.log(outlier_frac / (1.0 - outlier_frac))))
        #: Learned detector noise (:class:`~torchsx.merging.likelihood.detector_noise.DetectorNoise`):
        #: the counting variance becomes ``A + B (lam + sig_b2)`` per panel,
        #: fitted with the network.  ``None`` is the plain detection model.
        #: The outlier Cauchy keeps the noise-free detector SD as its scale.
        self.detector_noise = detector_noise
        self._noise_cache: tuple | None = None

    def noise_terms(self, meta) -> tuple[Tensor, Tensor] | None:
        """``(A, B)`` per observation of ``meta`` (gathered by panel), or ``None``.

        Cached on the batch object (held, so its id cannot be reused), the grad
        mode and the parameters' versions: the locate scans and outer nodes of one
        step share one gather and one graph, and a no-grad gather is never reused
        by the loss pass.
        """
        dn = self.detector_noise
        if dn is None:
            return None
        versions = tuple(getattr(dn, f"log_{t}")._version for t in "AB")
        c = self._noise_cache
        if (c is not None and c[0] is meta and c[1] == torch.is_grad_enabled()
                and c[2] == versions):
            return c[3]
        group = getattr(meta, "panel_id", None)
        if group is not None and dn.n_groups == 1:
            group = None
        terms = dn.terms_obs(group, meta.hkl_idx.numel(), meta.hkl_idx.device)
        self._noise_cache = (meta, torch.is_grad_enabled(), versions, terms)
        return terms

    def log_outlier(self, n_ph: Tensor, bg: Tensor) -> Tensor:
        """``log p_out(n)``: a Cauchy about the detection background mean, scale
        ``outlier_scale`` times the detector SD at zero signal."""
        bg_mean, sig_b2 = self.likelihood.detection_params(bg)
        lam0 = bg_mean.clamp(min=1e-6)
        var0 = lam0 + (sig_b2 if sig_b2 is not None else 0.0)
        gamma = self.outlier_scale * var0.sqrt()
        t = (n_ph.to(lam0.dtype) - lam0) / gamma
        return -math.log(math.pi) - gamma.log() - torch.log1p(t * t)

    def _log_miss(self, n_ph: Tensor, bg: Tensor, noise=None) -> Tensor:
        """``log p(n | S = 0)`` on the kernel's scale (Poisson about its peak,
        with the background-variance term when the likelihood carries one).

        Delegated to the quadrature so the miss term and the marginal it is
        mixed with can never disagree about the detection model's background --
        ``SplitLogGrid`` hands back the ``log_L0`` it already computed for the
        analytic tail, at no cost, on the same cached grid.
        """
        q = getattr(self, "quadrature", None)
        if noise is not None:
            return q.log_miss(self.likelihood, n_ph, bg, noise=noise)
        if q is not None and hasattr(q, "log_miss"):
            return q.log_miss(self.likelihood, n_ph, bg)
        bg_mean, sig_b2 = self.likelihood.detection_params(bg)
        return poisson_log_prob_at_zero_signal(n_ph, bg_mean, sig_b2)

    def _mix_miss(self, lp: Tensor, n_ph: Tensor, bg: Tensor, noise=None) -> Tensor:
        if self.miss_logit is not None:
            log_pi = torch.nn.functional.logsigmoid(self.miss_logit)
            log_1m = torch.nn.functional.logsigmoid(-self.miss_logit)
            lp = torch.logaddexp(log_1m + lp, log_pi + self._log_miss(n_ph, bg, noise))
        if self.outlier_logit is not None:
            log_r = torch.nn.functional.logsigmoid(self.outlier_logit)
            log_1r = torch.nn.functional.logsigmoid(-self.outlier_logit)
            lo = self.log_outlier(n_ph, bg)
            lp = torch.logaddexp(log_1r + lp, log_r + lo)
        return lp

    def _marginal(self, scatter, params, log_mean, n_ph, bg, meta, obs_range, noise=None) -> Tensor:
        """``log p(n | log_mean)`` with the scatter integrated out.

        The quadrature owns mixture integration, including any fused path.
        ``noise`` (detector noise terms) is passed only when present, so
        quadratures that do not implement it refuse rather than drop it.
        """
        kw = {} if noise is None else {"noise": noise}
        if isinstance(scatter, PredictedMixtureScatter):
            return self.quadrature.marginalize_mixture(
                scatter, self.likelihood, log_mean, n_ph, bg, params, meta,
                obs_range=obs_range, **kw,
            )
        return self.quadrature.marginalize_signal(
            scatter, self.likelihood, log_mean, n_ph, bg, params, meta, obs_range=obs_range, **kw,
        )

    def log_prob(
        self, scatter, params, log_mean: Tensor, n_ph: Tensor, bg: Tensor, meta
    ) -> Tensor:
        n = log_mean.shape[0]
        cs = self.chunk_size or n
        if cs >= n:
            noise = self.noise_terms(meta)
            lp = self._marginal(scatter, params, log_mean, n_ph, bg, meta, (0, n), noise)
            return self._mix_miss(lp, n_ph, bg, noise)
        parts = []
        for s in range(0, n, cs):
            e = min(s + cs, n)
            sub = meta.slice(s, e)

            def _chunk(lm, nn, bb, _sub=sub, _s=s, _e=e):
                noise = self.noise_terms(_sub)
                lp = self._marginal(scatter, params, lm, nn, bb, _sub, (_s, _e), noise)
                return self._mix_miss(lp, nn, bb, noise)

            args = (log_mean[s:e], n_ph[s:e], bg[s:e])
            if self.checkpoint_chunks and torch.is_grad_enabled():
                lp = torch.utils.checkpoint.checkpoint(
                    _chunk, *args, use_reentrant=False
                )
            else:
                lp = _chunk(*args)
            parts.append(lp)
        return torch.cat(parts)

    def forward(
        self, scatter, params, log_mean: Tensor, n_ph: Tensor, bg: Tensor, meta,
        *, mask: Tensor | None = None,
    ) -> Tensor:
        lp = self.log_prob(scatter, params, log_mean, n_ph, bg, meta)
        if mask is not None:
            lp = lp[mask]
        return -lp.sum()
