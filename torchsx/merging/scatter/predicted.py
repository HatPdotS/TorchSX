"""GenGauss scatter whose (α, β) are predicted per observation by a network.

The one override — :meth:`log_alpha_beta_obs` — is enough.  Everything
downstream routes through that hook, so this class is accepted unchanged by:

* the eager quadrature path,
* both Triton kernels (``_can_use_triton`` tests
  ``isinstance(scatter, GenGaussLogScatter)``, and this is a subclass, so the
  fast path is kept — the kernel already takes per-observation
  ``log_alpha_obs`` / ``log_beta_obs`` and returns analytic gradients for both),
* :func:`~torchsx.merging.posterior.compute_posterior_moments`,
* and therefore :class:`~torchsx.merging.merger.Merger` itself.

Predictions are addressed by ``ObsInfo.obs_id``, not by position, so an
arbitrary crystal-batched gather resolves correctly — and so does a contiguous
chunk taken by ``DataNLLTarget``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter.gengauss import GenGaussLogScatter


class PredictedGenGaussScatter(GenGaussLogScatter):
    """Per-observation (α, β) from a network, addressed by ``obs_id``.

    Usage per training step::

        scatter.set_prediction(batch.obs_id, log_alpha, log_beta)   # grad-carrying
        loss = quadrature.marginalize_signal(scatter, likelihood, ...)

    ``log_alpha`` / ``log_beta`` keep their autograd graph, so gradients flow
    back into the network through the Triton kernel's analytic
    ``d log_p / d log_alpha_obs`` and ``d log_p / d log_beta_obs``.

    Parameters
    ----------
    strict
        If ``True`` (default) a query for an ``obs_id`` outside the current
        prediction raises.  Silently substituting a stale or default value
        there would corrupt the loss in a way no metric would reveal.
    """

    def __init__(self, *, strict: bool = True) -> None:
        super().__init__()
        self.strict = bool(strict)
        self._sorted_ids: Tensor | None = None
        self._sort_perm: Tensor | None = None
        self._log_alpha: Tensor | None = None
        self._log_beta: Tensor | None = None
        self._log_alpha_hi: Tensor | None = None

    # ── prediction plumbing ────────────────────────────────────────────

    def set_prediction(
        self, obs_id: Tensor, log_alpha: Tensor, log_beta: Tensor,
        log_alpha_hi: Tensor | None = None,
    ) -> None:
        """Install the current batch's per-observation (log α, log β[, log α_hi])."""
        if not (obs_id.shape == log_alpha.shape == log_beta.shape):
            raise ValueError(
                f"shape mismatch: obs_id={tuple(obs_id.shape)} "
                f"log_alpha={tuple(log_alpha.shape)} log_beta={tuple(log_beta.shape)}"
            )
        sorted_ids, perm = torch.sort(obs_id)
        self._sorted_ids = sorted_ids
        self._sort_perm = perm
        self._log_alpha = log_alpha
        self._log_beta = log_beta
        self._log_alpha_hi = log_alpha_hi

    def clear_prediction(self) -> None:
        self._sorted_ids = self._sort_perm = None
        self._log_alpha = self._log_beta = self._log_alpha_hi = None

    @property
    def has_prediction(self) -> bool:
        return self._log_alpha is not None

    def _positions(self, obs_id: Tensor) -> Tensor:
        """Map ``obs_id`` to rows of the installed prediction."""
        assert self._sorted_ids is not None and self._sort_perm is not None
        loc = torch.searchsorted(self._sorted_ids, obs_id)
        loc = loc.clamp(max=self._sorted_ids.numel() - 1)
        found = self._sorted_ids[loc] == obs_id
        if self.strict and not bool(found.all()):
            n_missing = int((~found).sum())
            raise KeyError(
                f"{n_missing} of {obs_id.numel()} obs_id values are outside the "
                f"installed prediction. The scatter model and the scale model "
                f"must be given the same observation set in the same step."
            )
        return self._sort_perm[loc]

    # ── the seam ───────────────────────────────────────────────────────

    def log_alpha_beta_obs(
        self, params: Parameters, meta: ObsInfo
    ) -> tuple[Tensor, Tensor]:
        if self._log_alpha is None:
            raise RuntimeError(
                "PredictedGenGaussScatter has no prediction installed; call "
                "set_prediction(obs_id, log_alpha, log_beta) first."
            )
        if meta.obs_id is None:
            raise ValueError(
                "PredictedGenGaussScatter needs ObsInfo.obs_id to address its "
                "prediction (prepare_observations populates it)."
            )
        pos = self._positions(meta.obs_id)
        return self._log_alpha[pos], self._log_beta[pos]

    def log_alpha_hi_obs(self, params: Parameters, meta: ObsInfo) -> Tensor | None:
        if self._log_alpha_hi is None:
            return None
        return self._log_alpha_hi[self._positions(meta.obs_id)]
