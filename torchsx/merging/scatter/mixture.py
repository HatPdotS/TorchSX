"""Mixture of two-width GenGauss components on the log-scale scatter.

One two-width generalised Gaussian cannot represent a distribution with a
sharp edge on one side and a long tail on the other without mislocating its
centre, and the mean correction computed from a wrong shape is then wrong
too.  This family keeps each component a two-width GenGauss -- so every
existing quadrature path and kernel is reused per component -- and lets a
small number of components with their own weights, offsets and side widths
describe the edge, the bulk and the tail separately.

The mixture is centred so that its mean multiplicative factor is one:
``log E[exp(eps)]`` has a closed form as the log-sum-exp of the per-component
corrections, and the caller subtracts it from ``log K + log I0`` exactly as
for the single family.  Component ``k`` is then centred at
``log_mean + offset_k``.

Per-observation parameters are addressed by ``ObsInfo.obs_id`` like the
single predicted family.  This class deliberately does not subclass
``GenGaussLogScatter``: the fast paths that test for that class would treat
the mixture as one component. The quadrature can integrate the mixture in a
single pass; other paths retain per-component calls on the child views.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter.base import ScatterModel
from torchsx.merging.scatter.gengauss import (
    gengauss_log_prob_from_obs,
    log_e_s_gengauss,
)
from torchsx.merging.scatter.predicted import PredictedGenGaussScatter

__all__ = [
    "PredictedMixtureScatter", "split_mix", "mixture_log_e_s",
    "split_gaussian_log_e_s", "two_width_moments", "mixture_log_sd", "N_BLOCKS",
]


N_BLOCKS = 5


def split_mix(mix: Tensor, n_components: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(n, 5K) -> (log_pi, offset, log_alpha_lo, log_alpha_hi, log_beta)``, each ``(n, K)``."""
    K = int(n_components)
    if mix.shape[-1] != N_BLOCKS * K:
        raise ValueError(f"mix has {mix.shape[-1]} columns, expected {N_BLOCKS} * {K}")
    return tuple(mix[:, i * K:(i + 1) * K] for i in range(N_BLOCKS))


def split_gaussian_log_e_s(log_alpha_lo: Tensor, log_alpha_hi: Tensor) -> Tensor:
    """``log E[exp(eps)]`` of a two-width Gaussian (GenGauss at ``beta = 2``), exact.

    Each side is a half normal with ``sigma = alpha / sqrt(2)`` carrying mass
    ``alpha_side / (alpha_lo + alpha_hi)``, so::

        E[exp(eps)] = 2 w_lo exp(s_lo^2 / 2) Phi(-s_lo) + 2 w_hi exp(s_hi^2 / 2) Phi(s_hi)

    The symmetric case reduces to ``alpha^2 / 4``.  Same value as
    :func:`log_e_s_gengauss` at ``beta = 2`` without its ``(n, 257)`` grid.

    The lower side is evaluated as ``exp(s^2 / 2) Phi(-s) = erfcx(s / sqrt 2) / 2``:
    written as ``s^2 / 2 + log_ndtr(-s)`` it cancels two terms of size ``s^2``,
    which in float32 loses every digit (and returns NaN gradients) once the
    lower width is in the thousands.  The upper side has no cancellation.
    """
    log_norm = torch.logaddexp(log_alpha_lo, log_alpha_hi) - math.log(2.0)
    s_hi = log_alpha_hi.exp() * math.sqrt(0.5)
    lo = log_alpha_lo - log_norm + torch.log(torch.special.erfcx(0.5 * log_alpha_lo.exp())) - math.log(2.0)
    hi = log_alpha_hi - log_norm + 0.5 * s_hi * s_hi + torch.special.log_ndtr(s_hi)
    return torch.logaddexp(lo, hi)


def mixture_log_e_s(
    log_pi: Tensor, offset: Tensor, log_alpha_lo: Tensor, log_alpha_hi: Tensor, beta: Tensor,
    *, gaussian: bool = False,
) -> Tensor:
    """``log E[exp(eps)]`` of the mixture, per observation.

    ``E[exp(eps)] = sum_k pi_k exp(offset_k) E_k[exp(eps_k)]`` with the
    per-component expectation from :func:`log_e_s_gengauss`.  ``beta`` is
    ``(n, K)`` per component or ``(n,)`` shared.  ``gaussian=True`` asserts
    that every component has ``beta = 2`` and uses the closed form
    :func:`split_gaussian_log_e_s`; ``beta`` is then not read.
    """
    if gaussian:
        return torch.logsumexp(
            log_pi + offset + split_gaussian_log_e_s(log_alpha_lo, log_alpha_hi), dim=1)
    terms = [
        log_pi[:, k] + offset[:, k]
        + log_e_s_gengauss(log_alpha_lo[:, k], beta[:, k] if beta.dim() == 2 else beta,
                           log_alpha_hi=log_alpha_hi[:, k])
        for k in range(log_pi.shape[1])
    ]
    return torch.logsumexp(torch.stack(terms, dim=1), dim=1)


def two_width_moments(log_alpha_lo: Tensor, log_alpha_hi: Tensor,
                      beta: Tensor | None) -> tuple[Tensor, Tensor]:
    """First and second raw moments of a two-width GenGauss centred at zero.

    With side widths ``a_lo`` below and ``a_hi`` above the centre and one
    shape ``b``::

        E[eps]   = (a_hi^2 - a_lo^2) / (a_lo + a_hi) * G(2/b) / G(1/b)
        E[eps^2] = (a_hi^3 + a_lo^3) / (a_lo + a_hi) * G(3/b) / G(1/b)
    """
    a_lo, a_hi = log_alpha_lo.exp(), log_alpha_hi.exp()
    if beta is None:                      # beta = 2: G(1)/G(1/2), G(3/2)/G(1/2)
        r2, r3 = 1.0 / math.sqrt(math.pi), 0.5
    else:
        g1 = torch.lgamma(1.0 / beta)
        r2 = torch.exp(torch.lgamma(2.0 / beta) - g1)
        r3 = torch.exp(torch.lgamma(3.0 / beta) - g1)
    s = a_lo + a_hi
    m1 = (a_hi * a_hi - a_lo * a_lo) / s * r2
    m2 = (a_hi ** 3 + a_lo ** 3) / s * r3
    return m1, m2


def mixture_log_sd(
    log_pi: Tensor, offset: Tensor, log_alpha_lo: Tensor, log_alpha_hi: Tensor, beta: Tensor,
    *, gaussian: bool = False,
) -> Tensor:
    """Log standard deviation of the mixture in log space, per observation.

    ``gaussian=True`` asserts ``beta = 2`` for every component (not read)."""
    pi = log_pi.exp()
    b = None if gaussian else (beta if beta.dim() == 2 else beta[:, None].expand_as(log_alpha_lo))
    m1, m2 = two_width_moments(log_alpha_lo, log_alpha_hi, b)
    mean_k = m1 + offset
    var_k = m2 - m1 * m1
    mean = (pi * mean_k).sum(dim=1)
    var = (pi * (var_k + mean_k * mean_k)).sum(dim=1) - mean * mean
    return 0.5 * torch.log(var.clamp(min=1e-12))


class PredictedMixtureScatter(ScatterModel):
    """Per-observation mixture of two-width GenGauss components.

    ``set_prediction(obs_id, mix)`` installs the batch: ``mix`` is ``(n, 5K)``
    with normalised ``log_pi``, ``offset``, ``log_alpha_lo``, ``log_alpha_hi``
    and ``log_beta`` blocks (see :func:`split_mix`), so every component has
    its own two widths and shape.  ``components(meta)`` yields, per component,
    the weight and offset gathered for ``meta`` and a child
    ``PredictedGenGaussScatter`` view that any GenGauss-only quadrature accepts.

    ``fixed_gaussian=True`` declares that beta is fixed at 2, enabling the
    fused split-grid kernel. It must not be set for a learned shape that
    merely happens to equal 2 at the current step.
    """

    def __init__(self, n_components: int, *, strict: bool = True,
                 fixed_gaussian: bool = False) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.strict = bool(strict)
        # An explicit model contract, never inferred from the current beta
        # values: a learned shape can pass through beta=2 during training.
        self.fixed_gaussian = bool(fixed_gaussian)
        self._mix: Tensor | None = None
        self.views = nn.ModuleList(
            [PredictedGenGaussScatter(strict=strict) for _ in range(self.n_components)]
        )
        self._sorted_ids: Tensor | None = None
        self._sort_perm: Tensor | None = None
        self._log_pi: Tensor | None = None
        self._offset: Tensor | None = None
        self._log_alpha_lo: Tensor | None = None
        self._log_alpha_hi: Tensor | None = None
        self._log_beta: Tensor | None = None

    # ── prediction plumbing ────────────────────────────────────────────

    def set_prediction(self, obs_id: Tensor, mix: Tensor) -> None:
        log_pi, offset, la_lo, la_hi, log_beta = split_mix(mix, self.n_components)
        if obs_id.shape != log_pi.shape[:1]:
            raise ValueError(f"shape mismatch: obs_id={tuple(obs_id.shape)} mix={tuple(mix.shape)}")
        if self.fixed_gaussian and not bool((log_beta == math.log(2.0)).all()):
            raise ValueError("fixed_gaussian requires every component beta to be fixed at 2")
        self._mix = mix
        for k, view in enumerate(self.views):
            view.set_prediction(obs_id, la_lo[:, k], log_beta[:, k], la_hi[:, k])
        sorted_ids, perm = torch.sort(obs_id)
        self._sorted_ids, self._sort_perm = sorted_ids, perm
        self._log_pi, self._offset = log_pi, offset
        self._log_alpha_lo, self._log_alpha_hi, self._log_beta = la_lo, la_hi, log_beta

    def clear_prediction(self) -> None:
        for view in self.views:
            view.clear_prediction()
        self._sorted_ids = self._sort_perm = None
        self._log_pi = self._offset = None
        self._log_alpha_lo = self._log_alpha_hi = self._log_beta = None
        self._mix = None

    @property
    def has_prediction(self) -> bool:
        return self._log_pi is not None

    def _positions(self, obs_id: Tensor) -> Tensor:
        if self._sorted_ids is None or self._sort_perm is None:
            raise RuntimeError(
                "PredictedMixtureScatter has no prediction installed; call "
                "set_prediction(obs_id, mix) first."
            )
        loc = torch.searchsorted(self._sorted_ids, obs_id)
        loc = loc.clamp(max=self._sorted_ids.numel() - 1)
        found = self._sorted_ids[loc] == obs_id
        if self.strict and not bool(found.all()):
            n_missing = int((~found).sum())
            raise KeyError(
                f"{n_missing} of {obs_id.numel()} obs_id values are outside the "
                f"installed prediction."
            )
        return self._sort_perm[loc]

    def components(self, meta: ObsInfo) -> list[tuple[Tensor, Tensor, PredictedGenGaussScatter]]:
        """``[(log_pi_k, offset_k, view_k)]`` with the weights and offsets gathered for ``meta``."""
        if meta.obs_id is None:
            raise ValueError("PredictedMixtureScatter needs ObsInfo.obs_id to address its prediction")
        pos = self._positions(meta.obs_id)
        return [
            (self._log_pi[pos, k], self._offset[pos, k], view)
            for k, view in enumerate(self.views)
        ]

    def mixture_obs(self, meta: ObsInfo) -> Tensor:
        """Gather all five parameter blocks with one observation lookup."""
        if meta.obs_id is None:
            raise ValueError("PredictedMixtureScatter needs ObsInfo.obs_id to address its prediction")
        pos = self._positions(meta.obs_id)
        return self._mix.index_select(0, pos)

    # ── the base contract: the dominant component stands in for the family ──

    def _dominant(self, meta: ObsInfo) -> tuple[Tensor, Tensor]:
        pos = self._positions(meta.obs_id)
        k = self._log_pi[pos].argmax(dim=1)
        return pos, k

    def log_alpha_beta_obs(self, params: Parameters, meta: ObsInfo) -> tuple[Tensor, Tensor]:
        pos, k = self._dominant(meta)
        return self._log_alpha_lo[pos, k], self._log_beta[pos, k]

    def log_alpha_hi_obs(self, params: Parameters, meta: ObsInfo) -> Tensor | None:
        pos, k = self._dominant(meta)
        return self._log_alpha_hi[pos, k]

    def log_prob(self, eps: Tensor, params: Parameters, meta: ObsInfo) -> Tensor:
        """Mixture log-density at ``eps`` ``(M, K_grid)`` for the observations in ``meta``."""
        pos = self._positions(meta.obs_id)
        terms = []
        for k in range(self.n_components):
            lp = gengauss_log_prob_from_obs(
                eps - self._offset[pos, k][:, None],
                self._log_alpha_lo[pos, k], self._log_beta[pos, k], self._log_alpha_hi[pos, k],
            )
            terms.append(self._log_pi[pos, k][:, None] + lp)
        return torch.logsumexp(torch.stack(terms, dim=0), dim=0)
