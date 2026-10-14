"""The unity merge and the per-observation record of a batch merge.

**Everything happens in observation space.  The data are never scaled.**
``I_true`` is scaled *up* by ``K`` to meet the observation; the observation is
never divided down to meet ``I_true``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torchsx.nn.pooling import group_sum

__all__ = ["MergeTargetOut", "unity_merge", "plain_mean", "add_unity_sums",
           "unity_from_sums", "add_plain_sums", "plain_mean_from_sums"]


def unity_merge(
    I: Tensor, sigma: Tensor, hkl_id: Tensor, n_hkl: int, *, floor: float = 1e-3
) -> Tensor:
    """Inverse-variance merge at unit scale: the start of the merge grid and
    the per-reflection brightness the peer features carry.

    Deliberately **linear and over all observations**, including the negatives.
    ``PreparedObs.mu_init`` is a log-space MLE fitted to *positive* observations
    only; with a large share of observations negative, exponentiating it gives
    a target biased high, and every observation then sits systematically low
    against it.  That is not a subtle effect — seeding from ``exp(mu_init)``
    puts the initial normalised residual an order of magnitude above its
    target of 1.
    """
    # The K == 1 case of the same observation-space regression.
    w = 1.0 / sigma.clamp(min=1e-6) ** 2
    num = group_sum(w * I, hkl_id, n_hkl)
    den = group_sum(w, hkl_id, n_hkl)
    return unity_from_sums(num, den, floor=floor)


def add_unity_sums(num: Tensor, den: Tensor, I: Tensor, sigma: Tensor, hkl_id: Tensor) -> None:
    """Add one block of observations to the sums of :func:`unity_merge`.

    Blocks added in order give the sums of the whole set, to the last bit.
    """
    w = 1.0 / sigma.clamp(min=1e-6) ** 2
    num.index_add_(0, hkl_id, w * I)
    den.index_add_(0, hkl_id, w)


def unity_from_sums(num: Tensor, den: Tensor, *, floor: float = 1e-3) -> Tensor:
    """:func:`unity_merge` from its sums ``sum(w I)`` and ``sum(w)``."""
    return torch.where(den > 0, num / den.clamp(min=1e-12), torch.zeros_like(num)).clamp(
        min=floor
    )


def plain_mean(I: Tensor, hkl_id: Tensor, n_hkl: int, *, floor: float = 1e-3) -> Tensor:
    """Per-reflection mean of the background-subtracted counts, every observation
    weighted equally, floored at ``floor``.

    Unweighted, so it is not pulled toward the weakest observations of a
    reflection the way an inverse-variance mean is when the counting variance
    grows with the signal.
    """
    num = group_sum(I.float(), hkl_id, n_hkl)
    cnt = group_sum(torch.ones_like(I, dtype=torch.float32), hkl_id, n_hkl)
    return plain_mean_from_sums(num, cnt, floor=floor)


def add_plain_sums(num: Tensor, cnt: Tensor, I: Tensor, hkl_id: Tensor) -> None:
    """Add one block of observations to the sums of :func:`plain_mean`."""
    num.index_add_(0, hkl_id, I.float())
    cnt.index_add_(0, hkl_id, torch.ones_like(I, dtype=torch.float32))


def plain_mean_from_sums(num: Tensor, cnt: Tensor, *, floor: float = 1e-3) -> Tensor:
    """:func:`plain_mean` from its sums ``sum(I)`` and the observation count."""
    return torch.where(cnt > 0, num / cnt.clamp(min=1.0), torch.zeros_like(num)).clamp(min=floor)


@dataclass
class MergeTargetOut:
    """Per-observation batch merge plus everything the metrics want."""

    log_I0: Tensor       # (n,) log of the reflection's batch posterior mean
    I0: Tensor           # (n,) the target itself
    resid: Tensor        # (n,) n' - K*I0, in OBSERVATION space
    var: Tensor          # (n,) V_j, the observation-space variance
    z: Tensor            # (n,) resid / sqrt(V) — the normalised residual
    valid: Tensor        # (n,) bool — a usable posterior mean existed
    n_floored: int       # how many targets hit the positivity floor
    var_frac_scatter: Tensor  # (n,) share of V from predicted scatter
