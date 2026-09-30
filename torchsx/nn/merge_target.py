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

__all__ = ["MergeTargetOut", "unity_merge"]


def unity_merge(
    I: Tensor, sigma: Tensor, hkl_id: Tensor, n_hkl: int, *, floor: float = 1e-3
) -> Tensor:
    """Inverse-variance merge at unit scale: the start of the merge grid and
    the per-reflection brightness the peer features carry.

    Deliberately **linear and over all observations**, including the negatives.
    ``PreparedObs.mu_init`` is a log-space MLE fitted to *positive* observations
    only; with 32.4 % of this dataset negative, exponentiating it gives a target
    biased high, and every observation then sits systematically low against it.
    That is not a subtle effect — seeding from ``exp(mu_init)`` put the initial
    normalised residual at ``var_z = 31`` on the real dark, against a target
    of 1.

    At ``K == 1`` this is the unity merge; ``partialator --model=unity``
    reaches CC1/2 0.985 on this data.
    """
    # The K == 1 case of the same observation-space regression.
    w = 1.0 / sigma.clamp(min=1e-6) ** 2
    num = group_sum(w * I, hkl_id, n_hkl)
    den = group_sum(w, hkl_id, n_hkl)
    return torch.where(den > 0, num / den.clamp(min=1e-12), torch.zeros_like(num)).clamp(
        min=floor
    )


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
