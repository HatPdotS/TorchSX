"""Group reductions over flat observation tables, and the fold assignment.

Observations are flat tensors with a group index (a crystal or a reflection)
per row.  :func:`group_sum` and :func:`group_count` reduce over it with one
``index_add_``; the merge targets build their sums on them.
:func:`fold_ids` deals each crystal's observations into ``K`` folds, the
assignment the fold transformer masks one fold at a time.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["group_sum", "group_count", "fold_ids"]


def group_sum(x: Tensor, gid: Tensor, n_groups: int) -> Tensor:
    """``out[g] = Σ_{j : gid[j] == g} x[j]``.

    ``x`` is ``(n,)`` or ``(n, D)``; the result is ``(n_groups,)`` or
    ``(n_groups, D)`` with ``x``'s dtype.
    """
    if x.ndim == 1:
        out = x.new_zeros(n_groups)
        return out.index_add_(0, gid, x)
    if x.ndim == 2:
        out = x.new_zeros(n_groups, x.shape[1])
        return out.index_add_(0, gid, x)
    raise ValueError(f"x must be 1-D or 2-D, got shape {tuple(x.shape)}")


def group_count(gid: Tensor, n_groups: int) -> Tensor:
    """Members per group, ``(n_groups,)`` int64."""
    return torch.bincount(gid, minlength=n_groups)


def fold_ids(gid: Tensor, n_groups: int, n_folds: int) -> Tensor:
    """Deterministic fold label per row: its rank within its group, mod ``n_folds``.

    Rows of one group are dealt round-robin into the folds **in the order they
    arrive**, so the assignment depends only on the relative order of a group's
    rows and not on how groups are interleaved.  A crystal-grouped batch and
    the hkl-sorted full table therefore give the same folds.

    Because the rows of a crystal arrive in hkl order, the folds interleave in
    resolution: both halves of a crystal span its whole diffraction pattern.
    """
    if gid.ndim != 1:
        raise ValueError(f"gid must be 1-D, got shape {tuple(gid.shape)}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    order = torch.argsort(gid, stable=True)
    counts = torch.bincount(gid, minlength=n_groups)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.empty_like(gid)
    pos[order] = (
        torch.arange(gid.numel(), device=gid.device) - starts.index_select(0, gid[order])
    )
    return pos % n_folds
