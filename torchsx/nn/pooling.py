"""Leave-one-out group reductions — the architectural centrepiece.

**Every quantity used to score observation ``j`` is computed from everything
except ``j``.**  Applied consistently this turns the training objective into a
cross-fitted, out-of-sample prediction, which is what stops the whole scheme
becoming a tautology.  It appears at two levels, and both are the same
sum-minus-self trick on an accumulator that has to be computed anyway:

============== ============================ =======================================
level          what is excluded             what it prevents
============== ============================ =======================================
context pool   ``j`` from its own crystal's the network reading ``j``'s own
               pooled embedding             intensity to set ``j``'s own scale
merge target   ``j`` from ``ȳ`` for its     the network lowering the loss by
               reflection                   dragging the merged estimate toward ``j``
============== ============================ =======================================

Without it both halves are degenerate.  A scale predicted from an observation's
own intensity fits that observation perfectly and means nothing; a merge that
includes the observation being scored can always be pulled toward it.  With
leave-one-out the only way to lower the loss is to make observations *mutually
predictable* — predict ``j`` from the crystal's other reflections and have that
agree with what the other crystals say about that reflection.

Two caveats worth carrying:

* **The guarantee is per forward pass, not per parameter.**  The shared weights
  were trained on ``j`` at earlier steps.  That residual leakage is what the
  held-out-*crystals* diagnostic is for.
* **"Independent" is doing real work.**  Observations of one reflection from one
  crystal share that crystal's scale error, so their agreement is inflated.
  Check for systematic sources (Friedel mates mapping to the same ASU index,
  multi-crystal frames indexed twice).

Numerics — float32 throughout
-----------------------------
Everything here runs in float32; float64 appears only as a test/diagnostic
option.  Two float32 facts shape the API.

**1. Sum-minus-self is exact in real arithmetic but not in floating point.**
The group sum ``S`` includes ``x[j]``, so perturbing ``x[j]`` perturbs how ``S``
rounds and ``S − x[j]`` moves by a rounding-scale amount (measured: ~1.7e-5 for
float32 activations under a gross 1e3 perturbation; 0.0 if accumulated in
float64).  So the guarantee has two strengths, and the strong one is the one
that matters:

* **Gradient independence is exact, always.**  ``∂out[j]/∂x[j] = 1/(c−1) −
  1/(c−1) = 0`` cancels bit-exactly, in float32 as in float64.  An optimiser
  moves only along gradients, so there is no path by which training can *learn*
  to read an observation's own value.  Asserted in the tests.
* **Forward independence holds to rounding.**  The residue is deterministic
  rounding noise, far below any real feature variation, and carries no gradient.

**2. Cancellation is the one real float32 hazard, and it is detected, not
avoided.**  The summands ``w·y`` span several decades (99.4 % of this dataset's
variance sits in I/σ > 20 observations).  On realistic groups float32 holds a
median relative error of ~1e-7 against float64, so it is simply fine.  It fails
only when one observation carries essentially the whole group weight — and
there :func:`loo_weighted_mean` marks the observation **invalid** rather than
returning silent garbage, because such a group carries no independent peer
information about that observation anyway.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "group_sum",
    "group_count",
    "leave_one_out_mean",
    "fold_ids",
    "leave_fold_out_mean",
    "loo_weighted_mean",
]


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


def leave_one_out_mean(
    x: Tensor,
    gid: Tensor,
    n_groups: int,
    *,
    fill: float = 0.0,
    accum_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Mean of each member's group **excluding that member**.

    ``out[j] = (Σ_{i : gid[i] == gid[j]} x[i]  −  x[j]) / (count[gid[j]] − 1)``

    Parameters
    ----------
    x
        ``(n,)`` or ``(n, D)`` — the values to pool.
    gid
        ``(n,)`` int64 group index (e.g. ``crystal_id``), values in
        ``[0, n_groups)``.
    n_groups
        Size of the accumulator.  May exceed the number of groups actually
        present; empty groups simply never get read.
    fill
        Value written for members of **singleton** groups, where the
        leave-one-out mean is undefined.  ``0.0`` is the right default for a
        context embedding: "no information from my peers" is the same statement
        as a zero bias into an additive branch.
    accum_dtype
        Accumulate in this dtype (e.g. ``torch.float64``) then cast back.
        ``None`` (default) keeps ``x``'s dtype — safe for O(1) activations.

    Returns
    -------
    (out, valid)
        ``out`` matches ``x``'s shape and dtype.  ``valid`` is ``(n,)`` bool,
        ``False`` exactly where the group has a single member and ``fill`` was
        used — callers that must not silently consume the fill can check it.

    Notes
    -----
    ``∂out[j]/∂x[j] == 0`` exactly (the two ``1/(c−1)`` terms cancel bit-for-bit),
    while ``∂out[peer]/∂x[j] == 1/(c−1)``: a member shapes its peers' contexts
    but never its own.  The forward value carries a rounding-scale dependence on
    ``x[j]`` unless ``accum_dtype=torch.float64`` — see the module docstring.
    """
    if gid.ndim != 1:
        raise ValueError(f"gid must be 1-D, got shape {tuple(gid.shape)}")
    if x.shape[0] != gid.shape[0]:
        raise ValueError(
            f"x has {x.shape[0]} rows but gid has {gid.shape[0]} entries"
        )

    work = x if accum_dtype is None else x.to(accum_dtype)
    sums = group_sum(work, gid, n_groups)
    counts = group_count(gid, n_groups)

    num = sums.index_select(0, gid) - work
    den = (counts - 1).to(work.dtype)
    valid = counts.index_select(0, gid) > 1
    den_j = den.index_select(0, gid).clamp(min=1.0)

    if work.ndim == 2:
        out = num / den_j[:, None]
        out = torch.where(valid[:, None], out, torch.full_like(out, fill))
    else:
        out = num / den_j
        out = torch.where(valid, out, torch.full_like(out, fill))

    return out.to(x.dtype), valid


def fold_ids(gid: Tensor, n_groups: int, n_folds: int) -> Tensor:
    """Deterministic fold label per row: its rank within its group, mod ``n_folds``.

    Rows of one group are dealt round-robin into the folds **in the order they
    arrive**, so the assignment depends only on the relative order of a group's
    rows and not on how groups are interleaved.  A crystal-grouped batch and
    the hkl-sorted full table therefore give the same folds, and so does every
    chunk of a chunked pass -- which is what lets the fold pool replace
    leave-one-out under :meth:`AmortizedScale.predict` without changing a value.

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


def leave_fold_out_mean(
    x: Tensor,
    gid: Tensor,
    n_groups: int,
    fold: Tensor,
    n_folds: int,
    *,
    fill: float = 0.0,
    accum_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean over each group **excluding one fold**, as a per-(group, fold) table.

    ``table[g * n_folds + f] = mean{ x[i] : gid[i] == g, fold[i] != f }``

    The crystal-level counterpart of :func:`leave_one_out_mean`.  Leave-one-out
    gives every member its own context -- the group mean minus itself -- which
    differs between two members of one group by a single term but still has to
    be materialised and processed per member.  Leaving a *fold* out instead
    gives every member of fold ``f`` the same context, so the context is a
    group-level object: ``n_groups * n_folds`` rows, processed once, and read
    back per member with ``table.index_select(0, gid * n_folds + fold)``.

    The exclusion is **stronger** than leave-one-out, not weaker.  Member ``j``
    never enters the sum it is scored against, so both the forward value and
    the gradient are independent of ``x[j]`` bit-exactly -- there is no
    sum-minus-self cancellation to round.  What is given up is peers:
    ``(n_folds - 1) / n_folds`` of the group instead of ``n - 1``.  At ~180
    observations per crystal (SwissFEL) or ~2000 (SACLA) that is immaterial.

    Returns
    -------
    (table, count, valid)
        ``table`` is ``(n_groups * n_folds, D)`` in ``x``'s dtype; ``count`` the
        ``(n_groups * n_folds,)`` number of members each mean was taken over;
        ``valid`` is ``False`` where that count is zero and ``fill`` was
        written -- a group with fewer members than folds, or a group absent
        from this batch entirely.
    """
    if gid.ndim != 1 or fold.ndim != 1:
        raise ValueError("gid and fold must be 1-D")
    if not (x.shape[0] == gid.shape[0] == fold.shape[0]):
        raise ValueError(
            f"length mismatch: x={x.shape[0]} gid={gid.shape[0]} fold={fold.shape[0]}"
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    work = x if accum_dtype is None else x.to(accum_dtype)
    cell = gid * n_folds + fold
    n_cells = n_groups * n_folds
    cell_sum = group_sum(work, cell, n_cells)
    cell_cnt = group_count(cell, n_cells)

    shape = (n_groups, n_folds) + tuple(work.shape[1:])
    cell_sum = cell_sum.view(shape)
    cell_cnt = cell_cnt.view(n_groups, n_folds)
    # Complement of fold f = the sum over the OTHER folds, formed directly.
    # ``total - fold_f`` is the same number in exact arithmetic but not in
    # floating point: the total contains fold f's own values, so their
    # rounding would leak into the complement at 1e-7 -- exactly the
    # sum-minus-self residue leave-one-out has and this pool must not.
    others = [
        [f2 for f2 in range(n_folds) if f2 != f] for f in range(n_folds)
    ]
    comp_sum = torch.stack([cell_sum[:, o].sum(dim=1) for o in others], dim=1)
    comp_cnt = torch.stack([cell_cnt[:, o].sum(dim=1) for o in others], dim=1)

    count = comp_cnt.reshape(n_cells)
    valid = count > 0
    den = count.to(work.dtype).clamp(min=1.0)
    table = comp_sum.reshape(n_cells, *work.shape[1:])
    if table.ndim == 2:
        table = table / den[:, None]
        table = torch.where(valid[:, None], table, torch.full_like(table, fill))
    else:
        table = table / den
        table = torch.where(valid, table, torch.full_like(table, fill))
    return table.to(x.dtype), count, valid


def loo_weighted_mean(
    y: Tensor,
    w: Tensor,
    gid: Tensor,
    n_groups: int,
    *,
    fill: float | Tensor = 0.0,
    min_weight: float = 1e-12,
    max_rel_error: float = 1e-3,
    prior_num: Tensor | None = None,
    prior_den: Tensor | None = None,
    accum_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Inverse-variance group mean **excluding each member** — the merge target.

    ``ȳ⁻ʲ = (Σ_i w_i y_i − w_j y_j) / (Σ_i w_i − w_j)``  over ``i`` in ``j``'s group.

    This is the leave-one-out merged intensity: every observation is scored
    against a target its peers set, not one it helped set.  ``w`` is ``1/τ²`` —
    a *linear* inverse-variance regression, deliberately not a log-space mean:
    32.5 % of observations here are negative, and any merge that takes logs
    discards them and underestimates the spread (measured: α 1.25 log-space
    positives-only vs 1.65 Poisson all-observations).

    Runs in float32.  On realistic data (77 observations per reflection,
    heavy-tailed ``w``) that costs a median relative error of ~1e-7 against a
    float64 reference.  It only breaks when a *single* observation carries
    essentially the whole group weight, and ``max_rel_error`` detects exactly
    that — see Notes.

    Parameters
    ----------
    y, w
        ``(n,)`` values and non-negative weights.
    gid
        ``(n,)`` int64 group index (e.g. ``hkl_id``).
    fill
        Scalar or ``(n,)`` tensor used wherever the estimate is undefined.  Pass
        a per-observation prior/EMA estimate here rather than a constant.
    min_weight
        Floor on the leave-one-out denominator below which the estimate is
        treated as undefined.
    max_rel_error
        Cancellation guard.  Observations whose estimated relative rounding
        error exceeds this are marked **invalid** and take ``fill``.  Set to
        ``inf`` to disable.
    prior_num, prior_den
        Optional ``(n_groups,)`` pseudo-observation added to the numerator and
        denominator *before* the leave-one-out subtraction.  This is how the
        EMA blend is applied::

            prior_den = lambda * den_EMA
            prior_num = lambda * den_EMA * I0_EMA

        giving ``(den_batch·ȳ_batch + λ·den_EMA·I0_EMA)/(den_batch + λ·den_EMA)``
        exactly, while still excluding observation ``j`` from its own target.
        Adding a term that does not depend on ``j`` cannot break leave-one-out.
        λ→∞ recovers a fixed EM target, λ→0 full in-batch merging.
    accum_dtype
        ``None`` (default) keeps ``y``'s dtype — float32 in production.
        float64 is available for reference checks in tests.

    Returns
    -------
    (ybar_loo, valid)
        Both ``(n,)``; ``ybar_loo`` has ``y``'s dtype.

    Notes
    -----
    **The cancellation guard.**  ``Σ w·y − w_j·y_j`` is a difference of
    like-signed numbers, so it loses precision in proportion to how much of the
    sum ``j`` itself contributes.  With ``w_j`` dominant, float32 returns
    garbage *silently* — in an adversarial check (one member at ``w = 1e9``,
    ``y = 1e6``) the true LOO target 1.0 comes back as 0.0.  The guard estimates
    the surviving precision as ``ε·|Σ w·y| / |Σ w·y − w_j·y_j|`` and invalidates
    the observation when that exceeds ``max_rel_error``.

    This is the statistically right call as well as the numerically right one:
    a group in which one observation holds ~all the weight carries no
    independent peer information about that observation, so there is no target
    to score it against.  Such observations take ``fill`` (the EMA I₀) and are
    reported through ``valid`` rather than quietly corrupting the loss.
    """
    if not (y.ndim == w.ndim == gid.ndim == 1):
        raise ValueError("y, w and gid must all be 1-D")
    if not (y.shape[0] == w.shape[0] == gid.shape[0]):
        raise ValueError(
            f"length mismatch: y={y.shape[0]} w={w.shape[0]} gid={gid.shape[0]}"
        )

    yy = y if accum_dtype is None else y.to(accum_dtype)
    ww = w if accum_dtype is None else w.to(accum_dtype)

    wy = ww * yy
    acc_num = group_sum(wy, gid, n_groups)
    acc_den = group_sum(ww, gid, n_groups)
    if prior_num is not None:
        acc_num = acc_num + prior_num.to(acc_num.dtype)
    if prior_den is not None:
        acc_den = acc_den + prior_den.to(acc_den.dtype)

    sum_wy = acc_num.index_select(0, gid)
    num = sum_wy - wy
    den = acc_den.index_select(0, gid) - ww

    valid = den > min_weight
    if max_rel_error < float("inf"):
        eps = torch.finfo(yy.dtype).eps
        tiny = torch.finfo(yy.dtype).tiny
        rel_err = eps * sum_wy.abs() / num.abs().clamp(min=tiny)
        valid = valid & (rel_err < max_rel_error)

    out = num / den.clamp(min=min_weight)

    fill_t = (
        fill.to(out.dtype) if isinstance(fill, Tensor)
        else torch.full_like(out, float(fill))
    )
    out = torch.where(valid, out, fill_t)
    return out.to(y.dtype), valid
