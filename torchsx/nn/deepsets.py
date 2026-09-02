"""Deep Sets encoder over a crystal's reflections — the abismal architecture.

Two branches meet at a pooled crystal context::

    per observation:  features
                         |
            +------------+------------+
            | CONTEXT branch          | SELF branch
            | (sees I, sigma)         | (metadata only)
            | obs blocks              | Linear(D)
            | POOL over crystal_id  --+--> added as bias
            |  (leave-one-out, or     |
            |   leave-one-FOLD-out)   |
            +------------+------------+
                         | obs blocks
                         v
                     embedding (n, D)

The asymmetry between the branches *is* the design.  The self branch may not
see an observation's own intensity — that would let the network fit ``K_j =
I_j / I0`` exactly and mean nothing.  The context branch may see every
intensity, because the pool removes observation ``j`` from the context it is
scored against.  So a crystal's mosaic state and partiality get inferred from
the pattern of its *other* reflections, which for this dataset is the only
place that information exists: integration used ``--fix-profile-radius=4e7``,
so ``profile_radius`` is constant across all crystals and the geometry carries
none of it.

Two pooling modes
-----------------
``context_folds = 0`` — **leave-one-out** (the original abismal port).  Every
observation's context is "crystal mean minus me".  Two observations of one
crystal differ in context by a single term in ~180, but they differ, so the
context is an ``(n, D)`` tensor and *every* block runs per observation.

``context_folds = K >= 2`` — **leave-one-fold-out**.  Observations are dealt
round-robin into ``K`` folds within their crystal; the context for fold ``f``
is the mean over the crystal's *other* folds.  The context is now a
``(crystals * K, D)`` table, computed once and read back by gather, and the
blocks that act on the context alone run on that table -- ~14 k rows instead of
~13 M.  The exclusion is *stronger* than leave-one-out (``j`` never enters the
sum it is scored against, so value and gradient are independent of ``x_ctx[j]``
bit-exactly) at the price of ``(K-1)/K`` of the peers instead of ``n-1``.

Where the depth lives is then a choice.  ``obs_context_layers`` blocks run per
observation before the pool (forming the products the pool needs, e.g. observed
scale x excitation error); the remaining context blocks run on the table.
Likewise ``n_layers - obs_self_layers`` self blocks run on the table before
the geometry is added and ``obs_self_layers`` after it.  Total depth, widths and
parameter names are unchanged, so the same ``state_dict`` describes both modes.

Crystal-level input head
------------------------
Per-crystal metadata (``ObsInfo.crystal_feat``: cell, photon energy, resolution
limit, reflection count, detector shift, peak count) enters the context table
once per crystal through ``input_crystal``, together with ``log(count)`` of the
peers the context was built from.  Zero-initialised, so it changes nothing at
step 0 and old checkpoints load.

Ported from abismal's ``ImageScaler``.  The ragged-tensor plumbing that
dominates the TensorFlow version collapses to an ``index_add_`` here, because
torchsx already represents observations as flat tensors plus a ``crystal_id``.

Swapping the encoder
--------------------
:class:`ScaleEncoder` is the seam.  Anything with that ``forward`` signature
drops in — e.g. a transformer over the padded ``(B, L)`` batches
:class:`~torchsx.nn.batch.StreamBatcher` already emits — without touching the
heads, the objective, or the merge target.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.nn.pooling import fold_ids, leave_fold_out_mean, leave_one_out_mean

__all__ = ["ScaleEncoder", "ResidualFF", "DeepSetsEncoder"]


@runtime_checkable
class ScaleEncoder(Protocol):
    """Anything that turns per-observation features into an embedding.

    ``forward(x_self, x_ctx, group_id, n_groups, x_crys=None) -> (n, d_model)``

    Implementations **must** guarantee that row ``j`` of the output does not
    depend on row ``j`` of ``x_ctx`` (only on its peers within ``group_id``).
    ``x_crys`` is an optional ``(n_groups, F)`` per-group metadata table.
    """

    d_model: int

    def __call__(
        self,
        x_self: Tensor,
        x_ctx: Tensor,
        group_id: Tensor,
        n_groups: int,
        x_crys: Tensor | None = None,
    ) -> Tensor: ...


class ResidualFF(nn.Module):
    """ResNet-v2 residual feed-forward block: ``x + W2·act(W1·act(norm(x)))``.

    Pre-norm, and the second projection is **zero-initialised** so the block
    starts as the identity.  A stack of these is therefore exactly a linear map
    at step 0, which is what makes the zero-init heads give ``K == 1`` on the
    first batch — i.e. the first merge is a unity merge, and
    ``partialator --model=unity`` already reaches CC1/2 0.985 on this data, so
    training starts from a serviceable point with no special-casing.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        *,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_hidden = d_hidden if d_hidden is not None else 2 * d_model
        self.norm = nn.RMSNorm(d_model)
        self.act = activation()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm(x))
        h = self.act(self.fc1(h))
        return x + self.dropout(self.fc2(h))


class DeepSetsEncoder(nn.Module):
    """abismal's ``ImageScaler`` trunk: context pool → self stack.

    Parameters
    ----------
    n_self_features
        Width of ``x_self`` — metadata the scale head may see directly.  Must
        **not** include the observation's own intensity.
    n_context_features
        Width of ``x_ctx`` — what goes into the pooled branch.  May include
        intensity and σ, because the pool removes ``j`` from its own context.
    d_model
        Embedding width.  Default 128.  (abismal's class default is 32 and its
        CLI default 256 — a factor of 8 apart, with the CLI value coming from a
        benchmark config that was subsequently reverted.  Scale up only if
        held-out likelihood says capacity binds.)
    n_layers
        Residual blocks per stack; ``2 * n_layers`` total.
    n_crystal_features
        Width of the per-crystal table ``x_crys`` read by the crystal-level
        input head.  ``0`` disables the metadata part of the head; the
        ``log(count)`` column is always present.
    context_folds
        ``0`` = leave-one-out (per-observation context, the original path).
        ``K >= 2`` = leave-one-fold-out with a per-(crystal, fold) context
        table.  See the module docstring.
    obs_context_layers, obs_self_layers
        How many of the ``n_layers`` context blocks run per observation
        *before* the pool, and how many of the ``n_layers`` self blocks run per
        observation *after* the geometry is added.  The rest run on the pooled
        context.  Defaults: all of them when ``context_folds == 0`` (the
        original architecture, bit-identical), ``2`` and ``2`` otherwise.
    pool_accum_dtype
        Accumulation dtype for the pooled mean.  ``None`` (default) keeps
        float32; see :mod:`torchsx.nn.pooling` on what that costs.
    checkpoint
        Recompute each **per-observation** residual block in the backward pass
        instead of storing its activations.  On by default: activations are the
        entire memory cost here (parameters are ~2.7 M weights, i.e. noise),
        and checkpointing turns ~10.9 MB per crystal into ~1.8 MB at
        D=256/L=5, which is what makes an 8192-crystal batch fit.  The
        crystal-level blocks are never checkpointed; they are too small to
        matter.
    """

    def __init__(
        self,
        n_self_features: int,
        n_context_features: int,
        *,
        d_model: int = 128,
        n_layers: int = 5,
        d_hidden: int | None = None,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        n_crystal_features: int = 0,
        context_folds: int = 0,
        obs_context_layers: int | None = None,
        obs_self_layers: int | None = None,
        pool_accum_dtype: torch.dtype | None = None,
        checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_self_features = int(n_self_features)
        self.n_context_features = int(n_context_features)
        self.n_crystal_features = int(n_crystal_features)
        self.pool_accum_dtype = pool_accum_dtype
        self.checkpoint = bool(checkpoint)

        folds = int(context_folds)
        if folds < 0 or folds == 1:
            raise ValueError(
                f"context_folds must be 0 (leave-one-out) or >= 2, got {context_folds}"
            )
        self.context_folds = folds
        # Fold-mode default 2+2, measured on SwissFEL dark against the
        # leave-one-out baseline (R_free 0.1269 / 0.1252 nll / ml):
        #   1+1  R_free 0.1252 / 0.1229 but weak-quartile slope 0.87 / 0.74
        #   2+2  R_free 0.1242 / 0.1221 and slope 0.98 / 0.99, CC1/2 0.9979
        # One per-observation block on each side is short of capacity for the
        # weak end; two match or beat the ten-block original at 2.4x less
        # encoder time.
        if obs_context_layers is None:
            obs_context_layers = self.n_layers if folds == 0 else min(2, self.n_layers)
        if obs_self_layers is None:
            obs_self_layers = self.n_layers if folds == 0 else min(2, self.n_layers)
        for name, v in (("obs_context_layers", obs_context_layers),
                        ("obs_self_layers", obs_self_layers)):
            if not (0 <= int(v) <= self.n_layers):
                raise ValueError(f"{name}={v} must lie in [0, n_layers={self.n_layers}]")
        self.obs_context_layers = int(obs_context_layers)
        self.obs_self_layers = int(obs_self_layers)

        self.input_context = nn.Linear(n_context_features, d_model)
        self.input_self = nn.Linear(n_self_features, d_model)
        # Crystal-level head: [x_crys, log(count)] -> D.  Zero-init so the
        # first forward is unchanged and pre-existing checkpoints stay valid.
        self.input_crystal = nn.Linear(self.n_crystal_features + 1, d_model)
        nn.init.zeros_(self.input_crystal.weight)
        nn.init.zeros_(self.input_crystal.bias)

        def _stack() -> nn.ModuleList:
            return nn.ModuleList(
                ResidualFF(
                    d_model, d_hidden, activation=activation, dropout=dropout
                )
                for _ in range(n_layers)
            )

        self.context_blocks = _stack()
        self.self_blocks = _stack()
        # The architecture travels WITH the weights.  Both pooling modes and
        # every depth split share the same parameter names, so a state dict
        # loads silently into the wrong forward -- which is exactly what
        # happened when to_mtz built a default encoder for a fold-trained
        # checkpoint: CC1/2 fine in training, R_free 0.127 -> 0.149 at merge.
        self.register_buffer(
            "_arch",
            torch.tensor([self.context_folds, self.obs_context_layers,
                          self.obs_self_layers], dtype=torch.long),
        )

    # ── checkpoint compatibility ───────────────────────────────────────

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints written before the crystal head lack these two keys.  The
        # head is zero-init, so filling zeros reproduces the old model exactly.
        for name in ("weight", "bias"):
            key = f"{prefix}input_crystal.{name}"
            if key not in state_dict:
                state_dict[key] = torch.zeros_like(getattr(self.input_crystal, name))
        key = f"{prefix}_arch"
        if key not in state_dict:
            # Checkpoint predates the record: nothing to check against, so the
            # caller's construction is taken at its word (to_mtz flags).
            state_dict[key] = self._arch.clone()
        want = self._arch.tolist()
        have = state_dict[key].tolist()
        if have != want:
            raise RuntimeError(
                "DeepSetsEncoder architecture mismatch: checkpoint has "
                f"(context_folds, obs_context_layers, obs_self_layers)={tuple(have)} "
                f"but this encoder is {tuple(want)}. Build the encoder with the "
                "checkpoint's settings (to_mtz reads them from ckpt['encoder'])."
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ── forward ────────────────────────────────────────────────────────

    def forward(
        self,
        x_self: Tensor,        # (n, n_self_features)
        x_ctx: Tensor,         # (n, n_context_features)
        group_id: Tensor,      # (n,) int64 crystal index
        n_groups: int,
        x_crys: Tensor | None = None,   # (n_groups, n_crystal_features)
    ) -> Tensor:               # (n, d_model)
        n = x_self.shape[0]
        if x_ctx.shape[0] != n or group_id.shape[0] != n:
            raise ValueError(
                f"row mismatch: x_self={n} x_ctx={x_ctx.shape[0]} "
                f"group_id={group_id.shape[0]}"
            )
        if self.n_crystal_features:
            if x_crys is None:
                raise ValueError(
                    f"encoder was built with n_crystal_features="
                    f"{self.n_crystal_features} but x_crys is None"
                )
            if x_crys.shape[1] != self.n_crystal_features:
                raise ValueError(
                    f"x_crys has {x_crys.shape[1]} columns, expected "
                    f"{self.n_crystal_features}"
                )
        L_c, L_s = self.obs_context_layers, self.obs_self_layers

        # Per-observation part of the context branch.
        h = self._run(self.context_blocks[:L_c], self.input_context(x_ctx), obs=True)

        if self.context_folds == 0:
            # Leave-one-out: the context is per observation, so everything
            # downstream of the pool is per observation too.  ``fill=0.0`` is
            # the right value for a singleton crystal: "my peers tell me
            # nothing" is the same statement as a zero bias into an additive
            # branch.
            ctx, _ = leave_one_out_mean(
                h, group_id, n_groups, fill=0.0, accum_dtype=self.pool_accum_dtype
            )
            cnt = (torch.bincount(group_id, minlength=n_groups) - 1).index_select(0, group_id)
            crys = x_crys.index_select(0, group_id) if x_crys is not None else None
            ctx = ctx + self._crystal_head(crys, cnt)
            ctx = self._run(self.context_blocks[L_c:], ctx, obs=True)
            ctx = self._run(self.self_blocks[: self.n_layers - L_s], ctx, obs=True)
        else:
            K = self.context_folds
            fold = fold_ids(group_id, n_groups, K)
            # Compact to the crystals present in this batch: ``n_groups`` is the
            # dataset count (~150 k), and under chunked prediction the table
            # would otherwise be rebuilt at full size for every chunk.
            present = torch.bincount(group_id, minlength=n_groups) > 0
            local = torch.cumsum(present.long(), 0) - 1
            lid = local.index_select(0, group_id)
            orig = present.nonzero().squeeze(1)
            G = int(orig.numel())
            table, cnt, _ = leave_fold_out_mean(
                h, lid, G, fold, K, fill=0.0, accum_dtype=self.pool_accum_dtype
            )
            crys = (
                x_crys.index_select(0, orig).repeat_interleave(K, dim=0)
                if x_crys is not None else None
            )
            table = table + self._crystal_head(crys, cnt)
            # Crystal-level depth: on ~G*K rows, never checkpointed.
            table = self._run(self.context_blocks[L_c:], table, obs=False)
            table = self._run(self.self_blocks[: self.n_layers - L_s], table, obs=False)
            ctx = table.index_select(0, lid * K + fold)

        return self._run(self.self_blocks[self.n_layers - L_s:],
                         self.input_self(x_self) + ctx, obs=True)

    def _crystal_head(self, crys: Tensor | None, count: Tensor) -> Tensor:
        logc = torch.log(count.to(self.input_crystal.weight.dtype).clamp(min=1.0))[:, None]
        if crys is None:
            crys = logc.new_zeros(logc.shape[0], self.n_crystal_features)
        return self.input_crystal(torch.cat([crys, logc], dim=1))

    def _run(self, blocks, x: Tensor, *, obs: bool) -> Tensor:
        use_ckpt = obs and self.checkpoint and self.training and torch.is_grad_enabled()
        for blk in blocks:
            x = (
                torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                if use_ckpt else blk(x)
            )
        return x
