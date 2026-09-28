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

Peer cross-attention
--------------------
``peer_attention_layers = N > 0`` adds ``N`` cross-attention blocks between the
self input and the last self blocks.  Each observation's **query** -- its own
geometry plus the pooled context above -- reads the **keys and values** of the
other observations of its crystal, which are their per-observation context
encodings (the output of the ``obs_context_layers`` blocks, before any pool).

The mean pool gives every member of a fold the same summary of its crystal, so
anything specific to where ``j`` sits (whether the peers that need nearly the
same wavelength as ``j`` came out strong, say) has to survive the averaging.
Attention lets the weights depend on ``j``.

The exclusion is exact, and needs no folds: keys and values are functions of
one observation each and never see another token, only the query stream is
updated from layer to layer, and the diagonal is masked.  So no path leads
from ``x_ctx[j]`` to output ``j`` -- the same bit-exact guarantee as the fold
pool, over all ``n - 1`` peers.  (Updating the peers' own representations by
attention among themselves would break it: peer ``i`` would carry ``x_ctx[j]``
back to ``j``.)  Crystals are packed into padded ``(B, L)`` blocks, bucketed
by size so that ``B * L * L`` stays within ``peer_budget`` whatever the batch.
The output projection is zero-initialised: at step 0 the encoder is exactly
the one without attention.

Swapping the encoder
--------------------
:class:`ScaleEncoder` is the seam.  Anything with that ``forward`` signature
drops in — e.g. a transformer over padded ``(B, L)`` batches — without touching
the heads, the objective, or the merge target.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.nn.pooling import fold_ids, leave_fold_out_mean, leave_one_out_mean

__all__ = ["ScaleEncoder", "ResidualFF", "DeepSetsEncoder", "PeerCrossAttention", "peer_layout",
           "peer_masks"]


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


def peer_layout(group_id: Tensor, budget: int = 1 << 24) -> list[tuple[Tensor, Tensor]]:
    """Pack each crystal's rows into padded ``(B, L)`` index blocks for attention.

    Crystals are sorted by size and dealt into buckets.  A bucket's padded
    length ``L`` is its largest crystal rounded up to a multiple of 8, and a
    bucket closes before ``B * L * L`` would exceed ``budget``, so the
    ``(B, 1, L, L)`` mask is bounded whatever the batch (a crystal too large
    for the budget gets a bucket of its own).  Rows need not be grouped by
    crystal.  Returns ``[(idx, valid)]``: ``idx`` the ``(B, L)`` flat row of
    each slot (a valid row of the same crystal where padded), ``valid`` the
    ``(B, L)`` slots that hold a row.  Every row appears in exactly one valid
    slot.
    """
    order = torch.argsort(group_id, stable=True)
    _, counts = torch.unique_consecutive(group_id[order], return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    by_size = torch.argsort(counts, stable=True)
    sizes = counts[by_size].tolist()             # one host sync per forward
    blocks: list[tuple[Tensor, Tensor]] = []
    i, G = 0, len(sizes)
    while i < G:
        j, L = i, 0
        while j < G:
            L_j = -(-sizes[j] // 8) * 8
            if j > i and (j - i + 1) * L_j * L_j > budget:
                break
            L, j = L_j, j + 1
        sel = by_size[i:j]
        c, st = counts[sel], starts[sel]
        ar = torch.arange(L, device=group_id.device)
        valid = ar[None, :] < c[:, None]
        pos = st[:, None] + torch.minimum(ar[None, :], (c - 1)[:, None])
        blocks.append((order[pos], valid))
        i = j
    return blocks


def peer_masks(valid: Tensor) -> tuple[Tensor, Tensor]:
    """``(allow, empty)`` of one packed block: keys a slot may attend to, and slots with none.

    ``allow`` is ``(B, 1, L, L)``: valid slots of the same crystal, never the
    slot itself.  A slot with no peer (a single-row crystal, or padding) is
    allowed only itself, so its softmax stays finite; ``empty`` (``(B, L, 1)``)
    marks it for zeroing.
    """
    L = valid.shape[1]
    eye = torch.eye(L, dtype=torch.bool, device=valid.device)
    allow = valid[:, None, :] & ~eye[None]
    empty = ~allow.any(dim=-1)
    allow = allow | (empty[:, :, None] & eye[None])
    return allow[:, None], empty[..., None]


class PeerCrossAttention(nn.Module):
    """Pre-norm cross-attention of the query stream onto the peers, plus a residual FF.

    ``forward(q, kv, allow, empty)`` on packed ``(B, L, D)`` blocks (one
    crystal per row, :func:`peer_layout`), masks from :func:`peer_masks`: each
    slot attends to the valid slots of its crystal except itself, and a slot
    with no peer gets no attention output.  The output projection is
    zero-initialised, so the block starts as the identity.
    """

    def __init__(self, d_model: int, n_heads: int, d_hidden: int | None = None, *,
                 activation: type[nn.Module] = nn.SiLU, dropout: float = 0.0,
                 bf16: bool = False, d_attn: int | None = None) -> None:
        super().__init__()
        #: width inside the block: queries, keys and values are projected from
        #: ``d_model`` down to ``d_attn`` and the output back up, and the block's
        #: feed-forward is ``2 d_attn`` wide -- about ``8 d_model d_attn``
        #: parameters, so a narrow block buys depth at a fixed budget
        d_attn = int(d_attn or d_model)
        if d_attn % n_heads:
            raise ValueError(f"attention width {d_attn} not divisible by n_heads {n_heads}")
        self.d_attn = d_attn
        self.n_heads, self.d_head = int(n_heads), d_attn // n_heads
        #: attention in bfloat16 on CUDA (tensor cores); projections, norms and
        #: the residual stream stay float32, and the mask keeps the exclusion exact
        self.bf16 = bool(bf16)
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_kv = nn.RMSNorm(d_model)
        self.w_q = nn.Linear(d_model, d_attn, bias=False)
        self.w_k = nn.Linear(d_model, d_attn, bias=False)
        self.w_v = nn.Linear(d_model, d_attn, bias=False)
        self.w_o = nn.Linear(d_attn, d_model)
        nn.init.zeros_(self.w_o.weight)
        nn.init.zeros_(self.w_o.bias)
        self.ff = ResidualFF(d_model, d_hidden if d_attn == d_model else 2 * d_attn,
                             activation=activation, dropout=dropout)

    def forward(self, q: Tensor, kv: Tensor, allow: Tensor, empty: Tensor) -> Tensor:
        B, L, _ = q.shape
        H, dh, D = self.n_heads, self.d_head, self.d_attn
        heads = lambda x: x.view(B, L, H, dh).transpose(1, 2)   # noqa: E731
        kvn = self.norm_kv(kv)
        Q, K, V = heads(self.w_q(self.norm_q(q))), heads(self.w_k(kvn)), heads(self.w_v(kvn))
        if self.bf16 and q.is_cuda:
            Q, K, V = Q.to(torch.bfloat16), K.to(torch.bfloat16), V.to(torch.bfloat16)
        a = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=allow).to(q.dtype)
        a = a.transpose(1, 2).reshape(B, L, D).masked_fill(empty, 0.0)
        return self.ff(q + self.w_o(a))


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
    peer_attention_layers, peer_heads, peer_dim, peer_budget
        Cross-attention blocks from each observation onto the other
        observations of its crystal (module docstring), their head count, their
        internal width (``None`` = ``d_model``; one width, or one per layer for a
        tapered stack, wide first; see :class:`PeerCrossAttention`; a narrow layer
        runs ``min(peer_heads, width // 8)`` heads, so heads stay 8 wide),
        and the ``B * L * L`` element budget of one packed attention call.
        ``0`` layers is the encoder without attention.
    peer_compile, peer_bf16
        Run the packed attention stack through ``torch.compile(dynamic=True)``
        on CUDA (the stack is many small elementwise ops around one attention
        call per layer), and the attention itself in bfloat16 on CUDA.
        ``None`` reads ``TORCHSX_PEER_COMPILE`` / ``TORCHSX_PEER_BF16``
        (default on).  Runtime choices, not part of the architecture or the
        state dict.
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
        peer_attention_layers: int = 0,
        peer_heads: int = 8,
        peer_dim: int | list[int] | None = None,
        peer_budget: int = 1 << 24,
        peer_compile: bool | None = None,
        peer_bf16: bool | None = None,
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
        self.peer_attention_layers = int(peer_attention_layers)
        if self.peer_attention_layers < 0:
            raise ValueError(f"peer_attention_layers must be >= 0, got {peer_attention_layers}")
        self.peer_heads = int(peer_heads) if self.peer_attention_layers else 0
        N = self.peer_attention_layers
        dims = ([int(v) for v in peer_dim] if isinstance(peer_dim, (list, tuple))
                else [int(peer_dim or d_model)] * N)
        if len(dims) != N:
            raise ValueError(f"{len(dims)} attention widths for {N} attention layers")
        self.peer_dims = dims
        #: the width record: one int for a uniform stack, the per-layer list for a tapered one
        self.peer_dim = (dims[0] if len(set(dims)) == 1 else list(dims)) if N else 0
        self.peer_budget = int(peer_budget)
        if peer_compile is None:
            peer_compile = os.environ.get("TORCHSX_PEER_COMPILE", "1") != "0"
        self.peer_compile = bool(peer_compile)
        self._peer_compiled = None
        if peer_bf16 is None:
            peer_bf16 = os.environ.get("TORCHSX_PEER_BF16", "1") != "0"
        self.peer_blocks = nn.ModuleList(
            PeerCrossAttention(d_model, self.peer_heads if w == d_model else min(self.peer_heads, max(1, w // 8)),
                               d_hidden, activation=activation, dropout=dropout, bf16=peer_bf16, d_attn=w)
            for w in dims
        )
        # The architecture travels WITH the weights.  Both pooling modes and
        # every depth split share the same parameter names, so a state dict
        # loads silently into the wrong forward -- which is exactly what
        # happened when to_mtz built a default encoder for a fold-trained
        # checkpoint: CC1/2 fine in training, R_free 0.127 -> 0.149 at merge.
        self.register_buffer(
            "_arch",
            torch.tensor([self.context_folds, self.obs_context_layers,
                          self.obs_self_layers, self.peer_attention_layers,
                          self.peer_heads, *self.peer_dims], dtype=torch.long),
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
        if len(have) == 3:
            # written before peer attention existed: no attention blocks
            have = have + [0, 0]
        # (folds, L_c, L_s, N, heads, width per layer); older records carry no
        # width (full width) or one width for every layer
        n_att, widths = have[3], have[5:]
        if n_att == 0:
            have = have[:5]
        elif not widths:
            have = have[:5] + [self.d_model] * n_att
        elif len(widths) == 1 and n_att > 1:
            have = have[:5] + widths * n_att
        state_dict[key] = torch.tensor(have, dtype=torch.long)
        if have != want:
            raise RuntimeError(
                "DeepSetsEncoder architecture mismatch: checkpoint has "
                "(context_folds, obs_context_layers, obs_self_layers, "
                f"peer_attention_layers, peer_heads, width per layer)={tuple(have)} "
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

        z = self.input_self(x_self) + ctx
        if self.peer_attention_layers:
            # keys/values: the per-observation encodings h, which have seen no
            # other row; only the query stream z is updated.  Each bucket is
            # packed once and runs every layer before it is unpacked.
            run = self._peer_stack
            if self.peer_compile and z.is_cuda:
                if self._peer_compiled is None:
                    self._peer_compiled = torch.compile(self._peer_stack, dynamic=True)
                run = self._peer_compiled
            use_ckpt = self.checkpoint and self.training and torch.is_grad_enabled()
            for idx, valid in peer_layout(group_id, self.peer_budget):
                zq, hk = z[idx], h[idx]
                zq = (torch.utils.checkpoint.checkpoint(run, zq, hk, valid, use_reentrant=False)
                      if use_ckpt else run(zq, hk, valid))
                z = z.index_put((idx[valid],), zq[valid])
        return self._run(self.self_blocks[self.n_layers - L_s:], z, obs=True)

    def _peer_stack(self, zq: Tensor, hk: Tensor, valid: Tensor) -> Tensor:
        """Every peer-attention layer on one packed ``(B, L, D)`` block."""
        allow, empty = peer_masks(valid)
        for blk in self.peer_blocks:
            zq = blk(zq, hk, allow, empty)
        return zq

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
