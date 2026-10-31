"""Fold transformer: a crystal's observations attend to each other, one fold masked at a time.

Each crystal's observations are dealt round-robin into ``K`` folds in the order
they arrive (the assignment of :func:`~torchsx.nn.pooling.fold_ids`), and the
crystal is encoded ``K`` times.  In run ``f`` every observation is a token:
the observations of fold ``f`` carry their geometry and one learned mask
embedding in place of their measurement; every other observation carries
geometry and measurement.  One more token carries the crystal's metadata and
the number of observations whose measurement the run shows::

    run f:   [ g+m  g+m  g+e  g+m  g+m  g+e  ...  crystal ]     g+e = fold f
                               │
        ctx_layers  x [ self-attention + FF ]      every token of the run reads every other
                               │
                           memory_f
                               │ keys / values
        query j of fold f = j's own token after the stack
        query_layers x [ cross-attention  q_j -> memory_f  + FF ]     queries never see each other
        last query layer, with query_peers: also q_j -> { token_i : i != j }
                               │
                              z_j

Every observation is read out from the one run in which it is masked.  No
input of run ``f`` carries a measurement of a fold-``f`` observation, so the
embedding of every fold-``f`` observation is independent of those
measurements, in value and in gradient, bit for bit.  Within a run the
context tokens therefore exchange information through any number of layers,
and each query reads a context that has already been worked over as a whole.
An observation's context is the ``(K - 1) / K`` of its crystal outside its
fold, and each crystal is encoded ``K`` times.

With ``query_peers`` the last query layer reads, next to ``memory_f``, every
other observation of the crystal through its own token: its geometry and
measurement embedded alone, through one residual block that sees no other
observation.  Observation ``j``'s own token is masked, so this read is exact
in the same sense, over all ``n - 1`` peers including ``j``'s fold mates.
The run memories of the other runs are never read: there ``j`` is visible,
and after one self-attention layer every token of such a run carries it.

With ``refine_geometry`` the geometry itself is corrected inside each run.
The excitation error and lab-frame direction of an observation follow from
its scattering vector ``q`` and the beam's ``k0 = 1/lambda`` as
``|q + k0 z| - k0``, and the commonest errors in them are one small rotation
of the crystal and one photon-energy offset.  After every context block a
head on the run's crystal token adds to a running correction (rotation
``delta``, relative energy shift), every token's excitation error and
direction are recomputed from the rotated ``q`` and the shifted ``k0``, and
the change is added to the residual stream through the geometry embedding.
Later blocks and the queries read corrected geometry.  The correction of run
``f`` comes from run ``f``'s crystal token, so it carries no fold-``f``
measurement either.

Geometry comes from ``x_self`` for every token.  The measurement is the part
of ``x_ctx`` that ``x_self`` lacks (``meas_idx``): the context-only channels
of :class:`~torchsx.nn.obs_features.ObsFeatureSpec`.

Every block's output projection is zero-initialised, so at step 0 an
observation's embedding is its masked token: its geometry plus the mask
embedding.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from torchsx.nn.deepsets import ResidualFF

__all__ = ["AttentionBlock", "FoldTransformerEncoder", "fold_layout", "key_prefix_mask",
           "scatter_positions"]


def fold_layout(group_id: Tensor, folds: int, tokens: int = 1 << 19) -> list[tuple[Tensor, Tensor]]:
    """Pack each crystal's rows into padded ``(B, L)`` blocks, ``L`` a multiple of ``folds``.

    Crystals are sorted by size and dealt into blocks.  A block's ``L`` is its
    largest crystal rounded up to a multiple of 8 and then of ``folds``, and a
    block closes before its ``folds`` runs would hold more than ``tokens``
    tokens (``folds * B * L``); a crystal larger than that gets a block of its
    own.  Within a crystal the rows keep their arrival order, so slot ``l``
    holds the crystal's ``l``-th row and its fold is ``l % folds``.  One host
    sync per call.  Returns ``[(idx, valid)]`` as
    :func:`~torchsx.nn.deepsets.peer_layout` does.
    """
    order = torch.argsort(group_id, stable=True)
    _, counts = torch.unique_consecutive(group_id[order], return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    by_size = torch.argsort(counts, stable=True)
    sizes = counts[by_size].tolist()

    def padded(n: int) -> int:
        n = -(-n // 8) * 8
        return n + (-n % folds)

    blocks: list[tuple[Tensor, Tensor]] = []
    i, G = 0, len(sizes)
    while i < G:
        j, L = i, 0
        while j < G:
            L_j = padded(sizes[j])
            if j > i and folds * (j - i + 1) * L_j > tokens:
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


def scatter_positions(dest: list[Tensor], n: int) -> Tensor:
    """Where each of ``n`` rows sits among the concatenated slots of the packed blocks.

    ``dest`` holds, per block, the row of every flattened slot, ``n`` for a
    padding slot; every row fills exactly one slot.  Gathering the
    concatenated slot outputs at the result puts them back in row order,
    without a host sync.
    """
    flat = torch.cat(dest)
    pos = torch.zeros(n + 1, dtype=torch.long, device=flat.device)
    pos.index_put_((flat,), torch.arange(flat.numel(), device=flat.device))
    return pos[:n]


def key_prefix_mask(n_keys: Tensor, q_len, kv_len, mask_mod=None, block: int = 128) -> BlockMask:
    """Flex-attention block mask in which row ``b`` reads its first ``n_keys[b]`` keys.

    Built from the counts alone, with no dense mask, and equal to
    ``create_block_mask`` on the same rule: in every complete query block of
    row ``b`` the key blocks wholly inside the prefix are full (no mask
    evaluated), the block holding its end is partial, and the rest are
    skipped, so padding costs no attention work; a last, incomplete query
    block reads every key block of the prefix as partial (the kernel masks
    only partial blocks, and that block runs past ``q_len``).  The tables
    hold one block a side more than the lengths (the kernels index them by
    their strides and visit only the blocks of the lengths): a block count of
    one is a size of one, specialised when compiled.  ``mask_mod`` (same signature as
    flex attention's) is the element mask of the partial blocks; it must keep
    the same prefix, and it reads the counts by default.
    """
    B, dev, i32 = n_keys.shape[0], n_keys.device, torch.int32
    nq, nk = -(-q_len // block), -(-kv_len // block)
    tq, tk = nq + 1, nk + 1                                                   # table sizes
    n_complete = q_len // block                                               # query blocks inside q_len
    full = (n_keys // block).to(i32)[:, None, None]                          # (B, 1, 1)
    part = (n_keys % block != 0).to(i32)[:, None, None]
    k_blocks = torch.arange(tk, device=dev, dtype=i32)
    q_blocks = torch.arange(tq, device=dev, dtype=i32)
    # key side, per query block: in a complete one full blocks 0 .. full-1 and the
    # partial one at `full`; in the incomplete one blocks 0 .. full+part-1, all partial
    complete = (torch.zeros(B, 1, tq, dtype=i32, device=dev) + q_blocks < n_complete).to(i32)
    full_kv_num = full * complete
    kv_num = part * complete + (full + part) * (1 - complete)
    blocks = torch.zeros(B, 1, tq, tk, dtype=i32, device=dev) + k_blocks
    full_kv_idx = blocks % nk
    kv_idx = (blocks + (full * complete)[..., None]) % nk
    # query side (the backward), per key block: the complete query blocks 0 ..
    # n_complete-1 when it is full or partial there, then the incomplete one
    blk = torch.zeros(B, 1, tk, dtype=i32, device=dev) + k_blocks
    edge = ((blk == full) & (part > 0)).to(i32)
    ragged = torch.zeros(1, dtype=i32, device=dev) + (nq - n_complete)         # 1 if q_len % block
    tail = (blk < full + part).to(i32) * ragged
    full_q_num = (blk < full).to(i32) * n_complete
    q_num = edge * n_complete + tail
    blocks = torch.zeros(B, 1, tk, tq, dtype=i32, device=dev) + q_blocks
    full_q_idx = blocks % nq
    q_idx = (blocks + ((1 - edge) * n_complete)[..., None]) % nq

    if mask_mod is None:
        def mask_mod(b, h, q, kv):
            return kv < n_keys[b]
    return BlockMask((q_len, kv_len), kv_num, kv_idx, full_kv_num, full_kv_idx,
                     q_num, q_idx, full_q_num, full_q_idx, (block, block), mask_mod)


class AttentionBlock(nn.Module):
    """Pre-norm multi-head attention of a query stream onto a key/value set, plus a residual FF.

    ``forward(q, kv, keep)`` on ``(B, Lq, D)`` queries and ``(B, Lk, D)``
    keys/values; ``keep`` is the ``(B, 1, 1, Lk)`` set of keys every query may
    read.  ``block_mask``, when given, is the same set as a flex-attention
    :class:`BlockMask` and is used in place of ``keep``.  With
    ``self_attention`` the block reads its own input (``kv`` is ignored)
    through one norm.  The output projection is zero-initialised, so the block
    starts as the identity.
    """

    def __init__(self, d_model: int, n_heads: int, d_hidden: int | None = None, *,
                 self_attention: bool, activation: type[nn.Module] = nn.SiLU,
                 bf16: bool = False) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads, self.d_head = int(n_heads), d_model // n_heads
        self.self_attention = bool(self_attention)
        #: attention in bfloat16 on CUDA; projections, norms and the residual
        #: stream stay float32
        self.bf16 = bool(bf16)
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_kv = None if self.self_attention else nn.RMSNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.w_o.weight)
        nn.init.zeros_(self.w_o.bias)
        self.ff = ResidualFF(d_model, d_hidden, activation=activation)

    def forward(self, q: Tensor, kv: Tensor | None, keep: Tensor,
                block_mask: BlockMask | None = None) -> Tensor:
        B, Lq, D = q.shape
        H, dh = self.n_heads, self.d_head
        qn = self.norm_q(q)
        kvn = qn if self.self_attention else self.norm_kv(kv)
        Lk = kvn.shape[1]
        Q = self.w_q(qn).view(B, Lq, H, dh).transpose(1, 2)
        K = self.w_k(kvn).view(B, Lk, H, dh).transpose(1, 2)
        V = self.w_v(kvn).view(B, Lk, H, dh).transpose(1, 2)
        if self.bf16 and q.is_cuda:
            Q, K, V = Q.to(torch.bfloat16), K.to(torch.bfloat16), V.to(torch.bfloat16)
        if block_mask is not None:
            a = flex_attention(Q, K, V, block_mask=block_mask).to(q.dtype)
        else:
            a = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=keep).to(q.dtype)
        return self.ff(q + self.w_o(a.transpose(1, 2).reshape(B, Lq, D)))


class FoldTransformerEncoder(nn.Module):
    """Masked-fold transformer over each crystal's observations (module docstring).

    Parameters
    ----------
    n_self_features, n_context_features
        Widths of ``x_self`` (geometry, every token) and ``x_ctx``.
    meas_idx
        Columns of ``x_ctx`` that make up the measurement: shown on visible
        tokens, replaced by the mask embedding on the masked fold.
    folds
        ``K >= 2``: runs per crystal; each run masks one fold.
    ctx_layers
        Self-attention blocks over the tokens of a run.
    query_layers
        Cross-attention blocks from each masked observation onto its run's
        output.  ``0`` reads the embedding straight off the masked token.
    query_peers
        The last query block also reads every other observation's own token
        (module docstring).  Needs ``query_layers >= 1``.
    geometry
        Where the geometry sits in ``x_self`` and how it was standardised:
        ``{"cols": [ewald_offset, qx, qy, qz, s_sq, beam_energy], "mean",
        "std"}`` (columns of ``x_self`` and the six statistics).  Needed by
        ``refine_geometry``.
    refine_geometry
        Correct each run's geometry from its context (module docstring).
        Needs ``ctx_layers >= 1``.
    n_heads
        Attention heads in every block.
    n_crystal_features
        Width of the per-crystal table ``x_crys`` read by the crystal token.
    tokens
        Bound on the tokens of one packed block, ``folds * B * L``
        (:func:`fold_layout`).
    compile, bf16
        Run each packed block (:meth:`encode`) through
        ``torch.compile(dynamic=True)`` on CUDA, and the attention itself in
        bfloat16 on CUDA.  ``None`` reads ``TORCHSX_PEER_COMPILE`` /
        ``TORCHSX_PEER_BF16`` (default on).
    flex
        Inside the compiled block, with autograd on, attention reads each
        run's keys through flex attention, which skips the padding instead of
        masking it (the same keys; heads at least 16 wide, a power of two).
        Without autograd, and in the last query layer under ``query_peers``,
        the masked kernel runs: inductor inlines the block-mask arithmetic into
        a forward-only flex kernel, which then reads out of bounds.  ``None``
        reads ``TORCHSX_FOLD_FLEX`` (default on).
    checkpoint
        Recompute each packed block in the backward pass instead of storing
        its activations.
    """

    kind = "fold"

    def __init__(
        self,
        n_self_features: int,
        n_context_features: int,
        meas_idx: list[int] | tuple[int, ...],
        *,
        d_model: int = 128,
        n_heads: int = 8,
        folds: int = 5,
        ctx_layers: int = 2,
        query_layers: int = 1,
        query_peers: bool = False,
        geometry: dict | None = None,
        refine_geometry: bool = False,
        d_hidden: int | None = None,
        activation: type[nn.Module] = nn.SiLU,
        n_crystal_features: int = 0,
        tokens: int = 1 << 19,
        compile: bool | None = None,
        bf16: bool | None = None,
        flex: bool | None = None,
        checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if int(folds) < 2:
            raise ValueError(f"folds must be >= 2, got {folds}")
        if ctx_layers < 0 or query_layers < 0:
            raise ValueError("ctx_layers and query_layers must be >= 0")
        if query_peers and query_layers < 1:
            raise ValueError("query_peers reads the peers in the last query layer; it needs query_layers >= 1")
        if refine_geometry and (ctx_layers < 1 or geometry is None):
            raise ValueError("refine_geometry corrects the geometry from the context; it needs "
                             "ctx_layers >= 1 and the geometry columns")
        meas = [int(i) for i in meas_idx]
        if not meas or min(meas) < 0 or max(meas) >= n_context_features:
            raise ValueError(f"meas_idx {meas} must select columns of x_ctx (width {n_context_features})")
        self.d_model = int(d_model)
        self.n_self_features = int(n_self_features)
        self.n_context_features = int(n_context_features)
        self.n_crystal_features = int(n_crystal_features)
        #: K, recorded under the same name as the Deep Sets fold count
        self.context_folds = int(folds)
        self.ctx_layers, self.query_layers = int(ctx_layers), int(query_layers)
        self.query_peers = bool(query_peers)
        self.refine_geometry = bool(refine_geometry)
        self.n_heads = int(n_heads)
        self.tokens = int(tokens)
        self.checkpoint = bool(checkpoint)
        self.register_buffer("meas_idx", torch.tensor(meas, dtype=torch.long), persistent=False)

        self.input_geometry = nn.Linear(n_self_features, d_model)
        self.input_measurement = nn.Linear(len(meas), d_model)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        # [x_crys, log(visible count)] -> the crystal token
        self.input_crystal = nn.Linear(self.n_crystal_features + 1, d_model)

        if compile is None:
            compile = os.environ.get("TORCHSX_PEER_COMPILE", "1") != "0"
        if bf16 is None:
            bf16 = os.environ.get("TORCHSX_PEER_BF16", "1") != "0"
        self.compile = bool(compile)
        if flex is None:
            flex = os.environ.get("TORCHSX_FOLD_FLEX", "1") != "0"
        d_head = self.d_model // self.n_heads
        self.flex = bool(flex) and d_head >= 16 and d_head & (d_head - 1) == 0
        self._compiled = None
        self.ctx_blocks = nn.ModuleList(
            AttentionBlock(d_model, n_heads, d_hidden, self_attention=True,
                           activation=activation, bf16=bf16)
            for _ in range(self.ctx_layers)
        )
        self.query_blocks = nn.ModuleList(
            AttentionBlock(d_model, n_heads, d_hidden, self_attention=False,
                           activation=activation, bf16=bf16)
            for _ in range(self.query_layers)
        )
        if self.query_peers:
            #: marks a key as a peer's own token rather than a run-memory token
            self.peer_embedding = nn.Parameter(torch.zeros(d_model))
            #: each peer's own encoding: one residual block on its token alone
            self.peer_ff = ResidualFF(d_model, d_hidden, activation=activation)
        #: last correction per crystal-run, ``(runs, 4)``: rotation (rad) and relative energy shift
        self.last_correction: Tensor | None = None
        if self.refine_geometry:
            cols = [int(c) for c in geometry["cols"]]
            if len(cols) != 6:
                raise ValueError("geometry needs the columns of ewald_offset, qx, qy, qz, s_sq, beam_energy")
            self.register_buffer("geo_cols", torch.tensor(cols, dtype=torch.long), persistent=False)
            # persistent: x_self is standardised with the statistics the model was trained under
            self.register_buffer("geo_mean", torch.as_tensor(geometry["mean"], dtype=torch.float32).clone())
            self.register_buffer("geo_std", torch.as_tensor(geometry["std"], dtype=torch.float32).clone())
            #: one head per context block on the crystal token: an increment of
            #: (rotation x 3, relative energy shift); zero-initialised, so no correction at step 0
            self.geometry_heads = nn.ModuleList(nn.Linear(d_model, 4) for _ in range(self.ctx_layers))
            for h in self.geometry_heads:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

    #: radians and relative energy shift per unit of the geometry heads' output
    ROT_SCALE = 1e-2
    ENERGY_SCALE = 1e-3
    #: hc in eV Angstrom: k0 = E / HC
    HC = 12398.4

    def forward(
        self,
        x_self: Tensor,        # (n, n_self_features)
        x_ctx: Tensor,         # (n, n_context_features)
        group_id: Tensor,      # (n,) int64 crystal index
        n_groups: int,
        x_crys: Tensor | None = None,   # (n_groups, n_crystal_features)
    ) -> Tensor:               # (n, d_model)
        if isinstance(x_self, (tuple, list)):
            raise NotImplementedError("the fold transformer takes one x_self table "
                                      "(no pseudo-translation profiles)")
        n = x_self.shape[0]
        if x_ctx.shape[0] != n or group_id.shape[0] != n:
            raise ValueError(f"row mismatch: x_self={n} x_ctx={x_ctx.shape[0]} "
                             f"group_id={group_id.shape[0]}")
        if self.n_crystal_features:
            if x_crys is None or x_crys.shape[1] != self.n_crystal_features:
                raise ValueError(f"encoder was built with n_crystal_features="
                                 f"{self.n_crystal_features}; x_crys does not match")
        run = self.encode
        if self.compile and x_self.is_cuda:
            if self._compiled is None:
                self._compiled = torch.compile(self.encode, dynamic=True)
            run = self._compiled
        use_ckpt = self.checkpoint and self.training and torch.is_grad_enabled()
        zs, dest, corrections = [], [], []
        for idx, valid in self.layout(group_id):
            args = (x_self, x_ctx, x_crys, group_id, idx, valid)
            zb, corr = (torch.utils.checkpoint.checkpoint(run, *args, use_reentrant=False)
                        if use_ckpt else run(*args))
            zs.append(zb.flatten(0, 1))
            dest.append(torch.where(valid, idx, n).flatten())
            corrections.append(corr.detach())
        if self.refine_geometry:
            self.last_correction = torch.cat(corrections)
        return torch.cat(zs).index_select(0, scatter_positions(dest, n))

    def layout(self, group_id: Tensor) -> list[tuple[Tensor, Tensor]]:
        """The packed blocks of one batch (:func:`fold_layout` under ``tokens``)."""
        return fold_layout(group_id, self.context_folds, self.tokens)

    def encode(self, x_self: Tensor, x_ctx: Tensor, x_crys: Tensor | None, group_id: Tensor,
               idx: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """One packed block from the batch tables: ``(B, L, D)`` embeddings of its slots
        and each run's geometry correction (:meth:`_block`)."""
        rows_self = x_self[idx]
        g = self.input_geometry(rows_self)
        m = self.input_measurement(x_ctx[idx].index_select(-1, self.meas_idx))
        crys = (x_crys.index_select(0, group_id[idx[:, 0]]) if self.n_crystal_features
                else g.new_zeros(idx.shape[0], 0))
        xs = rows_self if self.refine_geometry else g.new_zeros(0)
        return self._block(g, m, crys, valid, xs)

    def _block(self, g: Tensor, m: Tensor, crys: Tensor, valid: Tensor,
               xs: Tensor) -> tuple[Tensor, Tensor]:
        """Every run of one packed block: ``(B, L, D)`` embeddings, slot ``l`` read from
        run ``l % K``, and each run's geometry correction ``(K * B, 4)`` (empty without
        ``refine_geometry``)."""
        K = self.context_folds
        B, L, D = g.shape
        runs = torch.arange(K, device=g.device)
        masked = (torch.arange(L, device=g.device) % K)[None, :] == runs[:, None]    # (K, L)
        tok = g[None] + torch.where(masked[:, None, :, None], self.mask_embedding, m[None])
        tok = tok.masked_fill(~valid[None, :, :, None], 0.0)                          # (K, B, L, D)
        n_vis = (valid[None] & ~masked[:, None]).sum(-1).clamp(min=1)                 # (K, B)
        ct = self.input_crystal(torch.cat(
            [crys[None].expand(K, -1, -1), n_vis.to(g.dtype).log()[..., None]], dim=-1))
        # the crystal token first (slot 0, so no index depends on L), the
        # observations, empty keys up to a multiple of 8: at least one for
        # folds <= 8 (fold_layout's L), so no size is specialised when compiled
        pad = -(L + 1) % 8
        x = torch.cat([ct[:, :, None], tok, tok.new_zeros(K, B, pad, D)], dim=2)
        x = x.reshape(K * B, 1 + L + pad, D)
        keep = torch.cat([valid.new_ones(B, 1), valid, valid.new_zeros(B, pad)], dim=1)
        keep = keep[None].expand(K, -1, -1).reshape(K * B, 1, 1, -1)
        ctx_mask = query_mask = None
        if self.flex and g.is_cuda and torch.is_grad_enabled() and torch.compiler.is_compiling():
            # every run reads its crystal token and its observations: a prefix of the keys
            n_keys = (1 + valid.sum(-1)).repeat(K)                                    # (K * B,)

            def is_key(b, h, q, kv):
                # read off ``valid`` itself, a graph input: inductor cannot index
                # a buffer computed inside the graph from a flex kernel everywhere
                return (kv == 0) | ((kv <= L) & valid[b % B, (kv - 1).clamp(0, L - 1)])
            ctx_mask = key_prefix_mask(n_keys, 1 + L + pad, 1 + L + pad, is_key)
            query_mask = key_prefix_mask(n_keys, L // K, 1 + L + pad, is_key)
        correction = x.new_zeros(0, 4)
        if self.refine_geometry:
            q0, k0 = self._raw_geometry(xs)                                           # (B, L, 3), (B, L)
            w_geo = self.input_geometry.weight.index_select(1, self.geo_cols[:4])     # (D, 4)
            live = valid[None].expand(K, -1, -1).reshape(K * B, L, 1)
            correction = x.new_zeros(K * B, 4)
            applied = None
        for i, blk in enumerate(self.ctx_blocks):
            x = blk(x, None, keep, ctx_mask)
            if self.refine_geometry:
                correction = correction + self.geometry_heads[i](x[:, 0])
                emb = (self._geometry_change(q0, k0, correction, K) @ w_geo.T) * live   # (K * B, L, D)
                step = emb if applied is None else emb - applied
                x = torch.cat([x[:, :1], x[:, 1:1 + L] + step, x[:, 1 + L:]], dim=1)
                applied = emb
        # run f's queries are its own fold's slots f, f + K, f + 2K, ...
        Lq = L // K
        q = x[:, 1:1 + L].reshape(K, B, Lq, K, D)[runs, :, :, runs].reshape(K * B, Lq, D)
        last_kv, last_keep = x, keep
        if self.query_peers:
            # every observation's own token, encoded alone, in the run memory's slots
            own = self.peer_ff(g + m + self.peer_embedding).masked_fill(~valid[:, :, None], 0.0)
            own = torch.cat([own.new_zeros(B, 1, D), own, own.new_zeros(B, pad, D)], dim=1)
            slot = torch.arange(1 + L + pad, device=g.device)
            own_slot = 1 + torch.arange(Lq, device=g.device)[None, :] * K + runs[:, None]  # (K, Lq)
            peer_keep = (keep.reshape(K, B, 1, -1) & (slot != 0)
                         & (slot[None, None, None, :] != own_slot[:, None, :, None]))  # (K, B, Lq, Lp)
            last_kv = torch.cat([x, own[None].expand(K, -1, -1, -1).reshape(K * B, -1, D)], dim=1)
            last_keep = torch.cat([keep.expand(-1, -1, Lq, -1),
                                   peer_keep.reshape(K * B, 1, Lq, -1)], dim=-1)
        for i, blk in enumerate(self.query_blocks):
            last = i == len(self.query_blocks) - 1
            q = (blk(q, last_kv, last_keep) if last and self.query_peers
                 else blk(q, x, keep, query_mask))
        # [f, b, j] -> slot j K + f
        z = q.reshape(K, B, L // K, D).permute(1, 2, 0, 3).reshape(B, L, D)
        if self.refine_geometry:
            correction = torch.cat([correction[:, :3] * self.ROT_SCALE,
                                    correction[:, 3:] * self.ENERGY_SCALE], dim=1)
        return z, correction

    def _raw_geometry(self, xs: Tensor) -> tuple[Tensor, Tensor]:
        """Scattering vector ``q`` (lab frame, 1/Angstrom) and ``k0`` of each slot from its
        standardised self features (a constant column reads back as its mean)."""
        raw = xs.index_select(-1, self.geo_cols) * self.geo_std + self.geo_mean
        direction = raw[..., 1:4]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        q = 2.0 * raw[..., 4:5].clamp(min=0.0).sqrt() * direction      # |q| = 1/d = 2 sqrt(s_sq)
        k0 = raw[..., 5].clamp(min=1.0) / self.HC
        return q, k0

    def _geometry_change(self, q0: Tensor, k0: Tensor, correction: Tensor, K: int) -> Tensor:
        """Standardised change of (ewald_offset, qx, qy, qz) under each run's correction,
        ``(K * B, L, 4)``: the crystal rotated by ``delta`` (to second order) and ``k0``
        scaled by ``1 + epsilon``."""
        B, L, _ = q0.shape
        delta = (correction[:, :3] * self.ROT_SCALE).view(K, B, 1, 3)
        eps = (correction[:, 3] * self.ENERGY_SCALE).view(K, B, 1)
        q = q0[None].expand(K, -1, -1, -1)
        dq = torch.cross(delta.expand_as(q), q, dim=-1)
        q1 = q + dq + 0.5 * torch.cross(delta.expand_as(q), dq, dim=-1)
        k1 = k0[None] * (1.0 + eps)

        def ewald(v: Tensor, k: Tensor) -> Tensor:
            return torch.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2 + (v[..., 2] + k) ** 2) - k

        def unit(v: Tensor) -> Tensor:
            return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        de = ewald(q1, k1) - ewald(q, k0[None])
        ddir = unit(q1) - unit(q)
        change = torch.cat([de[..., None], ddir], dim=-1) / self.geo_std[:4]
        return change.reshape(K * B, L, 4)
