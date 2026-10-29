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

from torchsx.nn.deepsets import ResidualFF, peer_layout

__all__ = ["AttentionBlock", "FoldTransformerEncoder", "fold_layout"]


def fold_layout(group_id: Tensor, folds: int, budget: int = 1 << 24) -> list[tuple[Tensor, Tensor]]:
    """Pack each crystal's rows into padded ``(B, L)`` blocks, ``L`` a multiple of ``folds``.

    :func:`~torchsx.nn.deepsets.peer_layout` with the budget shared by the
    ``folds`` runs, each block padded on the right to a multiple of ``folds``.
    Within a crystal the rows keep their arrival order, so slot ``l`` holds the
    crystal's ``l``-th row and its fold is ``l % folds``.  Returns
    ``[(idx, valid)]`` as :func:`peer_layout` does.
    """
    blocks = []
    for idx, valid in peer_layout(group_id, max(1, budget // folds)):
        pad = -idx.shape[1] % folds
        if pad:
            idx = torch.cat([idx, idx[:, -1:].expand(-1, pad)], dim=1)
            valid = torch.cat([valid, valid.new_zeros(valid.shape[0], pad)], dim=1)
        blocks.append((idx, valid))
    return blocks


class AttentionBlock(nn.Module):
    """Pre-norm multi-head attention of a query stream onto a key/value set, plus a residual FF.

    ``forward(q, kv, keep)`` on ``(B, Lq, D)`` queries and ``(B, Lk, D)``
    keys/values; ``keep`` is the ``(B, 1, 1, Lk)`` set of keys every query may
    read.  With ``self_attention`` the block reads its own input (``kv`` is
    ignored) through one norm.  The output projection is zero-initialised, so
    the block starts as the identity.
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

    def forward(self, q: Tensor, kv: Tensor | None, keep: Tensor) -> Tensor:
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
    n_heads
        Attention heads in every block.
    n_crystal_features
        Width of the per-crystal table ``x_crys`` read by the crystal token.
    budget
        Bound on ``folds * B * L * L`` for one packed call
        (:func:`fold_layout`).
    compile, bf16
        Run the per-block stack through ``torch.compile(dynamic=True)`` on
        CUDA, and the attention itself in bfloat16 on CUDA.  ``None`` reads
        ``TORCHSX_PEER_COMPILE`` / ``TORCHSX_PEER_BF16`` (default on).
    checkpoint
        Recompute each packed block's stack in the backward pass instead of
        storing its activations.
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
        d_hidden: int | None = None,
        activation: type[nn.Module] = nn.SiLU,
        n_crystal_features: int = 0,
        budget: int = 1 << 24,
        compile: bool | None = None,
        bf16: bool | None = None,
        checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if int(folds) < 2:
            raise ValueError(f"folds must be >= 2, got {folds}")
        if ctx_layers < 0 or query_layers < 0:
            raise ValueError("ctx_layers and query_layers must be >= 0")
        if query_peers and query_layers < 1:
            raise ValueError("query_peers reads the peers in the last query layer; it needs query_layers >= 1")
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
        self.n_heads = int(n_heads)
        self.budget = int(budget)
        self.checkpoint = bool(checkpoint)
        #: one encoder chunk of ``n`` crystals holds ``K`` runs of each, so
        #: :meth:`AmortizedScale.set_encoder_chunking` divides by this
        self.chunk_divisor = self.context_folds
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
        g = self.input_geometry(x_self)
        m = self.input_measurement(x_ctx.index_select(1, self.meas_idx))
        run = self._block
        if self.compile and g.is_cuda:
            if self._compiled is None:
                self._compiled = torch.compile(self._block, dynamic=True)
            run = self._compiled
        use_ckpt = self.checkpoint and self.training and torch.is_grad_enabled()
        z = g.new_zeros(n, self.d_model)
        for idx, valid in fold_layout(group_id, self.context_folds, self.budget):
            crys = (x_crys.index_select(0, group_id[idx[:, 0]]) if self.n_crystal_features
                    else g.new_zeros(idx.shape[0], 0))
            args = (g[idx], m[idx], crys, valid)
            zb = (torch.utils.checkpoint.checkpoint(run, *args, use_reentrant=False)
                  if use_ckpt else run(*args))
            z = z.index_put((idx[valid],), zb[valid])
        return z

    def _block(self, g: Tensor, m: Tensor, crys: Tensor, valid: Tensor) -> Tensor:
        """Every run of one packed block: ``(B, L, D)`` embeddings, slot ``l`` read from run ``l % K``."""
        K = self.context_folds
        B, L, D = g.shape
        runs = torch.arange(K, device=g.device)
        masked = (torch.arange(L, device=g.device) % K)[None, :] == runs[:, None]    # (K, L)
        tok = g[None] + torch.where(masked[:, None, :, None], self.mask_embedding, m[None])
        tok = tok.masked_fill(~valid[None, :, :, None], 0.0)                          # (K, B, L, D)
        n_vis = (valid[None] & ~masked[:, None]).sum(-1).clamp(min=1)                 # (K, B)
        ct = self.input_crystal(torch.cat(
            [crys[None].expand(K, -1, -1), n_vis.to(g.dtype).log()[..., None]], dim=-1))
        # observations, the crystal token, empty keys up to a multiple of 8
        pad = -(L + 1) % 8
        x = torch.cat([tok, ct[:, :, None], tok.new_zeros(K, B, pad, D)], dim=2)
        x = x.reshape(K * B, L + 1 + pad, D)
        keep = torch.cat([valid, valid.new_ones(B, 1), valid.new_zeros(B, pad)], dim=1)
        keep = keep[None].expand(K, -1, -1).reshape(K * B, 1, 1, -1)
        for blk in self.ctx_blocks:
            x = blk(x, None, keep)
        # run f's queries are its own fold's slots f, f + K, f + 2K, ...
        Lq = L // K
        q = x[:, :L].reshape(K, B, Lq, K, D)[runs, :, :, runs].reshape(K * B, Lq, D)
        last_kv, last_keep = x, keep
        if self.query_peers:
            # every observation's own token, encoded alone; padded to a multiple of 8 keys
            own = self.peer_ff(g + m + self.peer_embedding).masked_fill(~valid[:, :, None], 0.0)
            pad_p = -L % 8
            own = torch.cat([own, own.new_zeros(B, pad_p, D)], dim=1)
            slot = torch.arange(L + pad_p, device=g.device)
            own_slot = torch.arange(Lq, device=g.device)[None, :] * K + runs[:, None]      # (K, Lq)
            peer_keep = (torch.cat([valid, valid.new_zeros(B, pad_p)], dim=1)[None, :, None, :]
                         & (slot[None, None, None, :] != own_slot[:, None, :, None]))  # (K, B, Lq, Lp)
            last_kv = torch.cat([x, own[None].expand(K, -1, -1, -1).reshape(K * B, -1, D)], dim=1)
            last_keep = torch.cat([keep.expand(-1, -1, Lq, -1),
                                   peer_keep.reshape(K * B, 1, Lq, -1)], dim=-1)
        for i, blk in enumerate(self.query_blocks):
            last = i == len(self.query_blocks) - 1
            q = blk(q, last_kv if last else x, last_keep if last else keep)
        # [f, b, j] -> slot j K + f
        return q.reshape(K, B, L // K, D).permute(1, 2, 0, 3).reshape(B, L, D)
