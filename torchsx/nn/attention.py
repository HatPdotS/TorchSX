"""Token-per-reflection attention encoder — each reflection queries its peers.

The mean-pooled Deep Sets context is a crystal-level summary: it can say "this
crystal is bright", which is why a model built on it recovers G and essentially
nothing else (measured on the real dark: ``between_frac`` 0.98,
``within_frac`` 0.02).  Per-reflection partiality is not a crystal-level
quantity — to know whether reflection ``j`` was well positioned you have to ask
what happened to the peers *near j* on the Ewald sphere, and a mean over peer
embeddings deliberately destroys that locality.

Why the peers beat ``j``'s own geometry at all: ``j``'s excitation error is
computed from a refined orientation that is itself in error, and it carries no
information about the pulse spectrum, which at SASE jitters shot to shot.  The
other reflections *measure the crystal's state* — which of them came out strong
or weak, given where they nominally sat, constrains the orientation error and
the spectral content, and those are what set ``j``'s partiality.  That is a
spatial regression over the peer set, and attention is the operator for it.

The leave-one-out contract
--------------------------
One embedding per reflection covering everything (hkl, geometry, Ewald offset,
detector position, intensity, sigma).  For ``j``'s own prediction:

* ``j``'s **query** is built from an embedding with the intensity channels
  zeroed, and
* the **diagonal of the attention is masked**, so ``j`` never attends to itself.

Both are needed.  Zeroing the query alone is not enough — if ``j`` attends to
itself its own intensity returns through the *value*, and the whole scheme
becomes the tautology ``K_j = n'_j / I0_j``.  With the diagonal masked, ``j``'s
geometry still enters (as the query, which is how it localises) while its
intensity cannot reach its own output by any path.

This subsumes the Deep Sets encoder: mean pooling is attention with uniform
weights and a masked diagonal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.nn.deepsets import ResidualFF
from torchsx.nn.pad import group_offsets_from_counts, pack, unpack

__all__ = ["PeerAttentionEncoder"]


class _PeerAttentionBlock(nn.Module):
    """Pre-norm attention over peers (diagonal masked) + residual FF.

    Uses ``scaled_dot_product_attention`` rather than ``nn.MultiheadAttention``
    for a reason that is about the *merge*, not the encoder:
    ``MultiheadAttention`` materialises the ``(B, H, L, L)`` scores, which
    forces a small crystal batch — and the in-batch leave-one-out merge needs
    the batch to be **large**.  With 14,128 reflections, a batch of N crystals
    gives only ~N/81 observations per reflection, so dropping from 8200 to 2300
    crystals cuts the merge multiplicity from ~101 to ~28 and degrades the very
    target that everything else depends on.  SDPA's memory-efficient backend
    keeps the scores off the heap, so the encoder no longer dictates the batch
    size.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = float(dropout)
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_kv = nn.RMSNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.ff = ResidualFF(d_model, dropout=dropout)
        # Zero-init the output projection so the block starts as the identity,
        # matching ResidualFF and keeping K == 1 at step 0.
        nn.init.zeros_(self.w_o.weight)
        nn.init.zeros_(self.w_o.bias)

    def _heads(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self, q: Tensor, kv: Tensor, attn_mask: Tensor, key_padding_mask: Tensor
    ) -> Tensor:
        B, L, _ = q.shape
        qn, kvn = self.norm_q(q), self.norm_kv(kv)
        Q, K, V = self._heads(self.w_q(qn)), self._heads(self.w_k(kvn)), self._heads(self.w_v(kvn))

        # allow = peer, not self, not padding
        allow = (~attn_mask)[None, None, :, :] & (~key_padding_mask)[:, None, None, :]
        # A row with no permitted key (singleton crystal) would give NaN from a
        # fully -inf softmax; let it attend to itself and zero the result after.
        empty = ~allow.any(dim=-1, keepdim=True)
        allow = allow | (empty & torch.eye(L, dtype=torch.bool, device=q.device))

        a = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=allow,
            dropout_p=self.dropout if self.training else 0.0,
        )
        a = a.masked_fill(empty, 0.0)
        a = a.transpose(1, 2).reshape(B, L, -1)
        return self.ff(q + self.w_o(a))


class PeerAttentionEncoder(nn.Module):
    """Reflections as tokens; each attends to the other reflections of its crystal.

    Satisfies :class:`~torchsx.nn.deepsets.ScaleEncoder`, so it drops in wherever
    ``DeepSetsEncoder`` is used.

    Parameters
    ----------
    n_self_features
        Width of the **query** embedding — the token with its intensity
        channels zeroed.
    n_context_features
        Width of the **key/value** embedding — the full token.
    max_tokens
        Crystals with more observations than this are truncated for the
        attention (a warning-free cap; attention is O(L^2) and the tail of the
        reflections-per-crystal distribution is long).  ``None`` disables.
    """

    def __init__(
        self,
        n_self_features: int,
        n_context_features: int,
        *,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_tokens: int | None = 512,
        checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_self_features = int(n_self_features)
        self.n_context_features = int(n_context_features)
        self.max_tokens = max_tokens
        self.checkpoint = bool(checkpoint)

        self.embed_token = nn.Linear(n_context_features, d_model)
        if n_self_features == n_context_features:
            # Token formulation: one embedding, the query copy arrives with its
            # measurement channels already zeroed.  Sharing the weights means
            # geometry is embedded identically in both roles.
            self.embed_query = self.embed_token
            self.shared_embedding = True
        else:
            self.embed_query = nn.Linear(n_self_features, d_model)
            self.shared_embedding = False
        self.blocks = nn.ModuleList(
            _PeerAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        )

    def forward(
        self,
        x_self: Tensor,        # (n, F_self)  — intensity channels zeroed
        x_ctx: Tensor,         # (n, F_ctx)   — full token
        group_id: Tensor,      # (n,) crystal index
        n_groups: int,
    ) -> Tensor:               # (n, d_model)
        order = torch.argsort(group_id, stable=True)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)

        gid_sorted = group_id[order]
        _, counts = torch.unique_consecutive(gid_sorted, return_counts=True)
        offsets = group_offsets_from_counts(counts)

        q_pad, pad_mask = pack(
            self.embed_query(x_self[order]), offsets, max_len=self.max_tokens
        )
        kv_pad, _ = pack(
            self.embed_token(x_ctx[order]), offsets, max_len=self.max_tokens
        )
        L = q_pad.shape[1]

        # Mask the diagonal: j must not attend to itself, or its own intensity
        # returns through the value and the objective becomes a tautology.
        no_self = torch.eye(L, dtype=torch.bool, device=q_pad.device)

        q = q_pad
        for blk in self.blocks:
            if self.checkpoint and self.training and torch.is_grad_enabled():
                q = torch.utils.checkpoint.checkpoint(
                    blk, q, kv_pad, no_self, pad_mask, use_reentrant=False
                )
            else:
                q = blk(q, kv_pad, no_self, pad_mask)

        flat = unpack(q, offsets)
        if flat.shape[0] != x_self.shape[0]:      # truncation dropped tokens
            out = q.new_zeros(x_self.shape[0], self.d_model)
            out[order[: flat.shape[0]]] = flat
            return out
        return flat[inv]
