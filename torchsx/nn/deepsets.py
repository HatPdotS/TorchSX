"""Deep Sets encoder over a crystal's reflections — the abismal architecture.

Two branches meet at a leave-one-out pool::

    per observation:  features
                         |
            +------------+------------+
            | CONTEXT branch          | SELF branch
            | (sees I, sigma)         | (metadata only)
            | L residual FF blocks    | Linear(D)
            | LEAVE-ONE-OUT mean      |
            |   over crystal_id  -----+--> added as bias
            +------------+------------+
                         | L residual FF blocks
                         v
                     embedding (n, D)

The asymmetry between the branches *is* the design.  The self branch may not
see an observation's own intensity — that would let the network fit ``K_j =
I_j / I0`` exactly and mean nothing.  The context branch may see every
intensity, because leave-one-out removes observation ``j`` from its own pooled
context before it is used.  So a crystal's mosaic state and partiality get
inferred from the pattern of its *other* reflections, which for this dataset is
the only place that information exists: integration used
``--fix-profile-radius=4e7``, so ``profile_radius`` is constant across all
crystals and the geometry carries none of it.

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

from torchsx.nn.pooling import leave_one_out_mean

__all__ = ["ScaleEncoder", "ResidualFF", "DeepSetsEncoder"]


@runtime_checkable
class ScaleEncoder(Protocol):
    """Anything that turns per-observation features into an embedding.

    ``forward(x_self, x_ctx, group_id, n_groups) -> (n, d_model)``

    Implementations **must** guarantee that row ``j`` of the output does not
    depend on row ``j`` of ``x_ctx`` (only on its peers within ``group_id``).
    """

    d_model: int

    def __call__(
        self, x_self: Tensor, x_ctx: Tensor, group_id: Tensor, n_groups: int
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
        intensity and σ, because leave-one-out removes ``j`` from its own pool.
    d_model
        Embedding width.  Default 128.  (abismal's class default is 32 and its
        CLI default 256 — a factor of 8 apart, with the CLI value coming from a
        benchmark config that was subsequently reverted.  Scale up only if
        held-out likelihood says capacity binds.)
    n_layers
        Residual blocks per stack; ``2 * n_layers`` total.
    pool_accum_dtype
        Accumulation dtype for the leave-one-out mean.  ``None`` (default)
        keeps float32; see :mod:`torchsx.nn.pooling` on what that costs.
    checkpoint
        Recompute each residual block in the backward pass instead of storing
        its activations.  On by default: activations are the entire memory cost
        here (parameters are ~2.7 M weights, i.e. noise), and checkpointing
        turns ~10.9 MB per crystal into ~1.8 MB at D=256/L=5, which is what
        makes an 8192-crystal batch fit.
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
        pool_accum_dtype: torch.dtype | None = None,
        checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_self_features = int(n_self_features)
        self.n_context_features = int(n_context_features)
        self.pool_accum_dtype = pool_accum_dtype
        self.checkpoint = bool(checkpoint)

        self.input_context = nn.Linear(n_context_features, d_model)
        self.input_self = nn.Linear(n_self_features, d_model)

        def _stack() -> nn.ModuleList:
            return nn.ModuleList(
                ResidualFF(
                    d_model, d_hidden, activation=activation, dropout=dropout
                )
                for _ in range(n_layers)
            )

        self.context_blocks = _stack()
        self.self_blocks = _stack()

    def forward(
        self,
        x_self: Tensor,        # (n, n_self_features)
        x_ctx: Tensor,         # (n, n_context_features)
        group_id: Tensor,      # (n,) int64 crystal index
        n_groups: int,
    ) -> Tensor:               # (n, d_model)
        if x_self.shape[0] != x_ctx.shape[0] or x_self.shape[0] != group_id.shape[0]:
            raise ValueError(
                f"row mismatch: x_self={x_self.shape[0]} x_ctx={x_ctx.shape[0]} "
                f"group_id={group_id.shape[0]}"
            )

        h = self._run(self.context_blocks, self.input_context(x_ctx))

        # The leave-one-out step.  ``fill=0.0`` is the right value for a
        # singleton crystal: "my peers tell me nothing" is the same statement
        # as a zero bias into an additive branch.
        ctx, _ = leave_one_out_mean(
            h, group_id, n_groups, fill=0.0, accum_dtype=self.pool_accum_dtype
        )

        return self._run(self.self_blocks, self.input_self(x_self) + ctx)

    def _run(self, blocks: nn.ModuleList, x: Tensor) -> Tensor:
        use_ckpt = self.checkpoint and self.training and torch.is_grad_enabled()
        for blk in blocks:
            x = (
                torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                if use_ckpt else blk(x)
            )
        return x
