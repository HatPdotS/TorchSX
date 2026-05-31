"""Shared string-table (categorical) encode/decode helpers.

Both :class:`torchsx.io._metadata.StreamMetadata` and :class:`torchsx.levels.Level`
store lists of strings as an order-preserving unique table plus an ``int32`` index
tensor. That ~3-line encode/decode core lived in two places; it lives here now.

These are intentionally tiny, pure functions — the two callers keep their own
storage containers (a JSON-serialisable schema dict vs. a typed ``ColumnSpec``),
which legitimately differ. Only the leaf encode/decode/remap is shared.

The on-disk shape is load-bearing: the table is the order-preserving unique list,
the index tensor is ``int32``. Do not change either without a format migration.
"""

from __future__ import annotations

import torch


def encode_categorical(values: list[str]) -> tuple[list[str], torch.Tensor]:
    """Encode strings as an order-preserving unique table + int32 index tensor.

    Returns ``(table, idx)`` where ``table`` is the deduplicated list (first
    occurrence wins) and ``idx`` is an ``int32`` tensor of positions into it.
    """
    table = list(dict.fromkeys(values))  # unique, order-preserving
    lookup = {s: i for i, s in enumerate(table)}
    idx = torch.tensor([lookup[s] for s in values], dtype=torch.int32)
    return table, idx


def decode_categorical(table: list[str], idx: torch.Tensor) -> list[str]:
    """Inverse of :func:`encode_categorical`: index ``table`` by ``idx``."""
    return [table[i] for i in idx.tolist()]


def remap_to_unified_table(
    tables: list[list[str]], idx_tensors: list[torch.Tensor]
) -> tuple[list[str], list[torch.Tensor]]:
    """Merge several ``(table, idx)`` shards onto one unified string table.

    Used when concatenating categorical columns whose shards may carry different
    local tables. Returns ``(unified_table, remapped_idx_tensors)``; each output
    tensor preserves its input's device and is ``int32``.
    """
    unified: list[str] = []
    lookup: dict[str, int] = {}
    remapped: list[torch.Tensor] = []
    for table, old in zip(tables, idx_tensors):
        for s in table:
            if s not in lookup:
                lookup[s] = len(unified)
                unified.append(s)
        remap = torch.tensor(
            [lookup[s] for s in table], dtype=torch.int32, device=old.device
        )
        remapped.append(remap[old.long()].to(torch.int32))
    return unified, remapped
