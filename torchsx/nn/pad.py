"""Ragged ↔ padded conversion for grouped reflection batches.

A flat tensor of ``N`` rows partitioned into ``G`` contiguous groups (by an
``offsets`` table, ``offsets[g]:offsets[g+1]`` is group ``g``) is packed into a
dense ``(G, L_max, *trailing)`` tensor plus a boolean padding mask, ready for
``nn.TransformerEncoder(src_key_padding_mask=...)``. The reverse (:func:`unpack`)
recovers the flat layout.

Index tensors (``g``/``within``/``valid``) are computed once and reused across
every column, so this works for any trailing shape: ``(N,)``, ``(N, 3)``,
``(N, D)``, ``(N, 21, 21)``. Vectorised — no per-group Python loop. Generalises
the hand-rolled pack/unpack in ``deprecated/merge/_neural_correction.py``.
"""

from __future__ import annotations

import torch


def group_offsets_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """Build a ``(G+1,)`` offsets table from per-group row counts ``(G,)``.

    ``offsets[g]:offsets[g+1]`` indexes the rows of group ``g`` in flat order.
    """
    offsets = counts.new_zeros(counts.shape[0] + 1)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets


def _pack_index(offsets: torch.Tensor, max_len: int | None):
    """Shared (g, within, valid, L) indices for one (offsets, max_len) pair."""
    counts = offsets[1:] - offsets[:-1]
    n = int(offsets[-1].item())
    g_full = torch.repeat_interleave(
        torch.arange(counts.shape[0], device=offsets.device), counts
    )
    within_full = torch.arange(n, device=offsets.device) - offsets[:-1].repeat_interleave(counts)
    L = int(counts.max().item()) if counts.numel() else 0
    if max_len is not None:
        L = min(L, max_len) if counts.numel() else 0
        valid = within_full < L
    else:
        valid = torch.ones(n, dtype=torch.bool, device=offsets.device)
    return g_full, within_full, valid, L, counts.shape[0]


def pack(
    flat: torch.Tensor, offsets: torch.Tensor, *, max_len: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a flat ``(N, *T)`` tensor into ``(G, L, *T)`` + padding mask ``(G, L)``.

    Parameters
    ----------
    flat : ``(N, *T)`` — rows in group order.
    offsets : ``(G+1,)`` — group boundaries (see :func:`group_offsets_from_counts`).
    max_len : optional cap on ``L``; rows beyond it are dropped (logged by the
        caller — :func:`pack` is silent).

    Returns
    -------
    padded : ``(G, L, *T)`` zero-padded.
    padding_mask : ``(G, L)`` bool, ``True`` at PAD positions
        (== ``src_key_padding_mask`` convention).
    """
    g, within, valid, L, G = _pack_index(offsets, max_len)
    trailing = flat.shape[1:]
    padded = flat.new_zeros((G, L, *trailing))
    padding_mask = torch.ones((G, L), dtype=torch.bool, device=flat.device)
    gv, wv = g[valid], within[valid]
    padded[gv, wv] = flat[valid]
    padding_mask[gv, wv] = False
    return padded, padding_mask


def unpack(padded: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack` (with ``max_len=None``): ``(G, L, *T)`` → ``(N, *T)``."""
    g, within, _, _, _ = _pack_index(offsets, None)
    return padded[g, within]
