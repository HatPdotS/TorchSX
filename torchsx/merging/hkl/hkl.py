"""HKL indexing: map Miller indices to contiguous merged-reflection indices."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HKLIndex:
    """Mapping between (h,k,l) tuples and contiguous integer indices.

    Attributes
    ----------
    unique_hkl : (N_unique, 3) int16 — sorted unique Miller indices
    hkl_id : (N_obs,) int64 — index into unique_hkl for each observation
    n_unique : int — number of unique reflections
    """

    unique_hkl: torch.Tensor
    hkl_id: torch.Tensor
    n_unique: int


def compute_hkl_index(hkl: torch.Tensor) -> HKLIndex:
    """Map (N_obs, 3) Miller indices to contiguous indices [0, N_unique).

    Uses torch.unique to find the set of unique HKL triplets and assigns
    each observation an index into that set.  Unique HKLs are sorted
    lexicographically (h, then k, then l).
    """
    # torch.unique(dim=0) sorts (N_obs, 3) rows lexicographically and is by far
    # the most expensive step in prepare_observations -- the bulk of it at
    # ~1e9 observations, where the .long() cast alone materialises tens of GB.  Miller
    # indices span a few hundred values at most, so pack them into ONE int64
    # key.  The encoding is positional with h most significant, so ordering by
    # key is exactly ordering lexicographically by (h, k, l): the returned
    # unique_hkl and hkl_id are identical to the dim=0 path, just far cheaper.
    lo = torch.stack([hkl[:, i].min() for i in range(3)]).long()
    hi = torch.stack([hkl[:, i].max() for i in range(3)]).long()
    span = (hi - lo + 1)
    if int(span.prod()) > 2 ** 62 or bool((span <= 0).any()):
        unique, inverse = torch.unique(hkl.long(), dim=0, sorted=True,
                                       return_inverse=True)
    else:
        key = (hkl[:, 0].long() - lo[0]) * span[1] + (hkl[:, 1].long() - lo[1])
        key = key * span[2] + (hkl[:, 2].long() - lo[2])
        uniq_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
        del key
        l = uniq_key % span[2]
        rest = uniq_key.div(span[2], rounding_mode="floor")
        k = rest % span[1]
        h = rest.div(span[1], rounding_mode="floor")
        unique = torch.stack([h + lo[0], k + lo[1], l + lo[2]], dim=1)
    return HKLIndex(
        unique_hkl=unique.to(torch.int16),
        hkl_id=inverse.to(torch.int64),
        n_unique=unique.shape[0],
    )


def merge_intensities(
    hkl_id: torch.Tensor,
    I: torch.Tensor,
    sigma: torch.Tensor,
    n_unique: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted merge: I_merged = Σ(I/σ²) / Σ(1/σ²) per HKL group."""
    w = 1.0 / sigma.clamp(min=1e-10).pow(2)

    sum_wI = torch.zeros(n_unique, dtype=torch.float64, device=I.device)
    sum_w = torch.zeros(n_unique, dtype=torch.float64, device=I.device)
    count = torch.zeros(n_unique, dtype=torch.int32, device=I.device)

    sum_wI.scatter_add_(0, hkl_id, (w * I).double())
    sum_w.scatter_add_(0, hkl_id, w.double())
    count.scatter_add_(0, hkl_id, torch.ones_like(hkl_id, dtype=torch.int32))

    I_merged = (sum_wI / sum_w.clamp(min=1e-30)).float()
    sigma_merged = (1.0 / sum_w.clamp(min=1e-30).sqrt()).float()
    return I_merged, sigma_merged, count


def merge_median(
    hkl_id: torch.Tensor,
    I: torch.Tensor,
    n_unique: int,
    min_mult: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-HKL median merge, robust to heavy-tailed multiplicative noise.

    Returns (I_median, sigma_MAD, count); NaN where count < min_mult.
    """
    order = hkl_id.argsort()
    I_sorted = I[order]
    count = torch.zeros(n_unique, dtype=torch.int32, device=I.device)
    count.scatter_add_(0, hkl_id, torch.ones_like(hkl_id, dtype=torch.int32))

    offsets = torch.zeros(n_unique + 1, dtype=torch.long, device=I.device)
    offsets[1:] = count.long().cumsum(0)

    I_merged = torch.full((n_unique,), float("nan"), device=I.device)
    sigma_merged = torch.full((n_unique,), float("nan"), device=I.device)

    for h in range(n_unique):
        lo, hi = offsets[h].item(), offsets[h + 1].item()
        n = hi - lo
        if n < min_mult:
            continue
        vals = I_sorted[lo:hi]
        med = vals.median()
        I_merged[h] = med
        if n >= 2:
            mad = (vals - med).abs().median()
            sigma_merged[h] = 1.4826 * mad / (n ** 0.5)

    return I_merged, sigma_merged, count
