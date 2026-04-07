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

    unique_hkl: torch.Tensor  # (N_unique, 3)
    hkl_id: torch.Tensor  # (N_obs,)
    n_unique: int


def compute_hkl_index(hkl: torch.Tensor) -> HKLIndex:
    """Map (N_obs, 3) Miller indices to contiguous indices [0, N_unique).

    Uses torch.unique to find the set of unique HKL triplets and assigns
    each observation an index into that set. The unique HKLs are sorted
    lexicographically (h, then k, then l).

    Parameters
    ----------
    hkl : (N_obs, 3) int16 or int32 tensor of Miller indices

    Returns
    -------
    HKLIndex with unique_hkl, hkl_id, and n_unique

    Examples
    --------
    >>> hkl = torch.tensor([[1,0,0],[0,1,0],[1,0,0],[2,0,0],[0,1,0]], dtype=torch.int16)
    >>> idx = compute_hkl_index(hkl)
    >>> idx.n_unique
    3
    >>> idx.hkl_id
    tensor([1, 0, 1, 2, 0])
    """
    # torch.unique with dim=0 returns sorted unique rows + inverse mapping
    unique, inverse = torch.unique(hkl.long(), dim=0, sorted=True, return_inverse=True)

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
    """Weighted merge of unmerged intensities by HKL group.

    Performs inverse-variance weighted mean: I_merged = Σ(I/σ²) / Σ(1/σ²).

    Parameters
    ----------
    hkl_id : (N_obs,) int64 — index into merged array
    I : (N_obs,) float32 — observed intensities
    sigma : (N_obs,) float32 — sigma values
    n_unique : number of unique reflections

    Returns
    -------
    I_merged : (N_unique,) float32 — merged intensities
    sigma_merged : (N_unique,) float32 — merged sigma
    multiplicity : (N_unique,) int32 — observations per reflection
    """
    # Inverse-variance weights (clamp sigma to avoid division by zero)
    w = 1.0 / sigma.clamp(min=1e-10).pow(2)

    # Weighted sum of I and sum of weights
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
    """Merge intensities by per-HKL median.

    The median is robust to the heavy-tailed multiplicative noise in SFX
    data (GenGauss with alpha > 0.5) where the mean is biased high by
    E[exp(epsilon)] >> 1.

    Parameters
    ----------
    hkl_id : (N_obs,) int64 — index into merged array
    I : (N_obs,) float32 — observed intensities
    n_unique : number of unique reflections
    min_mult : minimum observations per HKL (set to NaN below this)

    Returns
    -------
    I_merged : (N_unique,) float32 — merged intensities (NaN where mult < min_mult)
    sigma_merged : (N_unique,) float32 — MAD-based sigma (1.4826 * MAD)
    multiplicity : (N_unique,) int32 — observations per reflection
    """
    order = hkl_id.argsort()
    I_sorted = I[order]
    hkl_sorted = hkl_id[order]
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
        # MAD-based robust sigma: 1.4826 * median(|x - median(x)|)
        if n >= 2:
            mad = (vals - med).abs().median()
            sigma_merged[h] = 1.4826 * mad / (n ** 0.5)

    return I_merged, sigma_merged, count
