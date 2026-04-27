"""Pair construction and B/G correction for rank concordance refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class RefineData:
    """Per-observation data for explained-variance orientation refinement."""

    hkl_obs: Tensor       # (N, 3) int16 — Miller indices
    crystal_id: Tensor    # (N,) int32
    hkl_id: Tensor        # (N,) int64 — unique HKL index
    log_resid: Tensor     # (N,) float32 — log(I / (I0 × scale)), fixed
    weights: Tensor       # (N,) float32 — 1/Var(log_resid), Poisson-aware
    UB: Tensor            # (C, 3, 3) float32
    lambda_0: Tensor      # (C,) float32 — per-crystal wavelength (Å)
    n_crystals: int
    n_hkl: int

    def to(self, device):
        fields = {}
        for k in ["hkl_obs", "crystal_id", "hkl_id", "log_resid", "weights",
                   "UB", "lambda_0"]:
            fields[k] = getattr(self, k).to(device)
        return RefineData(**fields, n_crystals=self.n_crystals, n_hkl=self.n_hkl)


@dataclass
class PairData:
    """Precomputed pair data for rank concordance refinement."""

    hkl_obs: Tensor       # (N, 3) int16 — Miller indices
    crystal_id: Tensor    # (N,) int32
    hkl_id: Tensor        # (N,) int64 — unique HKL index
    I_corr: Tensor        # (N,) float32 — B/G-corrected intensity
    s_sq: Tensor          # (N,) float32 — (1/(2d))²

    # Per-crystal
    UB: Tensor            # (C, 3, 3) float32
    lambda_0: Tensor      # (C,) float32 — per-crystal wavelength (Å)

    # Pair index structure
    pair_i: Tensor        # (P,) int64 — obs index of first in pair
    pair_j: Tensor        # (P,) int64 — obs index of second in pair
    pair_sign: Tensor     # (P,) float32 — sign(I_i - I_j)
    pair_weight: Tensor   # (P,) float32

    n_crystals: int
    n_obs: int
    n_pairs: int

    def to(self, device):
        """Move all tensors to device."""
        fields = {}
        for k in ["hkl_obs", "crystal_id", "hkl_id", "I_corr", "s_sq",
                   "UB", "lambda_0", "pair_i", "pair_j", "pair_sign", "pair_weight"]:
            fields[k] = getattr(self, k).to(device)
        return PairData(**fields, n_crystals=self.n_crystals,
                        n_obs=self.n_obs, n_pairs=self.n_pairs)


def fit_BG(I_obs, hkl_id, crystal_id, s_sq, n_unique, n_crystals):
    """Fit per-crystal G and B via closed-form log-space OLS.

    I_obs = G_c × exp(-2 B_c s²) × I_true_h × noise
    log(I_obs) = log(G_c) - 2 B_c s² + log(I_true_h) + ε

    Iterates: estimate I_true from current corrections, re-fit G/B.

    Returns (I_corrected, log_G, B).
    """
    pos = I_obs > 0
    log_I = torch.zeros_like(I_obs)
    log_I[pos] = torch.log(I_obs[pos])

    # Initial merged I (mean of positives per HKL)
    I_sum = torch.zeros(n_unique)
    I_cnt = torch.zeros(n_unique)
    I_sum.scatter_add_(0, hkl_id[pos], I_obs[pos])
    I_cnt.scatter_add_(0, hkl_id[pos], torch.ones_like(I_obs[pos]))
    I_merged = I_sum / I_cnt.clamp(min=1)

    log_G = torch.zeros(n_crystals)
    B = torch.zeros(n_crystals)

    for iteration in range(3):
        # Residuals in log space
        log_I_merged = torch.log(I_merged.clamp(min=1e-6))
        resid = torch.zeros_like(I_obs)
        resid[pos] = log_I[pos] - log_I_merged[hkl_id[pos]]

        # Per-crystal 2×2 OLS: resid = a + b×(-2s²) where a=log_G, b=B
        cid = crystal_id.long()
        x = -2.0 * s_sq

        sum_1 = torch.zeros(n_crystals)
        sum_x = torch.zeros(n_crystals)
        sum_xx = torch.zeros(n_crystals)
        sum_r = torch.zeros(n_crystals)
        sum_rx = torch.zeros(n_crystals)

        sum_1.scatter_add_(0, cid[pos], torch.ones_like(I_obs[pos]))
        sum_x.scatter_add_(0, cid[pos], x[pos])
        sum_xx.scatter_add_(0, cid[pos], x[pos] ** 2)
        sum_r.scatter_add_(0, cid[pos], resid[pos])
        sum_rx.scatter_add_(0, cid[pos], resid[pos] * x[pos])

        ridge = 0.01 * sum_1.clamp(min=1)
        det = (sum_1 + ridge) * (sum_xx + ridge) - sum_x ** 2
        det = det.clamp(min=1e-10)

        log_G = ((sum_xx + ridge) * sum_r - sum_x * sum_rx) / det
        B = ((sum_1 + ridge) * sum_rx - sum_x * sum_r) / det

        log_G = log_G - log_G.mean()

        # Update I_merged with corrections applied
        correction = torch.exp(log_G[cid] + B[cid] * x)
        I_corrected = I_obs / correction.clamp(min=1e-6)
        I_sum.zero_()
        I_cnt.zero_()
        pos_c = I_corrected > 0
        I_sum.scatter_add_(0, hkl_id[pos_c], I_corrected[pos_c])
        I_cnt.scatter_add_(0, hkl_id[pos_c], torch.ones_like(I_corrected[pos_c]))
        I_merged = I_sum / I_cnt.clamp(min=1)

    I_corr = I_obs / torch.exp(log_G[cid] + B[cid] * x).clamp(min=1e-6)
    return I_corr, log_G, B


def build_pairs(I_corr, hkl_id, crystal_id, n_unique, *,
                I_threshold=0.0, min_mult=3, max_pairs_per_hkl=500,
                sigma=None):
    """Build (pair_i, pair_j, pair_sign, pair_weight) for all qualifying HKLs.

    Parameters
    ----------
    I_corr : (N,) corrected intensities
    hkl_id : (N,) unique HKL index per observation
    crystal_id : (N,) crystal index (unused, kept for interface compat)
    n_unique : number of unique HKLs
    I_threshold : minimum intensity to include (default 0 = use all)
    min_mult : minimum observations per HKL
    max_pairs_per_hkl : cap on pairs (subsampled by weight if exceeded)
    sigma : (N,) optional per-obs Poisson sigma. When provided, pairs are
        weighted by the Z-score of the intensity difference:
        w_ij = |I_i - I_j| / sqrt(σ²_i + σ²_j).  When None, falls back
        to sqrt(min(I_i, I_j)).
    """
    hkl_np = hkl_id.numpy()
    I_np = I_corr.numpy()
    sigma_np = sigma.numpy() if sigma is not None else None

    order = np.argsort(hkl_np)
    boundaries = np.searchsorted(hkl_np[order], np.arange(n_unique + 1))

    all_i, all_j, all_sign, all_weight = [], [], [], []
    rng = np.random.RandomState(42)

    for uid in range(n_unique):
        lo, hi = boundaries[uid], boundaries[uid + 1]
        if hi - lo < min_mult:
            continue

        idx = order[lo:hi]
        I_h = I_np[idx]

        if I_threshold > 0:
            keep = I_h > I_threshold
            if keep.sum() < min_mult:
                continue
            idx = idx[keep]
            I_h = I_h[keep]

        n = len(idx)
        if n < min_mult:
            continue

        ii, jj = np.triu_indices(n, k=1)

        # Compute pair weights before subsampling so we can sample by weight
        sign_ij = np.sign(I_h[ii] - I_h[jj]).astype(np.float32)
        nonzero = sign_ij != 0
        if nonzero.sum() == 0:
            continue

        if sigma_np is not None:
            sig_i = sigma_np[idx[ii]]
            sig_j = sigma_np[idx[jj]]
            # Z-score: |ΔI| / sqrt(σ²_i + σ²_j)
            w = np.abs(I_h[ii] - I_h[jj]) / np.sqrt(
                sig_i ** 2 + sig_j ** 2).clip(min=1.0)
        else:
            w = np.sqrt(np.minimum(I_h[ii], I_h[jj]).clip(min=1))

        # Apply nonzero filter
        ii = ii[nonzero]
        jj = jj[nonzero]
        sign_ij = sign_ij[nonzero]
        w = w[nonzero]

        # Subsample if too many pairs — weighted sampling keeps informative pairs
        if len(ii) > max_pairs_per_hkl:
            prob = w / w.sum()
            sel = rng.choice(len(ii), max_pairs_per_hkl, replace=False, p=prob)
            ii, jj, sign_ij, w = ii[sel], jj[sel], sign_ij[sel], w[sel]

        all_i.append(idx[ii])
        all_j.append(idx[jj])
        all_sign.append(sign_ij)
        all_weight.append(w.astype(np.float32))

    pair_i = torch.from_numpy(np.concatenate(all_i)).long()
    pair_j = torch.from_numpy(np.concatenate(all_j)).long()
    pair_sign = torch.from_numpy(np.concatenate(all_sign))
    pair_weight = torch.from_numpy(np.concatenate(all_weight))

    return pair_i, pair_j, pair_sign, pair_weight
