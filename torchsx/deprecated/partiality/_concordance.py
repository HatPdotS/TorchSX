"""Losses for partiality-free orientation refinement."""

from __future__ import annotations

import torch
from torch import nn, Tensor


class ExplainedVarianceLoss(nn.Module):
    """Maximize variance of log-residuals explained by |r|.

    For each HKL, computes weighted Corr²(|r|, e) via scatter ops,
    where e = log(I/(I0·scale)) is fixed and |r| is differentiable
    through the crystal orientations.

    No partiality model is assumed — only that partiality correlates
    with |r|.  Poisson-noise-aware weighting naturally downweights
    noise-dominated weak observations.
    """

    def forward(
        self,
        abs_r: Tensor,
        hkl_id: Tensor,
        log_resid: Tensor,
        weights: Tensor,
        n_hkl: int,
    ) -> Tensor:
        """Compute negative mean R² across HKLs (minimise to maximise R²).

        Parameters
        ----------
        abs_r : (N,) differentiable |excitation error|
        hkl_id : (N,) int64 — HKL index per observation
        log_resid : (N,) fixed log-residuals log(I/(I0·scale))
        weights : (N,) 1/Var(log_resid), Poisson+GenGauss-aware
        n_hkl : number of unique HKLs
        """
        dev = abs_r.device

        # Per-HKL weighted means
        sum_wr = torch.zeros(n_hkl, device=dev).scatter_add_(0, hkl_id, abs_r * weights)
        sum_we = torch.zeros(n_hkl, device=dev).scatter_add_(0, hkl_id, log_resid * weights)
        sum_w = torch.zeros(n_hkl, device=dev).scatter_add_(0, hkl_id, weights).clamp(min=1e-8)
        mean_r = sum_wr / sum_w
        mean_e = sum_we / sum_w

        # Center observations
        dr = abs_r - mean_r[hkl_id]
        de = log_resid - mean_e[hkl_id]

        # Per-HKL weighted covariance and variances
        cov = torch.zeros(n_hkl, device=dev).scatter_add_(0, hkl_id, dr * de * weights) / sum_w
        var_r = torch.zeros(n_hkl, device=dev).scatter_add_(0, hkl_id, dr ** 2 * weights) / sum_w
        var_e = torch.zeros(n_hkl, device=dev).scatter_add_(0, hkl_id, de ** 2 * weights) / sum_w

        # R² = Corr² = cov² / (var_r × var_e)
        denom = (var_r * var_e).clamp(min=1e-12)
        r_squared = cov ** 2 / denom

        # Weighted mean R² (weight by total per-HKL weight)
        return -(r_squared * sum_w).sum() / sum_w.sum()

    def mean_r_squared(
        self,
        abs_r: Tensor,
        hkl_id: Tensor,
        log_resid: Tensor,
        weights: Tensor,
        n_hkl: int,
    ) -> float:
        """Non-differentiable diagnostic: weighted mean R² across HKLs."""
        with torch.no_grad():
            return -self.forward(abs_r, hkl_id, log_resid, weights, n_hkl).item()


class RankConcordanceLoss(nn.Module):
    """Differentiable rank concordance between intensity ordering and |r| ordering.

    For precomputed pairs (i, j) with known intensity ordering sign_ij
    and SNR-based weight w_ij, the loss is:

        L = -Σ w_ij × sign_ij × σ(β × (|r_j| - |r_i|)) / Σ w_ij

    Minimising L maximises concordance: brighter observations should
    have smaller |r| (closer to the Ewald sphere).

    No partiality function is assumed — only monotonicity of p(|r|).

    Parameters
    ----------
    beta : float
        Sigmoid sharpness. Anneal from soft (~5) to hard (~50).
    """

    def __init__(self, beta: float = 5.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        abs_r: Tensor,
        pair_i: Tensor,
        pair_j: Tensor,
        pair_sign: Tensor,
        pair_weight: Tensor,
    ) -> Tensor:
        """Compute rank concordance loss.

        Parameters
        ----------
        abs_r : (N_obs,) differentiable |excitation error|
        pair_i, pair_j : (P,) int64 — observation indices
        pair_sign : (P,) float — sign(I_i - I_j), +1 if i is brighter
        pair_weight : (P,) float — SNR-based pair weight

        Returns scalar loss (minimise to maximise concordance).
        """
        r_i = abs_r[pair_i]
        r_j = abs_r[pair_j]

        # If i is brighter (sign=+1), we want |r_i| < |r_j| → (r_j - r_i) > 0
        delta_r = r_j - r_i
        concordance = torch.sigmoid(self.beta * delta_r)

        weighted = pair_weight * pair_sign * concordance
        return -weighted.sum() / pair_weight.sum().clamp(min=1e-8)

    def concordance_fraction(
        self,
        abs_r: Tensor,
        pair_i: Tensor,
        pair_j: Tensor,
        pair_sign: Tensor,
    ) -> float:
        """Hard concordance fraction (non-differentiable diagnostic).

        Returns fraction of pairs where intensity and |r| orderings agree.
        """
        with torch.no_grad():
            r_i = abs_r[pair_i]
            r_j = abs_r[pair_j]
            r_sign = torch.sign(r_j - r_i)
            concordant = (pair_sign * r_sign > 0).float()
            tied = (r_sign == 0).float()
            return (concordant.sum() / (concordant.numel() - tied.sum()).clamp(min=1)).item()
