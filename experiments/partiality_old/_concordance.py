"""Rank concordance loss for partiality-free orientation refinement."""

from __future__ import annotations

import torch
from torch import nn, Tensor


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
