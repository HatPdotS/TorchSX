"""Poisson NLL loss at sparse pixel positions after per-pixel signal aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor

from torchsx.forward._pixel_matching import PixelReflectionPairs, SparseForwardResult


@dataclass
class SparsePixelResult:
    """Loss result with diagnostics."""

    loss: Tensor              # scalar — training loss
    n_pixels: int             # number of sparse pixels evaluated
    n_pairs: int              # number of (reflection, pixel) pairs
    mean_responsibility: float  # mean signal/(signal+bg) at sparse pixels
    mean_pairs_per_refl: float  # diagnostic
    val_loss: Tensor | None = None
    n_val: int = 0


class SparsePixelLoss(nn.Module):
    """Poisson NLL at sparse pixel positions.

    Multiple reflections may contribute signal to the same pixel (via scatter_add).
    The total predicted rate is: lambda_j = gain * (bg_j + sum_i n_signal_ij)

    Parameters
    ----------
    eps : float
        Minimum predicted rate (prevents log(0)).
    val_frac : float
        Fraction of pixels held out for validation (0 = no validation).
    """

    def __init__(self, eps: float = 1e-3, val_frac: float = 0.0):
        super().__init__()
        self.eps = eps
        self.val_frac = val_frac

    def forward(
        self,
        result: SparseForwardResult,
        pairs: PixelReflectionPairs,
    ) -> SparsePixelResult:
        """Compute Poisson NLL per pixel.

        Parameters
        ----------
        result : SparseForwardResult from forward_pairs()
        pairs : PixelReflectionPairs

        Returns
        -------
        SparsePixelResult with loss and diagnostics
        """
        lam = result.n_pred_pixels  # (N_pixels,) — already clamped
        k = pairs.pixel_counts.float()  # (N_pixels,)

        # Poisson NLL: lambda - k * log(lambda) + log(k!)
        # We drop the log(k!) constant since it doesn't affect gradients
        nll = lam - k * torch.log(lam)  # (N_pixels,)

        N = pairs.n_pixels

        # Train/val split
        val_loss = None
        n_val = 0
        if self.val_frac > 0 and N >= 10:
            n_val = max(1, int(N * self.val_frac))
            # Deterministic split based on pixel index (stable across epochs)
            perm = torch.randperm(N, device=lam.device, generator=torch.Generator(device=lam.device).manual_seed(0))
            val_mask = torch.zeros(N, dtype=torch.bool, device=lam.device)
            val_mask[perm[:n_val]] = True
            train_mask = ~val_mask

            loss = nll[train_mask].mean()
            val_loss = nll[val_mask].mean().detach()
        else:
            loss = nll.mean()

        # Diagnostics
        with torch.no_grad():
            # Signal contribution per pixel (aggregated)
            n_signal_per_pixel = torch.zeros(N, device=lam.device)
            n_signal_per_pixel.scatter_add_(
                0, pairs.pixel_idx, result.n_signal_pairs.detach()
            )
            bg = pairs.pixel_bg
            resp = n_signal_per_pixel / (n_signal_per_pixel + bg).clamp(min=1e-6)
            mean_resp = resp.mean().item()

            n_pairs = pairs.n_pairs
            n_refl = pairs.n_refl
            mean_pairs = n_pairs / max(n_refl, 1)

        return SparsePixelResult(
            loss=loss,
            n_pixels=N,
            n_pairs=n_pairs,
            mean_responsibility=mean_resp,
            mean_pairs_per_refl=mean_pairs,
            val_loss=val_loss,
            n_val=n_val,
        )
