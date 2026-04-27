"""Attraction + Poisson loss for sparse SFX data.

Two-term loss that solves the bootstrap/overlap problem:

1. **Attraction term**: ``-Σ k_i × log(blur(n_signal)_i + ε)``
   Rewards predicted signal *near* observed photons.  The Gaussian blur
   creates a wide basin of attraction — even if the predicted spot is
   several pixels away from the observed sparse pixels, the convolution
   tail provides gradient to pull the prediction closer.  Weighting by
   k_i (observed counts) means high-count pixels (real signal) dominate
   over background fluctuations.

2. **Poisson term**: standard Poisson NLL at sparse pixels for precise
   fitting once overlap develops.

The balance is controlled by ``beta``: L = attraction + beta × poisson.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass
class AttractionResult:
    """Result from AttractionLoss."""

    loss: Tensor              # scalar — combined loss per observation
    attraction: Tensor        # scalar — attraction term per observation
    poisson: Tensor           # scalar — Poisson NLL term per observation
    n_obs: int
    mean_responsibility: float  # signal/(signal+bg) at sparse pixels — overlap diagnostic
    mean_smoothed_signal: float  # mean blurred signal at sparse pixels

    val_loss: Tensor | None = None
    n_val: int = 0


class AttractionLoss(nn.Module):
    """Attraction + Poisson loss for sparse pixel data.

    Parameters
    ----------
    blur_sigma : float
        Gaussian blur width (pixels) for the attraction term.
        Controls the basin of attraction — how far away the predicted
        spot can be and still receive gradient toward the observed photons.
        Typically 3–5 pixels.
    beta : float
        Weight of the Poisson NLL term relative to attraction.
        Start small (0.01–0.1) to let attraction dominate for alignment,
        increase later for precise fitting.
    eps : float
        Floor for log stability and rate clamping.
    val_frac : float
        Fraction of observations held out for validation.
    """

    def __init__(
        self,
        blur_sigma: float = 3.0,
        beta: float = 0.1,
        eps: float = 1e-3,
        val_frac: float = 0.0,
    ):
        super().__init__()
        self.blur_sigma = blur_sigma
        self.beta = beta
        self.eps = eps
        self.val_frac = val_frac
        self._blur_kernel: Tensor | None = None

    def _get_blur_kernel(self, device: torch.device) -> Tensor:
        """Build or retrieve cached Gaussian blur kernel."""
        if self._blur_kernel is not None:
            return self._blur_kernel.to(device)
        sigma = self.blur_sigma
        ks = max(3, int(math.ceil(sigma * 6)) | 1)
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        k = k / k.sum()
        self._blur_kernel = k[None, None, :, :]
        return self._blur_kernel.to(device)

    def _blur(self, x: Tensor) -> Tensor:
        """Apply Gaussian blur to (B, ph, pw) tensor."""
        k = self._get_blur_kernel(x.device)
        pad = k.shape[-1] // 2
        return F.conv2d(x.unsqueeze(1), k, padding=pad).squeeze(1)

    def forward(
        self,
        n_signal: Tensor,       # (B, ph, pw) predicted signal
        background: Tensor,     # (B, ph, pw) per-pixel background
        is_sparse: Tensor,      # (B, ph, pw) bool
        sparse_counts: Tensor,  # (B, ph, pw) int32
        threshold: Tensor,      # (B, ph, pw) float — unused, API compat
        valid: Tensor,          # (B, ph, pw) bool
        gain: Tensor,           # (B, 1, 1) or (B, ph, pw)
    ) -> AttractionResult:
        B, ph, pw = n_signal.shape
        dev = n_signal.device

        sparse_bool = is_sparse > 0
        valid_bool = valid > 0
        mask = sparse_bool & valid_bool  # only sparse + valid pixels
        k = sparse_counts.float()

        # ── Attraction term ─────────────────────────────────────────
        # Blur the predicted signal to create a wide basin of attraction.
        # The blur spreads predicted signal so sparse pixels several
        # pixels away from the predicted center still "see" it.
        smoothed_signal = self._blur(gain * n_signal)  # (B, ph, pw)

        # -k_i × log(smoothed_signal_i + eps) at sparse pixels
        # Weighted by k: high-count pixels pull harder
        log_smooth = torch.log(smoothed_signal + self.eps)
        attract_per_pixel = -k * log_smooth * mask.float()
        attract_per_obs = attract_per_pixel.sum(dim=(1, 2))  # (B,)

        # Normalize by total counts per observation to make scale-invariant
        k_total = (k * mask.float()).sum(dim=(1, 2)).clamp(min=1.0)
        attract_per_obs = attract_per_obs / k_total

        # ── Poisson term ────────────────────────────────────────────
        # Standard Poisson NLL at sparse pixels (for fine-tuning)
        lam_signal = gain * n_signal
        lam_bg = gain * background
        lam = (lam_signal + lam_bg).clamp(min=self.eps)
        poisson_per_pixel = (lam - k * torch.log(lam)) * mask.float()
        poisson_per_obs = poisson_per_pixel.sum(dim=(1, 2))  # (B,)

        # ── Combined loss per observation ───────────────────────────
        loss_per_obs = attract_per_obs + self.beta * poisson_per_obs

        # ── Diagnostics ─────────────────────────────────────────────
        with torch.no_grad():
            if mask.any():
                r = (lam_signal / lam).detach()
                mean_r = r[mask].mean().item()
                mean_smooth = smoothed_signal[mask].mean().item()
            else:
                mean_r = 0.0
                mean_smooth = 0.0

        # ── Train/val split ─────────────────────────────────────────
        if self.val_frac > 0 and B > 4:
            n_val = max(1, int(B * self.val_frac))
            perm = torch.randperm(B, device=dev)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            train_loss = loss_per_obs[train_idx].mean()
            train_attract = attract_per_obs[train_idx].mean()
            train_poisson = poisson_per_obs[train_idx].mean()
            with torch.no_grad():
                val_loss = loss_per_obs[val_idx].mean()

            return AttractionResult(
                loss=train_loss,
                attraction=train_attract,
                poisson=train_poisson,
                n_obs=train_idx.numel(),
                mean_responsibility=mean_r,
                mean_smoothed_signal=mean_smooth,
                val_loss=val_loss,
                n_val=val_idx.numel(),
            )

        return AttractionResult(
            loss=loss_per_obs.mean(),
            attraction=attract_per_obs.mean(),
            poisson=poisson_per_obs.mean(),
            n_obs=B,
            mean_responsibility=mean_r,
            mean_smoothed_signal=mean_smooth,
        )
