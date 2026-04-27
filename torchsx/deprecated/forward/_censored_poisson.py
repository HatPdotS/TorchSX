"""Censored Poisson NLL for sparse pixel data.

Two-term loss for refinement against sparse-compressed SFX data:
1. Sparse pixels: standard Poisson NLL for all observed sparse pixels
2. Censored + predicted: penalizes hallucinated signal where pixel was censored

Optional spatial blur: when blur_sigma > 0, both predicted rates and observed
counts are convolved with a Gaussian kernel before computing the loss. This
creates a smooth gradient landscape that allows the optimizer to align predicted
and observed spots even when they are offset by several pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class CensoredPoissonResult:
    """Result from CensoredPoissonLoss evaluation.

    All loss values are **per-pixel averages** (normalized by the number of
    valid pixels in the respective split).  This ensures gradients are
    independent of batch size and train/val losses are directly comparable.
    """

    loss: Tensor        # scalar — mean training loss per valid train pixel
    term1: Tensor       # scalar — mean Poisson NLL on sparse pixels (train)
    term2: Tensor       # scalar — mean censored NLL on censored+predicted pixels (train)
    n_term1: int        # pixel count for term 1
    n_term2: int        # pixel count for term 2
    n_train: int        # total valid train pixels (normalization denominator)

    # Validation (held-out pixels)
    val_loss: Tensor | None = None   # scalar — mean val loss per valid val pixel
    val_term1: Tensor | None = None  # scalar — mean val Poisson NLL
    val_term2: Tensor | None = None  # scalar — mean val censored NLL
    n_val_pixels: int = 0            # total held-out valid pixels


def compute_censoring_threshold(
    background: Tensor,
    z_threshold: float = 4.0,
) -> Tensor:
    """Compute per-pixel censoring threshold from Anscombe z-score.

    Inverts the Anscombe transform to find the smallest photon count k
    where z = (sqrt(k + 3/8) - sqrt(bg + 3/8)) / 0.5 exceeds z_threshold.

    Parameters
    ----------
    background : (...) float — per-pixel background rate
    z_threshold : float — z-score threshold used during sparsification

    Returns
    -------
    threshold : (...) float — smallest k that would be flagged as signal
    """
    t = torch.ceil(
        (z_threshold * 0.5 + torch.sqrt(background + 0.375)) ** 2 - 0.375
    )
    return t.clamp(min=1.0)


class CensoredPoissonLoss(nn.Module):
    """Censored Poisson NLL for sparse pixel data.

    Operates on pre-built (B, ph, pw) tensors — no data loading or batch logic.

    Term 1 (sparse pixels) includes ALL sparse pixels regardless of whether
    the model predicts signal there.  This ensures weak reflections receive
    gradient even when their predicted signal is near zero.

    Term 2 (censored + predicted) only activates where the model predicts
    non-trivial signal on a censored pixel (hallucinated signal penalty).

    Parameters
    ----------
    eps : float
        Minimum rate clamp to avoid log(0).
    val_frac : float
        Fraction of valid pixels to hold out for validation (0 = no holdout).
        Held-out pixels are excluded from all training loss terms and their
        loss is reported separately in val_loss/val_term1/val_term2.
    """

    def __init__(
        self,
        eps: float = 1e-3,
        val_frac: float = 0.0,
        use_term2: bool = True,
        blur_sigma: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        self.val_frac = val_frac
        self.use_term2 = use_term2
        self.blur_sigma = blur_sigma
        self._blur_kernel: Tensor | None = None

    def _get_blur_kernel(self, device: torch.device) -> Tensor:
        """Build or retrieve cached Gaussian blur kernel."""
        if self._blur_kernel is not None:
            return self._blur_kernel.to(device)
        sigma = self.blur_sigma
        # Kernel size: cover ±3σ, must be odd
        ks = max(3, int(math.ceil(sigma * 6)) | 1)
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        k = k / k.sum()
        # Shape for conv2d: (1, 1, ks, ks)
        self._blur_kernel = k[None, None, :, :]
        return self._blur_kernel.to(device)

    def _blur(self, x: Tensor) -> Tensor:
        """Apply Gaussian blur to (B, ph, pw) tensor."""
        if self.blur_sigma <= 0:
            return x
        k = self._get_blur_kernel(x.device)
        pad = k.shape[-1] // 2
        # conv2d expects (B, C, H, W)
        return torch.nn.functional.conv2d(
            x.unsqueeze(1), k, padding=pad,
        ).squeeze(1)

    def forward(
        self,
        n_signal: Tensor,       # (B, ph, pw) predicted signal from forward model
        background: Tensor,     # (B, ph, pw) per-pixel radial background
        is_sparse: Tensor,      # (B, ph, pw) bool — pixel in sparse set
        sparse_counts: Tensor,  # (B, ph, pw) int32 — observed k_i (0 where censored)
        threshold: Tensor,      # (B, ph, pw) float — censoring threshold t_i
        valid: Tensor,          # (B, ph, pw) bool — on-detector + good pixel
        gain: Tensor,           # (B, 1, 1) or (B, ph, pw) — detector gain
    ) -> CensoredPoissonResult:
        """Evaluate the two-term censored Poisson loss.

        When val_frac > 0, a random subset of valid pixels is held out from
        training and its loss is returned separately. The random mask is
        regenerated each call, so held-out pixels vary across batches/epochs
        (averaging out over training).
        """
        # Predicted rate: λ = gain * (background + signal)
        lam = (gain * (background + n_signal)).clamp(min=self.eps)

        valid_bool = valid > 0
        sparse_bool = is_sparse > 0
        censored = ~sparse_bool  # pixel was below threshold → censored

        # ── Observed counts: sparse pixels get real counts, censored ≈ bg
        observed = torch.where(
            sparse_bool,
            sparse_counts.float(),
            (gain * background).squeeze(-1).squeeze(-1)
            if gain.dim() == 3 and gain.shape[-1] == 1
            else gain * background,
        )

        # ── Spatial blur (scale-space): blur both sides so the loss
        # "sees" overlap even when spots are offset by several pixels.
        lam_eff = self._blur(lam)
        obs_eff = self._blur(observed)
        # Clamp after blur to keep rates positive
        lam_eff = lam_eff.clamp(min=self.eps)
        obs_eff = obs_eff.clamp(min=0)

        # Predicted signal exceeds noise floor → only used for term 2
        has_signal = n_signal > self.eps  # (B, ph, pw)

        # ── Validation holdout mask ────────────────────────────────
        if self.val_frac > 0:
            val_hold = torch.rand_like(background) < self.val_frac
            train_valid = valid_bool & ~val_hold
            val_valid = valid_bool & val_hold
        else:
            train_valid = valid_bool
            val_hold = None

        # ── Term 1: Poisson NLL on (blurred) observations ─────────
        # With blur: computed at ALL valid pixels (blurred signal spreads)
        # Without blur: only at sparse pixels (original behaviour)
        if self.blur_sigma > 0:
            mask1 = train_valid
        else:
            mask1 = train_valid & sparse_bool
        n1 = mask1.sum().item()
        if n1 > 0:
            lam1 = lam_eff[mask1]
            k1 = obs_eff[mask1]
            term1 = (lam1 - k1 * torch.log(lam1)).sum()
        else:
            term1 = lam.new_tensor(0.0)

        # ── Term 2: censored + predicted (Poisson CDF penalty) ────
        # Only where model predicts signal — censored pixels with no
        # predicted signal contribute no useful gradient (constant).
        # Can be disabled (use_term2=False) to prevent the model from
        # learning to predict nothing to avoid the censored penalty.
        if self.use_term2:
            mask2 = train_valid & censored & has_signal
            n2 = mask2.sum().item()
            if n2 > 0:
                lam2 = lam[mask2]
                t2 = threshold[mask2].float()
                log_cdf = torch.log(
                    torch.special.gammaincc(t2 + 1.0, lam2).clamp(min=1e-20)
                )
                term2 = -log_cdf.sum()
            else:
                term2 = lam.new_tensor(0.0)
        else:
            n2 = 0
            term2 = lam.new_tensor(0.0)

        # ── Normalize by number of observations (B), not pixels ─────
        # Per-pixel normalization dilutes F² gradients ~20x because most
        # pixels are background-only.  Per-observation normalization gives
        # each reflection's Poisson NLL its natural magnitude.
        B = n_signal.shape[0]
        n_train = train_valid.sum().item()
        if B > 0:
            norm = 1.0 / B
            loss = (term1 + term2) * norm
            term1_normed = term1 * norm
            term2_normed = term2 * norm
        else:
            loss = lam.new_tensor(0.0)
            term1_normed = term1
            term2_normed = term2

        # ── Validation terms (same computation on held-out pixels) ─
        val_loss = None
        val_term1 = None
        val_term2 = None

        n_val_pixels = 0
        if val_hold is not None:
            n_val_pixels = val_valid.sum().item()
            with torch.no_grad():
                if self.blur_sigma > 0:
                    vmask1 = val_valid
                else:
                    vmask1 = val_valid & sparse_bool
                if vmask1.any():
                    vlam1 = lam_eff[vmask1]
                    vk1 = obs_eff[vmask1]
                    val_term1 = (vlam1 - vk1 * torch.log(vlam1)).sum()
                else:
                    val_term1 = lam.new_tensor(0.0)

                if self.use_term2:
                    vmask2 = val_valid & censored & has_signal
                    if vmask2.any():
                        vlam2 = lam[vmask2]
                        vt2 = threshold[vmask2].float()
                        vlog_cdf = torch.log(
                            torch.special.gammaincc(vt2 + 1.0, vlam2).clamp(min=1e-20)
                        )
                        val_term2 = -vlog_cdf.sum()
                    else:
                        val_term2 = lam.new_tensor(0.0)
                else:
                    val_term2 = lam.new_tensor(0.0)

                # Normalize val by B (same as training)
                if B > 0:
                    vnorm = 1.0 / B
                    val_loss = (val_term1 + val_term2) * vnorm
                    val_term1 = val_term1 * vnorm
                    val_term2 = val_term2 * vnorm
                else:
                    val_loss = lam.new_tensor(0.0)

        return CensoredPoissonResult(
            loss=loss,
            term1=term1_normed,
            term2=term2_normed,
            n_term1=n1,
            n_term2=n2,
            n_train=n_train,
            val_loss=val_loss,
            val_term1=val_term1,
            val_term2=val_term2,
            n_val_pixels=n_val_pixels,
        )
