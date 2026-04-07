"""Profile-fitted (matched filter) loss for sparse SFX data.

Instead of per-pixel Poisson NLL, integrates observed and predicted signal
using a Gaussian matched-filter kernel.  This collapses each (B, ph, pw)
ROI into one high-SNR scalar per observation, dramatically improving
sensitivity to structure factors and crystal scales — especially for weak
reflections where per-pixel signal is barely above background.

The filter kernel is an isotropic 2-D Gaussian centred on the predicted
spot position (patch centre).  This is a first-order approximation to the
true anisotropic relp profile projected onto the detector.

Loss per observation:

    L_i = (I_obs_i - I_pred_i)^2 / (2 * var_i)

where:
    I_obs  = sum_j  w_j * (k_j - gain_j * bg_j)    # filtered observed residual
    I_pred = sum_j  w_j * gain_j * signal_j          # filtered predicted signal
    var    = sum_j  w_j^2 * gain_j * bg_j             # propagated Poisson variance

Censored pixels (not in sparse set) contribute k_j ≈ bg_j, so their
residual is ~0 — they add noise variance but not bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class ProfileFittedResult:
    """Result from ProfileFittedLoss evaluation."""

    loss: Tensor          # scalar — mean Gaussian NLL per observation (train)
    n_obs: int            # number of training observations
    mean_snr: float       # mean |I_obs| / sqrt(var) — diagnostic
    frac_positive: float  # fraction with I_obs > 0 — diagnostic

    # Validation (held-out observations)
    val_loss: Tensor | None = None
    n_val: int = 0


class ProfileFittedLoss(nn.Module):
    """Matched-filter profile-fitted loss for sparse pixel data.

    Parameters
    ----------
    sigma_kernel : float
        Width of the isotropic Gaussian kernel in pixels.  Should roughly
        match the spot FWHM on the detector (~1–2 pixels for typical SFX).
    val_frac : float
        Fraction of observations to hold out for validation (0 = none).
    min_var : float
        Floor for per-observation variance to avoid division by zero.
    """

    def __init__(
        self,
        sigma_kernel: float = 1.5,
        val_frac: float = 0.0,
        min_var: float = 1e-6,
    ):
        super().__init__()
        self.sigma_kernel = sigma_kernel
        self.val_frac = val_frac
        self.min_var = min_var
        self._kernel: Tensor | None = None
        self._kernel_shape: tuple[int, int] | None = None

    def _get_kernel(self, ph: int, pw: int, device: torch.device) -> Tensor:
        """Lazily build and cache the Gaussian kernel."""
        if self._kernel is not None and self._kernel_shape == (ph, pw):
            return self._kernel.to(device)

        y = torch.arange(ph, device=device, dtype=torch.float32) - ph // 2
        x = torch.arange(pw, device=device, dtype=torch.float32) - pw // 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        w = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * self.sigma_kernel ** 2))
        w = w / w.sum()  # normalise so sum(w) = 1
        self._kernel = w
        self._kernel_shape = (ph, pw)
        return w

    def forward(
        self,
        n_signal: Tensor,       # (B, ph, pw) predicted signal from forward model
        background: Tensor,     # (B, ph, pw) per-pixel radial background
        is_sparse: Tensor,      # (B, ph, pw) bool — pixel in sparse set
        sparse_counts: Tensor,  # (B, ph, pw) int32 — observed k_i
        threshold: Tensor,      # (B, ph, pw) float — unused, kept for API compat
        valid: Tensor,          # (B, ph, pw) bool — on-detector + good pixel
        gain: Tensor,           # (B, 1, 1) or (B, ph, pw) — detector gain
    ) -> ProfileFittedResult:
        """Evaluate the profile-fitted Gaussian NLL loss.

        Parameters match CensoredPoissonLoss.forward() for drop-in replacement.
        ``threshold`` is accepted but unused (API compatibility).
        """
        B, ph, pw = n_signal.shape
        w = self._get_kernel(ph, pw, n_signal.device)  # (ph, pw)

        valid_f = valid.float()
        sparse_f = (is_sparse > 0).float()

        # Fill censored pixels with gain*background (their expected value).
        # Sparse pixels use their actual observed count.
        gain_bg = gain * background  # (B, ph, pw) — expected count in detector units
        observed = sparse_f * sparse_counts.float() + (1.0 - sparse_f) * gain_bg

        # Residual in detector units: k - gain*bg, masked by valid
        residual = (observed - gain_bg) * valid_f  # (B, ph, pw)

        # Predicted signal in detector units
        pred_signal = gain * n_signal  # (B, ph, pw)

        # Profile-weighted integration over each ROI
        w_valid = w * valid_f  # (ph, pw) broadcast → (B, ph, pw)
        I_obs = (residual * w_valid).sum(dim=(1, 2))        # (B,)
        I_pred = (pred_signal * w_valid).sum(dim=(1, 2))    # (B,)

        # Propagated Poisson variance: var(I) = sum w^2 * var(k)
        # var(k) ≈ gain * bg for background-dominated pixels
        var = (w_valid ** 2 * gain_bg.clamp(min=0.1)).sum(dim=(1, 2))  # (B,)
        var = var.clamp(min=self.min_var)

        # Gaussian NLL per observation (no log-var term — var is data, not param)
        nll = (I_obs - I_pred) ** 2 / (2.0 * var)  # (B,)

        # ── Diagnostics ─────────────────────────────────────────────
        with torch.no_grad():
            snr = (I_obs.abs() / var.sqrt()).mean().item()
            frac_pos = (I_obs > 0).float().mean().item()

        # ── Train/val split (per-observation) ────────────────────────
        if self.val_frac > 0 and B > 4:
            n_val = max(1, int(B * self.val_frac))
            perm = torch.randperm(B, device=n_signal.device)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            train_loss = nll[train_idx].mean()
            with torch.no_grad():
                val_loss = nll[val_idx].mean()

            return ProfileFittedResult(
                loss=train_loss,
                n_obs=train_idx.numel(),
                mean_snr=snr,
                frac_positive=frac_pos,
                val_loss=val_loss,
                n_val=val_idx.numel(),
            )

        return ProfileFittedResult(
            loss=nll.mean(),
            n_obs=B,
            mean_snr=snr,
            frac_positive=frac_pos,
        )
