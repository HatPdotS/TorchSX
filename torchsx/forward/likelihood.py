"""PixelLikelihood: per-pixel NLL for the forward model."""

from __future__ import annotations

import torch
from torch import nn, Tensor


class PixelLikelihood(nn.Module):
    """Pixel-level negative log-likelihood.

    Supports per-ROI weighting by spectral integral, which naturally
    down-weights reflections that barely touch the Ewald sphere.

    Parameters
    ----------
    mode : "gaussian" or "poisson"
    var_floor : float
        Minimum variance (photon²). Default 1.0.
    """

    def __init__(
        self,
        mode: str = "gaussian",
        var_floor: float = 1.0,
    ):
        super().__init__()
        if mode not in ("gaussian", "poisson"):
            raise ValueError(f"Unknown mode {mode!r}, expected 'gaussian' or 'poisson'")
        self.mode = mode
        self.var_floor = var_floor

    def forward(
        self,
        n_pred: Tensor,
        n_obs: Tensor,
        valid: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Compute pixel-level NLL.

        Parameters
        ----------
        n_pred : (B, ph, pw) float — predicted counts (must be > 0)
        n_obs : (B, ph, pw) float — observed counts
        valid : (B, ph, pw) bool or uint8 — pixel mask
        weights : (B,) float, optional — per-ROI weight (e.g., from spectral
            integral). Broadcast to all pixels in each ROI.

        Returns
        -------
        loss : scalar — weighted total NLL over valid pixels
        """
        mask = valid > 0
        pred = n_pred[mask].clamp(min=1e-3)
        obs = n_obs[mask]

        if self.mode == "gaussian":
            var = pred.clamp(min=self.var_floor)
            nll = 0.5 * (obs - pred) ** 2 / var + 0.5 * torch.log(var)
        else:
            # Poisson NLL: -obs * log(pred) + pred
            nll = -obs * torch.log(pred) + pred

        if weights is not None:
            # Broadcast per-ROI weights to per-pixel
            w = weights[:, None, None].expand_as(n_pred)[mask]
            nll = nll * w

        return nll.sum()
