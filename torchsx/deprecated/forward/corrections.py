"""Geometric corrections: polarization, solid angle, background estimation."""

from __future__ import annotations

import torch
from torch import Tensor


def polarization_correction(two_theta: Tensor, phi: Tensor) -> Tensor:
    """Polarization correction for horizontally polarized beam.

    P = 1 - sin^2(2theta) * cos^2(phi)

    Parameters
    ----------
    two_theta : (...,) float — scattering angle 2theta in radians
    phi : (...,) float — azimuthal angle in radians

    Returns
    -------
    P : (...,) float — polarization factor in (0, 1]
    """
    return 1.0 - torch.sin(two_theta) ** 2 * torch.cos(phi) ** 2


def solid_angle_correction(s_hat: Tensor) -> Tensor:
    """Solid angle correction: cos^3(alpha).

    alpha is the angle between the beam axis (+z) and the sample-to-pixel vector.

    Parameters
    ----------
    s_hat : (..., 3) float — unit scattering vectors

    Returns
    -------
    omega : (...,) float — solid angle factor
    """
    cos_alpha = s_hat[..., 2]  # dot with z-axis
    return cos_alpha.abs() ** 3


def estimate_background(
    patches: Tensor, valid: Tensor, border_width: int = 2,
) -> Tensor:
    """Estimate per-ROI background from median of border pixels.

    Parameters
    ----------
    patches : (B, ph, pw) float — observed pixel counts
    valid : (B, ph, pw) bool or uint8 — pixel mask
    border_width : int — width of border strip

    Returns
    -------
    background : (B,) float — scalar background per ROI
    """
    B, ph, pw = patches.shape

    # Build border mask
    border = torch.zeros(ph, pw, dtype=torch.bool, device=patches.device)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    border = border.unsqueeze(0).expand(B, -1, -1)  # (B, ph, pw)

    mask = border & (valid > 0)

    # Use masked_fill to compute median per ROI
    bg = torch.zeros(B, device=patches.device)
    # Fill invalid with nan, then nanmedian per ROI over spatial dims
    vals = patches.clone()
    vals[~mask] = float("nan")
    vals_flat = vals.reshape(B, -1)  # (B, ph*pw)
    bg = torch.nanmedian(vals_flat, dim=1).values
    # Replace nan (all-invalid border) with 0
    bg = torch.nan_to_num(bg, nan=0.0)
    return bg


def signal_significance(
    patches: Tensor, valid: Tensor, background: Tensor,
    center_radius: int = 2,
) -> Tensor:
    """Per-ROI signal-to-noise under the null hypothesis of background only.

    Computes how unlikely the center-pixel counts are if the ROI contained
    only Poisson background. This is purely data-driven — no model needed.

    For each ROI, sums the excess counts in a center disk of given radius
    and divides by the expected Poisson noise:

        SNR = sum_{center}(obs - bg) / sqrt(n_center * max(bg, 1))

    Parameters
    ----------
    patches : (B, ph, pw) float — observed pixel counts
    valid : (B, ph, pw) bool or uint8 — pixel mask
    background : (B,) float — estimated per-ROI background
    center_radius : int — radius of the center region (in pixels from center)

    Returns
    -------
    snr : (B,) float — signal significance per ROI (≈ sigma units)
    """
    B, ph, pw = patches.shape
    cy, cx = ph // 2, pw // 2

    # Build center mask
    yy = torch.arange(ph, device=patches.device).float() - cy
    xx = torch.arange(pw, device=patches.device).float() - cx
    yg, xg = torch.meshgrid(yy, xx, indexing="ij")
    center = (yg ** 2 + xg ** 2) <= center_radius ** 2  # (ph, pw)
    center = center.unsqueeze(0).expand(B, -1, -1)  # (B, ph, pw)

    mask = center & (valid > 0)
    n_center = mask.sum(dim=(1, 2)).float().clamp(min=1)  # (B,)

    # Excess counts in center
    excess = (patches.float() - background[:, None, None]) * mask.float()
    total_excess = excess.sum(dim=(1, 2))  # (B,)

    # Expected noise under null: sqrt(n_pix * bg)
    noise = (n_center * background.clamp(min=1.0)).sqrt()

    return total_excess / noise
