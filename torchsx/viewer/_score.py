"""On-demand matched-filter score map for the displayed frame.

Computes the oriented matched-filter score ``S`` (the "convoluted map") for one
frame via :func:`torchsx.datareduction.peakfinder.bayes_score_map`, on GPU when
available (``--device auto``) else CPU. To stay fast and dependency-light it uses
a **constant anisotropic profile** and the empirical per-radius σ (no run
calibration); the kernel ``(sigma_radial, sigma_azimuthal)`` is adjustable from
the UI, and the profile/operator are cached per kernel so re-tuning is cheap.

Output is float16 with masked/non-finite pixels set to 0, matching the frontend's
HALF_FLOAT score texture.
"""

from __future__ import annotations

import math

import numpy as np


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class ScoreEngine:
    DEFAULT_SR = 2.0   # radial kernel σ (spectral bandwidth)
    DEFAULT_SA = 0.7   # azimuthal kernel σ (divergence/mosaicity)

    def __init__(self, geom, device: str = "auto"):
        import torch

        from torchsx.datareduction.peakfinder import TorchPeakFinderParams

        self.device = torch.device(resolve_device(device))
        self.geom = geom
        self.params = TorchPeakFinderParams()
        P = geom.n_panels
        nss, nfs = 514, geom.detector_shape[1]
        self._shape = (P, nss, nfs)
        self._radius = geom.pixel_radius_px.reshape(P, nss, nfs).to(torch.float32).to(self.device)
        self._angle = geom.pixel_angle_rad.reshape(P, nss, nfs).to(torch.float32).to(self.device)
        self._profile_op: dict[tuple, tuple] = {}  # (sr,sa) -> (profile, operator)

    def _get_profile(self, sr: float, sa: float):
        key = (round(sr, 3), round(sa, 3))
        if key in self._profile_op:
            return self._profile_op[key]
        from torchsx.datareduction.spot_profile import SpotProfile

        kernel_radius = max(3, int(math.ceil(3.0 * sr)))
        profile = SpotProfile.constant(sr, sa, kernel_radius=kernel_radius, n_orient=12)
        self._profile_op[key] = profile
        return profile

    def score(self, panels, good, sigma_radial: float | None = None,
              sigma_azimuthal: float | None = None):
        """Return ``(S float16 (P,nss,nfs) numpy, smin, smax)`` for one frame.

        ``panels`` and ``good`` are float32/bool tensors of shape ``(P,nss,nfs)`` on
        this engine's device (the server decodes the frame via ``reader.panels_gpu``).
        """
        import torch

        from torchsx.datareduction.peakfinder import bayes_score_map

        sr = self.DEFAULT_SR if sigma_radial is None else float(sigma_radial)
        sa = self.DEFAULT_SA if sigma_azimuthal is None else float(sigma_azimuthal)
        profile = self._get_profile(sr, sa)

        # operator=None → GPU convolution path (matched_filter_score), avoiding the
        # fused Triton kernel; identical score, fast enough for one on-demand frame.
        S, _bg, _good = bayes_score_map(
            panels, good, self._radius, self._angle, self.params,
            noise_model=None, profile=profile, operator=None,
        )
        S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
        arr = S.detach().cpu().numpy()
        smin = float(arr.min()) if arr.size else 0.0
        smax = float(arr.max()) if arr.size else 1.0
        return arr.astype(np.float16), smin, smax
