"""Reflection integration with per-pixel noise propagation.

The standalone integration step of the data-extraction pipeline.  Detection
(:mod:`peakfinder` / :mod:`spotfinding`) finds peaks and indexing
(:mod:`indexing`) assigns the lattice; *this* module then integrates each
indexed reflection at its centroid and produces the science quantities the rest
of the stack consumes — integrated intensity ``I``, its propagated ``sigma``,
the per-observation ``background`` level, the ``peak`` height, and ``npix``.

Model (v1, locked with the user):

* **Footprint** — a fixed-radius disk ``D`` (radius ``r_int`` px) about the
  reflection centroid (CrystFEL-style). An annulus is reserved for a future
  local-background model (Phase C).
* **Background** — the per-frame **global radial** background ``b(ρ)``
  (:func:`~torchsx.datareduction.peakfinder.radial_background`), evaluated per
  pixel by its radius.  No local annulus fit in v1.
* **Noise** — the calibrated, detector-specific polynomial
  :class:`~torchsx.datareduction.noise_model.NoiseModel` ``σ²(μ)`` is the single
  authoritative noise source; everything is in detector-native units (au) and we
  assume the Gaussian approximation always holds.

Per reflection, over the valid (in-bounds, good) disk pixels ``p∈D``::

    I      = Σ_p ( v_p − b(ρ_p) )                 # NOT clamped: weak/absent → may be ≤ 0
    npix   = |{p∈D : valid}|
    peak   = max_p v_p                            # peak height (raw)
    bg     = mean_p b(ρ_p)                         # per-obs background level
    σ_I²   = Σ_p NoiseModel.variance(μ_p) ,  μ_p = max(v_p, b(ρ_p), 0)
    σ      = sqrt(σ_I²)

This is exactly "per-pixel detector noise propagated to the integrated
intensity".  The background-*level* estimate variance (``npix²·Var(b̄)``) is
negligible for the global-radial model — ``b(ρ)`` is an azimuthal mean over a
whole ring (hundreds–thousands of pixels) — so it is omitted in v1; it would
matter only for a small local annulus (Phase C).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._schema import FS, PANEL, SS
from .noise_model import NoiseModel
from .peakfinder import radial_background


@dataclass
class IntegrationResult:
    """Per-reflection integration outputs, all shape ``(M,)`` (``npix`` int64)."""

    I: Tensor
    sigma: Tensor
    background: Tensor
    peak: Tensor
    npix: Tensor

    def cpu(self) -> "IntegrationResult":
        """Return a copy with every field moved to host (chunk-building boundary)."""
        return IntegrationResult(
            I=self.I.cpu(), sigma=self.sigma.cpu(), background=self.background.cpu(),
            peak=self.peak.cpu(), npix=self.npix.cpu(),
        )


def integrate_reflections(
    panels: Tensor,           # (P, n_ss, n_fs) frame data (au)
    good: Tensor,             # (P, n_ss, n_fs) bool, True = good pixel
    radius: Tensor,           # (P, n_ss, n_fs) radial distance from beam center (px)
    spots: Tensor,            # (M, ≥3) [panel, assembled_ss, fs, ...] centroids
    noise_model: NoiseModel | None = None,
    *,
    r_int: int = 4,
    radial_bg: Tensor | None = None,
    n_bg_iters: int = 5,
    bin_width: float = 1.0,
    sigma_floor: float = 1e-6,
) -> IntegrationResult:
    """Integrate ``spots`` on ``panels`` over a fixed-radius disk with propagated σ.

    ``spots`` carries **assembled** ``ss`` (panel offset folded in, as emitted by
    the finders); it is converted back to panel-local coordinates here.  Pixels
    outside the panel or flagged bad are dropped from each disk.  ``noise_model``
    defaults to a pure-Poisson ``NoiseModel()`` (σ²=μ) when not supplied.
    """
    if noise_model is None:
        noise_model = NoiseModel()
    device = panels.device
    P, n_ss, n_fs = panels.shape
    M = spots.shape[0]
    if M == 0:
        z = torch.zeros(0, device=device)
        return IntegrationResult(z, z, z, z, z.to(torch.int64))

    val = panels.to(torch.float32)
    good = good.to(torch.bool)
    if radial_bg is None:
        radial_bg = radial_background(
            val.reshape(-1), good.reshape(-1), radius.reshape(-1).to(torch.float32),
            n_iters=n_bg_iters, bin_width=bin_width,
        ).view(P, n_ss, n_fs)
    bg = radial_bg.to(torch.float32)

    # centroid → panel-local center pixel
    panel = spots[:, PANEL].to(torch.int64)
    c_ss = (spots[:, SS] - panel.to(torch.float32) * n_ss).round().to(torch.int64)
    c_fs = spots[:, FS].round().to(torch.int64)

    # fixed-radius disk offsets (shared across reflections)
    R = int(r_int)
    o = torch.arange(-R, R + 1, device=device)
    doff_ss, doff_fs = torch.meshgrid(o, o, indexing="ij")
    doff_ss = doff_ss.reshape(-1)                      # (W,)
    doff_fs = doff_fs.reshape(-1)
    in_disk = (doff_ss * doff_ss + doff_fs * doff_fs) <= R * R   # (W,)

    win_ss = c_ss[:, None] + doff_ss[None, :]          # (M, W)
    win_fs = c_fs[:, None] + doff_fs[None, :]
    in_bounds = (win_ss >= 0) & (win_ss < n_ss) & (win_fs >= 0) & (win_fs < n_fs)

    ss_c = win_ss.clamp(0, n_ss - 1)
    fs_c = win_fs.clamp(0, n_fs - 1)
    flat = panel[:, None] * (n_ss * n_fs) + ss_c * n_fs + fs_c   # (M, W)

    v = val.reshape(-1)[flat]                          # (M, W)
    b = bg.reshape(-1)[flat]
    g = good.reshape(-1)[flat]
    m = in_bounds & g & in_disk[None, :]               # (M, W) valid disk pixels
    mf = m.to(torch.float32)

    npix = m.sum(dim=1)
    I = ((v - b) * mf).sum(dim=1)                      # NOT clamped (signed)

    mu = torch.maximum(v, b).clamp(min=0.0)            # per-pixel mean estimate
    var = noise_model.variance(mu)                     # (M, W)
    sigma = torch.sqrt((var * mf).sum(dim=1).clamp(min=sigma_floor))

    background = (b * mf).sum(dim=1) / npix.clamp(min=1).to(torch.float32)
    peak = torch.where(m, v, torch.full_like(v, float("-inf"))).max(dim=1).values
    peak = torch.where(npix > 0, peak, torch.zeros_like(peak))

    return IntegrationResult(I=I, sigma=sigma, background=background,
                             peak=peak, npix=npix)
