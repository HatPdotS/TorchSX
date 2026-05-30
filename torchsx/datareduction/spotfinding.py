"""Spot finding with peakfinder8 (Cheetah's algorithm) + reciprocal mapping.

Wraps the compiled ``ssc.peakfinder8_extension`` from the ``submodules/pf8`` clone.
We call the extension's per-panel ``peakfinder_8`` **directly** rather than going
through ``ssc.peak`` (whose module-level ``skimage`` / ``pypsalg`` imports are not
installed) and we pass a real per-pixel radius map so peakfinder8 can do its
radius-binned local-background estimate (the cloned wrapper hard-codes ``pix_r``
to a constant, which disables that).

The detector data is already gain-corrected to photons, so thresholds are in
*photon* units (small), not raw ADU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..geometry import Geometry
from .frames import JF_PANEL_SS

# ── locate + import the compiled peakfinder8 extension ────────────────────────
_PF8_DIR = Path(__file__).resolve().parents[2] / "submodules" / "pf8"


def _load_peakfinder8():
    if str(_PF8_DIR) not in sys.path:
        sys.path.insert(0, str(_PF8_DIR))
    try:
        from ssc.peakfinder8_extension import peakfinder_8  # type: ignore
    except ImportError as e:  # pragma: no cover - depends on build state
        raise ImportError(
            "peakfinder8 extension is not built. Build it with:\n"
            f"  cd {_PF8_DIR} && python build_ext.py build_ext --inplace\n"
            f"(original error: {e})"
        ) from e
    return peakfinder_8


@dataclass
class PeakFinder8Params:
    """peakfinder8 parameters, tuned for JUNGFRAU hit-finding.

    The detector data is already gain-corrected (``meta/conversion_factor`` ≈
    12.96 keV/photon), so values are in keV-ish units, not raw ADU. These
    defaults were calibrated against the SwissFEL online-monitor ``.dap`` log on
    the lyso run: blank frames yield ~0 peaks while real crystal hits yield tens
    to ~120, and the resulting spots index at ~90 % inlier fraction. Loosen
    ``adc_thresh`` (e.g. 20–40) if you want more spots to feed the indexer.
    """

    adc_thresh: float = 150.0
    min_snr: float = 8.0
    min_pix_count: int = 2
    max_pix_count: int = 40
    local_bg_radius: int = 4
    max_num_peaks: int = 5000


# columns of the returned peak table
PANEL, SS, FS, INTENSITY = 0, 1, 2, 3


def find_spots(
    panels: torch.Tensor,
    mask_panels: torch.Tensor,
    pix_r_panels: torch.Tensor,
    params: PeakFinder8Params | None = None,
) -> torch.Tensor:
    """Run peakfinder8 over a module stack.

    Parameters
    ----------
    panels : (P, ss, fs) tensor — per-module detector data (photons).
    mask_panels : (P, ss, fs) tensor — True/1 = good pixel.
    pix_r_panels : (P, ss, fs) tensor — radial distance from beam center (px).
    params : :class:`PeakFinder8Params`, optional.

    Returns
    -------
    spots : (M, 4) float32 tensor with columns
        ``[panel, ss, fs, intensity]`` in **assembled** detector coordinates
        (``ss`` includes the ``panel * 514`` row offset).
    """
    params = params or PeakFinder8Params()
    peakfinder_8 = _load_peakfinder8()

    p = panels.detach().cpu().numpy()
    m = mask_panels.detach().cpu().numpy()
    r = pix_r_panels.detach().cpu().numpy()
    n_panels, n_ss, n_fs = p.shape

    rows: list[np.ndarray] = []
    for pi in range(n_panels):
        data = np.ascontiguousarray(p[pi], dtype=np.float32)
        mask = np.ascontiguousarray(m[pi].astype(np.int8))  # 1 = active
        pix_r = np.ascontiguousarray(r[pi], dtype=np.float32)
        peak_x, peak_y, peak_val = peakfinder_8(
            params.max_num_peaks,
            data, mask, pix_r,
            n_fs, n_ss, 1, 1,  # asic_nx, asic_ny, nasics_x, nasics_y
            params.adc_thresh, params.min_snr,
            params.min_pix_count, params.max_pix_count,
            params.local_bg_radius,
        )
        if len(peak_val) == 0:
            continue
        peak_x = np.asarray(peak_x, dtype=np.float32)  # fast (column)
        peak_y = np.asarray(peak_y, dtype=np.float32)  # slow (row, panel-local)
        peak_val = np.asarray(peak_val, dtype=np.float32)
        block = np.empty((len(peak_val), 4), dtype=np.float32)
        block[:, PANEL] = pi
        block[:, SS] = peak_y + pi * JF_PANEL_SS  # → assembled slow-scan row
        block[:, FS] = peak_x
        block[:, INTENSITY] = peak_val
        rows.append(block)

    if not rows:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.from_numpy(np.vstack(rows))


def find_spots_on_frame(
    reader,
    i: int,
    geom: Geometry,
    params: PeakFinder8Params | None = None,
) -> torch.Tensor:
    """Convenience: read frame ``i`` and find spots, using the geometry's
    cached pixel-radius map for ``pix_r``."""
    panels = reader.panels(i).to(torch.float32)
    mask = reader.mask_panels()
    n_panels, n_ss, n_fs = panels.shape
    pix_r = geom.pixel_radius_px.reshape(n_panels, n_ss, n_fs)
    return find_spots(panels, mask, pix_r, params)


def spots_to_q(spots: torch.Tensor, geom: Geometry) -> torch.Tensor:
    """Map a ``(M, 4)`` spot table to ``(M, 3)`` reciprocal q-vectors (Å⁻¹)."""
    if spots.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32)
    ss = spots[:, SS].to(torch.float64)
    fs = spots[:, FS].to(torch.float64)
    q = geom.to_reciprocal(ss, fs)
    return q.to(torch.float32)
