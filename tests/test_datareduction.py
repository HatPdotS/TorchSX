"""Tests for the datareduction pipeline (spotfinding + indexing).

The peakfinder8 extension and the PyTorch TORO clone are optional native/vendored
deps; tests that need them skip cleanly when they are not built/cloned.
"""

import math

import numpy as np
import pytest
import torch

from torchsx.datareduction import spotfinding as sf
from torchsx.datareduction.frames import JF_PANEL_SS
from torchsx.geometry import Geometry, Panel


def _has_peakfinder8():
    try:
        sf._load_peakfinder8()
        return True
    except ImportError:
        return False


def _has_toro():
    try:
        from torchsx.datareduction.indexing import _load_toro
        _load_toro()
        return True
    except ImportError:
        return False


pf8 = pytest.mark.skipif(not _has_peakfinder8(), reason="peakfinder8 not built")
toro = pytest.mark.skipif(not _has_toro(), reason="PyTorch TORO not cloned")


def _planted_panels(centers, n_panels=2, ss=514, fs=400, amp=300.0):
    """A module stack with Gaussian peaks planted at `centers` = [(p, y, x), ...]."""
    panels = torch.zeros(n_panels, ss, fs)
    yy, xx = torch.meshgrid(torch.arange(ss).float(), torch.arange(fs).float(),
                            indexing="ij")
    for (pi, cy, cx) in centers:
        panels[pi] += amp * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2))
    mask = torch.ones(n_panels, ss, fs)
    pix_r = torch.hypot(yy - ss / 2, xx - fs / 2).expand(n_panels, ss, fs).clone()
    return panels, mask, pix_r


@pf8
def test_find_spots_recovers_planted_peaks():
    centers = [(0, 100, 200), (0, 300, 50), (1, 250, 350)]
    panels, mask, pix_r = _planted_panels(centers)
    params = sf.PeakFinder8Params(adc_thresh=20, min_snr=5)
    spots = sf.find_spots(panels, mask, pix_r, params)
    assert spots.shape[1] == 4
    assert spots.shape[0] == len(centers)
    # each planted peak should be matched (assembled ss = panel*514 + y)
    found = {(int(round(float(r[sf.SS]))), int(round(float(r[sf.FS])))) for r in spots}
    for (pi, cy, cx) in centers:
        assert (pi * JF_PANEL_SS + cy, cx) in found


def test_spots_to_q_shape_and_empty():
    panel = Panel(name="p0", min_fs=0, max_fs=399, min_ss=0, max_ss=513,
                  corner_x=-200.0, corner_y=-256.0,
                  fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0)
    geom = Geometry(panels=[panel], photon_energy_eV=12960.0, clen=0.1, res=10000.0)
    spots = torch.tensor([[0.0, 10.0, 20.0, 99.0], [0.0, 30.0, 40.0, 88.0]])
    q = sf.spots_to_q(spots, geom)
    assert q.shape == (2, 3)
    assert sf.spots_to_q(torch.empty(0, 4), geom).shape == (0, 3)


@toro
def test_index_synthetic_lattice():
    from torchsx.datareduction.indexing import HEWL_CELL, index_spots

    torch.manual_seed(0)
    a, b, c = 79.0, 79.0, 38.0
    cell = torch.tensor([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=torch.float32)
    recip = torch.linalg.inv(cell).T  # columns a*,b*,c*
    q_, _ = torch.linalg.qr(torch.randn(3, 3))
    R = q_ @ torch.diag(torch.sign(torch.diag(q_)))
    hkls = torch.tensor(
        [[h, k, l] for h in range(-4, 5) for k in range(-4, 5) for l in range(-6, 7)],
        dtype=torch.float32,
    )
    q = (R @ recip @ hkls.T).T
    q = q[(q.norm(dim=1) < 0.35) & (q.norm(dim=1) > 0.02)][:90]

    res = index_spots(q, HEWL_CELL, device="cpu", min_num_spots=8)
    assert res.success
    assert res.n_indexed >= 50
    cl = res.cell_lengths
    assert float(cl[0]) == pytest.approx(79.0, abs=2.0)
    assert float(cl[2]) == pytest.approx(38.0, abs=2.0)
    assert res.hkl.shape == (q.shape[0], 3)
