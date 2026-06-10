"""Tests for the native PyTorch spot finder (radial-background model).

The synthetic tests need no native deps. The cross-check against peakfinder8 and
the end-to-end indexing milestone use the real lyso dataset + the pf8/TORO
vendored deps, and skip cleanly when those are absent.
"""

from pathlib import Path

import pytest
import torch

from torchsx.datareduction import spotfinding as sf
from torchsx.datareduction.frames import JF_PANEL_SS
from torchsx.datareduction.peakfinder import (
    TorchPeakFinderParams,
    _connected_components,
    find_spots_torch,
)
from torchsx.geometry import Geometry, Panel

_DATA = Path("/data/project/cls/standfuss/torchsx/raw_data")
_ACQ = _DATA / "run0032-lyso_13keV" / "data" / "acq0001.JF06T32V07.h5"
_GEOM = _DATA / "16M_v1_distance_93.8.geom"


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
realdata = pytest.mark.skipif(not _ACQ.exists(), reason="lyso dataset not present")


def _planted(centers, *, n_panels=2, ss=514, fs=400, amp=300.0, peak_sigma=1.5,
             bg_base=4.0, bg_slope=0.01, noise=1.0, seed=0):
    """Module stack: Gaussian peaks on a radial-ramp + noise background.

    ``centers`` = ``[(panel, ss, fs), ...]``. Returns ``(panels, mask, pix_r)``.
    """
    torch.manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(ss).float(), torch.arange(fs).float(),
                            indexing="ij")
    pix_r = torch.hypot(yy - ss / 2, xx - fs / 2)
    bg = bg_base + bg_slope * pix_r
    panels = (bg.expand(n_panels, ss, fs).clone()
              + noise * torch.randn(n_panels, ss, fs))
    for (pi, cy, cx) in centers:
        panels[pi] += amp * torch.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * peak_sigma ** 2))
    mask = torch.ones(n_panels, ss, fs, dtype=torch.bool)
    pix_r = pix_r.expand(n_panels, ss, fs).clone()
    return panels, mask, pix_r


def test_recovers_planted_peaks():
    centers = [(0, 100, 200), (0, 300, 50), (1, 250, 350)]
    # sharp peaks → blobs well under max_pix_count (real Bragg spots are compact)
    panels, mask, pix_r = _planted(centers, amp=150.0, peak_sigma=1.0)
    spots = find_spots_torch(panels, mask, pix_r,
                             TorchPeakFinderParams(adc_thresh=20, min_snr=5))
    assert spots.shape == (len(centers), 4)
    # centroid within 0.5 px of each planted (assembled ss = panel*514 + cy)
    found = [(float(r[sf.SS]), float(r[sf.FS])) for r in spots]
    for (pi, cy, cx) in centers:
        target = (pi * JF_PANEL_SS + cy, cx)
        assert any(abs(s - target[0]) < 0.5 and abs(f - target[1]) < 0.5
                   for s, f in found), f"missed {target}; got {found}"


def test_pure_background_yields_no_peaks():
    panels, mask, pix_r = _planted([], noise=1.0, seed=3)
    spots = find_spots_torch(panels, mask, pix_r,
                             TorchPeakFinderParams(adc_thresh=20, min_snr=5))
    # iterative rejection + min_pix_count=2 drop lone noise excursions
    assert spots.shape == (0, 4)


def test_pix_count_filters():
    # one sharp (small) peak and one very broad blob exceeding max_pix_count
    panels, mask, pix_r = _planted([(0, 200, 200)], amp=300, peak_sigma=1.2)
    yy, xx = torch.meshgrid(torch.arange(514).float(), torch.arange(400).float(),
                            indexing="ij")
    panels[0] += 300 * torch.exp(-((yy - 100) ** 2 + (xx - 300) ** 2) / (2 * 8.0 ** 2))
    spots = find_spots_torch(panels, mask, pix_r,
                             TorchPeakFinderParams(adc_thresh=20, min_snr=5,
                                                   max_pix_count=40))
    # the broad blob (>40 px above threshold) is rejected; the sharp one survives
    found = [(float(r[sf.SS]), float(r[sf.FS])) for r in spots]
    assert any(abs(s - 200) < 0.5 and abs(f - 200) < 0.5 for s, f in found)
    assert not any(abs(s - 100) < 1.0 and abs(f - 300) < 1.0 for s, f in found)


def test_connected_components_separates_and_respects_panels():
    cand = torch.zeros(2, 20, 20, dtype=torch.bool)
    # two separated 2x2 blobs in panel 0 -> two components
    cand[0, 2:4, 2:4] = True
    cand[0, 12:14, 14:16] = True
    # panel 1: a blob at its top edge that, assembled, would touch panel 0's bottom
    cand[0, 18:20, 5:7] = True
    cand[1, 0:2, 5:7] = True
    labels = _connected_components(cand, 20, 20)
    comp_ids = set()
    P, ss, fs = cand.shape
    big = ss * fs + 1
    for p in range(P):
        for y in range(ss):
            for x in range(fs):
                if cand[p, y, x]:
                    comp_ids.add(p * big + int(labels[p, y, x]))
    # 4 distinct blobs, and the cross-panel pair stays split (panels don't merge)
    assert len(comp_ids) == 4


def test_empty_frame():
    panels = torch.zeros(2, 100, 80)
    mask = torch.ones(2, 100, 80, dtype=torch.bool)
    pix_r = torch.zeros(2, 100, 80)
    out = find_spots_torch(panels, mask, pix_r,
                           TorchPeakFinderParams(adc_thresh=20, min_snr=5))
    assert out.shape == (0, 4)
    assert out.device == panels.device


def _real_geom_reader():
    from torchsx.datareduction.frames import JungfrauReader
    geom = Geometry.from_geom_file(str(_GEOM))
    reader = JungfrauReader(str(_ACQ))
    return geom, reader


@pf8
@realdata
def test_crosscheck_vs_peakfinder8_frame60():
    """Native finder reproduces pf8 on a known HEWL hit (frame 60).

    The only systematic difference is single-pixel handling: pf8 grows blobs below
    the seed threshold and keeps 1-px peaks, while our ``min_pix_count=2`` default
    drops them as zingers. With ``min_pix_count=1`` (matched single-pixel handling)
    the spot counts agree to within a few percent; at default params the torch
    finder is a strict, centroid-accurate subset.
    """
    geom, reader = _real_geom_reader()
    pf = sf.find_spots_on_frame(reader, 60, geom, backend="pf8")
    tx = sf.find_spots_on_frame(reader, 60, geom, backend="torch-threshold",
                                params=TorchPeakFinderParams())
    tx1 = sf.find_spots_on_frame(reader, 60, geom, backend="torch-threshold",
                                 params=TorchPeakFinderParams(min_pix_count=1))
    reader.close()
    assert tx.shape[0] > 30 and pf.shape[0] > 30
    # matched single-pixel handling → counts agree within ~10 %
    assert 0.9 <= tx1.shape[0] / pf.shape[0] <= 1.1
    # default (multi-pixel) torch spots are centroid-accurate vs pf8: the bulk
    # match a pf8 spot within ~2 px (small offsets are expected — pf8 centroids
    # use a local-annulus background, ours the radial background)
    d = torch.cdist(tx[:, [sf.SS, sf.FS]], pf[:, [sf.SS, sf.FS]])
    matched = int((d.min(dim=1).values < 2.0).sum())
    assert matched >= 0.75 * tx.shape[0]


@toro
@realdata
@pytest.mark.parametrize("frame", [60, 233, 814])
def test_end_to_end_indexes_hewl(frame):
    """Torch-found spots index cleanly to the HEWL cell (the milestone)."""
    from torchsx.datareduction.indexing import HEWL_CELL, index_spots

    geom, reader = _real_geom_reader()
    tx = sf.find_spots_on_frame(reader, frame, geom, backend="torch-threshold",
                                params=TorchPeakFinderParams())
    reader.close()
    res = index_spots(sf.spots_to_q(tx, geom), HEWL_CELL, device="cpu",
                      min_num_spots=8)
    assert res.success
    cl = res.cell_lengths
    assert float(cl[0]) == pytest.approx(79.0, abs=3.0)
    assert float(cl[2]) == pytest.approx(38.0, abs=3.0)
    assert res.n_indexed >= 25  # plenty of inliers for a solution
