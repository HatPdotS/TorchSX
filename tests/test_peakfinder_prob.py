"""Tests for the thresholdless probabilistic finder + weighted indexer.

Synthetic tests need no native deps. The weighted-indexer and end-to-end tests
use the TORO clone / real lyso data and skip cleanly when those are absent.
"""

from pathlib import Path

import pytest
import torch

from torchsx.datareduction import spotfinding as sf
from torchsx.datareduction.frames import JF_PANEL_SS
from torchsx.datareduction.peakfinder import (
    PROB,
    TorchPeakFinderParams,
    find_peaks_prob,
)
from torchsx.geometry import Geometry, Panel

_DATA = Path("/data/project/cls/standfuss/torchsx/raw_data")
_ACQ = _DATA / "run0032-lyso_13keV" / "data" / "acq0001.JF06T32V07.h5"
_GEOM = _DATA / "16M_v1_distance_93.8.geom"


def _has_toro():
    try:
        from torchsx.datareduction.indexing import _load_toro
        _load_toro()
        return True
    except ImportError:
        return False


toro = pytest.mark.skipif(not _has_toro(), reason="PyTorch TORO not cloned")
realdata = pytest.mark.skipif(not _ACQ.exists(), reason="lyso dataset not present")


def _planted(centers, *, n_panels=2, ss=514, fs=400, amp=300.0, peak_sigma=1.0,
             bg_base=4.0, bg_slope=0.01, noise=1.0, seed=0):
    torch.manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(ss).float(), torch.arange(fs).float(),
                            indexing="ij")
    pix_r = torch.hypot(yy - ss / 2, xx - fs / 2)
    bg = bg_base + bg_slope * pix_r
    panels = (bg.expand(n_panels, ss, fs).clone()
              + noise * torch.randn(n_panels, ss, fs))
    for (pi, cy, cx, a) in centers:
        panels[pi] += a * torch.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * peak_sigma ** 2))
    mask = torch.ones(n_panels, ss, fs, dtype=torch.bool)
    pix_r = pix_r.expand(n_panels, ss, fs).clone()
    return panels, mask, pix_r


def test_thresholdless_recovers_and_ranks_by_probability():
    # bright + faint planted peaks; NO adc_thresh set anywhere
    centers = [(0, 100, 200, 600.0), (0, 300, 50, 600.0),
               (1, 250, 350, 90.0)]  # last one is faint
    panels, mask, pix_r = _planted(centers)
    spots = find_peaks_prob(panels, mask, pix_r, TorchPeakFinderParams(top_n=200))
    assert spots.shape[1] == 5
    # every planted peak is recovered within ~0.5 px
    found = [(float(r[sf.SS]), float(r[sf.FS]), float(r[PROB])) for r in spots]
    for (pi, cy, cx, _a) in centers:
        target = (pi * JF_PANEL_SS + cy, cx)
        assert any(abs(s - target[0]) < 0.5 and abs(f - target[1]) < 0.5
                   for s, f, _p in found), f"missed {target}"
    # probabilities in [0,1]; the bright peaks score higher than the faint one
    probs = spots[:, PROB]
    assert bool(((probs >= 0) & (probs <= 1)).all())
    # the two bright peaks should be near p=1
    bright = [p for s, f, p in found
              if (abs(s - 100) < 1 and abs(f - 200) < 1)
              or (abs(s - 300) < 1 and abs(f - 50) < 1)]
    assert len(bright) == 2 and min(bright) > 0.9


def test_always_returns_top_n_even_on_blank():
    # pure background: still returns top_n rows, but all low probability
    panels, mask, pix_r = _planted([], noise=1.0, seed=5)
    params = TorchPeakFinderParams(top_n=200, prob_midpoint=5.0, prob_scale=1.5)
    spots = find_peaks_prob(panels, mask, pix_r, params)
    assert spots.shape == (200, 5)
    # noise maxima sit well below the strong-peak regime
    assert float(spots[:, PROB].max()) < 0.5


def test_top_n_cap_and_empty():
    centers = [(0, 100 + 7 * k, 50 + 5 * k, 400.0) for k in range(40)]
    panels, mask, pix_r = _planted(centers, n_panels=1)
    spots = find_peaks_prob(panels, mask, pix_r, TorchPeakFinderParams(top_n=10))
    assert spots.shape[0] == 10  # capped
    # fully masked → (0, 5)
    out = find_peaks_prob(torch.zeros(1, 50, 50),
                          torch.zeros(1, 50, 50, dtype=torch.bool),
                          torch.zeros(1, 50, 50), TorchPeakFinderParams())
    assert out.shape == (0, 5)


def _synthetic_lattice(n_noise=30, seed=0):
    torch.manual_seed(seed)
    a, b, c = 79.0, 79.0, 38.0
    cell = torch.tensor([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=torch.float32)
    recip = torch.linalg.inv(cell).T
    qmat, _ = torch.linalg.qr(torch.randn(3, 3))
    R = qmat @ torch.diag(torch.sign(torch.diag(qmat)))
    hkls = torch.tensor(
        [[h, k, l] for h in range(-4, 5) for k in range(-4, 5) for l in range(-6, 7)],
        dtype=torch.float32)
    q = (R @ recip @ hkls.T).T
    q = q[(q.norm(dim=1) < 0.35) & (q.norm(dim=1) > 0.02)][:90]
    n_real = q.shape[0]
    # off-lattice "noise" peaks (random reciprocal points)
    noise = (torch.rand(n_noise, 3) - 0.5) * 0.5
    q_all = torch.cat([q, noise], 0)
    weights = torch.cat([torch.ones(n_real), torch.full((n_noise,), 0.05)])
    return q_all, weights, n_real


@toro
def test_weighted_indexer_downweights_noise_and_parity():
    from torchsx.datareduction.indexing import HEWL_CELL, index_spots

    q_all, weights, n_real = _synthetic_lattice()

    # weighted: low-weight noise peaks should be ignored, cell recovered
    rw = index_spots(q_all, HEWL_CELL, weights=weights, device="cpu", min_num_spots=8)
    assert rw.success
    assert float(rw.cell_lengths[0]) == pytest.approx(79.0, abs=2.0)
    assert float(rw.cell_lengths[2]) == pytest.approx(38.0, abs=2.0)
    assert rw.weighted_inlier_mass is not None and rw.weighted_inlier_mass > 40

    # parity: uniform weights reproduce the unweighted indexer's solution
    ones = torch.ones(q_all.shape[0])
    r_uniform = index_spots(q_all, HEWL_CELL, weights=ones, device="cpu", min_num_spots=8)
    r_plain = index_spots(q_all, HEWL_CELL, device="cpu", min_num_spots=8)
    assert r_uniform.success == r_plain.success
    assert torch.allclose(r_uniform.basis, r_plain.basis, atol=1e-4)
    assert int(r_uniform.n_indexed) == int(r_plain.n_indexed)


@toro
@realdata
@pytest.mark.parametrize("frame", [60, 233, 814])
def test_end_to_end_probabilistic_indexes_hewl(frame):
    from torchsx.datareduction.frames import JungfrauReader
    from torchsx.datareduction.indexing import HEWL_CELL, index_spots

    geom = Geometry.from_geom_file(str(_GEOM))
    reader = JungfrauReader(str(_ACQ))
    spots = sf.find_spots_on_frame(reader, frame, geom, backend="torch")
    reader.close()
    assert spots.shape[1] == 5  # has the PROB column
    q = sf.spots_to_q(spots, geom)
    res = index_spots(q, HEWL_CELL, weights=spots[:, PROB], device="cpu",
                      min_num_spots=8)
    assert res.success
    assert float(res.cell_lengths[0]) == pytest.approx(79.0, abs=3.0)
    assert float(res.cell_lengths[2]) == pytest.approx(38.0, abs=3.0)
    assert res.n_indexed >= 20


@toro
@realdata
def test_blank_frame_low_hit_score():
    """A blank frame yields a low weighted inlier mass / no solution."""
    from torchsx.datareduction.frames import JungfrauReader
    from torchsx.datareduction.indexing import HEWL_CELL, index_spots

    geom = Geometry.from_geom_file(str(_GEOM))
    reader = JungfrauReader(str(_ACQ))
    # frame 0 is (per the .dap) a blank / non-hit
    spots = sf.find_spots_on_frame(reader, 0, geom, backend="torch")
    reader.close()
    q = sf.spots_to_q(spots, geom)
    res = index_spots(q, HEWL_CELL, weights=spots[:, PROB], device="cpu",
                      min_num_spots=8)
    # either no solution, or a very small weighted inlier mass relative to a hit
    if res.success:
        assert res.weighted_inlier_mass < 20
