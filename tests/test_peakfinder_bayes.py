"""Tests for the fully-Bayesian matched-filter finder (``find_peaks_bayes``).

The defining property over the per-pixel z-finder (``find_peaks_prob``) is
**spatial coupling**: a pixel's significance is the oriented matched-filter score
``S = (g ⋆ w)/sqrt(g² ⋆ good)``, so a coherent multi-pixel blob outscores an
isolated hot pixel of equal height. Synthetic tests need no native deps; the
profile-bootstrap test uses a tiny in-memory fake reader; the end-to-end test
uses the TORO clone / real lyso data and skips cleanly when absent.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torchsx.datareduction import spotfinding as sf
from torchsx.datareduction.frames import JF_PANEL_SS
from torchsx.datareduction.noise_model import NoiseModel, calibrate_spot_profile
from torchsx.datareduction.peakfinder import (
    PROB,
    TorchPeakFinderParams,
    find_peaks_bayes,
    find_peaks_prob,
)
from torchsx.datareduction.spot_profile import SpotProfile, matched_filter_score
from torchsx.geometry import Geometry

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


def _planted(centers, *, n_panels=2, ss=514, fs=400, peak_sigma=1.0,
             bg_base=4.0, bg_slope=0.01, noise=1.0, seed=0, zingers=()):
    """Planted Gaussian blobs (``centers``) + optional single-pixel ``zingers``.

    ``centers`` / ``zingers`` are ``(panel, cy, cx, height)``. Returns
    ``panels, mask, pix_r, angle`` — the four inputs ``find_peaks_bayes`` needs.
    """
    torch.manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(ss).float(), torch.arange(fs).float(),
                            indexing="ij")
    pix_r = torch.hypot(yy - ss / 2, xx - fs / 2)
    angle = torch.atan2(yy - ss / 2, xx - fs / 2)
    bg = bg_base + bg_slope * pix_r
    panels = (bg.expand(n_panels, ss, fs).clone()
              + noise * torch.randn(n_panels, ss, fs))
    for (pi, cy, cx, a) in centers:
        panels[pi] += a * torch.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * peak_sigma ** 2))
    for (pi, cy, cx, a) in zingers:
        panels[pi, cy, cx] += a
    mask = torch.ones(n_panels, ss, fs, dtype=torch.bool)
    pix_r = pix_r.expand(n_panels, ss, fs).clone()
    angle = angle.expand(n_panels, ss, fs).clone()
    return panels, mask, pix_r, angle


def test_recovers_planted_peaks_ranked_by_probability():
    centers = [(0, 100, 200, 600.0), (0, 300, 50, 600.0), (1, 250, 350, 120.0)]
    panels, mask, pix_r, angle = _planted(centers)
    spots = find_peaks_bayes(panels, mask, pix_r, angle, TorchPeakFinderParams(top_n=200))
    assert spots.shape[1] == 5
    found = [(float(r[sf.SS]), float(r[sf.FS]), float(r[PROB])) for r in spots]
    for (pi, cy, cx, _a) in centers:
        target = (pi * JF_PANEL_SS + cy, cx)
        assert any(abs(s - target[0]) < 0.5 and abs(f - target[1]) < 0.5
                   for s, f, _p in found), f"missed {target}"
    probs = spots[:, PROB]
    assert bool(((probs >= 0) & (probs <= 1)).all())
    bright = [p for s, f, p in found
              if (abs(s - 100) < 1 and abs(f - 200) < 1)
              or (abs(s - 300) < 1 and abs(f - 50) < 1)]
    assert len(bright) == 2 and min(bright) > 0.9


def test_blob_outscores_zinger():
    """Core regression: a multi-pixel blob beats a single hot pixel of equal height.

    The matched filter accumulates the blob's neighbour evidence, so its score is
    ~π× a same-height delta's. The per-pixel z-finder has no such coupling, so it
    rates the two ~equally — we assert both, the property `find_peaks_prob` lacks.
    """
    from torchsx.datareduction.peakfinder import bayes_score_map

    height = 200.0
    blob = [(0, 150, 150, height)]
    zinger = [(0, 150, 300, height)]  # same panel, same height, far apart
    panels, mask, pix_r, angle = _planted(blob, n_panels=1, peak_sigma=1.0,
                                          zingers=zinger, seed=1)
    params = TorchPeakFinderParams(top_n=50)
    # Use a calibrated-style noise law (σ from background level, not the empirical
    # per-shell σ, which an extended bright blob inflates at its own radius).
    nm = NoiseModel(a0=1.0, a1=0.0)

    # matched-filter score: the blob clearly outscores the equal-height zinger
    S, _b, _g = bayes_score_map(panels, mask, pix_r, angle, params, noise_model=nm)
    s_blob = float(S[0, 150, 150])
    s_zing = float(S[0, 150, 300])
    assert s_blob > 1.8 * s_zing, (s_blob, s_zing)

    # both are still found, and the blob's probability is never below the zinger's
    spots = find_peaks_bayes(panels, mask, pix_r, angle, params, noise_model=nm)

    def _prob_at(spots, cx):
        rows = [(abs(float(r[sf.FS]) - cx), float(r[PROB])) for r in spots]
        return min(rows)[1]  # prob of the row nearest that column

    assert _prob_at(spots, 150) >= _prob_at(spots, 300)

    # the z-map finder has no spatial coupling → the two are ~equally significant
    zspots = find_peaks_prob(panels, mask, pix_r, params)
    assert abs(_prob_at(zspots, 150) - _prob_at(zspots, 300)) < 0.2


def test_always_returns_top_n_even_on_blank():
    panels, mask, pix_r, angle = _planted([], noise=1.0, seed=5)
    params = TorchPeakFinderParams(top_n=200, prob_midpoint=5.0, prob_scale=1.5)
    spots = find_peaks_bayes(panels, mask, pix_r, angle, params)
    assert spots.shape == (200, 5)
    assert float(spots[:, PROB].max()) < 0.5


def test_top_n_cap_and_empty():
    centers = [(0, 100 + 9 * k, 50 + 6 * k, 400.0) for k in range(30)]
    panels, mask, pix_r, angle = _planted(centers, n_panels=1)
    spots = find_peaks_bayes(panels, mask, pix_r, angle, TorchPeakFinderParams(top_n=10))
    assert spots.shape[0] == 10
    out = find_peaks_bayes(torch.zeros(1, 50, 50),
                           torch.zeros(1, 50, 50, dtype=torch.bool),
                           torch.zeros(1, 50, 50), torch.zeros(1, 50, 50),
                           TorchPeakFinderParams())
    assert out.shape == (0, 5)


def test_matched_filter_score_is_standardized_on_noise():
    # pure N(0,1) residual, all-good mask → S has ~unit variance
    torch.manual_seed(0)
    w = torch.randn(1, 200, 200)
    good = torch.ones(1, 200, 200, dtype=torch.bool)
    angle = torch.zeros(1, 200, 200)
    S = matched_filter_score(w, good, angle, SpotProfile.FALLBACK)
    assert abs(float(S.std()) - 1.0) < 0.15
    assert abs(float(S.mean())) < 0.1


def test_radius_banded_profile_recovers_peaks():
    # an explicit 2-band (radius-dependent) curve: tighter near the center,
    # broader at the edge. Planted isotropic peaks should still be recovered.
    centers = [(0, 100, 200, 600.0), (0, 300, 50, 600.0)]
    panels, mask, pix_r, angle = _planted(centers)
    prof = SpotProfile(
        radii=torch.tensor([50.0, 250.0]),
        sigma_radial=torch.tensor([1.0, 2.0]),
        sigma_azimuthal=torch.tensor([1.0, 2.5]),
        kernel_radius=5, n_orient=8,
    )
    assert prof.n_bands == 2
    spots = find_peaks_bayes(panels, mask, pix_r, angle, TorchPeakFinderParams(top_n=100),
                             profile=prof, noise_model=NoiseModel(a0=1.0, a1=0.0))
    found = [(float(r[sf.SS]), float(r[sf.FS])) for r in spots]
    for (pi, cy, cx, _a) in centers:
        target = (pi * JF_PANEL_SS + cy, cx)
        assert any(abs(s - target[0]) < 1.0 and abs(f - target[1]) < 1.0
                   for s, f in found), f"missed {target}"


class _FakeReader:
    """Minimal reader over a dict of planted frames for the bootstrap test."""

    def __init__(self, frames, mask):
        self._frames = frames
        self._mask = mask
        self.path = "fake"

    @property
    def n_frames(self):
        return len(self._frames)

    def panels(self, i):
        return self._frames[i]

    def mask_panels(self):
        return self._mask


def _fake_geom(ss, fs):
    yy, xx = torch.meshgrid(torch.arange(ss).float(), torch.arange(fs).float(),
                            indexing="ij")
    return SimpleNamespace(
        pixel_radius_px=torch.hypot(yy - ss / 2, xx - fs / 2),
        pixel_angle_rad=torch.atan2(yy - ss / 2, xx - fs / 2),
    )


def test_calibrate_spot_profile_recovers_widths_and_falls_back():
    ss, fs, sig = 120, 120, 1.3
    geom = _fake_geom(ss, fs)
    nm = NoiseModel(a0=1.0, a1=0.0)             # σ ≈ 1
    mask = torch.ones(1, ss, fs, dtype=torch.bool)

    # plant ~24 isotropic blobs over 2 frames
    frames = []
    for s in range(2):
        centers = [(0, 15 + 10 * (k // 10), 15 + 10 * (k % 10), 300.0) for k in range(12)]
        p, _m, _r, _a = _planted(centers, n_panels=1, ss=ss, fs=fs,
                                 peak_sigma=sig, seed=10 + s)
        frames.append(p)
    reader = _FakeReader(frames, mask)
    prof = calibrate_spot_profile(reader, geom, nm, [0, 1], min_spots=10, min_snr=6.0,
                                  n_radius_bins=2, min_per_bin=4)
    # isotropic input → both width curves ≈ planted σ at every control point
    assert float(prof.sigma_radial.mean()) == pytest.approx(sig, abs=0.45)
    assert float(prof.sigma_azimuthal.mean()) == pytest.approx(sig, abs=0.45)
    assert prof.n_bands >= 1

    # blank frames → too few spots → fall back to the isotropic init unchanged
    blank, _m, _r, _a = _planted([], n_panels=1, ss=ss, fs=fs, seed=99)
    blank_reader = _FakeReader([blank], mask)
    init = SpotProfile.constant(0.7, 0.7)
    fb = calibrate_spot_profile(blank_reader, geom, nm, [0], init=init, min_spots=10)
    assert fb is init


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def test_build_matched_filter_operator_shapes():
    """The operator builds (CPU ok) with the right tiny gtab + per-pixel brackets."""
    from torchsx.datareduction.spot_profile import build_matched_filter_operator
    ss, fs, P = 64, 80, 2
    yy, xx = torch.meshgrid(torch.arange(ss).float(), torch.arange(fs).float(), indexing="ij")
    angle = torch.atan2(yy - ss / 2, xx - fs / 2).expand(P, ss, fs).contiguous()
    radius = torch.hypot(yy - ss / 2, xx - fs / 2).expand(P, ss, fs).contiguous()
    good = torch.ones(P, ss, fs, dtype=torch.bool)
    prof = SpotProfile(radii=torch.tensor([10.0, 30.0]),
                       sigma_radial=torch.tensor([1.0, 2.0]),
                       sigma_azimuthal=torch.tensor([1.0, 1.5]),
                       kernel_radius=4, n_orient=8)
    op = build_matched_filter_operator(angle, radius, good, prof)
    Kw = (2 * 4 + 1) ** 2
    assert op.gtab.numel() == prof.n_bands * prof.n_orient * Kw
    assert op.rlo.shape == (P, ss, fs) and op.olo.shape == (P, ss, fs)
    assert op.a0.shape == (P, ss, fs) and op.a3.shape == (P, ss, fs)


@cuda
@pytest.mark.parametrize("profile_kind", ["multiband", "constant"])
def test_fused_matched_filter_equals_conv_path(profile_kind):
    """Fused Triton operator == conv ``matched_filter_score`` at true fp32.

    Ampere cuDNN conv uses TF32 (~1e-3 rel) by default; the fused path is exact
    fp32, so we disable TF32 to compare apples-to-apples and expect ~roundoff.
    Exercises a masked rectangle + stripe so the on-the-fly denominator and the
    panel/mask bounds are covered.
    """
    from torchsx.datareduction.spot_profile import (
        build_matched_filter_operator, matched_filter_score_fused,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    torch.manual_seed(0)
    P, ss, fs = 4, 514, 1030
    w = torch.randn(P, ss, fs, device=dev)
    mask = torch.ones(P, ss, fs, dtype=torch.bool, device=dev)
    mask[:, 100:160, 200:260] = False
    mask[:, 300, :] = False
    yy, xx = torch.meshgrid(torch.arange(ss, device=dev).float(),
                            torch.arange(fs, device=dev).float(), indexing="ij")
    yy = (yy - ss / 2)[None].expand(P, -1, -1)
    xx = (xx - fs / 2)[None].expand(P, -1, -1)
    angle = torch.atan2(yy, xx).contiguous()
    radius = torch.sqrt(xx * xx + yy * yy).contiguous()
    w = torch.where(mask, w, torch.zeros_like(w)).contiguous()
    if profile_kind == "multiband":
        prof = SpotProfile(radii=torch.tensor([50.0, 150.0, 300.0, 500.0]),
                           sigma_radial=torch.tensor([1.0, 1.5, 2.0, 2.5]),
                           sigma_azimuthal=torch.tensor([1.0, 1.2, 1.6, 2.0]),
                           kernel_radius=6, n_orient=12)
    else:
        prof = SpotProfile.constant(1.4, 1.0, kernel_radius=4, n_orient=12)

    op = build_matched_filter_operator(angle, radius, mask, prof)
    S_ref = matched_filter_score(w, mask, angle, prof, radius=radius)
    S_fast = matched_filter_score_fused(w, op)
    f = torch.isfinite(S_ref) & torch.isfinite(S_fast) & mask
    assert float((S_fast[f] - S_ref[f]).abs().max()) < 1e-3


@toro
@realdata
@pytest.mark.parametrize("frame", [60, 233, 814])
def test_end_to_end_bayes_indexes_hewl(frame):
    from torchsx.datareduction.frames import JungfrauReader
    from torchsx.datareduction.indexing import HEWL_CELL, index_spots

    geom = Geometry.from_geom_file(str(_GEOM))
    reader = JungfrauReader(str(_ACQ))
    spots = sf.find_spots_on_frame(reader, frame, geom, backend="bayes")
    reader.close()
    assert spots.shape[1] == 5
    q = sf.spots_to_q(spots, geom)
    res = index_spots(q, HEWL_CELL, weights=spots[:, PROB], device="cpu", min_num_spots=8)
    assert res.success
    assert float(res.cell_lengths[0]) == pytest.approx(79.0, abs=3.0)
    assert float(res.cell_lengths[2]) == pytest.approx(38.0, abs=3.0)
    assert res.n_indexed >= 20
