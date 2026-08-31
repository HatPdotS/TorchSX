"""Tests for the per-pixel empirical calibration (``torchsx.datareduction.calibration``).

The constant channel ``c_p``/``σ_p`` and the empirical noise ``P(e|μ)`` are validated on
synthetic frames (no native deps, CPU): a planted beam-fixed pattern must be recovered as
``c_p``; pure-background frames must whiten to ``N(0,1)`` via the normal-score transform;
and the finder must still recover planted peaks when driven through the ``calib`` path.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from torchsx.datareduction.calibration import (
    Calibration,
    EmpiricalNoise,
    PerPixelModel,
    _score_residual,
    calibrate_empirical_noise,
    calibrate_per_pixel,
    whitened_residual,
)
from torchsx.datareduction.noise_model import NoiseModel
from torchsx.datareduction.peakfinder import PROB, TorchPeakFinderParams, find_peaks_bayes


def _fake_geom(ss, fs):
    yy, xx = torch.meshgrid(torch.arange(ss).float(), torch.arange(fs).float(), indexing="ij")
    return SimpleNamespace(
        pixel_radius_px=torch.hypot(yy - ss / 2, xx - fs / 2),
        pixel_angle_rad=torch.atan2(yy - ss / 2, xx - fs / 2),
    )


class _FakeReader:
    """Frames = bg(r) + fixed_pattern(p) + N(0,σ) (+ optional planted peaks per frame)."""

    def __init__(self, *, ss=96, fs=96, n=20, sigma=1.0, fixed=None, bg_base=5.0,
                 bg_slope=0.02, peaks=(), seed=0):
        g = torch.manual_seed(seed)
        yy, xx = torch.meshgrid(torch.arange(ss).float(), torch.arange(fs).float(), indexing="ij")
        r = torch.hypot(yy - ss / 2, xx - fs / 2)
        bg = bg_base + bg_slope * r
        self.fixed = torch.zeros(ss, fs) if fixed is None else fixed
        self._frames = []
        for k in range(n):
            f = bg + self.fixed + sigma * torch.randn(ss, fs, generator=g)
            for (cy, cx, a, sg) in peaks:
                f += a * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg ** 2))
            self._frames.append(f.unsqueeze(0))            # (1, ss, fs)
        self._mask = torch.ones(1, ss, fs, dtype=torch.bool)
        self.path = "fake"
        self.n_panels = 1

    @property
    def n_frames(self):
        return len(self._frames)

    def panels(self, i):
        return self._frames[i]

    def mask_panels(self):
        return self._mask


def test_normal_score_identity_on_gaussian_table():
    # If the calibrated CDF is exactly N(0,1), Φ⁻¹(F(e)) must return e (the transform
    # is a no-op on truly Gaussian noise). Build a 1-μ-bin table from N(0,1).
    egrid = torch.linspace(-10.0, 60.0, 1401, dtype=torch.float64)
    ec = 0.5 * (egrid[1:] + egrid[:-1])
    pdf = torch.exp(-ec ** 2 / 2) / math.sqrt(2 * math.pi)
    Pn = (pdf / pdf.sum()).unsqueeze(0)
    emp = EmpiricalNoise(Pn=Pn, cdf_lo=torch.cumsum(Pn, 1) - Pn,
                         medges=torch.tensor([-1e9, 1e9], dtype=torch.float64),
                         egrid=egrid, noise_model=NoiseModel(a0=1.0))
    e = torch.linspace(-4, 4, 17, dtype=torch.float64)
    w = emp.normal_score(e, torch.zeros_like(e))
    assert float((w - e.float()).abs().max()) < 0.1


def test_score_residual_modes():
    v = torch.tensor([10.0, 10.0, 10.0]); b = torch.full((3,), 4.0); s = torch.full((3,), 2.0)
    fp = torch.tensor([0.5, 0.0, -0.5])
    assert torch.allclose(_score_residual(v, b, s), v - b)
    assert torch.allclose(_score_residual(v, b, s, fixed_pattern=fp), v - b - fp * s)
    c = torch.ones(3); sc = torch.full((3,), 2.0)
    assert torch.allclose(_score_residual(v, b, s, c=c, scale=sc), (v - b - c) / sc)


def test_calibrate_per_pixel_recovers_fixed_pattern_and_scale():
    ss = fs = 96
    geom = _fake_geom(ss, fs)
    # a beam-fixed, zero-shell-mean azimuthal pattern (radial_background leaves it in e)
    fixed = 2.0 * torch.sin(2 * geom.pixel_angle_rad)
    sigma = 1.2
    reader = _FakeReader(ss=ss, fs=fs, n=40, sigma=sigma, fixed=fixed, seed=1)
    nm = NoiseModel(a0=sigma ** 2, a1=0.0)
    ppm = calibrate_per_pixel(reader, geom, nm, list(range(reader.n_frames)))
    assert ppm.c.shape == (1, ss, fs) and ppm.sigma.shape == (1, ss, fs)
    c = ppm.c.reshape(-1); f = fixed.reshape(-1)
    # c_p tracks the planted fixed pattern (high correlation), and σ_p ≈ planted noise
    corr = torch.corrcoef(torch.stack([c, f]))[0, 1]
    assert float(corr) > 0.9, float(corr)
    assert float(ppm.sigma.median()) == pytest.approx(sigma, abs=0.35)


def test_calibrate_empirical_noise_whitens_blank_to_normal():
    ss = fs = 96
    geom = _fake_geom(ss, fs)
    sigma = 1.0
    reader = _FakeReader(ss=ss, fs=fs, n=40, sigma=sigma, seed=2)
    nm = NoiseModel(a0=sigma ** 2, a1=0.0)
    idx = list(range(reader.n_frames))
    ppm = calibrate_per_pixel(reader, geom, nm, idx)
    emp = calibrate_empirical_noise(reader, geom, nm, idx, per_pixel=ppm)
    good = reader.mask_panels()
    radius = geom.pixel_radius_px.reshape(1, ss, fs)
    # held-out-style blank (new noise draw) → whitened residual ≈ N(0,1)
    blank = _FakeReader(ss=ss, fs=fs, n=1, sigma=sigma, seed=999).panels(0)
    w = whitened_residual(blank, good, radius, noise_model=nm, empirical=emp, per_pixel=ppm)
    assert w.shape == (1, ss, fs)
    assert abs(float(w.mean())) < 0.1
    assert abs(float(w.std()) - 1.0) < 0.2


def test_find_peaks_bayes_with_calib_recovers_peak():
    # the full wiring on CPU: calibrate, then drive find_peaks_bayes through the calib
    # (normal-score) whitening branch and confirm a planted blob is found + ranks high.
    ss = fs = 96
    geom = _fake_geom(ss, fs)
    sigma = 1.0
    cal_reader = _FakeReader(ss=ss, fs=fs, n=30, sigma=sigma, seed=3)
    nm = NoiseModel(a0=sigma ** 2, a1=0.0)
    idx = list(range(cal_reader.n_frames))
    ppm = calibrate_per_pixel(cal_reader, geom, nm, idx)
    emp = calibrate_empirical_noise(cal_reader, geom, nm, idx, per_pixel=ppm)
    calib = Calibration(noise_model=nm, empirical=emp, per_pixel=ppm)

    peak = _FakeReader(ss=ss, fs=fs, n=1, sigma=sigma, peaks=[(48, 60, 80.0, 1.0)], seed=7)
    panels = peak.panels(0)
    mask = peak.mask_panels()
    radius = geom.pixel_radius_px.reshape(1, ss, fs)
    angle = geom.pixel_angle_rad.reshape(1, ss, fs)
    spots = find_peaks_bayes(panels, mask, radius, angle,
                             TorchPeakFinderParams(top_n=50), noise_model=nm, calib=calib)
    assert spots.shape[1] == 5
    found = [(float(r[1]), float(r[2])) for r in spots]      # (ss, fs)
    assert any(abs(s - 48) < 1.5 and abs(f - 60) < 1.5 for s, f in found), found[:5]


def test_bragg_logratio_boosts_positive_excursions():
    # the signal-model log-LR ℓ(e|μ) must be ≈0 on background, grow with positive e
    # (peak-like), and go negative for subtractive shadows (e<0).
    ss = fs = 96
    geom = _fake_geom(ss, fs)
    sigma = 1.0
    reader = _FakeReader(ss=ss, fs=fs, n=40, sigma=sigma, seed=4)
    nm = NoiseModel(a0=sigma ** 2, a1=0.0)
    idx = list(range(reader.n_frames))
    emp = calibrate_empirical_noise(reader, geom, nm, idx, bragg_mean=8.0)
    assert emp.logratio is not None and emp.logratio.shape == emp.Pn.shape
    # evaluate at a fixed mid-range background μ
    b = torch.full((5,), 5.0)
    e = torch.tensor([-3.0, 0.0, 3.0, 8.0, 20.0])
    ell = emp.bragg_loglr(e, b)
    assert ell[0] < 0.5                      # shadow / background: not boosted
    assert ell[-1] > ell[1] + 1.0            # bright peak strongly boosted vs background
    assert bool((ell[1:] - ell[:-1] >= -1e-3).all())   # monotone non-decreasing in e


def test_calibration_to_device_is_noop_on_cpu():
    emp = EmpiricalNoise(Pn=torch.ones(2, 3) / 3, cdf_lo=torch.zeros(2, 3),
                         medges=torch.tensor([0.0, 1.0, 2.0]),
                         egrid=torch.linspace(-1, 1, 4), noise_model=NoiseModel())
    ppm = PerPixelModel(c=torch.zeros(1, 4, 4), sigma=torch.ones(1, 4, 4),
                        n_samples=torch.ones(1, 4, 4))
    calib = Calibration(noise_model=NoiseModel(), empirical=emp, per_pixel=ppm)
    moved = calib.to("cpu")
    assert moved.empirical.Pn.device.type == "cpu"
    assert moved.per_pixel.c.device.type == "cpu"
