"""Tests for the per-run noise model + empirical-null peak calibration."""

import torch

from torchsx.datareduction.noise_model import (
    EmpiricalNull,
    NoiseModel,
    calibrate_noise,
    calibrate_null,
)
from torchsx.geometry import Geometry, Panel


def _geom(ss=128, fs=128):
    # beam center near panel middle so radius spans ~0..90 px
    panel = Panel(name="m0", min_fs=0, max_fs=fs - 1, min_ss=0, max_ss=ss - 1,
                  corner_x=-fs / 2, corner_y=-ss / 2,
                  fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0)
    return Geometry(panels=[panel], photon_energy_eV=12960.0, clen=0.1, res=10000.0)


class _FakeReader:
    """Minimal reader over a precomputed (n, P, ss, fs) stack."""
    def __init__(self, frames, mask):
        self._f = frames
        self._m = mask

    @property
    def n_frames(self): return self._f.shape[0]
    @property
    def n_panels(self): return self._f.shape[1]
    def panels(self, i): return self._f[i]
    def mask_panels(self): return self._m


def _make_stack(geom, *, a0, a1, a2, n_frames, amp_jitter=0.0, seed=0):
    """Frames = A·b0(r) + noise, noise ~ N(0, a0+a1·μ+a2·μ²) with μ=A·b0(r)."""
    torch.manual_seed(seed)
    ss, fs = geom.detector_shape
    r = geom.pixel_radius_px.reshape(1, ss, fs)
    b0 = 10.0 + 0.5 * r                                   # radial ramp
    frames = torch.empty(n_frames, 1, ss, fs)
    for i in range(n_frames):
        A = 1.0 + amp_jitter * (torch.rand(1).item() - 0.5) * 2
        mu = A * b0
        var = (a0 + a1 * mu + a2 * mu * mu).clamp(min=1e-6)
        frames[i] = mu + torch.sqrt(var) * torch.randn(1, ss, fs)
    mask = torch.ones(1, ss, fs, dtype=torch.bool)
    return _FakeReader(frames, mask)


def test_noise_model_eval():
    nm = NoiseModel(a0=4.0, a1=2.0, a2=0.05, a3=0.0)
    mu = torch.tensor([0.0, 10.0, 100.0])
    assert torch.allclose(nm.variance(mu),
                          torch.tensor([4.0, 4 + 20 + 5, 4 + 200 + 500]))
    assert torch.allclose(nm.sigma(mu), nm.variance(mu).sqrt())


def test_calibrate_noise_recovers_law():
    a0, a1, a2 = 4.0, 2.0, 0.05
    geom = _geom()
    reader = _make_stack(geom, a0=a0, a1=a1, a2=a2, n_frames=25, seed=1)
    nm, dbg = calibrate_noise(reader, geom, list(range(25)), degree=2)
    # predicted σ² within ~20% of truth across the μ range
    for mu in (15.0, 30.0, 55.0):
        true = a0 + a1 * mu + a2 * mu * mu
        pred = float(nm.variance(torch.tensor(mu)))
        assert abs(pred - true) / true < 0.20, (mu, pred, true)


def test_residual_pooled_beats_per_pixel_temporal_under_jitter():
    """The user's control: with shot-to-shot background amplitude jitter, the
    per-pixel temporal variance is inflated by real background change, while the
    residual-pooled calibrate_noise stays correct."""
    a0, a1, a2 = 4.0, 2.0, 0.0
    geom = _geom()
    reader = _make_stack(geom, a0=a0, a1=a1, a2=a2, n_frames=40,
                         amp_jitter=0.8, seed=2)  # ±40% background jitter
    nm, _ = calibrate_noise(reader, geom, list(range(40)), degree=2)

    # per-pixel temporal variance (the WRONG estimator)
    stack = reader._f[:, 0]                               # (n, ss, fs)
    per_pixel_var = stack.var(dim=0)                      # (ss, fs)
    mean_mu = stack.mean(dim=0)
    # compare at a mid background level
    sel = (mean_mu > 25) & (mean_mu < 35)
    temporal = float(per_pixel_var[sel].median())
    true_mid = a0 + a1 * 30.0
    residual_pred = float(nm.variance(torch.tensor(30.0)))
    # residual-pooled ≈ truth; temporal inflated by the shot-to-shot jitter
    assert abs(residual_pred - true_mid) / true_mid < 0.30
    assert temporal > 1.5 * true_mid


def test_empirical_null_multiplicity_correction():
    """Blank maxima → low prob; real (very-high-z) peaks → high prob."""
    torch.manual_seed(0)
    n_lmax = 50000
    null = EmpiricalNull(z_sorted=torch.sort(torch.randn(n_lmax))[0],
                         n_lmax_per_frame=float(n_lmax))

    # blank frame: top-200 of a fresh null draw
    blank = torch.randn(n_lmax)
    zsel_blank = torch.topk(blank, 200).values
    p_blank = null.prob(zsel_blank)
    assert float(p_blank.median()) < 0.2          # bulk of blank maxima ~0
    assert int((p_blank > 0.9).sum()) <= 5         # at most a couple extreme flukes

    # hit frame: same background maxima + 30 real peaks at z≈12
    hit = torch.cat([torch.randn(n_lmax), torch.full((30,), 12.0)])
    zsel_hit = torch.topk(hit, 200).values
    p_hit = null.prob(zsel_hit)
    assert int((p_hit > 0.9).sum()) >= 30          # real peaks recovered confidently


def test_calibrate_null_runs():
    geom = _geom()
    reader = _make_stack(geom, a0=4.0, a1=2.0, a2=0.0, n_frames=10, seed=3)
    nm, _ = calibrate_noise(reader, geom, list(range(10)), degree=2)
    null = calibrate_null(reader, geom, nm, list(range(10)), peak_min_distance=2)
    assert null.z_sorted.numel() > 0
    assert null.n_lmax_per_frame > 0
    assert bool(torch.isfinite(null.z_sorted).all())
    # (prob() behaviour is validated consistently in the multiplicity test above,
    # which feeds one frame's worth of top maxima — not the multi-frame pool.)
