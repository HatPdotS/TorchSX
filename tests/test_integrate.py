"""Unit tests for reflection integration + per-pixel noise propagation.

Validates :func:`torchsx.datareduction.integrate.integrate_reflections` against
analytic expectations on synthetic frames with a known signal and a known
:class:`NoiseModel`. The background is supplied explicitly (``radial_bg=``) so the
integration/propagation math is tested independently of the radial-background
estimator (which is degenerate on a *perfectly* flat frame — see the noisy test).
"""

from __future__ import annotations

import math

import torch

from torchsx.datareduction.integrate import IntegrationResult, integrate_reflections
from torchsx.datareduction.noise_model import NoiseModel
from torchsx.datareduction._schema import FS, PANEL, SS


def _disk_npix(R: int) -> int:
    o = torch.arange(-R, R + 1)
    dss, dfs = torch.meshgrid(o, o, indexing="ij")
    return int(((dss * dss + dfs * dfs) <= R * R).sum())


def _flat_frame(P=1, n_ss=64, n_fs=64, bg=10.0):
    panels = torch.full((P, n_ss, n_fs), float(bg))
    good = torch.ones((P, n_ss, n_fs), dtype=torch.bool)
    radial_bg = torch.full((P, n_ss, n_fs), float(bg))  # supplied → exact
    radius = torch.zeros((P, n_ss, n_fs))               # unused when radial_bg given
    return panels, good, radius, radial_bg


def test_integrate_known_box_signal_and_sigma():
    """A 3×3 +50 box on bg=10, var(μ)=2+μ, disk r=3 → exact I, σ, peak, npix, bg."""
    panels, good, radius, radial_bg = _flat_frame(bg=10.0)
    panels[0, 19:22, 29:32] += 50.0                      # 9 pixels, height 60
    nm = NoiseModel(a0=2.0, a1=1.0)                      # var = 2 + μ
    spots = torch.tensor([[0.0, 20.0, 30.0]])            # panel0, ss=20, fs=30

    r = integrate_reflections(panels, good, radius, spots, nm,
                              r_int=3, radial_bg=radial_bg)

    npix = _disk_npix(3)                                 # 29
    assert int(r.npix) == npix
    assert float(r.I) == 50.0 * 9                        # 9 spot px × (60−10); bg px → 0
    assert float(r.background) == 10.0
    assert float(r.peak) == 60.0
    # σ² = Σ var(μ): 9 spot px μ=60 (var 62) + (npix−9) bg px μ=10 (var 12)
    exp_sig = math.sqrt(9 * (2 + 60) + (npix - 9) * (2 + 10))
    assert float(r.sigma) == torch.tensor(exp_sig)


def test_flat_field_zero_signal():
    """No spot → I ≈ 0, σ ≈ sqrt(npix · var_bg)."""
    panels, good, radius, radial_bg = _flat_frame(bg=10.0)
    nm = NoiseModel(a0=0.0, a1=1.0)                      # var = μ
    spots = torch.tensor([[0.0, 32.0, 32.0]])
    r = integrate_reflections(panels, good, radius, spots, nm,
                              r_int=4, radial_bg=radial_bg)
    npix = _disk_npix(4)
    assert abs(float(r.I)) < 1e-4
    assert float(r.sigma) == torch.tensor(math.sqrt(npix * 10.0))


def test_sigma_grows_with_radius():
    """Larger integration disk → more pixels summed → larger σ on flat field."""
    panels, good, radius, radial_bg = _flat_frame(bg=20.0)
    nm = NoiseModel(a0=1.0, a1=1.0)
    spots = torch.tensor([[0.0, 32.0, 32.0]])
    s2 = integrate_reflections(panels, good, radius, spots, nm, r_int=2, radial_bg=radial_bg)
    s5 = integrate_reflections(panels, good, radius, spots, nm, r_int=5, radial_bg=radial_bg)
    assert float(s5.sigma) > float(s2.sigma)
    assert int(s5.npix) > int(s2.npix)


def test_negative_intensity_not_clamped():
    """A dip below background integrates to a negative I (unbiased, not clamped)."""
    panels, good, radius, radial_bg = _flat_frame(bg=30.0)
    panels[0, 31:34, 31:34] -= 20.0                      # a hole below bg
    nm = NoiseModel(a1=1.0)
    spots = torch.tensor([[0.0, 32.0, 32.0]])
    r = integrate_reflections(panels, good, radius, spots, nm, r_int=2, radial_bg=radial_bg)
    assert float(r.I) < 0.0


def test_masked_pixels_excluded():
    """Bad pixels in the disk are dropped from npix, I and σ."""
    panels, good, radius, radial_bg = _flat_frame(bg=10.0)
    panels[0, 19:22, 29:32] += 50.0
    nm = NoiseModel(a0=2.0, a1=1.0)
    spots = torch.tensor([[0.0, 20.0, 30.0]])
    full = integrate_reflections(panels, good, radius, spots, nm, r_int=3, radial_bg=radial_bg)
    good[0, 20, 30] = False                              # kill the central spot pixel
    masked = integrate_reflections(panels, good, radius, spots, nm, r_int=3, radial_bg=radial_bg)
    assert int(masked.npix) == int(full.npix) - 1
    assert float(masked.I) == float(full.I) - 50.0       # lost one +50 spot pixel


def test_empty_spots():
    panels, good, radius, radial_bg = _flat_frame()
    r = integrate_reflections(panels, good, radius, torch.zeros(0, 3),
                              NoiseModel(), r_int=3, radial_bg=radial_bg)
    assert r.I.numel() == 0 and r.sigma.numel() == 0 and r.npix.numel() == 0


def test_assembled_ss_maps_to_correct_panel():
    """Assembled ss (panel offset folded in) routes to the right panel's pixels."""
    P, n_ss, n_fs = 3, 40, 40
    panels = torch.full((P, n_ss, n_fs), 5.0)
    panels[2, 10, 12] += 100.0                           # spot on panel 2
    good = torch.ones((P, n_ss, n_fs), dtype=torch.bool)
    radial_bg = torch.full((P, n_ss, n_fs), 5.0)
    radius = torch.zeros((P, n_ss, n_fs))
    # assembled ss = local 10 + panel 2 * 40 = 90
    spots = torch.tensor([[2.0, 90.0, 12.0]])
    r = integrate_reflections(panels, good, radius, spots, NoiseModel(a1=1.0),
                              r_int=1, radial_bg=radial_bg)
    assert float(r.peak) == 105.0                        # found the spot on panel 2
    assert float(r.I) == 100.0


def test_radial_background_path_with_noise():
    """With a noisy flat frame the internal radial_background recovers bg≈true."""
    torch.manual_seed(0)
    P, n_ss, n_fs = 1, 96, 96
    bg = 25.0
    panels = torch.full((P, n_ss, n_fs), bg) + torch.randn(P, n_ss, n_fs) * 2.0
    panels[0, 40:43, 50:53] += 80.0
    good = torch.ones((P, n_ss, n_fs), dtype=torch.bool)
    yy, xx = torch.meshgrid(torch.arange(n_ss).float(), torch.arange(n_fs).float(), indexing="ij")
    radius = torch.sqrt((yy - 48) ** 2 + (xx - 48) ** 2).unsqueeze(0)
    spots = torch.tensor([[0.0, 41.0, 51.0]])
    r = integrate_reflections(panels, good, radius, spots, NoiseModel(a0=4.0, a1=1.0), r_int=3)
    # Global-radial bg is biased slightly HIGH near a bright spot (the spot inflates
    # its own radial rings even after outlier rejection), which biases I slightly
    # low — the known global-only limitation a local annulus (Phase C) would fix.
    assert abs(float(r.background) - bg) < 5.0           # radial bg ≈ true (noisy)
    assert float(r.I) > 80.0 * 9 * 0.7                   # most of the 9×80 signal recovered
    assert float(r.sigma) > 0
