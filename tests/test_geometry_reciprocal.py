"""Tests for the reciprocal-space transform added to Geometry."""

import math

import pytest
import torch

from torchsx.geometry import Geometry, Panel


def _flat_geom(photon_energy_eV=12398.419843320026, clen=0.1, res=10000.0):
    """A single flat 512x512 panel centered on the beam.

    With this photon energy the wavelength is exactly 1.0 Å, which makes the
    |q| = 2 sin(theta)/lambda check easy to read.
    """
    panel = Panel(
        name="p0", min_fs=0, max_fs=511, min_ss=0, max_ss=511,
        corner_x=-256.0, corner_y=-256.0,
        fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0,
    )
    return Geometry(panels=[panel], photon_energy_eV=photon_energy_eV,
                    clen=clen, res=res)


def test_wavelength():
    g = _flat_geom()
    assert g.wavelength_A == pytest.approx(1.0, rel=1e-6)
    g2 = _flat_geom(photon_energy_eV=12960.0)
    assert g2.wavelength_A == pytest.approx(0.95667, abs=1e-4)


def test_wavelength_requires_energy():
    g = _flat_geom(photon_energy_eV=0.0)
    with pytest.raises(ValueError):
        _ = g.wavelength_A


def test_beam_center_q_is_zero():
    g = _flat_geom()
    # beam center pixel: corner_x + fs = 0 -> fs = 256, ss = 256
    ss = torch.tensor([256.0])
    fs = torch.tensor([256.0])
    q = g.to_reciprocal(ss, fs)
    assert float(q.norm()) < 1e-6


def test_q_magnitude_matches_bragg():
    g = _flat_geom()  # lambda = 1.0 A
    # an off-center pixel
    ss = torch.tensor([256.0])
    fs = torch.tensor([456.0])  # 200 px from center along fs
    q = g.to_reciprocal(ss, fs)
    # expected |q| = 2 sin(theta)/lambda, with 2theta = atan(r/clen)
    r_m = 200.0 / g.res
    two_theta = math.atan2(r_m, g.clen)
    expected = 2 * math.sin(two_theta / 2) / g.wavelength_A
    assert float(q.norm()) == pytest.approx(expected, rel=1e-5)


def test_pixel_radius_px_shape_and_center():
    g = _flat_geom()
    prad = g.pixel_radius_px
    assert prad.shape == (512, 512)
    # the minimum radius pixel sits at the beam center (corner offset 256,256)
    flat_min = prad.argmin()
    ss_c, fs_c = int(flat_min // 512), int(flat_min % 512)
    assert (ss_c, fs_c) == (256, 256)
    assert float(prad[256, 256]) == pytest.approx(0.0, abs=1.0)
