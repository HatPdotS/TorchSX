"""Tests for Geometry."""

import numpy as np
import torch

from torchsx.geometry import Geometry, Panel


def test_geometry_from_stream():
    """Parse geometry from the real stream file."""
    from tests.conftest import STREAM_PATH

    geom = Geometry.from_stream(STREAM_PATH)
    assert geom.n_panels == 32
    assert geom.photon_energy_eV == 12960.0
    assert abs(geom.clen - 0.0939) < 1e-4
    assert abs(geom.res - 13333.3) < 1.0
    assert "m0" in geom.panel_name_to_id
    assert "m31" in geom.panel_name_to_id
    assert geom.panel_name_to_id["m0"] == 0


def test_geometry_panel_properties(sample_geometry):
    assert sample_geometry.n_panels == 2
    assert sample_geometry.panel_names == ["m0", "m1"]
    assert sample_geometry.panels[0].n_fs == 1030
    assert sample_geometry.panels[0].n_ss == 514


def test_geometry_dict_round_trip(sample_geometry):
    d = sample_geometry.to_dict()
    geom2 = Geometry.from_dict(d)
    assert geom2.n_panels == sample_geometry.n_panels
    assert geom2.photon_energy_eV == sample_geometry.photon_energy_eV
    assert geom2.panels[0].corner_x == sample_geometry.panels[0].corner_x


def test_pixel_to_lab(sample_geometry):
    """Basic pixel_to_lab sanity check."""
    fs = torch.tensor([500.0, 500.0])
    ss = torch.tensor([250.0, 750.0])
    pid = torch.tensor([0, 1])

    xyz = sample_geometry.pixel_to_lab(fs, ss, pid)
    assert xyz.shape == (2, 3)
    # z should be close to clen
    assert abs(xyz[0, 2].item() - 0.0939) < 0.01


# ── to_lab tests ─────────────────────────────────────────────────────────

def test_to_lab_auto_resolve(sample_geometry):
    """to_lab without panel_id auto-resolves from (ss, fs) coordinates."""
    ss = torch.tensor([250.0, 750.0])  # panel 0, panel 1
    fs = torch.tensor([500.0, 500.0])

    xyz = sample_geometry.to_lab(ss, fs)
    assert xyz.shape == (2, 3)
    assert abs(xyz[0, 2].item() - 0.0939) < 0.01


def test_to_lab_matches_pixel_to_lab(sample_geometry):
    """to_lab with explicit panel_id gives same result as pixel_to_lab."""
    ss = torch.tensor([250.0, 750.0])
    fs = torch.tensor([500.0, 500.0])
    pid = torch.tensor([0, 1])

    xyz_new = sample_geometry.to_lab(ss, fs, panel_id=pid)
    xyz_old = sample_geometry.pixel_to_lab(fs, ss, pid)
    assert torch.allclose(xyz_new, xyz_old)


def test_to_lab_auto_vs_explicit(sample_geometry):
    """Auto-resolved and explicitly-provided panel_id give same results."""
    ss = torch.tensor([250.0, 750.0])
    fs = torch.tensor([500.0, 500.0])
    pid = torch.tensor([0, 1])

    xyz_auto = sample_geometry.to_lab(ss, fs)
    xyz_explicit = sample_geometry.to_lab(ss, fs, panel_id=pid)
    assert torch.allclose(xyz_auto, xyz_explicit)


def test_to_lab_cross_panel_batch(sample_geometry):
    """A single batch of positions spanning both panels."""
    # 4 points: 2 on panel 0 (ss < 514), 2 on panel 1 (ss >= 514)
    ss = torch.tensor([0.0, 513.0, 514.0, 1027.0])
    fs = torch.tensor([0.0, 1029.0, 0.0, 1029.0])

    xyz = sample_geometry.to_lab(ss, fs)
    assert xyz.shape == (4, 3)
    # All z values should be close to clen (coffset=0)
    assert torch.allclose(xyz[:, 2], torch.full((4,), 0.0939), atol=0.01)


# ── detector layout tests ────────────────────────────────────────────────

def test_detector_shape(sample_geometry):
    total_ss, total_fs = sample_geometry.detector_shape
    assert total_ss == 1028  # max_ss=1027 + 1
    assert total_fs == 1030  # max_fs=1029 + 1


def test_panel_slices(sample_geometry):
    slices = sample_geometry.panel_slices
    assert len(slices) == 2
    ss0, ss1, fs0, fs1 = slices[0]
    assert (ss0, ss1, fs0, fs1) == (0, 514, 0, 1030)
    ss0, ss1, fs0, fs1 = slices[1]
    assert (ss0, ss1, fs0, fs1) == (514, 1028, 0, 1030)


def test_panel_id_map(sample_geometry):
    pid_map = sample_geometry._panel_id_map
    assert pid_map.shape == (1028, 1030)
    assert pid_map[0, 0] == 0
    assert pid_map[513, 500] == 0
    assert pid_map[514, 500] == 1
    assert pid_map[1027, 500] == 1


def test_radial_map_shape(sample_geometry):
    r_map = sample_geometry.radial_map()
    total_ss, total_fs = sample_geometry.detector_shape
    assert r_map.shape == (total_ss, total_fs)
    assert r_map.dtype == np.float32


def test_panel_corners_mm(sample_geometry):
    corners = sample_geometry.panels[0].corners_mm(sample_geometry.res)
    assert corners.shape == (4, 2)
    assert corners.dtype == np.float64


def test_build_panel_map(sample_geometry):
    pmap = sample_geometry.build_panel_map()
    assert pmap.shape == (1028,)
    assert pmap[0] == 0
    assert pmap[513] == 0
    assert pmap[514] == 1
    assert pmap[1027] == 1


def test_pixel_to_mm(sample_geometry):
    """pixel_to_mm should match to_lab output converted to mm."""
    ss = np.array([250, 750])
    fs = np.array([500, 500])
    x_mm, y_mm = sample_geometry.pixel_to_mm(ss, fs)

    ss_t = torch.tensor([250.0, 750.0])
    fs_t = torch.tensor([500.0, 500.0])
    xyz = sample_geometry.to_lab(ss_t, fs_t)
    np.testing.assert_allclose(x_mm, xyz[:, 0].numpy() * 1000, atol=1e-6)
    np.testing.assert_allclose(y_mm, xyz[:, 1].numpy() * 1000, atol=1e-6)
