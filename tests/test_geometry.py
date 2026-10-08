"""Tests for Geometry."""

import numpy as np
import torch

from torchsx.geometry import Geometry, Panel


def test_geometry_from_stream(real_stream):
    """Parse the geometry header of a real stream.

    Photon energy and camera length are 0 when the header refers to per-image
    values instead of giving numbers.
    """
    geom = Geometry.from_stream(str(real_stream))
    assert geom.n_panels > 0
    assert geom.res > 0
    assert np.isfinite(geom.photon_energy_eV) and geom.photon_energy_eV >= 0
    assert np.isfinite(geom.clen) and geom.clen >= 0
    assert [geom.panel_name_to_id[n] for n in geom.panel_names] == list(range(geom.n_panels))


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


def _max_steering_error_deg(geom: Geometry) -> float:
    """Max angle (deg) between the matched-filter kernel radial axis — the stored
    pixel-frame angle θ, mapped back to the lab via the real 3-D edge vectors as
    ``cosθ·fs + sinθ·ss`` — and the true in-plane radial ``R = P − beam_foot``.

    This is the actual quantity the anisotropic spot kernel is steered by, so it
    must be ~0 (down to the float32 quantization of the stored angle) for the
    matched filter to point radially. Lab-azimuth steering gives 90–180° here.
    """
    amap = geom.angle_map()
    worst = 0.0
    for p in geom.panels:
        a = np.asarray(p.fs_vec); b = np.asarray(p.ss_vec)
        n = np.cross(a, b); n = n / np.linalg.norm(n)
        dfs = np.arange(p.n_fs)[None, :]; dss = np.arange(p.n_ss)[:, None]
        Px = p.corner_x + dfs * a[0] + dss * b[0]
        Py = p.corner_y + dfs * a[1] + dss * b[1]
        Pz = p.coffset + dfs * a[2] + dss * b[2]
        d = p.corner_x * n[0] + p.corner_y * n[1] + p.coffset * n[2]
        foot_z = d / n[2]
        R = np.stack([Px + 0 * dss, Py + 0 * dfs, Pz - foot_z], axis=-1)
        th = amap[p.min_ss : p.max_ss + 1, p.min_fs : p.max_fs + 1].astype(np.float64)
        u = np.cos(th)[..., None] * a + np.sin(th)[..., None] * b   # kernel axis → lab
        cos = (u * R).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(R, axis=-1) + 1e-30)
        worst = max(worst, float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max()))
    return worst


def _one_panel_geom(fs_vec, ss_vec, *, cx=-300.0, cy=200.0, coffset=0.0, n=64) -> Geometry:
    p = Panel("t", 0, n - 1, 0, n - 1, cx, cy, tuple(fs_vec), tuple(ss_vec), coffset)
    return Geometry(panels=[p], photon_energy_eV=12000.0, clen=0.1, res=1e4)


def test_angle_map_steers_radially_real_geom(sample_geometry):
    """Kernel radial axis aligns with the true radial to float precision."""
    assert _max_steering_error_deg(sample_geometry) < 1e-4


def test_angle_map_steers_radially_rotated_skewed_tilted():
    """Exact for in-plane rotation, non-orthonormal (skew + non-unit) edges, and
    out-of-plane tilt — no orthonormality assumption."""
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    assert _max_steering_error_deg(_one_panel_geom((c, s, 0), (-s, c, 0))) < 1e-4
    assert _max_steering_error_deg(_one_panel_geom((1.007, 0.05, 0), (0.02, 0.987, 0))) < 1e-4
    t = np.radians(15)
    assert _max_steering_error_deg(_one_panel_geom((1, 0, 0), (0, np.cos(t), np.sin(t)))) < 1e-4


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
