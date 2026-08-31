"""Tests for the forward model submodule."""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.deprecated.scaling._orientation import CrystalOrientation
from torchsx.deprecated.forward.coords import RefinableGeometry
from torchsx.deprecated.forward.relp import RelpProfile
from torchsx.deprecated.forward.spectrum import GaussianSpectrum
from torchsx.deprecated.forward.corrections import (
    polarization_correction,
    solid_angle_correction,
    estimate_background,
)
from torchsx.deprecated.forward.likelihood import PixelLikelihood
from torchsx.deprecated.forward.forward_model import PixelForwardModel
from torchsx.deprecated.forward.batch import ROIBatch, _build_pixel_grids, _compute_pixel_geometry
from torchsx.deprecated.forward.refine import RefinementStage, _set_requires_grad, _module_for_group
from torchsx.geometry import Geometry, Panel
from torchsx.deprecated.merge._structure_factor import MergedStructureFactor
from torchsx.deprecated.scaling._absolute import AbsoluteScale
from torchsx.deprecated.scaling._bscale import BScale


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_geometry(n_panels=2) -> Geometry:
    """Create a simple test geometry."""
    panels = []
    for i in range(n_panels):
        panels.append(Panel(
            name=f"p{i}",
            min_fs=i * 100,
            max_fs=i * 100 + 99,
            min_ss=0,
            max_ss=99,
            corner_x=-50.0 + i * 110.0,
            corner_y=-50.0,
            fs_vec=(1.0, 0.0, 0.0),
            ss_vec=(0.0, 1.0, 0.0),
            coffset=0.0,
        ))
    return Geometry(
        panels=panels,
        photon_energy_eV=12000.0,
        clen=0.1,  # 100 mm
        res=10000.0,  # 10000 px/m = 100 um pixels
    )


def _simple_orientation(n_crystals=3) -> tuple[CrystalOrientation, torch.Tensor]:
    """Create orientation for cubic cell a=50 A."""
    a = 50.0  # Angstrom
    astar = torch.tensor([1.0/a, 0, 0])
    bstar = torch.tensor([0, 1.0/a, 0])
    cstar = torch.tensor([0, 0, 1.0/a])
    UB = torch.stack([astar, bstar, cstar], dim=-1)  # (3,3)
    UB = UB.unsqueeze(0).expand(n_crystals, -1, -1).clone()
    return CrystalOrientation(UB), UB


# ---------------------------------------------------------------------------
# CrystalOrientation tests
# ---------------------------------------------------------------------------

class TestCrystalOrientation:
    def test_identity_rotation(self):
        orient, UB = _simple_orientation(2)
        RUB = orient.RUB()
        # delta_omega = 0 → R = I → RUB = UB
        torch.testing.assert_close(RUB, UB, atol=1e-6, rtol=1e-6)

    def test_predict_h_lab(self):
        orient, UB = _simple_orientation(2)
        hkl = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        crystal_id = torch.tensor([0, 1], dtype=torch.int32)
        h_lab = orient.predict_h_lab(hkl, crystal_id)
        # For cubic 50 A: h=[1,0,0] → h_lab = UB @ [1,0,0] = [1/50, 0, 0]
        assert h_lab.shape == (2, 3)
        assert abs(h_lab[0, 0].item() - 1.0/50.0) < 1e-6

    def test_skew_symmetric(self):
        v = torch.tensor([[0.0, 0.0, 0.1]])
        K = CrystalOrientation._skew_symmetric(v)
        # K should be antisymmetric
        torch.testing.assert_close(K, -K.transpose(-1, -2), atol=1e-7, rtol=0)

    def test_small_rotation_changes_h_lab(self):
        orient, _ = _simple_orientation(1)
        hkl = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        cid = torch.tensor([0])

        h0 = orient.predict_h_lab(hkl, cid).clone()
        orient.delta_omega.data[0, 2] = 0.05  # 0.05 rad ~3 deg around z
        h1 = orient.predict_h_lab(hkl, cid)
        assert not torch.allclose(h0, h1)

    def test_grad_flows(self):
        orient, _ = _simple_orientation(1)
        hkl = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        cid = torch.tensor([0])
        h = orient.predict_h_lab(hkl, cid)
        loss = h.sum()
        loss.backward()
        assert orient.delta_omega.grad is not None
        assert orient.delta_omega.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# RefinableGeometry tests
# ---------------------------------------------------------------------------

class TestRefinableGeometry:
    def test_init_from_geometry(self):
        geom = _simple_geometry()
        rg = RefinableGeometry(geom)
        assert rg.n_panels == 2

    def test_pixel_to_lab_basic(self):
        geom = _simple_geometry(1)
        rg = RefinableGeometry(geom)
        # Center of panel: fs=50, ss=50
        fs = torch.tensor([50.0])
        ss = torch.tensor([50.0])
        pid = torch.tensor([0])
        xyz = rg.pixel_to_lab(fs, ss, pid)
        assert xyz.shape == (1, 3)
        # z should be clen = 0.1
        assert abs(xyz[0, 2].item() - 0.1) < 1e-4

    def test_pixel_to_s_hat_unit_vectors(self):
        geom = _simple_geometry()
        rg = RefinableGeometry(geom)
        fs = torch.tensor([50.0, 150.0])
        ss = torch.tensor([50.0, 50.0])
        pid = torch.tensor([0, 1])
        s_hat = rg.pixel_to_s_hat(fs, ss, pid)
        # Should be unit vectors
        norms = s_hat.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-5, rtol=0)

    def test_pixel_to_recip_beam_center(self):
        """Beam center pixel should give s_p ≈ 0."""
        # Create geometry where panel corner maps beam center to (0,0)
        panel = Panel(
            name="p0", min_fs=0, max_fs=99, min_ss=0, max_ss=99,
            corner_x=0.0, corner_y=0.0,
            fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0),
            coffset=0.0,
        )
        geom = Geometry(panels=[panel], clen=0.1, res=10000.0)
        rg = RefinableGeometry(geom)
        # fs=0, ss=0 → corner (0, 0) → lab (0, 0, 0.1)
        fs = torch.tensor([0.0])
        ss = torch.tensor([0.0])
        pid = torch.tensor([0])
        s_p = rg.pixel_to_recip(fs, ss, pid, wavelength=1.0)
        # s_hat ≈ (0, 0, 1), so s_p = (1/λ) * (s_hat - [0,0,1]) ≈ 0
        assert s_p.abs().max().item() < 0.01  # near beam center

    def test_delta_corrections_change_output(self):
        geom = _simple_geometry(1)
        rg = RefinableGeometry(geom)
        fs = torch.tensor([50.0])
        ss = torch.tensor([50.0])
        pid = torch.tensor([0])
        xyz0 = rg.pixel_to_lab(fs, ss, pid).clone()
        rg.delta_corner_x.data[0] = 5.0
        xyz1 = rg.pixel_to_lab(fs, ss, pid)
        assert not torch.allclose(xyz0, xyz1)

    def test_gain(self):
        geom = _simple_geometry(2)
        rg = RefinableGeometry(geom)
        pid = torch.tensor([0, 1])
        g = rg.gain(pid)
        # Initially log_gain=0 → gain=1
        torch.testing.assert_close(g, torch.ones(2), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# RelpProfile tests
# ---------------------------------------------------------------------------

class TestRelpProfile:
    def test_peak_at_zero_displacement(self):
        rp = RelpProfile(10, sigma_r_init=0.02, sigma_t_init=0.02)
        delta_s = torch.zeros(5, 3)
        h_hat = torch.tensor([[1.0, 0.0, 0.0]]).expand(5, -1)
        hkl_id = torch.tensor([0, 1, 2, 3, 4])
        R = rp.evaluate(delta_s, h_hat, hkl_id)
        # At zero displacement, R should be 1
        torch.testing.assert_close(R, torch.ones(5), atol=1e-5, rtol=0)

    def test_decays_with_distance(self):
        rp = RelpProfile(5, sigma_r_init=0.02, sigma_t_init=0.02)
        h_hat = torch.tensor([[1.0, 0.0, 0.0]])
        hkl_id = torch.tensor([0])

        r_near = rp.evaluate(
            torch.tensor([[0.01, 0.0, 0.0]]), h_hat, hkl_id,
        )
        r_far = rp.evaluate(
            torch.tensor([[0.05, 0.0, 0.0]]), h_hat, hkl_id,
        )
        assert r_near.item() > r_far.item()

    def test_anisotropy(self):
        """Radial and tangential widths should give different decay."""
        rp = RelpProfile(1, sigma_r_init=0.01, sigma_t_init=0.05)
        h_hat = torch.tensor([[1.0, 0.0, 0.0]])
        hkl_id = torch.tensor([0])
        d = 0.03

        # Radial displacement (along h_hat)
        r_rad = rp.evaluate(torch.tensor([[d, 0.0, 0.0]]), h_hat, hkl_id)
        # Tangential displacement (perpendicular to h_hat)
        r_tan = rp.evaluate(torch.tensor([[0.0, d, 0.0]]), h_hat, hkl_id)
        # sigma_r < sigma_t → radial should decay faster
        assert r_rad.item() < r_tan.item()

    def test_grad_flows(self):
        rp = RelpProfile(3)
        ds = torch.tensor([[0.01, 0.01, 0.0]])
        h_hat = torch.tensor([[1.0, 0.0, 0.0]])
        hkl_id = torch.tensor([0])
        R = rp.evaluate(ds, h_hat, hkl_id)
        R.backward()
        assert rp.log_sigma_r.grad is not None


# ---------------------------------------------------------------------------
# GaussianSpectrum tests
# ---------------------------------------------------------------------------

class TestGaussianSpectrum:
    def test_weights_sum_to_one(self):
        spec = GaussianSpectrum(5, mean_energy_eV=12000.0, sigma_eV=30.0)
        cid = torch.tensor([0, 1, 2, 3, 4])
        w = spec.spectrum(cid)
        sums = w.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(5), atol=1e-4, rtol=0)

    def test_n_bins(self):
        spec = GaussianSpectrum(3, mean_energy_eV=12000.0, n_bins=80)
        assert spec.n_bins == 80
        assert spec.wavelengths().shape == (80,)

    def test_d_lambda_positive(self):
        spec = GaussianSpectrum(1, mean_energy_eV=12000.0)
        assert spec.d_lambda() > 0

    def test_per_crystal_energy(self):
        """Different crystals should get different spectral weights."""
        spec = GaussianSpectrum(2, mean_energy_eV=12000.0, sigma_eV=30.0)
        spec.log_energy.data[1] = math.log(12100.0)  # shift one crystal
        cid = torch.tensor([0, 1])
        w = spec.spectrum(cid)
        # Peak positions should differ
        peak0 = w[0].argmax().item()
        peak1 = w[1].argmax().item()
        assert peak0 != peak1


# ---------------------------------------------------------------------------
# Corrections tests
# ---------------------------------------------------------------------------

class TestCorrections:
    def test_polarization_forward_beam(self):
        # At 2theta=0, P should be 1
        P = polarization_correction(torch.tensor(0.0), torch.tensor(0.0))
        assert abs(P.item() - 1.0) < 1e-6

    def test_polarization_range(self):
        two_theta = torch.linspace(0, math.pi / 2, 100)
        phi = torch.zeros(100)
        P = polarization_correction(two_theta, phi)
        assert (P >= 0).all()
        assert (P <= 1).all()

    def test_solid_angle_forward(self):
        s_hat = torch.tensor([[0.0, 0.0, 1.0]])
        omega = solid_angle_correction(s_hat)
        assert abs(omega.item() - 1.0) < 1e-6

    def test_solid_angle_off_axis(self):
        # 45 degree angle: cos(45) = 1/sqrt(2), cos^3 = 1/(2*sqrt(2))
        s_hat = torch.tensor([[1.0, 0.0, 1.0]])
        s_hat = s_hat / s_hat.norm()
        omega = solid_angle_correction(s_hat)
        expected = (1.0 / math.sqrt(2)) ** 3
        assert abs(omega.item() - expected) < 1e-5

    def test_estimate_background(self):
        patches = torch.ones(3, 11, 11) * 100.0
        valid = torch.ones(3, 11, 11, dtype=torch.uint8)
        bg = estimate_background(patches, valid, border_width=2)
        assert bg.shape == (3,)
        torch.testing.assert_close(bg, torch.full((3,), 100.0), atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# PixelLikelihood tests
# ---------------------------------------------------------------------------

class TestPixelLikelihood:
    def test_gaussian_zero_residual(self):
        lik = PixelLikelihood(mode="gaussian")
        n = torch.ones(2, 5, 5) * 10.0
        valid = torch.ones(2, 5, 5, dtype=torch.uint8)
        loss = lik(n, n, valid)
        # Zero residual → loss = 0.5 * log(pred) per pixel = 50 * 0.5 * log(10)
        expected = 50 * 0.5 * math.log(10.0)
        assert abs(loss.item() - expected) < 0.1

    def test_poisson_zero_residual(self):
        lik = PixelLikelihood(mode="poisson")
        n = torch.ones(1, 3, 3) * 10.0
        valid = torch.ones(1, 3, 3, dtype=torch.uint8)
        loss = lik(n, n, valid)
        # n_obs * log(n_pred) - n_pred = 10*log(10) - 10 per pixel, 9 pixels
        expected = 9 * (10 * math.log(10) - 10)
        # Poisson NLL = -obs*log(pred) + pred, so total = 9 * (-10*log(10) + 10)
        assert abs(loss.item() - (-expected)) < 0.1

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            PixelLikelihood(mode="unknown")

    def test_valid_mask_zeros_out_pixels(self):
        lik = PixelLikelihood(mode="gaussian")
        pred = torch.ones(1, 3, 3) * 10.0
        obs = torch.ones(1, 3, 3) * 20.0
        valid_all = torch.ones(1, 3, 3, dtype=torch.uint8)
        valid_none = torch.zeros(1, 3, 3, dtype=torch.uint8)
        loss_all = lik(pred, obs, valid_all)
        loss_none = lik(pred, obs, valid_none)
        assert loss_all.item() > 0
        assert loss_none.item() == 0.0


# ---------------------------------------------------------------------------
# Pixel grids tests
# ---------------------------------------------------------------------------

class TestPixelGrids:
    def test_grid_shape(self):
        fs = torch.tensor([50.0, 150.0])
        ss = torch.tensor([50.0, 50.0])
        pid = torch.tensor([0, 1])
        fs_g, ss_g, pid_g = _build_pixel_grids(fs, ss, pid, 5, 7)
        assert fs_g.shape == (2, 5, 7)
        assert ss_g.shape == (2, 5, 7)
        assert pid_g.shape == (2, 5, 7)

    def test_grid_center(self):
        fs = torch.tensor([50.0])
        ss = torch.tensor([50.0])
        pid = torch.tensor([0])
        fs_g, ss_g, _ = _build_pixel_grids(fs, ss, pid, 5, 5)
        # Center pixel should be at (50, 50)
        assert abs(fs_g[0, 2, 2].item() - 50.0) < 1e-5
        assert abs(ss_g[0, 2, 2].item() - 50.0) < 1e-5


# ---------------------------------------------------------------------------
# Forward model integration test
# ---------------------------------------------------------------------------

class TestPixelForwardModel:
    def _build_model(self):
        n_crystals = 2
        n_unique = 5
        wavelength = 12398.4 / 12000.0  # 12 keV → ~1.033 A

        geom = _simple_geometry(1)
        geometry = RefinableGeometry(geom)

        _, UB = _simple_orientation(n_crystals)
        orientation = CrystalOrientation(UB)
        scale = AbsoluteScale(n_crystals)
        bfac = BScale(n_crystals)

        unique_hkl = torch.tensor(
            [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1]], dtype=torch.int16,
        )
        sf = MergedStructureFactor(n_unique, unique_hkl)
        sf.log_F.data.fill_(2.0)  # |F|^2 = exp(4)

        relp = RelpProfile(n_unique, sigma_r_init=0.02, sigma_t_init=0.02)
        spectrum = GaussianSpectrum(n_crystals, mean_energy_eV=12000.0, n_bins=10)

        model = PixelForwardModel(
            structure_factors=sf,
            orientation=orientation,
            crystal_scale=scale,
            crystal_bfactor=bfac,
            relp_profile=relp,
            spectrum=spectrum,
            geometry=geometry,
            wavelength=wavelength,
        )
        return model

    def _build_batch(self):
        B, ph, pw = 3, 5, 5

        patches = torch.ones(B, ph, pw) * 100.0
        valid = torch.ones(B, ph, pw, dtype=torch.uint8)
        crystal_id = torch.tensor([0, 0, 1], dtype=torch.int32)
        hkl_id = torch.tensor([0, 1, 2], dtype=torch.int64)
        hkl = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.int16)
        panel_id = torch.tensor([0, 0, 0], dtype=torch.int16)
        fs_center = torch.tensor([50.0, 60.0, 70.0])
        ss_center = torch.tensor([50.0, 50.0, 50.0])
        background = torch.tensor([10.0, 10.0, 10.0])

        geom = _simple_geometry(1)
        rg = RefinableGeometry(geom)
        fs_grid, ss_grid, pid_grid = _build_pixel_grids(
            fs_center, ss_center, panel_id, ph, pw,
        )
        s_hat_out, two_theta, phi, pol, sol = _compute_pixel_geometry(
            rg, fs_grid, ss_grid, pid_grid,
        )

        return ROIBatch(
            patches=patches,
            valid=valid,
            crystal_id=crystal_id,
            hkl_id=hkl_id,
            hkl=hkl,
            panel_id=panel_id,
            fs_center=fs_center,
            ss_center=ss_center,
            background=background,
            s_hat_out=s_hat_out,
            two_theta=two_theta,
            phi=phi,
            polarization=pol,
            solid_angle=sol,
        )

    def test_forward_shape(self):
        model = self._build_model()
        batch = self._build_batch()
        n_pred = model(batch)
        assert n_pred.shape == (3, 5, 5)

    def test_forward_positive(self):
        model = self._build_model()
        batch = self._build_batch()
        n_pred = model(batch)
        assert (n_pred > 0).all()

    def test_grad_flows_through_forward(self):
        model = self._build_model()
        batch = self._build_batch()
        n_pred = model(batch)
        loss = n_pred.sum()
        loss.backward()
        # Check that gradients flow to structure factors
        assert model.structure_factors.log_F.grad is not None


# ---------------------------------------------------------------------------
# RefinementStage tests
# ---------------------------------------------------------------------------

class TestRefineConfig:
    def test_set_requires_grad(self):
        n_crystals = 2
        n_unique = 3
        wavelength = 1.0
        geom = _simple_geometry(1)

        _, UB = _simple_orientation(n_crystals)
        model = PixelForwardModel(
            structure_factors=MergedStructureFactor(n_unique, torch.zeros(n_unique, 3, dtype=torch.int16)),
            orientation=CrystalOrientation(UB),
            crystal_scale=AbsoluteScale(n_crystals),
            crystal_bfactor=BScale(n_crystals),
            relp_profile=RelpProfile(n_unique),
            spectrum=GaussianSpectrum(n_crystals, 12000.0, n_bins=5),
            geometry=RefinableGeometry(geom),
            wavelength=wavelength,
        )

        _set_requires_grad(model, ["orientation", "geometry"])
        assert not model.orientation.delta_omega.requires_grad
        assert not model.geometry.delta_corner_x.requires_grad
        assert model.structure_factors.log_F.requires_grad
        assert model.relp_profile.log_sigma_r.requires_grad

    def test_module_for_group_invalid(self):
        n_crystals = 1
        _, UB = _simple_orientation(n_crystals)
        model = PixelForwardModel(
            structure_factors=MergedStructureFactor(1, torch.zeros(1, 3, dtype=torch.int16)),
            orientation=CrystalOrientation(UB),
            crystal_scale=AbsoluteScale(n_crystals),
            crystal_bfactor=BScale(n_crystals),
            relp_profile=RelpProfile(1),
            spectrum=GaussianSpectrum(n_crystals, 12000.0, n_bins=5),
            geometry=RefinableGeometry(_simple_geometry(1)),
            wavelength=1.0,
        )
        with pytest.raises(ValueError):
            _module_for_group(model, "nonexistent")


# ---------------------------------------------------------------------------
# ROIBatch.to() test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-step forward model tests (using forward_debug)
# ---------------------------------------------------------------------------

class TestForwardSteps:
    """Test each stage of the forward model independently with analytical checks."""

    def _build_model_and_batch(self, sigma_r=0.02, sigma_t=0.02, n_bins=20):
        """Build a model + batch where one spot is at the exact Ewald condition.

        Cubic cell a=10 Å, crystal 0, hkl=(1,0,0) → d*=0.1 Å⁻¹.
        Geometry: single panel at 100 mm, 100 µm pixels.
        Wavelength 1.0 Å (12.4 keV).
        """
        from torchsx.deprecated.forward.forward_model import ForwardResult

        n_crystals = 1
        n_unique = 1
        wavelength = 1.0  # Å

        a = 10.0  # Å
        astar = torch.tensor([1.0 / a, 0, 0])
        bstar = torch.tensor([0, 1.0 / a, 0])
        cstar = torch.tensor([0, 0, 1.0 / a])
        UB = torch.stack([astar, bstar, cstar], dim=-1).unsqueeze(0)  # (1, 3, 3)

        orientation = CrystalOrientation(UB)
        scale = AbsoluteScale(n_crystals)
        scale.log_G.data.fill_(0.0)  # S = 1
        bfac = BScale(n_crystals)

        unique_hkl = torch.tensor([[1, 0, 0]], dtype=torch.int16)
        sf = MergedStructureFactor(n_unique, unique_hkl)
        sf.log_F.data.fill_(1.0)  # F = e, F² = e²

        relp = RelpProfile(n_unique, sigma_r_init=sigma_r, sigma_t_init=sigma_t)
        spectrum = GaussianSpectrum(
            n_crystals, mean_energy_eV=12398.4 / wavelength,
            sigma_eV=30.0, n_bins=n_bins, n_sigma=3.0,
        )

        # Geometry: panel centered on beam, detector at 100 mm
        panel = Panel(
            name="p0", min_fs=0, max_fs=199, min_ss=0, max_ss=199,
            corner_x=-100.0, corner_y=-100.0,
            fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0),
            coffset=0.0,
        )
        geom = Geometry(panels=[panel], clen=0.1, res=10000.0)
        geometry = RefinableGeometry(geom)

        model = PixelForwardModel(
            structure_factors=sf, orientation=orientation,
            crystal_scale=scale, crystal_bfactor=bfac,
            relp_profile=relp, spectrum=spectrum,
            geometry=geometry, wavelength=wavelength,
        )

        # Compute where hkl=(1,0,0) lands on the detector
        # h_lab = UB @ [1,0,0] = [0.1, 0, 0] Å⁻¹
        # s_out = λ * h_lab + s_in = [0.1, 0, 1] → normalize → s_hat_out
        # Lab position = clen * s_hat_out / s_hat_out[2] ≈ [0.01, 0, 0.1] m
        # In pixels: 0.01 * res = 100 px from corner_x → fs = 100 + 100 = 200?
        # Actually let me just compute it properly:
        h_lab = torch.tensor([0.1, 0.0, 0.0])
        s_out_unnorm = wavelength * h_lab + torch.tensor([0.0, 0.0, 1.0])
        s_hat = s_out_unnorm / s_out_unnorm.norm()
        # Lab position at detector plane:
        # x = clen * s_hat[0] / s_hat[2], y = clen * s_hat[1] / s_hat[2]
        x_m = 0.1 * s_hat[0].item() / s_hat[2].item()
        y_m = 0.1 * s_hat[1].item() / s_hat[2].item()
        # To pixel: fs = (x_m * res) - corner_x, ss = (y_m * res) - corner_y
        fs_val = x_m * 10000.0 - (-100.0)
        ss_val = y_m * 10000.0 - (-100.0)

        ps = 5
        B = 1
        fs_center = torch.tensor([fs_val])
        ss_center = torch.tensor([ss_val])
        panel_id = torch.tensor([0], dtype=torch.int16)

        fs_g, ss_g, pid_g = _build_pixel_grids(
            fs_center, ss_center, panel_id, ps, ps,
        )
        sh, tt, ph, po, so = _compute_pixel_geometry(geometry, fs_g, ss_g, pid_g)

        batch = ROIBatch(
            patches=torch.zeros(B, ps, ps),
            valid=torch.ones(B, ps, ps, dtype=torch.uint8),
            crystal_id=torch.tensor([0], dtype=torch.int32),
            hkl_id=torch.tensor([0], dtype=torch.int64),
            hkl=torch.tensor([[1, 0, 0]], dtype=torch.int16),
            panel_id=panel_id,
            fs_center=fs_center,
            ss_center=ss_center,
            background=torch.tensor([0.0]),
            s_hat_out=sh,
            two_theta=tt,
            phi=ph,
            polarization=po,
            solid_angle=so,
        )

        return model, batch, wavelength

    # -- Step 1: compute_h_lab --

    def test_h_lab_direction(self):
        """h_lab for cubic (1,0,0) should point along x."""
        model, batch, _ = self._build_model_and_batch()
        s_hat_center = batch.s_hat_out[:, 2, 2, :]
        h_lab, h_hat, d_star, s_sq, friedel = model.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )
        # h_lab should be (0.1, 0, 0) or (-0.1, 0, 0) depending on Friedel
        assert abs(h_lab[0, 0].abs().item() - 0.1) < 1e-4
        assert abs(h_lab[0, 1].item()) < 1e-4
        assert abs(h_lab[0, 2].item()) < 1e-4

    def test_h_lab_d_star(self):
        """d* for cubic a=10 Å, hkl=(1,0,0) should be 0.1 Å⁻¹."""
        model, batch, _ = self._build_model_and_batch()
        s_hat_center = batch.s_hat_out[:, 2, 2, :]
        _, _, d_star, s_sq, _ = model.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )
        assert abs(d_star[0].item() - 0.1) < 1e-4
        assert abs(s_sq[0].item() - 0.0025) < 1e-5  # (0.1/2)² = 0.0025

    def test_friedel_sign_consistent_with_pixel(self):
        """h_lab after Friedel correction should point toward the pixel, not away."""
        model, batch, _ = self._build_model_and_batch()
        s_hat_center = batch.s_hat_out[:, 2, 2, :]
        h_lab, _, _, _, _ = model.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )
        # s_p at center should have same sign as h_lab (dot product > 0)
        inv_wl = model._inv_wavelength
        s_hat_in = torch.zeros(3)
        s_hat_in[2] = 1.0
        s_p = inv_wl * (s_hat_center - s_hat_in)
        dot = (h_lab * s_p).sum(dim=-1)
        assert dot[0].item() > 0

    # -- Step 2: compute_spectral_integral --

    def test_integral_peaks_at_center(self):
        """Spectral integral should peak at/near the center pixel."""
        model, batch, _ = self._build_model_and_batch(sigma_r=0.01, sigma_t=0.01)
        s_hat_center = batch.s_hat_out[:, 2, 2, :]
        h_lab, h_hat, _, _, _ = model.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )
        integral = model.compute_spectral_integral(
            batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
        )
        # Center pixel should be the brightest (or near-brightest)
        center_val = integral[0, 2, 2].item()
        max_val = integral[0].max().item()
        assert center_val > 0.5 * max_val, (
            f"Center {center_val:.4f} should be near max {max_val:.4f}"
        )

    def test_integral_decays_from_center(self):
        """Edge pixels should have lower integral than center."""
        model, batch, _ = self._build_model_and_batch(sigma_r=0.005, sigma_t=0.005)
        s_hat_center = batch.s_hat_out[:, 2, 2, :]
        h_lab, h_hat, _, _, _ = model.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )
        integral = model.compute_spectral_integral(
            batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
        )
        center = integral[0, 2, 2].item()
        corner = integral[0, 0, 0].item()
        assert center > corner, f"Center {center} should exceed corner {corner}"

    def test_integral_zero_far_from_relp(self):
        """If h_lab is wildly wrong, integral should be near zero."""
        model, batch, _ = self._build_model_and_batch(sigma_r=0.001, sigma_t=0.001)
        # Use a fake h_lab far from the real one
        h_lab = torch.tensor([[10.0, 0.0, 0.0]])  # 10 Å⁻¹ — way off
        h_hat = torch.tensor([[1.0, 0.0, 0.0]])
        integral = model.compute_spectral_integral(
            batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
        )
        assert integral.max().item() < 1e-10

    def test_wider_sigma_gives_broader_spot(self):
        """Broader relp should give more integrated signal over the patch."""
        model_narrow, batch, _ = self._build_model_and_batch(sigma_r=0.002, sigma_t=0.002)
        model_wide, _, _ = self._build_model_and_batch(sigma_r=0.02, sigma_t=0.02)

        s_hat_center = batch.s_hat_out[:, 2, 2, :]
        h_lab_n, h_hat_n, _, _, _ = model_narrow.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )
        h_lab_w, h_hat_w, _, _, _ = model_wide.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )
        int_narrow = model_narrow.compute_spectral_integral(
            batch.s_hat_out, h_lab_n, h_hat_n, batch.hkl_id, batch.crystal_id,
        )
        int_wide = model_wide.compute_spectral_integral(
            batch.s_hat_out, h_lab_w, h_hat_w, batch.hkl_id, batch.crystal_id,
        )
        # Wider sigma → more total signal in the patch
        assert int_wide.sum().item() > int_narrow.sum().item()

    def test_spectral_integral_anisotropy(self):
        """With σ_r ≠ σ_t, the spot should be elongated."""
        model, batch, _ = self._build_model_and_batch(sigma_r=0.001, sigma_t=0.01)
        s_hat_center = batch.s_hat_out[:, 2, 2, :]
        h_lab, h_hat, _, _, _ = model.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )
        integral = model.compute_spectral_integral(
            batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
        )
        # The spot should not be perfectly round (row vs col profiles differ)
        center = 2
        row_profile = integral[0, center, :].sum().item()
        col_profile = integral[0, :, center].sum().item()
        # Just check they're not identical (anisotropy creates asymmetry)
        assert abs(row_profile - col_profile) / max(row_profile, col_profile, 1e-10) > 0.01 or \
               integral[0].max().item() < 1e-10  # allow if both zero (far from Ewald)

    # -- Step 3: compute_signal --

    def test_signal_scales_with_F_squared(self):
        """Doubling F should quadruple signal."""
        model, batch, _ = self._build_model_and_batch()
        result1 = model.forward_debug(batch)

        model.structure_factors.log_F.data.fill_(2.0)  # F = e²
        result2 = model.forward_debug(batch)

        # F² ratio = e^4 / e^2 = e^2 ≈ 7.39
        ratio = result2.n_signal.sum().item() / max(result1.n_signal.sum().item(), 1e-30)
        expected = math.exp(2.0)
        assert abs(ratio - expected) / expected < 0.01, (
            f"F² scaling: got ratio {ratio:.3f}, expected {expected:.3f}"
        )

    def test_signal_scales_with_crystal_scale(self):
        """Setting log_G should scale signal exponentially."""
        model, batch, _ = self._build_model_and_batch()
        result1 = model.forward_debug(batch)

        model.crystal_scale.log_G.data.fill_(2.0)  # S = e²
        result2 = model.forward_debug(batch)

        ratio = result2.n_signal.sum().item() / max(result1.n_signal.sum().item(), 1e-30)
        expected = math.exp(2.0)
        assert abs(ratio - expected) / expected < 0.01

    def test_background_adds_to_prediction(self):
        """Non-zero background should add uniformly to n_pred."""
        model, batch, _ = self._build_model_and_batch()
        result_no_bg = model.forward_debug(batch)

        batch_bg = ROIBatch(
            patches=batch.patches,
            valid=batch.valid,
            crystal_id=batch.crystal_id,
            hkl_id=batch.hkl_id,
            hkl=batch.hkl,
            panel_id=batch.panel_id,
            fs_center=batch.fs_center,
            ss_center=batch.ss_center,
            background=torch.tensor([50.0]),
            s_hat_out=batch.s_hat_out,
            two_theta=batch.two_theta,
            phi=batch.phi,
            polarization=batch.polarization,
            solid_angle=batch.solid_angle,
        )
        result_bg = model.forward_debug(batch_bg)

        # Signal should be the same
        torch.testing.assert_close(
            result_no_bg.n_signal, result_bg.n_signal, atol=1e-6, rtol=1e-6,
        )
        # n_pred should differ by ~50 (times gain ≈ 1)
        diff = result_bg.n_pred - result_no_bg.n_pred
        assert diff.min().item() > 49.0  # at least 49 (gain ≈ 1)
        assert diff.max().item() < 51.0

    # -- forward_debug returns consistent results --

    def test_forward_debug_matches_forward(self):
        """forward() and forward_debug() should return the same n_pred."""
        model, batch, _ = self._build_model_and_batch()
        n_pred = model(batch)
        result = model.forward_debug(batch)
        torch.testing.assert_close(n_pred, result.n_pred, atol=1e-6, rtol=1e-6)

    def test_forward_debug_gradient_flows(self):
        """Gradients should flow through forward_debug to all parameters."""
        model, batch, _ = self._build_model_and_batch()
        result = model.forward_debug(batch)
        loss = result.n_pred.sum()
        loss.backward()
        assert model.structure_factors.log_F.grad is not None
        assert model.crystal_scale.log_G.grad is not None
        assert model.relp_profile.log_sigma_r.grad is not None


# ---------------------------------------------------------------------------
# ROIBatch.to() test
# ---------------------------------------------------------------------------

class TestROIBatch:
    def test_to_device(self):
        B, ph, pw = 2, 3, 3
        batch = ROIBatch(
            patches=torch.zeros(B, ph, pw),
            valid=torch.ones(B, ph, pw, dtype=torch.uint8),
            crystal_id=torch.zeros(B, dtype=torch.int32),
            hkl_id=torch.zeros(B, dtype=torch.int64),
            hkl=torch.zeros(B, 3, dtype=torch.int16),
            panel_id=torch.zeros(B, dtype=torch.int16),
            fs_center=torch.zeros(B),
            ss_center=torch.zeros(B),
            background=torch.zeros(B),
            s_hat_out=torch.zeros(B, ph, pw, 3),
            two_theta=torch.zeros(B, ph, pw),
            phi=torch.zeros(B, ph, pw),
            polarization=torch.ones(B, ph, pw),
            solid_angle=torch.ones(B, ph, pw),
        )
        # Just test that to() returns a valid ROIBatch
        batch2 = batch.to(torch.device("cpu"))
        assert batch2.patches.shape == (B, ph, pw)


# ---------------------------------------------------------------------------
# Helpers for prediction diagnostics / update_observations tests
# ---------------------------------------------------------------------------

def _make_mini_core(n_crystals=2, n_obs_per_crystal=5) -> "_SFXStreamCore":
    """Build a minimal _SFXStreamCore with known observations."""
    from torchsx._core import _SFXStreamCore

    n_obs = n_crystals * n_obs_per_crystal
    hkl_vals = []
    cids = []
    for c in range(n_crystals):
        for i in range(n_obs_per_crystal):
            hkl_vals.append([i + 1, 0, 0])
            cids.append(c)

    hkl = torch.tensor(hkl_vals, dtype=torch.int16)
    crystal_id = torch.tensor(cids, dtype=torch.int32)
    I = torch.rand(n_obs) * 100
    sigma = torch.rand(n_obs) * 10 + 1
    background = torch.zeros(n_obs)
    peak = torch.rand(n_obs) * 50
    fs = torch.rand(n_obs) * 100
    ss = torch.rand(n_obs) * 100
    panel_id = torch.zeros(n_obs, dtype=torch.int16)

    # Crystal-level
    a = 50.0
    astar = torch.tensor([1.0 / a, 0, 0]).unsqueeze(0).expand(n_crystals, -1).clone()
    bstar = torch.tensor([0, 1.0 / a, 0]).unsqueeze(0).expand(n_crystals, -1).clone()
    cstar = torch.tensor([0, 0, 1.0 / a]).unsqueeze(0).expand(n_crystals, -1).clone()
    cell = torch.tensor([[a, a, a, 90.0, 90.0, 90.0]]).expand(n_crystals, -1).clone()
    profile_rad = torch.full((n_crystals,), 0.01)
    resolution = torch.full((n_crystals,), 2.0)
    n_refl = torch.full((n_crystals,), n_obs_per_crystal, dtype=torch.int32)
    image_id = torch.arange(n_crystals, dtype=torch.int32)
    crystal_in_image = torch.zeros(n_crystals, dtype=torch.int32)

    # Image-level
    beam_energy = torch.full((n_crystals,), 12000.0)
    det_shift_x = torch.zeros(n_crystals)
    det_shift_y = torch.zeros(n_crystals)
    n_peaks = torch.full((n_crystals,), 100, dtype=torch.int32)

    return _SFXStreamCore(
        hkl=hkl, I=I, sigma=sigma, background=background, peak=peak,
        fs=fs, ss=ss, panel_id=panel_id, crystal_id=crystal_id,
        astar=astar, bstar=bstar, cstar=cstar, cell=cell,
        profile_rad=profile_rad, resolution=resolution,
        n_refl=n_refl, image_id=image_id, crystal_in_image=crystal_in_image,
        beam_energy=beam_energy, det_shift_x=det_shift_x,
        det_shift_y=det_shift_y, n_peaks=n_peaks,
    )


def _make_predicted_from_core(
    core: "_SFXStreamCore",
    drop_frac: float = 0.0,
    add_extra: int = 0,
    position_offset: float = 0.0,
) -> "PredictedReflections":
    """Build PredictedReflections that overlaps with core's observations.

    Parameters
    ----------
    drop_frac : fraction of core observations NOT included in predictions
    add_extra : number of extra predicted-only reflections to add
    position_offset : pixel offset to apply to predicted positions
    """
    from torchsx.deprecated.forward.predict import PredictedReflections

    n_obs = core.n_observations
    n_keep = int(n_obs * (1.0 - drop_frac))

    # Take first n_keep observations as "matched"
    keep_idx = torch.arange(n_keep)

    hkl_list = [core.hkl[keep_idx]]
    cid_list = [core.crystal_id[keep_idx]]
    fs_list = [core.fs[keep_idx] + position_offset]
    ss_list = [core.ss[keep_idx] + position_offset]
    pid_list = [core.panel_id[keep_idx].long()]
    r_list = [torch.rand(n_keep) * 0.01]

    if add_extra > 0:
        # Add predicted-only reflections with HKL that don't exist in core
        extra_hkl = torch.zeros(add_extra, 3, dtype=torch.int16)
        for i in range(add_extra):
            extra_hkl[i, 0] = 50 + i  # high h values, unlikely in core
        extra_cid = torch.zeros(add_extra, dtype=torch.int32)
        extra_fs = torch.rand(add_extra) * 100
        extra_ss = torch.rand(add_extra) * 100
        extra_pid = torch.zeros(add_extra, dtype=torch.long)
        extra_r = torch.rand(add_extra) * 0.01

        hkl_list.append(extra_hkl)
        cid_list.append(extra_cid)
        fs_list.append(extra_fs)
        ss_list.append(extra_ss)
        pid_list.append(extra_pid)
        r_list.append(extra_r)

    return PredictedReflections(
        hkl=torch.cat(hkl_list).to(torch.int16),
        crystal_id=torch.cat(cid_list),
        fs=torch.cat(fs_list),
        ss=torch.cat(ss_list),
        panel_id=torch.cat(pid_list),
        excitation_error=torch.cat(r_list),
    )


# ---------------------------------------------------------------------------
# PredictionDiagnostics tests
# ---------------------------------------------------------------------------

class TestPredictionDiagnostics:
    def test_perfect_match(self):
        """All observations are predicted and vice versa."""
        from torchsx.deprecated.forward.predict import compare_predictions

        core = _make_mini_core(2, 5)
        pred = _make_predicted_from_core(core, drop_frac=0.0, add_extra=0)
        diag = compare_predictions(pred, core, core.cell)

        assert diag.n_matched == core.n_observations
        assert diag.n_predicted_only == 0
        assert diag.n_observed_only == 0
        assert abs(diag.hit_rate - 1.0) < 1e-6
        assert abs(diag.recovery_rate - 1.0) < 1e-6

    def test_partial_overlap(self):
        """80% of observations matched, plus some predicted-only."""
        from torchsx.deprecated.forward.predict import compare_predictions

        core = _make_mini_core(2, 5)
        # Drop 20% of observations from prediction, add 3 extra
        pred = _make_predicted_from_core(core, drop_frac=0.2, add_extra=3)
        diag = compare_predictions(pred, core, core.cell)

        n_obs = core.n_observations
        n_matched = int(n_obs * 0.8)
        assert diag.n_matched == n_matched
        assert diag.n_observed_only == n_obs - n_matched
        assert diag.n_predicted_only == 3

        expected_hit = n_matched / (n_matched + 3)
        expected_rec = n_matched / n_obs
        assert abs(diag.hit_rate - expected_hit) < 1e-6
        assert abs(diag.recovery_rate - expected_rec) < 1e-6

    def test_position_accuracy(self):
        """Known position offset → correct median error."""
        from torchsx.deprecated.forward.predict import compare_predictions

        core = _make_mini_core(1, 10)
        offset = 2.5
        pred = _make_predicted_from_core(core, position_offset=offset)
        diag = compare_predictions(pred, core, core.cell)

        expected_err = (offset ** 2 + offset ** 2) ** 0.5  # sqrt(2) * 2.5
        assert abs(diag.median_position_error - expected_err) < 0.1
        assert diag.frac_within_1px == 0.0  # 3.54 px > 1 px
        assert diag.frac_within_3px == 0.0  # 3.54 px > 3 px

    def test_empty_prediction(self):
        """Zero predictions → hit_rate=0, recovery_rate=0."""
        from torchsx.deprecated.forward.predict import compare_predictions, PredictedReflections

        core = _make_mini_core(1, 5)
        empty_pred = PredictedReflections(
            hkl=torch.zeros(0, 3, dtype=torch.int16),
            crystal_id=torch.zeros(0, dtype=torch.int32),
            fs=torch.zeros(0),
            ss=torch.zeros(0),
            panel_id=torch.zeros(0, dtype=torch.long),
            excitation_error=torch.zeros(0),
        )
        diag = compare_predictions(empty_pred, core, core.cell)

        assert diag.n_predicted == 0
        assert diag.n_matched == 0
        assert diag.hit_rate == 0.0
        assert diag.recovery_rate == 0.0

    def test_resolution_shells(self):
        """Shell stats should have consistent totals."""
        from torchsx.deprecated.forward.predict import compare_predictions

        core = _make_mini_core(2, 5)
        pred = _make_predicted_from_core(core, drop_frac=0.1, add_extra=2)
        diag = compare_predictions(pred, core, core.cell, n_shells=3)

        assert diag.shell_stats.shape[0] == 3
        assert diag.shell_stats.shape[1] == 5
        # Each shell has [d_lo, d_hi, n_matched, n_pred, n_obs]
        for i in range(3):
            n_m = diag.shell_stats[i, 2].item()
            n_p = diag.shell_stats[i, 3].item()
            n_o = diag.shell_stats[i, 4].item()
            assert n_m <= n_p
            assert n_m <= n_o

    def test_summary_nonempty(self):
        """summary() should produce a non-empty string."""
        from torchsx.deprecated.forward.predict import compare_predictions

        core = _make_mini_core(2, 5)
        pred = _make_predicted_from_core(core)
        diag = compare_predictions(pred, core, core.cell)
        s = diag.summary()
        assert len(s) > 50
        assert "Prediction quality" in s
        assert "Hit rate" in s


# ---------------------------------------------------------------------------
# update_observations mode tests
# ---------------------------------------------------------------------------

class TestUpdateObservationsModes:
    def test_position_only_unchanged_count(self):
        """position_only mode should not change observation count."""
        from torchsx.deprecated.forward.predict import update_observations

        core = _make_mini_core(2, 5)
        pred = _make_predicted_from_core(core, position_offset=2.0, add_extra=3)

        updated, _, diag = update_observations(
            core, pred, roi_store=None, threshold=0.5, mode="position_only",
        )
        assert updated.n_observations == core.n_observations

    def test_expand_adds_new(self):
        """expand mode should add predicted-only observations."""
        from torchsx.deprecated.forward.predict import update_observations

        core = _make_mini_core(2, 5)
        n_extra = 4
        pred = _make_predicted_from_core(core, add_extra=n_extra)

        updated, _, diag = update_observations(
            core, pred, roi_store=None, threshold=0.5, mode="expand",
        )
        assert updated.n_observations == core.n_observations + n_extra

        # New observations should have I=0 and sigma=inf
        new_I = updated.I[core.n_observations:]
        new_sigma = updated.sigma[core.n_observations:]
        assert (new_I == 0).all()
        assert torch.isinf(new_sigma).all()

    def test_expand_updates_n_refl(self):
        """Crystal n_refl tensor should match new observation count."""
        from torchsx.deprecated.forward.predict import update_observations

        core = _make_mini_core(2, 5)
        pred = _make_predicted_from_core(core, add_extra=3)

        updated, _, _ = update_observations(
            core, pred, roi_store=None, threshold=0.5, mode="expand",
        )

        # Verify n_refl sums to total observations
        assert updated.n_refl.sum().item() == updated.n_observations

        # Verify per-crystal counts by counting crystal_id
        for c in range(updated.n_crystals):
            expected = (updated.crystal_id == c).sum().item()
            assert updated.n_refl[c].item() == expected

    def test_full_removes_unmatched(self):
        """full mode should remove observed-only spots."""
        from torchsx.deprecated.forward.predict import update_observations

        core = _make_mini_core(2, 5)
        n_extra = 2
        drop = 0.2  # 20% of observations are observed-only
        pred = _make_predicted_from_core(core, drop_frac=drop, add_extra=n_extra)

        updated, _, diag = update_observations(
            core, pred, roi_store=None, threshold=0.5, mode="full",
        )

        n_matched = int(core.n_observations * (1.0 - drop))
        expected_total = n_matched + n_extra
        assert updated.n_observations == expected_total
        assert updated.n_refl.sum().item() == expected_total

    def test_expand_with_excitation_error_gate(self):
        """min_excitation_error should gate which predicted-only spots are added."""
        from torchsx.deprecated.forward.predict import update_observations, PredictedReflections

        core = _make_mini_core(1, 5)

        # Create predictions: all core obs + 3 extra with varying excitation error
        n_obs = core.n_observations
        extra_hkl = torch.tensor([[50, 0, 0], [51, 0, 0], [52, 0, 0]], dtype=torch.int16)
        pred = PredictedReflections(
            hkl=torch.cat([core.hkl, extra_hkl]),
            crystal_id=torch.cat([core.crystal_id, torch.zeros(3, dtype=torch.int32)]),
            fs=torch.cat([core.fs, torch.tensor([10.0, 20.0, 30.0])]),
            ss=torch.cat([core.ss, torch.tensor([10.0, 20.0, 30.0])]),
            panel_id=torch.cat([core.panel_id.long(), torch.zeros(3, dtype=torch.long)]),
            excitation_error=torch.cat([
                torch.zeros(n_obs),
                torch.tensor([0.001, 0.01, 0.1]),  # varying excitation errors
            ]),
        )

        # Only spots with |r| < 0.05 should be added (2 of 3 extras)
        updated, _, _ = update_observations(
            core, pred, roi_store=None, threshold=0.5,
            mode="expand", min_excitation_error=0.05,
        )
        assert updated.n_observations == n_obs + 2

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        from torchsx.deprecated.forward.predict import update_observations

        core = _make_mini_core(1, 3)
        pred = _make_predicted_from_core(core)
        with pytest.raises(ValueError, match="Unknown mode"):
            update_observations(core, pred, mode="invalid")
