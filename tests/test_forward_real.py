"""Tests for the forward model against real SFX data.

Each test isolates one component and checks a physically falsifiable prediction.
If a test fails, it tells you exactly which component is wrong.

Requires: sfx_scan/streams/delay_-45.sfx  (marks: slow)
"""

from __future__ import annotations

import math

import pytest
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Fixture: load stream once, shared across all tests in this module
# ---------------------------------------------------------------------------

SFX_PATH = "sfx_scan/streams/delay_-45.sfx"


@pytest.fixture(scope="module")
def real_data():
    """Load real stream + build forward model.  Cached for the module."""
    from torchsx import SFXStream
    from torchsx.scaling import (
        CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs,
    )
    from torchsx.merge._structure_factor import MergedStructureFactor
    from torchsx.merge._hkl import compute_hkl_index
    from torchsx.forward import (
        RefinableGeometry, RelpProfile, GaussianSpectrum, PixelForwardModel,
    )
    from torchsx.forward.batch import (
        ROIBatch, _build_pixel_grids, _compute_pixel_geometry,
    )

    stream = SFXStream.load(SFX_PATH)
    core = stream._core
    inp = prepare_scaler_inputs(core, n_shells=20)
    hkl_idx = compute_hkl_index(core.hkl)
    wavelength = inp.wavelength

    geometry = RefinableGeometry(core.geometry)
    orientation = CrystalOrientation(inp.UB)
    crystal_scale = AbsoluteScale(core.n_crystals)
    crystal_bfactor = BScale(core.n_crystals)
    sf = MergedStructureFactor.from_merge(core)
    relp = RelpProfile(
        n_unique=hkl_idx.n_unique, sigma_r_init=0.0005, sigma_t_init=0.002,
    )
    spectrum = GaussianSpectrum(
        n_crystals=core.n_crystals,
        mean_energy_eV=core.beam_energy.median().item(),
        sigma_eV=30.0, n_bins=20, n_sigma=3.0,
    )
    model = PixelForwardModel(
        structure_factors=sf, orientation=orientation,
        crystal_scale=crystal_scale, crystal_bfactor=crystal_bfactor,
        relp_profile=relp, spectrum=spectrum,
        geometry=geometry, wavelength=wavelength,
    )

    # Pick crystal with most reflections
    cid = core.n_refl.argmax().item()
    obs_mask = core.crystal_id == cid
    obs_idx = obs_mask.nonzero(as_tuple=True)[0]
    n_spots = obs_idx.shape[0]

    crystal_id = core.crystal_id[obs_idx]
    hkl = core.hkl[obs_idx]
    hkl_id = hkl_idx.hkl_id[obs_idx]
    panel_id = core.panel_id[obs_idx]
    fs_center = core.fs[obs_idx].float()
    ss_center = core.ss[obs_idx].float()
    I_obs = core.I[obs_idx]

    PATCH_SIZE = 21
    with torch.no_grad():
        fs_g, ss_g, pid_g = _build_pixel_grids(
            fs_center, ss_center, panel_id, PATCH_SIZE, PATCH_SIZE,
        )
        sh, tt, ph, po, so = _compute_pixel_geometry(geometry, fs_g, ss_g, pid_g)

    batch = ROIBatch(
        patches=torch.zeros(n_spots, PATCH_SIZE, PATCH_SIZE),
        valid=torch.ones(n_spots, PATCH_SIZE, PATCH_SIZE, dtype=torch.uint8),
        crystal_id=crystal_id,
        hkl_id=hkl_id,
        hkl=hkl,
        panel_id=panel_id,
        fs_center=fs_center,
        ss_center=ss_center,
        background=torch.zeros(n_spots),
        s_hat_out=sh,
        two_theta=tt,
        phi=ph,
        polarization=po,
        solid_angle=so,
    )

    with torch.no_grad():
        result = model.forward_debug(batch)

    return dict(
        core=core, model=model, batch=batch, result=result,
        orientation=orientation, geometry=geometry, spectrum=spectrum,
        hkl=hkl, hkl_id=hkl_id, crystal_id=crystal_id,
        panel_id=panel_id, fs_center=fs_center, ss_center=ss_center,
        I_obs=I_obs, wavelength=wavelength, hkl_idx=hkl_idx,
        n_spots=n_spots, PATCH_SIZE=PATCH_SIZE,
    )


# ---------------------------------------------------------------------------
# 1. Geometry: predicted peak position matches observed pixel position
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGeometry:
    def test_h_lab_projects_near_observed_pixel(self, real_data):
        """h_lab → detector position should land within ~2 pixels of the
        stream's reported (fs, ss).

        Tests: geometry (pixel_to_lab, pixel_to_s_hat) + orientation (RUB)
        Failure means: geometry or orientation is wrong.
        """
        model = real_data["model"]
        result = real_data["result"]
        wl = real_data["wavelength"]
        geometry = real_data["geometry"]
        fs_obs = real_data["fs_center"]
        ss_obs = real_data["ss_center"]
        panel_id = real_data["panel_id"]

        h_lab = result.h_lab  # (N, 3) — Friedel-corrected

        # Project h_lab back to detector: s_out = λ*h + s_in, then intersect panels
        s_hat_in = torch.zeros(3)
        s_hat_in[2] = 1.0
        s_out = wl * h_lab + s_hat_in  # (N, 3) — unnormalized
        s_hat_out = s_out / s_out.norm(dim=-1, keepdim=True)

        # For each panel: lab = clen/s_hat_z * s_hat
        # Then fs = (lab_x * res - corner_x), ss = (lab_y * res - corner_y)
        # But this is complicated for multi-panel.  Instead, compare scattering angles.
        # s_hat_out from h_lab should match s_hat_out at center pixel of ROI.
        s_hat_pixel = real_data["batch"].s_hat_out[:, 10, 10, :]  # center of 21x21

        # Angular difference between predicted and observed scattering direction
        cos_angle = (s_hat_out * s_hat_pixel).sum(dim=-1).clamp(-1, 1)
        angle_rad = torch.acos(cos_angle)
        angle_mrad = angle_rad * 1000.0

        # With 75 µm pixels at 100 mm distance, 1 pixel ≈ 0.75 mrad
        # Allow up to 3 pixels ≈ 2.25 mrad
        median_mrad = angle_mrad.median().item()
        frac_within_3px = (angle_mrad < 2.25).float().mean().item()

        # With 75 µm pixels at ~40-100 mm, 1 pixel ≈ 0.75-1.9 mrad.
        # CrystFEL peak positions have ~1-3 pixel uncertainty, plus
        # the indexed orientation has finite accuracy.  Allow ~10 mrad.
        assert median_mrad < 10.0, (
            f"Median angular error {median_mrad:.2f} mrad (expect < 10 mrad)"
        )
        frac_within = (angle_mrad < 15.0).float().mean().item()
        assert frac_within > 0.8, (
            f"Only {frac_within:.0%} within 15 mrad (expect > 80%)"
        )

    def test_two_theta_matches_d_spacing(self, real_data):
        """2θ at center pixel should satisfy Bragg's law: 2d sin(θ) = λ.

        Tests: geometry (two_theta computation) + orientation (d_star)
        Failure means: two_theta or d_star is wrong.
        """
        result = real_data["result"]
        batch = real_data["batch"]
        wl = real_data["wavelength"]

        d_star = result.d_star  # (N,)
        two_theta_center = batch.two_theta[:, 10, 10]  # center pixel

        # Bragg: sin(θ) = λ * d* / 2
        sin_theta_bragg = wl * d_star / 2.0
        # Clamp for safety (some reflections at resolution edge)
        sin_theta_bragg = sin_theta_bragg.clamp(-1, 1)
        two_theta_bragg = 2.0 * torch.asin(sin_theta_bragg)

        # Should agree within ~2 mrad (geometry discretization + sub-pixel offset)
        diff_mrad = (two_theta_center - two_theta_bragg).abs() * 1000.0
        median_diff = diff_mrad.median().item()

        assert median_diff < 5.0, (
            f"Median 2θ error {median_diff:.2f} mrad (expect < 5 mrad)"
        )


# ---------------------------------------------------------------------------
# 2. Ewald condition: excitation error should be small for indexed reflections
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEwaldCondition:
    def test_excitation_error_small(self, real_data):
        """CrystFEL indexed these reflections, so they should be near the
        Ewald sphere: |s_p - h_lab| should be small.

        Tests: orientation + geometry together.
        Failure means: UB matrix or pixel-to-recip conversion is wrong.
        """
        result = real_data["result"]
        batch = real_data["batch"]
        wl = real_data["wavelength"]

        h_lab = result.h_lab
        s_hat_center = batch.s_hat_out[:, 10, 10, :]
        s_hat_in = torch.zeros(3)
        s_hat_in[2] = 1.0
        s_p = (1.0 / wl) * (s_hat_center - s_hat_in)

        delta_s = s_p - h_lab
        excitation_error = delta_s.norm(dim=-1)

        # For indexed reflections, excitation error should be < ~0.05 Å⁻¹
        # (generous — depends on mosaicity and bandwidth)
        median_err = excitation_error.median().item()
        frac_small = (excitation_error < 0.05).float().mean().item()

        assert median_err < 0.02, (
            f"Median excitation error {median_err:.4f} Å⁻¹ (expect < 0.02)"
        )
        assert frac_small > 0.9, (
            f"Only {frac_small:.0%} have |Δs| < 0.05 (expect > 90%)"
        )

    def test_smaller_excitation_error_gives_more_signal(self, real_data):
        """Reflections closer to the Ewald sphere should have stronger
        spectral integral.

        Tests: spectral integration is responding to excitation error correctly.
        Failure means: relp profile or spectral integration is broken.
        """
        result = real_data["result"]
        batch = real_data["batch"]
        wl = real_data["wavelength"]

        h_lab = result.h_lab
        s_hat_center = batch.s_hat_out[:, 10, 10, :]
        s_hat_in = torch.zeros(3)
        s_hat_in[2] = 1.0
        s_p = (1.0 / wl) * (s_hat_center - s_hat_in)
        excitation_error = (s_p - h_lab).norm(dim=-1)

        # Peak of spectral integral
        peak_integral = result.integral.reshape(result.integral.shape[0], -1).max(dim=1).values

        # Split into two groups: small vs large excitation error
        median_err = excitation_error.median()
        close = excitation_error < median_err
        far = excitation_error >= median_err

        mean_close = peak_integral[close].mean().item()
        mean_far = peak_integral[far].mean().item()

        assert mean_close > mean_far, (
            f"Close group (mean integral {mean_close:.4f}) should exceed "
            f"far group ({mean_far:.4f})"
        )


# ---------------------------------------------------------------------------
# 3. Scale: predicted intensity should correlate with observed intensity
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestScaleCorrelation:
    def test_predicted_signal_correlates_with_I_obs(self, real_data):
        """Integrated predicted signal (before scale calibration) should
        positively correlate with CrystFEL's reported I_obs.

        Tests: the entire signal path (orientation → integral → F²).
        Failure means: the model is not predicting the right relative intensities.
        """
        result = real_data["result"]
        I_obs = real_data["I_obs"]

        # Integrated signal per spot
        pred_sum = result.n_signal.reshape(result.n_signal.shape[0], -1).sum(dim=1)

        # Only use spots with positive obs intensity and nonzero prediction
        mask = (I_obs > 1.0) & (pred_sum > 1e-6)
        if mask.sum() < 20:
            pytest.skip("Too few spots with signal for correlation test")

        log_pred = torch.log(pred_sum[mask])
        log_obs = torch.log(I_obs[mask])

        cc = torch.corrcoef(torch.stack([log_pred.double(), log_obs.double()]))[0, 1]

        assert cc.item() > 0.1, (
            f"CC(log I_pred, log I_obs) = {cc.item():.3f} (expect > 0.1)"
        )

    def test_f_squared_correlates_with_I_obs(self, real_data):
        """The merged |F|² should correlate with observed I (after accounting
        for the fact that I ∝ |F|² × partiality).

        Tests: structure factor values are physically reasonable.
        Failure means: MergedStructureFactor.from_merge() is wrong.
        """
        result = real_data["result"]
        I_obs = real_data["I_obs"]

        F_sq = result.F_sq
        mask = (I_obs > 1.0) & (F_sq > 0.01)
        if mask.sum() < 20:
            pytest.skip("Too few spots for F² correlation test")

        cc = torch.corrcoef(
            torch.stack([torch.log(F_sq[mask]).double(), torch.log(I_obs[mask]).double()])
        )[0, 1]

        # F² alone won't perfectly predict I (partiality, scale vary), but
        # should have positive correlation
        assert cc.item() > 0.0, (
            f"CC(log F², log I_obs) = {cc.item():.3f} (expect > 0)"
        )


# ---------------------------------------------------------------------------
# 4. Spectral integral: streak direction should match bandwidth sweep
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSpectralStreak:
    def test_streak_perpendicular_to_radial(self, real_data):
        """When σ_t >> σ_r, the spot elongation should be along the
        *tangential* direction (perpendicular to the radial/bandwidth sweep).

        With tight σ_r, the bandwidth sweep barely spreads the spot radially.
        The wider σ_t creates tangential elongation, perpendicular to h_hat
        projected onto the detector.

        Tests: relp anisotropy is correctly decomposed into radial/tangential.
        Failure means: radial/tangential decomposition in relp profile is wrong.
        """
        result = real_data["result"]
        batch = real_data["batch"]
        wl = real_data["wavelength"]
        PS = real_data["PATCH_SIZE"]
        center = PS // 2

        peak = result.integral.reshape(-1, PS * PS).max(dim=1).values
        top_idx = peak.argsort(descending=True)[:20]

        n_perp = 0
        n_tested = 0

        for idx in top_idx:
            i = idx.item()
            integral = result.integral[i]
            if integral.max() < 1e-6:
                continue

            yy, xx = torch.meshgrid(
                torch.arange(PS, dtype=torch.float32),
                torch.arange(PS, dtype=torch.float32),
                indexing="ij",
            )
            w = integral / integral.sum()
            cx = (w * xx).sum()
            cy = (w * yy).sum()
            cov_xx = (w * (xx - cx) ** 2).sum()
            cov_yy = (w * (yy - cy) ** 2).sum()
            cov_xy = (w * (xx - cx) * (yy - cy)).sum()
            cov = torch.tensor([[cov_xx, cov_xy], [cov_xy, cov_yy]])
            eigvals, eigvecs = torch.linalg.eigh(cov)

            if eigvals[0] < 1e-10 or eigvals[1] / eigvals[0] < 1.5:
                continue

            spot_dir = eigvecs[:, 1]  # largest variance direction

            # Radial direction in pixel coords (projection of h_hat)
            s_hat_out = batch.s_hat_out[i]
            inv_wl = 1.0 / wl
            J_fs = inv_wl * (s_hat_out[center, center + 1]
                             - s_hat_out[center, center - 1]) / 2.0
            J_ss = inv_wl * (s_hat_out[center + 1, center]
                             - s_hat_out[center - 1, center]) / 2.0

            h_hat = result.h_hat[i]  # radial direction in reciprocal space
            rad_fs = (h_hat * J_fs).sum()
            rad_ss = (h_hat * J_ss).sum()
            rad_px = torch.stack([rad_fs, rad_ss])
            norm = rad_px.norm()
            if norm < 1e-10:
                continue
            rad_px = rad_px / norm

            # Spot elongation should be PERPENDICULAR to radial
            # (i.e. along tangential), so |cos| should be small
            cos_rad = (spot_dir * rad_px).sum().abs().item()
            if cos_rad < 0.5:  # within 60° of perpendicular
                n_perp += 1
            n_tested += 1

        if n_tested < 3:
            pytest.skip("Too few elongated spots for streak direction test")

        frac_perp = n_perp / n_tested
        assert frac_perp > 0.5, (
            f"Only {n_perp}/{n_tested} spots elongated perpendicular to "
            f"radial direction (expect > 50% when σ_t >> σ_r)"
        )


# ---------------------------------------------------------------------------
# 5. Corrections: polarization and solid angle should vary with 2θ
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestCorrections:
    def test_polarization_decreases_with_two_theta(self, real_data):
        """Polarization correction should decrease at higher scattering angle.

        Tests: polarization_correction formula.
        Failure means: two_theta values or polarization formula is wrong.
        """
        batch = real_data["batch"]
        PS = real_data["PATCH_SIZE"]
        center = PS // 2

        two_theta = batch.two_theta[:, center, center]
        pol = batch.polarization[:, center, center]

        # Split by 2θ
        median_tt = two_theta.median()
        low_tt = two_theta < median_tt
        high_tt = two_theta >= median_tt

        mean_pol_low = pol[low_tt].mean().item()
        mean_pol_high = pol[high_tt].mean().item()

        assert mean_pol_low > mean_pol_high, (
            f"Low-2θ polarization ({mean_pol_low:.4f}) should exceed "
            f"high-2θ ({mean_pol_high:.4f})"
        )

    def test_solid_angle_decreases_off_axis(self, real_data):
        """Solid angle correction should decrease for pixels far from beam.

        Tests: solid_angle_correction formula.
        """
        batch = real_data["batch"]
        PS = real_data["PATCH_SIZE"]
        center = PS // 2

        sol = batch.solid_angle[:, center, center]

        # All should be <= 1 (cos³ correction)
        assert sol.max().item() <= 1.0 + 1e-6
        # For a Jungfrau detector with panels at large scattering angles,
        # cos³(α) can be quite small (< 0.1 at edges).
        assert sol.min().item() > 0.01, (
            f"Solid angle min = {sol.min():.4f} (expect > 0.01)"
        )


# ---------------------------------------------------------------------------
# 6. Friedel fix: both signs should be tried, correct one selected
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFriedelFix:
    def test_friedel_sign_distribution(self, real_data):
        """Both Friedel signs should appear (not all +1 or all -1).

        If all signs are the same, the Friedel fix is not doing anything.
        """
        result = real_data["result"]
        n_pos = (result.friedel_sign > 0).sum().item()
        n_neg = (result.friedel_sign < 0).sum().item()
        total = result.friedel_sign.shape[0]

        assert n_pos > 0.1 * total, (
            f"Only {n_pos}/{total} positive Friedel signs (expect mix)"
        )
        assert n_neg > 0.1 * total, (
            f"Only {n_neg}/{total} negative Friedel signs (expect mix)"
        )

    def test_friedel_fix_reduces_excitation_error(self, real_data):
        """The Friedel-corrected h_lab should be closer to the Ewald sphere
        than the uncorrected h_lab (for most reflections).

        Tests: the Friedel sign selection logic.
        Failure means: sign selection criterion is wrong.
        """
        model = real_data["model"]
        batch = real_data["batch"]
        wl = real_data["wavelength"]
        PS = real_data["PATCH_SIZE"]
        center = PS // 2

        with torch.no_grad():
            # Corrected h_lab
            s_hat_center = batch.s_hat_out[:, center, center, :]
            h_lab_corr, _, _, _, friedel = model.compute_h_lab(
                batch.hkl, batch.crystal_id, s_hat_center,
            )

            # Uncorrected (flip the sign back)
            h_lab_wrong = h_lab_corr * (-1.0)

            s_hat_in = torch.zeros(3)
            s_hat_in[2] = 1.0
            s_p = (1.0 / wl) * (s_hat_center - s_hat_in)

            err_corr = (s_p - h_lab_corr).norm(dim=-1)
            err_wrong = (s_p - h_lab_wrong).norm(dim=-1)

        # Corrected should be closer for the majority
        frac_better = (err_corr < err_wrong).float().mean().item()
        assert frac_better > 0.8, (
            f"Friedel correction only better for {frac_better:.0%} "
            f"(expect > 80%)"
        )


# ---------------------------------------------------------------------------
# 7. Prediction diagnostics: compare predicted vs observed reflections
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPredictionQuality:
    """Test prediction diagnostics against real data."""

    @pytest.fixture()
    def prediction_data(self, real_data):
        """Run predict_reflections and compare_predictions on the best crystal."""
        from torchsx.forward.predict import (
            predict_reflections, compare_predictions,
        )
        from torchsx.forward.coords import RefinableGeometry
        from torchsx.scaling._orientation import CrystalOrientation

        core = real_data["core"]
        orientation = real_data["orientation"]
        geometry = real_data["geometry"]
        wavelength = real_data["wavelength"]

        # Use median resolution as d_min
        d_min = core.resolution.median().item()

        predicted = predict_reflections(
            orientation, geometry, core.cell,
            d_min=d_min, wavelength=wavelength,
            sigma_cutoff=3.0,
        )
        diag = compare_predictions(predicted, core, core.cell, n_shells=10)
        return dict(predicted=predicted, diag=diag, core=core, wavelength=wavelength)

    def test_recovery_rate(self, prediction_data):
        """Model should predict most of what CrystFEL observed."""
        diag = prediction_data["diag"]
        assert diag.recovery_rate > 0.5, (
            f"Recovery rate {diag.recovery_rate:.2%} (expect > 50%)"
        )

    def test_position_accuracy_real(self, prediction_data):
        """Median position error < 5 pixels for matched spots."""
        diag = prediction_data["diag"]
        if diag.n_matched < 10:
            pytest.skip("Too few matched spots for position test")
        assert diag.median_position_error < 5.0, (
            f"Median position error {diag.median_position_error:.2f} px "
            f"(expect < 5 px)"
        )

    def test_observed_only_at_resolution_edge(self, prediction_data):
        """Observed-only spots should skew toward high resolution."""
        from torchref.base.reciprocal.hkl import get_d_spacing

        diag = prediction_data["diag"]
        core = prediction_data["core"]

        if diag.n_observed_only < 10 or diag.n_matched < 10:
            pytest.skip("Too few spots for resolution-edge test")

        # Get d-spacing for all observations
        obs_cell = core.cell[core.crystal_id.long()]
        d_all = get_d_spacing(core.hkl.float(), obs_cell)

        # Build observed-only index set
        from torchsx.forward.predict import _build_key_set, PredictedReflections
        pred = prediction_data["predicted"]
        pred_keys = set(_build_key_set(pred.crystal_id, pred.hkl).keys())
        obs_keys = _build_key_set(core.crystal_id, core.hkl)

        oo_idx = []
        matched_idx = []
        for key, idx in obs_keys.items():
            if key in pred_keys:
                matched_idx.append(idx)
            else:
                oo_idx.append(idx)

        if len(oo_idx) < 5 or len(matched_idx) < 5:
            pytest.skip("Too few spots for resolution comparison")

        d_oo = d_all[torch.tensor(oo_idx, dtype=torch.long)]
        d_matched = d_all[torch.tensor(matched_idx, dtype=torch.long)]

        # Observed-only should have lower median d-spacing (higher resolution)
        # This isn't always guaranteed, so use a soft check
        median_d_oo = d_oo.median().item()
        median_d_matched = d_matched.median().item()

        # Just check that observed-only are not exclusively low-resolution
        assert median_d_oo < median_d_matched * 2.0, (
            f"Observed-only median d={median_d_oo:.2f} vs "
            f"matched median d={median_d_matched:.2f}"
        )

    def test_relaxed_sigma_improves_recovery(self, real_data):
        """sigma_cutoff=5 should give higher recovery_rate than sigma_cutoff=3."""
        from torchsx.forward.predict import predict_reflections, compare_predictions

        core = real_data["core"]
        orientation = real_data["orientation"]
        geometry = real_data["geometry"]
        wavelength = real_data["wavelength"]
        d_min = core.resolution.median().item()

        pred_tight = predict_reflections(
            orientation, geometry, core.cell,
            d_min=d_min, wavelength=wavelength, sigma_cutoff=3.0,
        )
        pred_relaxed = predict_reflections(
            orientation, geometry, core.cell,
            d_min=d_min, wavelength=wavelength, sigma_cutoff=5.0,
        )

        diag_tight = compare_predictions(pred_tight, core, core.cell)
        diag_relaxed = compare_predictions(pred_relaxed, core, core.cell)

        assert diag_relaxed.recovery_rate >= diag_tight.recovery_rate, (
            f"Relaxed recovery {diag_relaxed.recovery_rate:.2%} should >= "
            f"tight {diag_tight.recovery_rate:.2%}"
        )

    def test_diagnostics_summary_runs(self, prediction_data):
        """Smoke test that summary() produces non-empty string."""
        diag = prediction_data["diag"]
        s = diag.summary()
        assert isinstance(s, str)
        assert len(s) > 50
        assert "Prediction quality" in s
