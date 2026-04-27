"""PixelForwardModel: differentiable path from parameters to predicted pixel counts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchsx.deprecated.scaling._orientation import CrystalOrientation
from torchsx.deprecated.scaling._absolute import AbsoluteScale
from torchsx.deprecated.scaling._bscale import BScale
from torchsx.deprecated.merge._structure_factor import MergedStructureFactor
from torchsx.deprecated.forward.coords import RefinableGeometry
from torchsx.deprecated.forward.relp import RelpProfile
from torchsx.deprecated.forward.spectrum import GaussianSpectrum, PCASpectrum
from torchsx.deprecated.forward.corrections import polarization_correction, solid_angle_correction
from torchsx.deprecated.forward.batch import _compute_pixel_geometry


@dataclass
class ForwardResult:
    """All intermediate quantities from a forward model evaluation.

    Useful for diagnostics, debugging, and per-component testing.
    """
    # Step 1: orientation
    h_lab: Tensor       # (B, 3) — relp center in lab frame (Friedel-corrected)
    h_hat: Tensor       # (B, 3) — unit vector along h_lab
    d_star: Tensor      # (B,) — |h_lab| in Å⁻¹
    s_sq: Tensor        # (B,) — (d*/2)² for B-factor
    friedel_sign: Tensor  # (B,) — +1 or -1

    # Step 2: spectral integral
    integral: Tensor    # (B, ph, pw) — ∫ w(λ) R(s_p(λ) - h) dλ

    # Step 3: signal components
    F_sq: Tensor        # (B,) — |F(hkl)|²
    scale: Tensor       # (B,) — exp(log_S + log_B)
    P: Tensor           # (B, ph, pw) — polarization correction
    omega: Tensor       # (B, ph, pw) — solid angle correction
    n_signal: Tensor    # (B, ph, pw) — signal before background

    # Final
    n_pred: Tensor      # (B, ph, pw) — predicted counts (bg + signal)


class PixelForwardModel(nn.Module):
    """Compose all sub-modules into the full pixel-level forward model.

    n_pred(p) = G * [ bg + S * exp(-B*s^2/2) * P(2th,phi) * Omega(2th)
                      * |F(hkl)|^2 * integral w(lambda) * R(s_p(lambda) - h; sigma) dlambda ]

    This module owns no parameters directly — it orchestrates the sub-modules.

    Parameters
    ----------
    structure_factors : MergedStructureFactor
    orientation : CrystalOrientation
    crystal_scale : AbsoluteScale
    crystal_bfactor : BScale
    relp_profile : RelpProfile
    spectrum : GaussianSpectrum
    geometry : RefinableGeometry
    wavelength : float — nominal beam wavelength in Angstroms
    """

    def __init__(
        self,
        structure_factors: MergedStructureFactor,
        orientation: CrystalOrientation,
        crystal_scale: AbsoluteScale,
        crystal_bfactor: BScale,
        relp_profile: RelpProfile,
        spectrum: GaussianSpectrum,
        geometry: RefinableGeometry,
        wavelength: float,
    ):
        super().__init__()
        self.structure_factors = structure_factors
        self.orientation = orientation
        self.crystal_scale = crystal_scale
        self.crystal_bfactor = crystal_bfactor
        self.relp_profile = relp_profile
        self.spectrum = spectrum
        self.geometry = geometry
        self.register_buffer(
            "_inv_wavelength",
            torch.tensor(1.0 / wavelength, dtype=torch.float32),
        )
        self.analytical_integral = True  # use closed-form spectral integral

    # ------------------------------------------------------------------
    # Step 1: Orientation — hkl → h_lab in lab frame
    # ------------------------------------------------------------------

    def compute_h_lab(
        self,
        hkl: Tensor,
        crystal_id: Tensor,
        s_hat_center: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute relp center in lab frame with Friedel correction.

        Parameters
        ----------
        hkl : (B, 3) int — Miller indices
        crystal_id : (B,) int — crystal index per observation
        s_hat_center : (B, 3) float — scattered beam unit vector at ROI center pixel

        Returns
        -------
        h_lab : (B, 3) — Friedel-corrected relp center in reciprocal space
        h_hat : (B, 3) — unit vector along h_lab
        d_star : (B,) — |h_lab| (d-star)
        s_sq : (B,) — (d*/2)² for B-factor
        friedel_sign : (B,) — +1 or -1 applied to h_lab
        """
        device = hkl.device

        # h_lab = RUB @ hkl
        h_lab = self.orientation.predict_h_lab(hkl.float(), crystal_id)  # (B, 3)

        # Friedel ambiguity fix: canonicalized hkl sign may not match
        # actual diffraction geometry.  Flip h_lab to match pixel direction.
        s_hat_in = torch.zeros(3, device=device)
        s_hat_in[2] = 1.0
        s_p_center = self._inv_wavelength * (s_hat_center - s_hat_in)  # (B, 3)
        friedel_sign = (h_lab * s_p_center).sum(dim=-1).sign()  # (B,)
        # Clamp zeros to +1 (degenerate case)
        friedel_sign = friedel_sign.clamp(min=0.0) * 2.0 - 1.0
        h_lab = h_lab * friedel_sign.unsqueeze(-1)

        # Normalize
        h_norm = h_lab.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        h_hat = h_lab / h_norm
        d_star = h_norm.squeeze(-1)
        s_sq = (d_star / 2.0) ** 2

        return h_lab, h_hat, d_star, s_sq, friedel_sign

    # ------------------------------------------------------------------
    # Step 2: Spectral integral — ∫ w(λ) R(s_p(λ) − h) dλ
    # ------------------------------------------------------------------

    def compute_spectral_integral(
        self,
        s_hat_out: Tensor,
        h_lab: Tensor,
        h_hat: Tensor,
        hkl_id: Tensor,
        crystal_id: Tensor,
    ) -> Tensor:
        """Compute the wavelength-integrated relp profile at each pixel.

        Parameters
        ----------
        s_hat_out : (B, ph, pw, 3) — scattered beam unit vectors per pixel
        h_lab : (B, 3) — relp center in reciprocal space
        h_hat : (B, 3) — unit vector along h_lab
        hkl_id : (B,) — unique reflection index per observation
        crystal_id : (B,) — crystal index per observation

        Returns
        -------
        integral : (B, ph, pw) — spectrally integrated relp profile value
        """
        B = h_lab.shape[0]
        ph, pw = s_hat_out.shape[1], s_hat_out.shape[2]
        device = s_hat_out.device

        wl_bins = self.spectrum.wavelengths()   # (n_bins,)
        d_wl = self.spectrum.d_lambda()
        weights = self.spectrum.spectrum(crystal_id)  # (B, n_bins)

        s_hat_in = torch.zeros(3, device=device)
        s_hat_in[2] = 1.0

        h_lab_4d = h_lab[:, None, None, :]   # (B, 1, 1, 3)
        h_hat_4d = h_hat[:, None, None, :]

        integral = torch.zeros(B, ph, pw, device=device)

        for k in range(wl_bins.shape[0]):
            inv_wl_k = 1.0 / wl_bins[k]
            s_p = inv_wl_k * (s_hat_out - s_hat_in)      # (B, ph, pw, 3)
            delta_s = s_p - h_lab_4d                       # (B, ph, pw, 3)

            ds_flat = delta_s.reshape(B * ph * pw, 3)
            hh_flat = h_hat_4d.expand(B, ph, pw, 3).reshape(B * ph * pw, 3)
            hkl_id_flat = hkl_id[:, None, None].expand(B, ph, pw).reshape(B * ph * pw)

            R_k = self.relp_profile.evaluate(ds_flat, hh_flat, hkl_id_flat)
            R_k = R_k.reshape(B, ph, pw)

            w_k = weights[:, k].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            integral = integral + w_k * R_k

        return integral * d_wl

    def compute_spectral_integral_analytical(
        self,
        s_hat_out: Tensor,
        h_lab: Tensor,
        h_hat: Tensor,
        hkl_id: Tensor,
        crystal_id: Tensor,
    ) -> Tensor:
        """Analytical spectral integral — closed-form Gaussian convolution.

        Replaces the numerical wavelength loop with a single evaluation per pixel.
        Valid when the spectrum and relp profile are both Gaussian and the
        reciprocal-space sweep is approximately linear in wavelength (good to
        ~0.3% for XFEL bandwidth ~0.2%).

        The key insight: s_p(λ) = (1/λ)(ŝ_out - ŝ_in) sweeps linearly through
        reciprocal space as λ varies.  The integral ∫ w(λ) R(s_p(λ) - h) dλ
        is then a 1D Gaussian integral with closed-form solution.
        """
        device = s_hat_out.device
        B = h_lab.shape[0]
        ph, pw = s_hat_out.shape[1], s_hat_out.shape[2]

        # Per-crystal wavelength center and sigma
        energy = torch.exp(self.spectrum.log_energy[crystal_id.long()])  # (B,)
        sigma_eV = torch.exp(self.spectrum.log_sigma_eV)
        from torchsx.deprecated.forward.spectrum import HC_EV_A
        wl_center = HC_EV_A / energy  # (B,) Å
        sigma_wl = wl_center ** 2 / HC_EV_A * sigma_eV  # (B,)
        d_wl = self.spectrum.d_lambda()

        s_hat_in = torch.zeros(3, device=device)
        s_hat_in[2] = 1.0

        # s_p at center wavelength: (B, ph, pw, 3)
        inv_wl_c = (1.0 / wl_center)[:, None, None, None]  # (B, 1, 1, 1)
        s_p0 = inv_wl_c * (s_hat_out - s_hat_in)

        # Sweep velocity: ds_p/dλ = -s_p/λ at λ_c
        # v = -s_p0 / wl_center  →  (B, ph, pw, 3)
        wl_c_4d = wl_center[:, None, None, None]
        v = -s_p0 / wl_c_4d  # (B, ph, pw, 3)

        # Displacement at center wavelength
        h_lab_4d = h_lab[:, None, None, :]  # (B, 1, 1, 3)
        delta_s0 = s_p0 - h_lab_4d  # (B, ph, pw, 3)

        # Relp profile widths per reflection
        sigma_r, sigma_t = self.relp_profile.get_sigma(hkl_id)  # (B,)

        # Decompose into radial/tangential
        h_hat_4d = h_hat[:, None, None, :]  # (B, 1, 1, 3)

        # Radial components
        ds_r0 = (delta_s0 * h_hat_4d).sum(dim=-1)  # (B, ph, pw)
        v_r = (v * h_hat_4d).sum(dim=-1)  # (B, ph, pw)

        # Tangential components
        ds_t0_vec = delta_s0 - ds_r0.unsqueeze(-1) * h_hat_4d  # (B, ph, pw, 3)
        v_t_vec = v - v_r.unsqueeze(-1) * h_hat_4d  # (B, ph, pw, 3)
        ds_t0_sq = (ds_t0_vec * ds_t0_vec).sum(dim=-1)  # (B, ph, pw)
        v_t_sq = (v_t_vec * v_t_vec).sum(dim=-1)  # (B, ph, pw)
        ds_t0_dot_v_t = (ds_t0_vec * v_t_vec).sum(dim=-1)  # (B, ph, pw)

        # Broadcast sigma to (B, 1, 1)
        sr2 = (sigma_r ** 2)[:, None, None]
        st2 = (sigma_t ** 2)[:, None, None]
        sl2 = (sigma_wl ** 2)[:, None, None]

        # Gaussian integral coefficients:
        # exponent = -A t² - B_coeff t - C  where t = λ - λ_c
        A = 0.5 / sl2 + 0.5 * v_r ** 2 / sr2 + 0.5 * v_t_sq / st2
        B_coeff = ds_r0 * v_r / sr2 + ds_t0_dot_v_t / st2
        C = 0.5 * ds_r0 ** 2 / sr2 + 0.5 * ds_t0_sq / st2

        # ∫ exp(-A t² - B t - C) dt = √(π/A) exp(B²/(4A) - C)
        # Numerical integral = d_wl * (above) / (σ_λ √(2π))
        # = d_wl / (σ_λ √(2A)) * exp(B²/(4A) - C)
        sigma_eff = 1.0 / torch.sqrt(2.0 * A)  # effective width
        sigma_wl_4d = sigma_wl[:, None, None]

        log_integral = (
            torch.log(sigma_eff)
            - torch.log(sigma_wl_4d)
            + B_coeff ** 2 / (4.0 * A)
            - C
        )

        integral = d_wl * torch.exp(log_integral)
        return integral

    def compute_spectral_integral_pca(
        self,
        s_hat_out: Tensor,
        h_lab: Tensor,
        h_hat: Tensor,
        hkl_id: Tensor,
        crystal_id: Tensor,
    ) -> Tensor:
        """PCA-based spectral integral using precomputed overlap grids.

        Factorizes the integral as:
            I = O(Δs_r, v_r, σ_r) × exp(-|Δs_t|²/(2σ_t²))

        where O is interpolated from a precomputed 3D grid via the
        PCASpectrum module. Requires self.spectrum to be a PCASpectrum.

        This is the most accurate method for real SASE spectra and is
        fast on GPU via hardware-accelerated trilinear interpolation.
        """
        if not isinstance(self.spectrum, PCASpectrum):
            raise TypeError(
                "compute_spectral_integral_pca requires spectrum to be PCASpectrum, "
                f"got {type(self.spectrum).__name__}"
            )

        device = s_hat_out.device
        B = h_lab.shape[0]
        ph, pw = s_hat_out.shape[1], s_hat_out.shape[2]

        s_hat_in = torch.zeros(3, device=device)
        s_hat_in[2] = 1.0

        # s_p at nominal wavelength: (B, ph, pw, 3)
        s_p0 = self._inv_wavelength * (s_hat_out - s_hat_in)

        # Displacement from relp center
        h_lab_4d = h_lab[:, None, None, :]
        h_hat_4d = h_hat[:, None, None, :]
        delta_s0 = s_p0 - h_lab_4d  # (B, ph, pw, 3)

        # Radial component: Δs_r = Δs · ĥ
        ds_r = (delta_s0 * h_hat_4d).sum(dim=-1)  # (B, ph, pw)

        # Tangential component: |Δs_t|²
        ds_t_vec = delta_s0 - ds_r.unsqueeze(-1) * h_hat_4d
        ds_t_sq = (ds_t_vec * ds_t_vec).sum(dim=-1)  # (B, ph, pw)

        # Radial sweep rate: v_r = |v · ĥ| where v = s_p/λ ≈ |s_p|/λ along ĥ
        # More precisely: v = -s_p/λ, v_r = |v · ĥ|
        # For the grid, we use v_r = d*/λ as a good approximation
        # (the per-pixel v_r variation is small across the ROI patch)
        d_star = h_lab.norm(dim=-1)  # (B,)
        inv_wl = self._inv_wavelength
        v_r = (d_star * inv_wl)[:, None, None].expand(B, ph, pw)  # (B, ph, pw)

        # Per-reflection σ_r (log-space for grid lookup)
        log_sigma_r = self.relp_profile.get_log_sigma_r(hkl_id)  # (B,)

        # Per-reflection σ_t for tangential Gaussian
        _, sigma_t = self.relp_profile.get_sigma(hkl_id)  # (B,)

        # Radial overlap from PCA grid
        overlap_r = self.spectrum.overlap(ds_r, v_r, log_sigma_r, crystal_id)

        # Tangential Gaussian
        st2 = (sigma_t ** 2)[:, None, None]
        tang = torch.exp(-ds_t_sq / (2.0 * st2))

        return overlap_r * tang

    # ------------------------------------------------------------------
    # Step 3: Signal — scale × corrections × F² × integral
    # ------------------------------------------------------------------

    def compute_signal(
        self,
        batch: "ROIBatch",
        integral: Tensor,
        s_sq: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Combine structure factors, scale, and corrections into signal.

        Parameters
        ----------
        batch : ROIBatch
        integral : (B, ph, pw) — spectral integral from step 2
        s_sq : (B,) — (d*/2)² for B-factor

        Returns
        -------
        n_signal : (B, ph, pw) — signal counts (no background)
        F_sq : (B,) — structure factor squared
        scale : (B,) — exp(log_S + log_B)
        """
        # Structure factors: |F(hkl)|²
        F_sq = self.structure_factors.predict_I(batch.hkl_id)  # (B,)

        # Per-crystal scale and B-factor
        log_S = self.crystal_scale.log_scale_per_obs(batch.crystal_id)
        log_B = self.crystal_bfactor.log_scale_per_obs(batch.crystal_id, s_sq)
        scale = torch.exp(log_S + log_B)  # (B,)

        # Corrections (precomputed in batch)
        P = batch.polarization      # (B, ph, pw)
        omega = batch.solid_angle   # (B, ph, pw)

        n_signal = (scale[:, None, None] * F_sq[:, None, None]
                    * P * omega * integral)
        return n_signal, F_sq, scale

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def forward(self, batch: "ROIBatch") -> Tensor:
        """Predict pixel counts for a batch of ROIs.

        Parameters
        ----------
        batch : ROIBatch dataclass

        Returns
        -------
        n_pred : (B, ph, pw) float — predicted counts per pixel
        """
        result = self.forward_debug(batch)
        return result.n_pred

    def _ensure_geometry(self, batch: "ROIBatch") -> "ROIBatch":
        """Compute pixel geometry on-the-fly if not already set.

        Optimization: geometry corrections are nearly constant across an
        11×11 patch, so we only need autograd through the CENTER pixel per
        ROI for geometry gradients (B pixels, not B×ph×pw). The per-pixel
        geometry is computed without grad, then the center-pixel gradient
        is broadcast as a uniform correction:

            s_hat_out = s_hat_all.detach() + (s_hat_center_grad - s_hat_center_detach)

        This reduces backward cost from 472ms to ~4ms per batch.
        """
        if batch.s_hat_out is not None:
            return batch
        if batch.fs_grid is None:
            raise ValueError(
                "ROIBatch has neither s_hat_out nor fs_grid — "
                "cannot compute pixel geometry"
            )
        shape = batch.fs_grid.shape  # (B, ph, pw)
        B, ph, pw = shape

        # Full pixel geometry without grad (fast: 8.5ms)
        with torch.no_grad():
            fs_flat = batch.fs_grid.reshape(-1)
            ss_flat = batch.ss_grid.reshape(-1)
            pid_flat = batch.pid_grid.reshape(-1)
            s_hat_all = self.geometry.pixel_to_s_hat(
                fs_flat, ss_flat, pid_flat,
            ).reshape(B, ph, pw, 3)

        # Center pixel WITH grad for geometry parameters (B pixels only)
        s_hat_center = self.geometry.pixel_to_s_hat(
            batch.fs_center, batch.ss_center, batch.panel_id.long(),
        )  # (B, 3)

        # Broadcast gradient: treat geometry shift as uniform across patch
        with torch.no_grad():
            s_hat_center_detach = s_hat_center.detach()
        delta = (s_hat_center - s_hat_center_detach)[:, None, None, :]  # (B, 1, 1, 3)
        batch.s_hat_out = s_hat_all + delta  # grad flows through delta only

        # Corrections from detached geometry (negligible sensitivity)
        with torch.no_grad():
            cos_2theta = s_hat_all[..., 2].clamp(-1, 1)
            batch.two_theta = torch.acos(cos_2theta)
            batch.phi = torch.atan2(s_hat_all[..., 1], s_hat_all[..., 0])
            batch.polarization = polarization_correction(batch.two_theta, batch.phi)
            batch.solid_angle = solid_angle_correction(s_hat_all)
        return batch

    def forward_sparse(self, batch: "SparseBatch") -> ForwardResult:
        """Forward pass for SparseBatch with per-pixel background.

        Like forward_debug but uses per-pixel background (B, ph, pw) from the
        sparse radial profile instead of scalar background (B,).

        Parameters
        ----------
        batch : SparseBatch — duck-types ROIBatch for geometry/orientation

        Returns
        -------
        ForwardResult with n_pred using per-pixel background
        """
        batch = self._ensure_geometry(batch)
        ph, pw = batch.background.shape[1], batch.background.shape[2]

        # Step 1: orientation + Friedel fix
        s_hat_center = batch.s_hat_out[:, ph // 2, pw // 2, :]
        h_lab, h_hat, d_star, s_sq, friedel_sign = self.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )

        # Step 2: spectral integral
        if isinstance(self.spectrum, PCASpectrum):
            integral = self.compute_spectral_integral_pca(
                batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
            )
        elif self.analytical_integral:
            integral = self.compute_spectral_integral_analytical(
                batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
            )
        else:
            integral = self.compute_spectral_integral(
                batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
            )

        # Step 3: signal
        n_signal, F_sq, scale = self.compute_signal(batch, integral, s_sq)

        # Step 4: per-pixel background + gain → final prediction
        bg = batch.background  # (B, ph, pw) — already per-pixel
        gain = self.geometry.gain(batch.panel_id)[:, None, None]
        n_pred = (gain * (bg + n_signal)).clamp(min=1e-3)

        return ForwardResult(
            h_lab=h_lab,
            h_hat=h_hat,
            d_star=d_star,
            s_sq=s_sq,
            friedel_sign=friedel_sign,
            integral=integral,
            F_sq=F_sq,
            scale=scale,
            P=batch.polarization,
            omega=batch.solid_angle,
            n_signal=n_signal,
            n_pred=n_pred,
        )

    def forward_debug(self, batch: "ROIBatch") -> ForwardResult:
        """Like forward() but returns all intermediate quantities.

        Useful for diagnostics, per-component testing, and plotting.
        """
        batch = self._ensure_geometry(batch)
        ph, pw = batch.patches.shape[1], batch.patches.shape[2]

        # Step 1: orientation + Friedel fix
        s_hat_center = batch.s_hat_out[:, ph // 2, pw // 2, :]
        h_lab, h_hat, d_star, s_sq, friedel_sign = self.compute_h_lab(
            batch.hkl, batch.crystal_id, s_hat_center,
        )

        # Step 2: spectral integral
        if isinstance(self.spectrum, PCASpectrum):
            integral = self.compute_spectral_integral_pca(
                batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
            )
        elif self.analytical_integral:
            integral = self.compute_spectral_integral_analytical(
                batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
            )
        else:
            integral = self.compute_spectral_integral(
                batch.s_hat_out, h_lab, h_hat, batch.hkl_id, batch.crystal_id,
            )

        # Step 3: signal
        n_signal, F_sq, scale = self.compute_signal(batch, integral, s_sq)

        # Step 4: background + gain → final prediction
        bg = batch.background[:, None, None]
        gain = self.geometry.gain(batch.panel_id)[:, None, None]
        n_pred = (gain * (bg + n_signal)).clamp(min=1e-3)

        return ForwardResult(
            h_lab=h_lab,
            h_hat=h_hat,
            d_star=d_star,
            s_sq=s_sq,
            friedel_sign=friedel_sign,
            integral=integral,
            F_sq=F_sq,
            scale=scale,
            P=batch.polarization,
            omega=batch.solid_angle,
            n_signal=n_signal,
            n_pred=n_pred,
        )

    # ------------------------------------------------------------------
    # Sparse pixel-level forward pass (pair-based)
    # ------------------------------------------------------------------

    def _ensure_pair_geometry(self, pairs: "PixelReflectionPairs"):
        """Compute per-pixel geometry if not already set."""
        if pairs.pixel_s_hat is not None:
            return

        s_hat = self.geometry.pixel_to_s_hat(
            pairs.pixel_fs.float(),
            pairs.pixel_ss.float(),
            pairs.pixel_panel.long(),
        )  # (N_pixels, 3)
        pairs.pixel_s_hat = s_hat

        with torch.no_grad():
            cos_2theta = s_hat[:, 2].clamp(-1, 1)
            two_theta = torch.acos(cos_2theta)
            phi = torch.atan2(s_hat[:, 1], s_hat[:, 0])
            pairs.pixel_two_theta = two_theta
            pairs.pixel_pol = polarization_correction(two_theta, phi)
            pairs.pixel_solid_angle = solid_angle_correction(s_hat)

    def _compute_spectral_integral_pca_flat(
        self,
        s_hat_p: Tensor,
        h_lab_p: Tensor,
        h_hat_p: Tensor,
        hkl_id_p: Tensor,
        crystal_id_p: Tensor,
        sigma_err_r: float = 0.0,
        sigma_err_t: float = 0.0,
    ) -> Tensor:
        """Flat PCA spectral integral for (N_pairs,) tensors.

        Same physics as compute_spectral_integral_pca but on flat tensors
        instead of (B, ph, pw) grids. Supports effective relp widths
        for position error broadening.

        Parameters
        ----------
        s_hat_p : (N, 3) — scattered beam unit vectors per pair
        h_lab_p : (N, 3) — relp center per pair
        h_hat_p : (N, 3) — unit vector along h_lab per pair
        hkl_id_p : (N,) — unique reflection index per pair
        crystal_id_p : (N,) — crystal index per pair
        sigma_err_r : float — radial position error width (A^-1)
        sigma_err_t : float — tangential position error width (A^-1)

        Returns
        -------
        integral : (N,) — spectral integral per pair
        """
        if not isinstance(self.spectrum, PCASpectrum):
            raise TypeError(
                "forward_pairs requires spectrum to be PCASpectrum"
            )

        device = s_hat_p.device
        N = s_hat_p.shape[0]

        s_hat_in = torch.zeros(3, device=device)
        s_hat_in[2] = 1.0

        # s_p at nominal wavelength: (N, 3)
        s_p0 = self._inv_wavelength * (s_hat_p - s_hat_in)

        # Displacement from relp center
        delta_s0 = s_p0 - h_lab_p  # (N, 3)

        # Radial component: ds_r = delta_s . h_hat
        ds_r = (delta_s0 * h_hat_p).sum(dim=-1)  # (N,)

        # Tangential component: |ds_t|^2
        ds_t_vec = delta_s0 - ds_r.unsqueeze(-1) * h_hat_p
        ds_t_sq = (ds_t_vec * ds_t_vec).sum(dim=-1)  # (N,)

        # Radial sweep rate: v_r = d*/lambda
        d_star_p = h_lab_p.norm(dim=-1)  # (N,)
        v_r = d_star_p * self._inv_wavelength  # (N,)

        # Per-reflection sigma_r (log-space for grid lookup)
        log_sigma_r = self.relp_profile.get_log_sigma_r(hkl_id_p)  # (N,)

        # Per-reflection sigma_t for tangential Gaussian
        _, sigma_t = self.relp_profile.get_sigma(hkl_id_p)  # (N,)

        # Apply effective widths: sigma_eff = sqrt(sigma^2 + sigma_err^2)
        if sigma_err_r > 0:
            sigma_r = torch.exp(log_sigma_r)
            sigma_r_eff = torch.sqrt(sigma_r ** 2 + sigma_err_r ** 2)
            log_sigma_r = torch.log(sigma_r_eff)

        if sigma_err_t > 0:
            sigma_t = torch.sqrt(sigma_t ** 2 + sigma_err_t ** 2)

        # Radial overlap from PCA grid (flat version)
        overlap_r = self._overlap_flat(ds_r, v_r, log_sigma_r, crystal_id_p)

        # Tangential Gaussian
        st2 = sigma_t ** 2
        tang = torch.exp(-ds_t_sq / (2.0 * st2))

        return overlap_r * tang

    def _overlap_flat(
        self,
        ds_r: Tensor,
        v_r: Tensor,
        log_sigma_r: Tensor,
        crystal_id: Tensor,
    ) -> Tensor:
        """PCA spectral overlap on flat (N,) tensors.

        Same as PCASpectrum.overlap but accepts flat tensors instead of
        (B, ph, pw) grids.

        Parameters
        ----------
        ds_r : (N,) — radial offset
        v_r : (N,) — sweep rate
        log_sigma_r : (N,) — log(sigma_r)
        crystal_id : (N,) — crystal index

        Returns
        -------
        overlap : (N,) — spectral overlap values
        """
        spectrum = self.spectrum
        N = ds_r.shape[0]

        # Normalize coordinates to [-1, 1]
        coords = spectrum._normalize_coords(ds_r, v_r, log_sigma_r)  # (N, 3)

        # grid_sample: input (1, C, D, H, W), grid (1, 1, N, 1, 3)
        grid = coords.reshape(1, 1, N, 1, 3)

        sampled = F.grid_sample(
            spectrum._grids,  # (1, 1+n_modes, n_ds, n_vr, n_sr)
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )  # (1, 1+n_modes, 1, N, 1)

        sampled = sampled.squeeze(0).squeeze(-1).squeeze(1)  # (1+n_modes, N)

        mean_val = sampled[0]  # (N,)
        mode_vals = sampled[1:]  # (n_modes, N)

        # Per-crystal coefficients
        c = spectrum.coeffs[crystal_id.long()]  # (N, n_modes)

        # Weighted sum: mean + sum_k c_k * mode_k
        overlap = mean_val + (c.T * mode_vals).sum(dim=0)  # (N,)

        return overlap

    def forward_pairs(
        self,
        pairs: "PixelReflectionPairs",
        sigma_err_r: float = 0.0,
        sigma_err_t: float = 0.0,
    ) -> "SparseForwardResult":
        """Forward pass on sparse pixel-reflection pairs.

        Evaluates the forward model directly at sparse pixel positions,
        aggregates signal per pixel via scatter_add, and returns predicted
        rates for Poisson NLL.

        Parameters
        ----------
        pairs : PixelReflectionPairs
        sigma_err_r : float — radial position error width (A^-1) for broadening
        sigma_err_t : float — tangential position error width (A^-1)

        Returns
        -------
        SparseForwardResult
        """
        from torchsx.deprecated.forward._pixel_matching import SparseForwardResult

        device = pairs.crystal_id.device

        # Step 1: Geometry (per-pixel, computed once)
        self._ensure_pair_geometry(pairs)

        # Step 2: Orientation (per-reflection)
        s_hat_center = self.geometry.pixel_to_s_hat(
            pairs.fs_pred, pairs.ss_pred, pairs.panel_id.long(),
        )  # (N_refl, 3)
        h_lab, h_hat, d_star, s_sq, friedel_sign = self.compute_h_lab(
            pairs.hkl, pairs.crystal_id, s_hat_center,
        )

        # Step 3: Expand to pair level
        h_lab_p = h_lab[pairs.refl_idx]          # (N_pairs, 3)
        h_hat_p = h_hat[pairs.refl_idx]          # (N_pairs, 3)
        s_hat_p = pairs.pixel_s_hat[pairs.pixel_idx]  # (N_pairs, 3)
        hkl_id_p = pairs.hkl_id[pairs.refl_idx]  # (N_pairs,)
        crystal_id_p = pairs.crystal_id[pairs.refl_idx]  # (N_pairs,)

        # Step 4: Spectral integral (per-pair, flat)
        integral = self._compute_spectral_integral_pca_flat(
            s_hat_p, h_lab_p, h_hat_p, hkl_id_p, crystal_id_p,
            sigma_err_r=sigma_err_r, sigma_err_t=sigma_err_t,
        )  # (N_pairs,)

        # Step 5: Signal per pair
        F_sq = self.structure_factors.predict_I(pairs.hkl_id)  # (N_refl,)
        log_S = self.crystal_scale.log_scale_per_obs(pairs.crystal_id)
        log_B = self.crystal_bfactor.log_scale_per_obs(pairs.crystal_id, s_sq)
        scale = torch.exp(log_S + log_B)  # (N_refl,)

        pol = pairs.pixel_pol[pairs.pixel_idx]          # (N_pairs,)
        omega = pairs.pixel_solid_angle[pairs.pixel_idx]  # (N_pairs,)

        n_signal_pairs = (
            scale[pairs.refl_idx] * F_sq[pairs.refl_idx]
            * pol * omega * integral
        )  # (N_pairs,)

        # Step 6: Aggregate per pixel via scatter_add
        N_pixels = pairs.n_pixels
        n_signal_per_pixel = torch.zeros(N_pixels, device=device)
        n_signal_per_pixel.scatter_add_(0, pairs.pixel_idx, n_signal_pairs)

        gain = self.geometry.gain(pairs.pixel_panel)  # (N_pixels,)
        n_pred = (gain * (pairs.pixel_bg + n_signal_per_pixel)).clamp(min=1e-3)

        return SparseForwardResult(
            n_signal_pairs=n_signal_pairs,
            n_pred_pixels=n_pred,
            F_sq=F_sq,
            scale=scale,
        )
