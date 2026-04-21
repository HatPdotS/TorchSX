#!/usr/bin/env python
"""Relp shape analysis: deconvolve spectral broadening from observed I(η).

Physics:
  - Each reflection sits at excitation error η from the Ewald sphere
  - The observed intensity I(η) ∝ relp_profile(η) ⊗ spectral_kernel(η)
  - The spectral kernel converts bandwidth Δλ to excitation error smearing:
      Δη = (d*/λ) × Δλ  (radial sweep rate × wavelength change)
  - Measured SASE spectra give us the kernel directly
  - Deconvolution → intrinsic relp shape → effective crystal domain size

Output: plots in the same directory.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from torchsx.stream import SFXStream
from torchsx.forward.coords import RefinableGeometry

HC_EV_A = 12398.4
OUT_DIR = Path(__file__).resolve().parent


def excitation_error(h_lab: torch.Tensor, wavelength: float) -> torch.Tensor:
    """Excitation error: distance from Ewald sphere (Å⁻¹)."""
    s_out = wavelength * h_lab.clone()
    s_out[:, 2] += 1.0
    return (s_out.norm(dim=-1) - 1.0).abs() / wavelength


def gaussian(x, A, sigma):
    return A * np.exp(-x**2 / (2 * sigma**2))


def lorentzian(x, A, gamma):
    return A / (1 + (x / gamma)**2)


def main():
    # ── Load SFX data ────────────────────────────────────────────────
    sfx_path = PROJECT_ROOT / "merged_368.sfx"
    print(f"Loading {sfx_path.name}...")
    stream = SFXStream.load(str(sfx_path), merge_friedel=False)
    core = stream._core
    geom = core.geometry
    profile_rad = core.profile_rad[0].item()

    # ── Load measured spectra ────────────────────────────────────────
    print("Loading SASE spectra...")
    spectra_dir = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/bs_data_files")
    all_y = []
    for i in range(1, 135):
        fn = spectra_dir / f"acq{i:04d}.BSDATA.h5"
        if fn.exists():
            with h5py.File(fn, "r") as f:
                all_y.append(f["SARFE10-PSSS059:SPECTRUM_Y/data"][:])
    spectra = np.concatenate(all_y, axis=0).astype(np.float64)
    with h5py.File(spectra_dir / "acq0001.BSDATA.h5", "r") as f:
        energy_eV = f["SARFE10-PSSS059:SPECTRUM_X/data"][0].astype(np.float64)
    n_shots = spectra.shape[0]
    print(f"  {n_shots} spectra, {energy_eV.min():.0f}–{energy_eV.max():.0f} eV")

    # Compute mean spectrum and per-shot bandwidth
    wl_bins = HC_EV_A / energy_eV  # Å
    d_wl = np.abs(np.diff(wl_bins).mean())  # Å per bin
    wl_center = HC_EV_A / 12960.0  # Å
    mean_spectrum = spectra.mean(axis=0)
    # Normalize to probability density in wavelength
    mean_spectrum_norm = mean_spectrum / (mean_spectrum.sum() * d_wl)

    # Convert spectrum from S(E) to S(λ): S(λ) = S(E) * |dE/dλ| = S(E) * hc/λ²
    # But since we have discrete bins, just renormalize
    # Spectral width in wavelength
    wl_var = np.average((wl_bins - wl_center)**2, weights=mean_spectrum.clip(0))
    sigma_wl = np.sqrt(wl_var)
    sigma_E = sigma_wl * HC_EV_A / wl_center**2
    print(f"  Mean spectrum σ_λ = {sigma_wl*1e4:.2f}×10⁻⁴ Å  (σ_E = {sigma_E:.1f} eV)")

    # ── Sample reflections ───────────────────────────────────────────
    print("Sampling reflections...")
    np.random.seed(42)
    n_refl_arr = core.n_refl.numpy()
    good = np.where(n_refl_arr >= 20)[0]
    sample_cids = np.random.choice(good, min(500, len(good)), replace=False)

    ref_geom = RefinableGeometry(geom)

    all_exc = []
    all_I = []
    all_bg = []
    all_peak = []
    all_d = []
    all_v_r = []  # radial sweep rate d*/(λ)

    for i, cid in enumerate(sample_cids):
        cid = int(cid)
        mask = core.crystal_id == cid
        if mask.sum() < 10:
            continue

        img_id = core.image_id[cid].item()
        en = core.beam_energy[img_id].item()
        wl = HC_EV_A / en
        cl = core.camera_length[img_id].item()
        if cl <= 0:
            cl = geom.clen

        UB = torch.stack([core.astar[cid], core.bstar[cid], core.cstar[cid]], dim=-1).float()
        hkl = core.hkl[mask].float()
        h_lab = (UB @ hkl.T).T
        d_star = h_lab.norm(dim=-1)  # |h| in Å⁻¹

        exc = excitation_error(h_lab, wl)
        d = 1.0 / d_star.clamp(min=1e-10)
        v_r = d_star / wl  # radial sweep rate (Å⁻²)

        all_exc.append(exc.numpy())
        all_I.append(core.I[mask].numpy())
        all_bg.append(core.background[mask].numpy())
        all_peak.append(core.peak[mask].numpy())
        all_d.append(d.numpy())
        all_v_r.append(v_r.numpy())

    exc = np.concatenate(all_exc)
    I_obs = np.concatenate(all_I)
    bg = np.concatenate(all_bg)
    peak = np.concatenate(all_peak)
    d_sp = np.concatenate(all_d)
    v_r = np.concatenate(all_v_r)
    N = len(exc)
    signal = I_obs - bg

    print(f"  {N} reflections from {len(sample_cids)} crystals")

    # ── Compute spectral broadening kernel ───────────────────────────
    # At resolution d*, the spectral smearing in excitation error is:
    #   Δη = v_r × Δλ = (d*/λ) × Δλ
    # The spectral kernel in η-space depends on resolution.
    # Use median v_r for the representative kernel.
    v_r_median = np.median(v_r)
    v_r_q25 = np.percentile(v_r, 25)
    v_r_q75 = np.percentile(v_r, 75)
    print(f"  v_r (d*/λ): median={v_r_median:.4f}, IQR=[{v_r_q25:.4f}, {v_r_q75:.4f}] Å⁻²")
    print(f"  Spectral broadening σ_η = v_r × σ_λ = {v_r_median * sigma_wl:.6f} Å⁻¹")
    print(f"  = {v_r_median * sigma_wl / profile_rad:.2f}× profile_rad")

    # Build the spectral kernel on an η grid (for median v_r)
    eta_max = 0.010  # Å⁻¹
    n_eta = 500
    eta_grid = np.linspace(-eta_max, eta_max, n_eta)
    d_eta = eta_grid[1] - eta_grid[0]

    # Convert: η = v_r × (λ - λ_center), so λ = λ_center + η/v_r
    wl_from_eta = wl_center + eta_grid / v_r_median
    # Interpolate mean spectrum onto this grid
    spec_kernel = np.interp(wl_from_eta, wl_bins[::-1] if wl_bins[0] > wl_bins[-1] else wl_bins,
                            mean_spectrum[::-1] if wl_bins[0] > wl_bins[-1] else mean_spectrum,
                            left=0, right=0)
    spec_kernel = spec_kernel / (spec_kernel.sum() * d_eta)  # normalize

    # ── Build observed I(η) profile ──────────────────────────────────
    n_bins = 80
    exc_edges = np.linspace(0, exc.max() * 1.01, n_bins + 1)
    exc_centers = 0.5 * (exc_edges[:-1] + exc_edges[1:])

    bin_signal_mean = np.zeros(n_bins)
    bin_signal_median = np.zeros(n_bins)
    bin_frac_mean = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        in_bin = (exc >= exc_edges[b]) & (exc < exc_edges[b + 1])
        n_in = in_bin.sum()
        bin_count[b] = n_in
        if n_in < 10:
            bin_signal_mean[b] = np.nan
            bin_signal_median[b] = np.nan
            bin_frac_mean[b] = np.nan
            continue
        bin_signal_mean[b] = signal[in_bin].mean()
        bin_signal_median[b] = np.median(signal[in_bin])
        bg_bin = bg[in_bin]
        valid_bg = bg_bin > 0
        if valid_bg.sum() > 5:
            bin_frac_mean[b] = np.median(signal[in_bin][valid_bg] / bg_bin[valid_bg])

    ok = bin_count >= 10

    # ── Fit models to I(η) ───────────────────────────────────────────
    # The observed profile is relp ⊗ spectral_kernel
    # For Gaussian relp: observed is also Gaussian with σ_obs² = σ_relp² + σ_spec²
    # Fit Gaussian to observed signal vs |η|
    fit_x = exc_centers[ok]
    fit_y = bin_signal_mean[ok]
    fit_y = fit_y / fit_y[0]  # normalize to peak

    try:
        popt_g, _ = curve_fit(gaussian, fit_x, fit_y, p0=[1.0, 0.003],
                              bounds=([0, 1e-5], [2, 0.05]))
        sigma_obs = popt_g[1]
        sigma_spec = v_r_median * sigma_wl
        sigma_relp_sq = sigma_obs**2 - sigma_spec**2
        if sigma_relp_sq > 0:
            sigma_relp = np.sqrt(sigma_relp_sq)
        else:
            sigma_relp = 0.0
        print(f"\n--- Gaussian fit ---")
        print(f"  σ_observed  = {sigma_obs:.6f} Å⁻¹  ({sigma_obs/profile_rad:.2f}× profile_rad)")
        print(f"  σ_spectral  = {sigma_spec:.6f} Å⁻¹  ({sigma_spec/profile_rad:.2f}× profile_rad)")
        print(f"  σ_relp      = {sigma_relp:.6f} Å⁻¹  ({sigma_relp/profile_rad:.2f}× profile_rad)")
        if sigma_relp > 0:
            # Crystal domain size: D ≈ 1 / (2π σ_relp) — for Gaussian shape transform
            D_nm = 1.0 / (2 * np.pi * sigma_relp) * 10  # Å → nm
            print(f"  Domain size ≈ 1/(2π σ_relp) = {D_nm:.1f} nm")
    except Exception as e:
        print(f"  Gaussian fit failed: {e}")
        sigma_obs = 0.003
        sigma_relp = 0.002
        sigma_spec = 0.001

    try:
        popt_l, _ = curve_fit(lorentzian, fit_x, fit_y, p0=[1.0, 0.003],
                              bounds=([0, 1e-5], [2, 0.05]))
        gamma_obs = popt_l[1]
        print(f"\n--- Lorentzian fit ---")
        print(f"  γ_observed  = {gamma_obs:.6f} Å⁻¹  ({gamma_obs/profile_rad:.2f}× profile_rad)")
    except Exception as e:
        print(f"  Lorentzian fit failed: {e}")
        gamma_obs = 0.003

    # ── Deconvolution via Wiener filter ──────────────────────────────
    # Build the observed profile on the symmetric η grid (mirror around 0)
    obs_profile = np.interp(np.abs(eta_grid), exc_centers, bin_signal_mean, right=0)
    obs_profile = np.nan_to_num(obs_profile, nan=0)

    # Wiener deconvolution
    F_obs = np.fft.fft(obs_profile)
    F_kernel = np.fft.fft(spec_kernel)
    noise_level = 0.01 * np.abs(F_obs).max()**2  # regularization
    F_relp = F_obs * np.conj(F_kernel) / (np.abs(F_kernel)**2 + noise_level)
    relp_deconv = np.fft.ifftshift(np.real(np.fft.ifft(F_relp)))

    # Also compute the reconvolution for validation
    reconv = np.real(np.fft.ifft(F_relp * F_kernel))

    # ── PLOTS ────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Observed I(η) with fits
    ax = axes[0, 0]
    ax.plot(exc_centers[ok] * 1000, bin_signal_mean[ok], "ko", markersize=4, label="data (mean)")
    eta_fit = np.linspace(0, exc_centers[ok].max(), 200)
    ax.plot(eta_fit * 1000, bin_signal_mean[ok][0] * gaussian(eta_fit, *popt_g),
            "C0-", lw=2, label=f"Gaussian σ={sigma_obs*1000:.2f}×10⁻³ Å⁻¹")
    ax.plot(eta_fit * 1000, bin_signal_mean[ok][0] * lorentzian(eta_fit, *popt_l),
            "C1--", lw=2, label=f"Lorentzian γ={gamma_obs*1000:.2f}×10⁻³ Å⁻¹")
    ax.axvline(profile_rad * 1000, color="red", ls=":", label=f"profile_rad={profile_rad*1000:.1f}×10⁻³")
    ax.set_xlabel("Excitation error η (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Mean (I − background) (ADU)")
    ax.set_title("Observed intensity vs excitation error")
    ax.legend(fontsize=8)

    # 2. Spectral kernel
    ax = axes[0, 1]
    ax.plot(eta_grid * 1000, spec_kernel / spec_kernel.max(), "C2-", lw=1.5,
            label=f"Spectral kernel (v_r={v_r_median:.3f})")
    # Also show Gaussian fit of kernel
    spec_sigma = v_r_median * sigma_wl
    ax.plot(eta_grid * 1000, gaussian(eta_grid, 1.0, spec_sigma) , "C3--", lw=1.5,
            label=f"Gaussian σ_spec={spec_sigma*1000:.2f}×10⁻³")
    ax.set_xlabel("η (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Spectral broadening kernel")
    ax.set_xlim(-3, 3)
    ax.legend(fontsize=8)

    # 3. Deconvolved relp shape
    ax = axes[0, 2]
    # Only plot central region
    center = len(eta_grid) // 2
    w = 150  # bins around center
    eta_c = eta_grid[center-w:center+w]
    relp_c = relp_deconv[center-w:center+w]
    relp_c = relp_c / relp_c.max()
    ax.plot(eta_c * 1000, relp_c, "C4-", lw=2, label="Deconvolved relp")
    if sigma_relp > 0:
        ax.plot(eta_c * 1000, gaussian(eta_c, 1.0, sigma_relp),
                "C0--", lw=1.5, label=f"Gaussian σ_relp={sigma_relp*1000:.2f}×10⁻³")
    ax.set_xlabel("η (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Normalized relp profile")
    ax.set_title("Intrinsic relp shape (deconvolved)")
    ax.legend(fontsize=8)

    # 4. Fractional intensity (I-bg)/bg
    ax = axes[1, 0]
    ax.plot(exc_centers[ok] / profile_rad, bin_frac_mean[ok], "s-", color="C5", markersize=4)
    ax.axvline(1.0, color="red", ls=":", label="1× profile_rad")
    ax.set_xlabel("Excitation error / profile_rad")
    ax.set_ylabel("Median (I − bg) / bg")
    ax.set_title("Fractional signal above background")
    ax.legend()

    # 5. Resolution dependence of spectral broadening
    ax = axes[1, 1]
    d_bins = np.linspace(d_sp.min(), min(d_sp.max(), 5), 20)
    d_centers = 0.5 * (d_bins[:-1] + d_bins[1:])
    sigma_eta_per_d = []
    for j in range(len(d_bins) - 1):
        in_d = (d_sp >= d_bins[j]) & (d_sp < d_bins[j + 1])
        if in_d.sum() > 50:
            sigma_eta_per_d.append(np.median(v_r[in_d]) * sigma_wl)
        else:
            sigma_eta_per_d.append(np.nan)
    sigma_eta_per_d = np.array(sigma_eta_per_d)
    ax.plot(d_centers, sigma_eta_per_d * 1000, "D-", color="C6", markersize=5)
    ax.axhline(profile_rad * 1000, color="red", ls=":", label="profile_rad")
    ax.set_xlabel("d-spacing (Å)")
    ax.set_ylabel("σ_spectral (×10⁻³ Å⁻¹)")
    ax.set_title("Spectral broadening vs resolution")
    ax.legend()
    ax.invert_xaxis()

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        f"Relp Shape Analysis\n"
        f"{'─'*35}\n"
        f"Reflections:       {N:,}\n"
        f"Crystals:          {len(sample_cids)}\n"
        f"profile_rad (CrystFEL): {profile_rad:.4f} Å⁻¹\n"
        f"\n"
        f"Beam:\n"
        f"  Energy:          12960 eV\n"
        f"  λ:               {wl_center:.4f} Å\n"
        f"  σ_λ:             {sigma_wl*1e4:.2f}×10⁻⁴ Å\n"
        f"  σ_E:             {sigma_E:.1f} eV\n"
        f"  Bandwidth:       {sigma_E/12960*100:.3f}%\n"
        f"\n"
        f"Observed profile (Gaussian fit):\n"
        f"  σ_obs:           {sigma_obs*1000:.3f}×10⁻³ Å⁻¹\n"
        f"\n"
        f"Spectral broadening (median v_r):\n"
        f"  v_r = d*/λ:      {v_r_median:.4f} Å⁻²\n"
        f"  σ_spec:          {spec_sigma*1000:.3f}×10⁻³ Å⁻¹\n"
        f"\n"
        f"Deconvolved relp:\n"
        f"  σ_relp:          {sigma_relp*1000:.3f}×10⁻³ Å⁻¹\n"
    )
    if sigma_relp > 0:
        D_nm = 1.0 / (2 * np.pi * sigma_relp) * 10
        summary += f"  Domain size:     {D_nm:.0f} nm\n"
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Relp shape from excitation error analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "relp_shape_analysis.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    plt.close(fig)

    # ── Also save the mean spectrum plot ─────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Mean spectrum in energy space
    ax1.plot(energy_eV, mean_spectrum / mean_spectrum.max(), "k-", lw=0.5, alpha=0.3)
    # Show a few individual shots
    for j in range(min(5, n_shots)):
        ax1.plot(energy_eV, spectra[j] / spectra[j].max(), lw=0.3, alpha=0.5)
    ax1.plot(energy_eV, mean_spectrum / mean_spectrum.max(), "C0-", lw=2, label="mean")
    ax1.set_xlabel("Photon energy (eV)")
    ax1.set_ylabel("Normalized intensity")
    ax1.set_title(f"SASE spectra ({n_shots} shots)")
    ax1.set_xlim(12880, 13040)
    ax1.legend()

    # Spectrum converted to η-space at different resolutions
    for d_val, color in [(3.0, "C1"), (2.0, "C2"), (1.5, "C3"), (1.0, "C4")]:
        d_star_v = 1.0 / d_val
        v_r_v = d_star_v / wl_center
        wl_from_eta_v = wl_center + eta_grid / v_r_v
        kern = np.interp(wl_from_eta_v, wl_bins[::-1] if wl_bins[0] > wl_bins[-1] else wl_bins,
                         mean_spectrum[::-1] if wl_bins[0] > wl_bins[-1] else mean_spectrum,
                         left=0, right=0)
        kern = kern / kern.max()
        ax2.plot(eta_grid * 1000, kern, color=color, lw=1.5, label=f"d={d_val:.1f} Å")
    ax2.axvline(profile_rad * 1000, color="red", ls=":", label="profile_rad")
    ax2.axvline(-profile_rad * 1000, color="red", ls=":")
    ax2.set_xlabel("Excitation error η (×10⁻³ Å⁻¹)")
    ax2.set_ylabel("Normalized spectral kernel")
    ax2.set_title("Spectral kernel in η-space (resolution-dependent)")
    ax2.set_xlim(-5, 5)
    ax2.legend(fontsize=9)

    fig2.tight_layout()
    out2 = OUT_DIR / "spectral_kernel.png"
    fig2.savefig(out2, dpi=150)
    print(f"Saved {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
