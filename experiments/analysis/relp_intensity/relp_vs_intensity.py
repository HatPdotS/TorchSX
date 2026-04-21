#!/usr/bin/env python
"""Fractional intensity above background as a function of excitation error.

For each observed reflection across many indexed crystals:
  1. Compute excitation error (distance from Ewald sphere) = minimum relp to excite
  2. Compute fractional intensity above background: (I - bg) / bg
  3. Bin by excitation error and plot the relationship

Output: plots in the same directory as this script.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from torchsx.stream import SFXStream
from torchsx.forward.coords import RefinableGeometry

HC_EV_A = 12398.4
OUT_DIR = Path(__file__).resolve().parent


def excitation_error(h_lab: torch.Tensor, wavelength: float) -> torch.Tensor:
    """Distance from Ewald sphere in reciprocal space (Å⁻¹)."""
    s_out = wavelength * h_lab.clone()
    s_out[:, 2] += 1.0
    return (s_out.norm(dim=-1) - 1.0).abs() / wavelength


def main():
    sfx_path = PROJECT_ROOT / "merged_368.sfx"
    print(f"Loading {sfx_path.name}...")
    stream = SFXStream.load(str(sfx_path), merge_friedel=False)
    core = stream._core
    geom = core.geometry
    profile_rad = core.profile_rad[0].item()

    # Sample 500 crystals with enough reflections
    np.random.seed(42)
    n_refl = core.n_refl.numpy()
    good = np.where(n_refl >= 20)[0]
    sample_cids = np.random.choice(good, min(500, len(good)), replace=False)

    print(f"Sampling {len(sample_cids)} crystals (profile_rad={profile_rad:.4f} Å⁻¹)...")

    ref_geom = RefinableGeometry(geom)

    all_exc = []       # excitation error (Å⁻¹)
    all_I = []         # integrated intensity
    all_bg = []        # background
    all_peak = []      # peak value
    all_d = []         # d-spacing
    all_pos_err = []   # position error (px)

    for i, cid in enumerate(sample_cids):
        cid = int(cid)
        mask = core.crystal_id == cid
        n = mask.sum().item()
        if n < 10:
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

        exc = excitation_error(h_lab, wl)
        d = 1.0 / h_lab.norm(dim=-1).clamp(min=1e-10)

        # Position error
        ref_geom._clen.fill_(cl)
        fs_obs = core.fs[mask].float()
        ss_obs = core.ss[mask].float()
        pid_obs = core.panel_id[mask]
        with torch.no_grad():
            pred_fs, pred_ss, pred_pid = ref_geom.recip_to_pixel(h_lab, wl)
        pos_err = torch.sqrt((pred_fs - fs_obs)**2 + (pred_ss - ss_obs)**2)

        all_exc.append(exc.numpy())
        all_I.append(core.I[mask].numpy())
        all_bg.append(core.background[mask].numpy())
        all_peak.append(core.peak[mask].numpy())
        all_d.append(d.numpy())
        all_pos_err.append(pos_err.numpy())

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sample_cids)} crystals processed...")

    exc = np.concatenate(all_exc)
    I_obs = np.concatenate(all_I)
    bg = np.concatenate(all_bg)
    peak = np.concatenate(all_peak)
    d_sp = np.concatenate(all_d)
    pos_err = np.concatenate(all_pos_err)
    N = len(exc)

    print(f"\n{N} total reflections from {len(sample_cids)} crystals")

    # Fractional intensity above background
    # Avoid division by zero: only use reflections where bg > 0
    valid = bg > 0
    frac_I = np.where(valid, (I_obs - bg) / bg, 0.0)
    signal = I_obs - bg  # net signal

    # Normalize excitation error by profile_rad
    exc_norm = exc / profile_rad

    # ── Binning ──
    n_bins = 40
    exc_edges = np.linspace(0, exc.max() * 1.01, n_bins + 1)
    exc_centers = 0.5 * (exc_edges[:-1] + exc_edges[1:])
    exc_norm_edges = exc_edges / profile_rad
    exc_norm_centers = exc_centers / profile_rad

    # Compute bin statistics
    bin_median_frac = np.zeros(n_bins)
    bin_mean_frac = np.zeros(n_bins)
    bin_median_I = np.zeros(n_bins)
    bin_mean_I = np.zeros(n_bins)
    bin_median_signal = np.zeros(n_bins)
    bin_median_peak = np.zeros(n_bins)
    bin_median_bg = np.zeros(n_bins)
    bin_median_d = np.zeros(n_bins)
    bin_median_pos_err = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    bin_q25_frac = np.zeros(n_bins)
    bin_q75_frac = np.zeros(n_bins)

    for b in range(n_bins):
        in_bin = (exc >= exc_edges[b]) & (exc < exc_edges[b + 1])
        n_in = in_bin.sum()
        bin_count[b] = n_in
        if n_in < 10:
            bin_median_frac[b] = np.nan
            bin_mean_frac[b] = np.nan
            bin_median_I[b] = np.nan
            bin_mean_I[b] = np.nan
            bin_median_signal[b] = np.nan
            bin_median_peak[b] = np.nan
            bin_median_bg[b] = np.nan
            bin_median_d[b] = np.nan
            bin_median_pos_err[b] = np.nan
            bin_q25_frac[b] = np.nan
            bin_q75_frac[b] = np.nan
            continue
        bin_median_frac[b] = np.median(frac_I[in_bin])
        bin_mean_frac[b] = np.mean(frac_I[in_bin])
        bin_q25_frac[b] = np.percentile(frac_I[in_bin], 25)
        bin_q75_frac[b] = np.percentile(frac_I[in_bin], 75)
        bin_median_I[b] = np.median(I_obs[in_bin])
        bin_mean_I[b] = np.mean(I_obs[in_bin])
        bin_median_signal[b] = np.median(signal[in_bin])
        bin_median_peak[b] = np.median(peak[in_bin])
        bin_median_bg[b] = np.median(bg[in_bin])
        bin_median_d[b] = np.median(d_sp[in_bin])
        bin_median_pos_err[b] = np.median(pos_err[in_bin])

    ok = bin_count >= 10

    # ── Plot 1: Fractional intensity above background vs excitation error ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.fill_between(exc_norm_centers[ok], bin_q25_frac[ok], bin_q75_frac[ok],
                     alpha=0.3, color="C0", label="IQR")
    ax.plot(exc_norm_centers[ok], bin_median_frac[ok], "o-", color="C0",
            markersize=4, label="median")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(1.0, color="red", ls=":", lw=1, label=f"1× profile_rad ({profile_rad:.4f} Å⁻¹)")
    ax.set_xlabel("Excitation error / profile_rad")
    ax.set_ylabel("(I − background) / background")
    ax.set_title("Fractional intensity above background")
    ax.legend(fontsize=9)

    # ── Plot 2: Median integrated intensity ──
    ax = axes[0, 1]
    ax.plot(exc_norm_centers[ok], bin_median_I[ok], "s-", color="C1",
            markersize=4, label="median I")
    ax.plot(exc_norm_centers[ok], bin_median_bg[ok], "^-", color="C2",
            markersize=4, label="median background")
    ax.plot(exc_norm_centers[ok], bin_median_signal[ok], "o-", color="C3",
            markersize=4, label="median (I − bg)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(1.0, color="red", ls=":", lw=1)
    ax.set_xlabel("Excitation error / profile_rad")
    ax.set_ylabel("Intensity (ADU)")
    ax.set_title("Intensity components vs excitation error")
    ax.legend(fontsize=9)

    # ── Plot 3: Position error vs excitation error ──
    ax = axes[1, 0]
    ax.plot(exc_norm_centers[ok], bin_median_pos_err[ok], "D-", color="C4", markersize=4)
    ax.axvline(1.0, color="red", ls=":", lw=1)
    ax.set_xlabel("Excitation error / profile_rad")
    ax.set_ylabel("Position error (px)")
    ax.set_title("Prediction–observation position mismatch")

    # ── Plot 4: Count per bin + CDF ──
    ax = axes[1, 1]
    ax.bar(exc_norm_centers, bin_count, width=np.diff(exc_norm_edges)[0] * 0.9,
           alpha=0.6, color="C5")
    ax.set_xlabel("Excitation error / profile_rad")
    ax.set_ylabel("Reflections per bin", color="C5")
    ax2 = ax.twinx()
    cdf = np.cumsum(bin_count) / bin_count.sum() * 100
    ax2.plot(exc_norm_centers, cdf, "k-", lw=2, label="CDF")
    ax2.set_ylabel("Cumulative %")
    ax2.axhline(50, color="gray", ls="--", lw=0.8)
    ax2.axhline(90, color="gray", ls="--", lw=0.8)
    ax.axvline(1.0, color="red", ls=":", lw=1)
    ax.set_title("Excitation error distribution")

    fig.suptitle(f"Relp analysis: {N} reflections from {len(sample_cids)} crystals\n"
                 f"profile_rad = {profile_rad:.4f} Å⁻¹ (fixed)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = OUT_DIR / "relp_vs_intensity.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)

    # ── Plot 5: 2D histogram — excitation error vs (I-bg)/bg ──
    fig, ax = plt.subplots(figsize=(10, 7))
    valid_mask = valid & np.isfinite(frac_I) & (np.abs(frac_I) < 50)
    h = ax.hist2d(exc[valid_mask] / profile_rad, frac_I[valid_mask],
                   bins=[80, 80], range=[[0, 2.0], [-2, 10]],
                   cmin=1, cmap="viridis")
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.axhline(0, color="white", ls="--", lw=0.8)
    ax.axvline(1.0, color="red", ls=":", lw=1.5, label="1× profile_rad")
    ax.set_xlabel("Excitation error / profile_rad")
    ax.set_ylabel("(I − background) / background")
    ax.set_title(f"Fractional signal vs excitation error ({N} reflections)")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "relp_vs_intensity_2d.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)

    # ── Print summary table ──
    print(f"\n{'Exc/profile_rad':>16s}  {'count':>6s}  {'(I-bg)/bg':>10s}  "
          f"{'I':>8s}  {'bg':>6s}  {'I-bg':>8s}  {'peak':>5s}  {'d(Å)':>6s}  {'pos_err':>8s}")
    print("-" * 90)
    for b in range(n_bins):
        if bin_count[b] < 10:
            continue
        print(f"  {exc_norm_centers[b]:13.2f}×  {bin_count[b]:6d}  "
              f"{bin_median_frac[b]:10.3f}  {bin_median_I[b]:8.1f}  {bin_median_bg[b]:6.1f}  "
              f"{bin_median_signal[b]:8.1f}  {bin_median_peak[b]:5.0f}  "
              f"{bin_median_d[b]:6.2f}  {bin_median_pos_err[b]:7.1f}px")


if __name__ == "__main__":
    main()
