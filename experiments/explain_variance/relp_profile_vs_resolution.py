#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Fit 2D Gaussian profiles to ROI patches, decompose into radial/tangential
widths, and plot vs resolution.

'Radial'     = along the direction from beam center to spot (radial streak direction)
'Tangential' = perpendicular to that on the detector surface

Method:
  1. Subtract background (edge ring)
  2. Compute intensity-weighted 2D covariance matrix
  3. Get eigenvalues → σ_major, σ_minor
  4. Get eigenvectors → orientation angle
  5. Use panel geometry to map fs/ss directions to lab frame
  6. Determine which eigenvector is radial (points toward/away from beam)
  7. Assign σ_radial and σ_tangential accordingly
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binned_statistic
import os, sys, glob

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

ROI_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
STREAM_DIR = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"


def load_one_stream():
    """Load one stream + geometry for panel info."""
    from torchsx.io._safetensors import load_sfx
    core = load_sfx(str(STREAM_DIR / "delay_-91.sfx"))
    return core


def load_all_streams_concat():
    """Load all 6 streams, concatenate obs-level tensors."""
    from torchsx.io._safetensors import load_sfx
    names = ["delay_-45", "delay_-91", "delay_-136",
             "delay_-182", "delay_-227", "delay_-273"]
    all_hkl, all_I, all_fs, all_ss, all_pid = [], [], [], [], []
    all_cid, all_astar, all_bstar, all_cstar = [], [], [], []
    cryst_offset = 0
    geo = None
    for name in names:
        c = load_sfx(str(STREAM_DIR / f"{name}.sfx"))
        all_hkl.append(c.hkl)
        all_I.append(c.I)
        all_fs.append(c.fs)
        all_ss.append(c.ss)
        all_pid.append(c.panel_id)
        all_cid.append(c.crystal_id + cryst_offset)
        all_astar.append(c.astar)
        all_bstar.append(c.bstar)
        all_cstar.append(c.cstar)
        cryst_offset += c.astar.shape[0]
        if geo is None:
            geo = c.geometry
        print(f"  {name}: {c.hkl.shape[0]:,} obs")
    hkl = torch.cat(all_hkl)
    I = torch.cat(all_I)
    fs = torch.cat(all_fs)
    ss = torch.cat(all_ss)
    pid = torch.cat(all_pid)
    cid = torch.cat(all_cid)
    astar = torch.cat(all_astar)
    bstar = torch.cat(all_bstar)
    cstar = torch.cat(all_cstar)
    print(f"  Total: {hkl.shape[0]:,} obs")
    return hkl, I, fs, ss, pid, cid, astar, bstar, cstar, geo


def compute_radial_direction_in_panel(fs, ss, panel_id, geometry):
    """
    For each observation, compute the radial unit vector in local (fs, ss) coords.

    The radial direction points from beam center (0,0 in lab) to the spot
    position on the detector. We convert this lab-frame direction into the
    panel-local (fs, ss) coordinate system.

    Returns (rad_fs, rad_ss) — unit vectors in panel-local coords.
    """
    # Get lab positions of each spot
    lab_xyz = geometry.pixel_to_lab(fs, ss, panel_id)  # (N, 3) meters
    # Radial direction in lab frame: (x, y, 0) normalized
    lab_xy = lab_xyz[:, :2]  # (N, 2) — x, y in detector plane
    lab_r = lab_xy.norm(dim=1, keepdim=True).clamp(min=1e-6)
    rad_lab = lab_xy / lab_r  # (N, 2) unit radial vector in lab

    # For each panel, the fs and ss vectors in lab frame are known.
    # We need to express rad_lab in the local (fs_hat, ss_hat) basis.
    panels = geometry.panels
    n_panels = len(panels)

    # Build panel fs/ss direction vectors (lab frame, 2D projection)
    fs_dirs = torch.zeros(n_panels, 2)
    ss_dirs = torch.zeros(n_panels, 2)
    for i, p in enumerate(panels):
        fs_dirs[i, 0] = p.fs_vec[0]
        fs_dirs[i, 1] = p.fs_vec[1]
        ss_dirs[i, 0] = p.ss_vec[0]
        ss_dirs[i, 1] = p.ss_vec[1]

    # Gather per-observation panel directions
    pid_long = panel_id.long()
    fs_hat = fs_dirs[pid_long]  # (N, 2)
    ss_hat = ss_dirs[pid_long]  # (N, 2)

    # Project radial lab direction onto (fs, ss) basis
    # rad_lab = alpha * fs_hat + beta * ss_hat
    # alpha = rad_lab · fs_hat, beta = rad_lab · ss_hat
    # (This is exact when fs and ss are orthonormal, good approx otherwise)
    rad_fs = (rad_lab * fs_hat).sum(dim=1)  # (N,)
    rad_ss = (rad_lab * ss_hat).sum(dim=1)  # (N,)

    # Normalize
    norm = torch.sqrt(rad_fs**2 + rad_ss**2).clamp(min=1e-6)
    rad_fs = rad_fs / norm
    rad_ss = rad_ss / norm

    return rad_fs, rad_ss


def fit_patches_batch(patches, valid, rad_fs, rad_ss, bg_border=3):
    """
    Fit 2D Gaussians via moments, decompose into radial/tangential.

    Returns dict of (N,) arrays.
    """
    N, H, W = patches.shape
    cy, cx = H // 2, W // 2

    # Pixel grids centered on patch center
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) - cy,
        torch.arange(W, dtype=torch.float32) - cx,
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(N, -1, -1)
    yy = yy.unsqueeze(0).expand(N, -1, -1)

    valid_bool = valid.bool()

    # Background from border ring
    border = torch.zeros(H, W, dtype=torch.bool)
    border[:bg_border, :] = True
    border[-bg_border:, :] = True
    border[:, :bg_border] = True
    border[:, -bg_border:] = True
    border = border.unsqueeze(0).expand(N, -1, -1)
    bv = border & valid_bool
    bg = (patches * bv.float()).sum(dim=(1, 2)) / bv.float().sum(dim=(1, 2)).clamp(min=1)

    # Signal = background-subtracted, inner region only, clamped
    signal = (patches - bg[:, None, None]) * valid_bool.float()
    inner = torch.zeros(H, W, dtype=torch.bool)
    inner[bg_border:H-bg_border, bg_border:W-bg_border] = True
    signal = signal * inner.unsqueeze(0)
    signal = signal.clamp(min=0)

    total = signal.sum(dim=(1, 2))

    # Weighted centroid
    mu_x = (signal * xx).sum(dim=(1, 2)) / total.clamp(min=1)
    mu_y = (signal * yy).sum(dim=(1, 2)) / total.clamp(min=1)

    # Weighted covariance
    dx = xx - mu_x[:, None, None]
    dy = yy - mu_y[:, None, None]
    Cxx = (signal * dx * dx).sum(dim=(1, 2)) / total.clamp(min=1)
    Cyy = (signal * dy * dy).sum(dim=(1, 2)) / total.clamp(min=1)
    Cxy = (signal * dx * dy).sum(dim=(1, 2)) / total.clamp(min=1)

    # Eigenvalues of 2x2 covariance [[Cxx, Cxy], [Cxy, Cyy]]
    trace = Cxx + Cyy
    det = Cxx * Cyy - Cxy * Cxy
    disc = (trace**2 / 4 - det).clamp(min=0)
    lam1 = trace / 2 + disc.sqrt()  # larger eigenvalue
    lam2 = trace / 2 - disc.sqrt()  # smaller eigenvalue

    sigma_major = lam1.clamp(min=0.01).sqrt()
    sigma_minor = lam2.clamp(min=0.01).sqrt()

    # Eigenvector of lam1: (Cxy, lam1 - Cxx) normalized
    ev1_x = Cxy
    ev1_y = lam1 - Cxx
    ev1_norm = (ev1_x**2 + ev1_y**2).clamp(min=1e-10).sqrt()
    ev1_x = ev1_x / ev1_norm  # major axis direction in (fs, ss)
    ev1_y = ev1_y / ev1_norm

    # Project major axis onto radial direction
    # cos(angle) between major axis eigenvector and radial direction
    # Note: xx is fs direction, yy is ss direction in the patch
    cos_angle = (ev1_x * rad_fs + ev1_y * rad_ss).abs()
    # cos_angle > 0.707 → major axis is closer to radial → σ_major = σ_radial
    # cos_angle < 0.707 → major axis is closer to tangential

    is_radial_major = cos_angle > 0.5  # use 0.5 for smoother transition
    sigma_r = torch.where(is_radial_major, sigma_major, sigma_minor)
    sigma_t = torch.where(is_radial_major, sigma_minor, sigma_major)

    peak = patches[:, cy, cx] - bg

    return {
        "sigma_r": sigma_r, "sigma_t": sigma_t,
        "sigma_major": sigma_major, "sigma_minor": sigma_minor,
        "cos_angle": cos_angle, "total": total, "peak": peak,
        "elongation": sigma_major / sigma_minor.clamp(min=0.1),
    }


def main():
    print("Loading stream metadata...")
    hkl, I, fs, ss, pid, cid, astar, bstar, cstar, geo = load_all_streams_concat()

    # d-spacing
    print("Computing d-spacing...")
    wavelength = 0.9567
    from torchsx.merge._neural_model import compute_excitation_error
    exc_err, q_lab = compute_excitation_error(hkl, astar, bstar, cstar, cid, wavelength)
    q_mag = q_lab.norm(dim=1)
    d = 1.0 / q_mag.clamp(min=1e-6)

    # Radial direction per observation in panel-local coords
    print("Computing radial directions...")
    rad_fs, rad_ss = compute_radial_direction_in_panel(fs, ss, pid, geo)

    # Process ROI parts
    n_parts = 126
    batch_size = 10

    all_sr, all_st, all_smaj, all_smin = [], [], [], []
    all_d, all_I, all_elong, all_cos = [], [], [], []

    obs_offset = 0
    for b0 in range(0, n_parts, batch_size):
        b1 = min(b0 + batch_size, n_parts)
        print(f"  Parts {b0}–{b1-1}...", end=" ", flush=True)

        from safetensors import safe_open
        patches_list, valid_list = [], []
        for idx in range(b0, b1):
            f = safe_open(str(ROI_DIR / f"rois_part_{idx:03d}.safetensors"), framework="pt")
            patches_list.append(f.get_tensor("patches"))
            valid_list.append(f.get_tensor("valid"))
        patches = torch.cat(patches_list)
        valid = torch.cat(valid_list)
        nb = patches.shape[0]

        rf = rad_fs[obs_offset:obs_offset + nb]
        rs = rad_ss[obs_offset:obs_offset + nb]
        d_b = d[obs_offset:obs_offset + nb]
        I_b = I[obs_offset:obs_offset + nb]

        res = fit_patches_batch(patches, valid, rf, rs)

        # Quality filter: strong spots with measurable shape
        good = (res["total"] > 500) & (I_b > 100) & (res["peak"] > 20) & \
               (res["sigma_major"] > 0.3) & (res["sigma_major"] < 8) & \
               (res["sigma_minor"] > 0.3)

        ng = good.sum().item()
        print(f"{nb} patches, {ng} good ({ng/nb*100:.1f}%)")

        all_sr.append(res["sigma_r"][good].numpy())
        all_st.append(res["sigma_t"][good].numpy())
        all_smaj.append(res["sigma_major"][good].numpy())
        all_smin.append(res["sigma_minor"][good].numpy())
        all_d.append(d_b[good].numpy())
        all_I.append(I_b[good].numpy())
        all_elong.append(res["elongation"][good].numpy())
        all_cos.append(res["cos_angle"][good].numpy())

        obs_offset += nb
        del patches, valid

    sr = np.concatenate(all_sr)
    st = np.concatenate(all_st)
    smaj = np.concatenate(all_smaj)
    smin = np.concatenate(all_smin)
    d_arr = np.concatenate(all_d)
    I_arr = np.concatenate(all_I)
    elong = np.concatenate(all_elong)
    cos_a = np.concatenate(all_cos)
    inv_d = 1.0 / d_arr

    print(f"\nTotal good: {len(sr):,}")

    # ── Figures ────────────────────────────────────────────────

    bins_d = np.linspace(0.05, 1.15, 30)

    # Fig 1: σ_r and σ_t vs resolution
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Relp profile: radial vs tangential — {len(sr):,} strong spots\n"
        f"2D Gaussian moments, panel-geometry radial decomposition",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    for vals, label, col in [(sr, "σ_radial", "#534AB7"), (st, "σ_tangential", "#D85A30")]:
        med, edges, _ = binned_statistic(inv_d, vals, statistic="median", bins=bins_d)
        q25, _, _ = binned_statistic(inv_d, vals, statistic=lambda x: np.percentile(x, 25), bins=bins_d)
        q75, _, _ = binned_statistic(inv_d, vals, statistic=lambda x: np.percentile(x, 75), bins=bins_d)
        c = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(c, q25, q75, alpha=0.15, color=col)
        ax.plot(c, med, "o-", color=col, lw=2, ms=4, label=label)
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("σ (pixels)")
    ax.set_title("Median σ_r, σ_t vs resolution")
    ax.legend()

    ax = axes[0, 1]
    ratio = sr / st.clip(min=0.1)
    med_ratio, edges, _ = binned_statistic(inv_d, ratio, statistic="median", bins=bins_d)
    q25_r, _, _ = binned_statistic(inv_d, ratio, statistic=lambda x: np.percentile(x, 25), bins=bins_d)
    q75_r, _, _ = binned_statistic(inv_d, ratio, statistic=lambda x: np.percentile(x, 75), bins=bins_d)
    c = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(c, q25_r, q75_r, alpha=0.2, color="#1D9E75")
    ax.plot(c, med_ratio, "o-", color="#1D9E75", lw=2, ms=4)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("σ_r / σ_t")
    ax.set_title("Radial/tangential ratio vs resolution")

    ax = axes[1, 0]
    med_elong, _, _ = binned_statistic(inv_d, elong, statistic="median", bins=bins_d)
    q25_e, _, _ = binned_statistic(inv_d, elong, statistic=lambda x: np.percentile(x, 25), bins=bins_d)
    q75_e, _, _ = binned_statistic(inv_d, elong, statistic=lambda x: np.percentile(x, 75), bins=bins_d)
    ax.fill_between(c, q25_e, q75_e, alpha=0.2, color="#B5338A")
    ax.plot(c, med_elong, "o-", color="#B5338A", lw=2, ms=4)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("σ_major / σ_minor")
    ax.set_title("Spot elongation vs resolution")

    ax = axes[1, 1]
    # Fraction where major axis is radial-aligned
    med_cos, _, _ = binned_statistic(inv_d, cos_a, statistic="median", bins=bins_d)
    frac_radial, _, _ = binned_statistic(inv_d, (cos_a > 0.707).astype(float),
                                          statistic="mean", bins=bins_d)
    ax.plot(c, frac_radial * 100, "o-", color="#534AB7", lw=2, ms=4,
            label="% major axis radial")
    ax.plot(c, med_cos, "s--", color="#D85A30", lw=2, ms=4,
            label="median |cos(angle)|")
    ax.axhline(50, color="gray", ls=":", lw=1, label="random expectation")
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("Fraction / cos")
    ax.set_title("Major axis alignment with radial direction")
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig_relp_profile_vs_resolution.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_relp_profile_vs_resolution.png'}")
    plt.close(fig)

    # Fig 2: distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Profile width distributions", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.hist(sr, bins=80, range=(0, 5), density=True, alpha=0.5, color="#534AB7",
            label=f"σ_r (med={np.median(sr):.2f})")
    ax.hist(st, bins=80, range=(0, 5), density=True, alpha=0.5, color="#D85A30",
            label=f"σ_t (med={np.median(st):.2f})")
    ax.set_xlabel("σ (px)")
    ax.legend()
    ax.set_title("Radial vs tangential")

    ax = axes[1]
    ax.hist(elong, bins=80, range=(1, 4), density=True, color="#B5338A", alpha=0.7)
    ax.set_xlabel("σ_major / σ_minor")
    ax.set_title(f"Elongation (median={np.median(elong):.2f})")

    ax = axes[2]
    ax.hist(cos_a, bins=60, range=(0, 1), density=True, color="#1D9E75", alpha=0.7)
    ax.axvline(0.707, color="gray", ls=":", lw=1, label="45°")
    ax.set_xlabel("|cos(major axis, radial)|")
    ax.set_title("Major axis alignment")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig_relp_profile_distributions.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_relp_profile_distributions.png'}")
    plt.close(fig)

    # Summary table
    print("\n" + "=" * 70)
    print("RELP PROFILE SUMMARY")
    print("=" * 70)
    print(f"  N: {len(sr):,}")
    print(f"  σ_radial:     med = {np.median(sr):.3f} px  IQR = [{np.percentile(sr,25):.3f}, {np.percentile(sr,75):.3f}]")
    print(f"  σ_tangential: med = {np.median(st):.3f} px  IQR = [{np.percentile(st,25):.3f}, {np.percentile(st,75):.3f}]")
    print(f"  σ_r/σ_t:     med = {np.median(ratio):.3f}")
    print(f"  elongation:   med = {np.median(elong):.3f}")
    d_bins = [0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    print(f"\n  {'d range':>12} {'σ_r':>6} {'σ_t':>6} {'r/t':>6} {'elong':>6} {'%rad':>6} {'N':>8}")
    for i in range(len(d_bins)-1):
        m = (d_arr >= d_bins[i]) & (d_arr < d_bins[i+1])
        if m.sum() < 50: continue
        print(f"  {d_bins[i]:5.1f}–{d_bins[i+1]:4.1f} Å "
              f"{np.median(sr[m]):6.3f} {np.median(st[m]):6.3f} "
              f"{np.median(ratio[m]):6.3f} {np.median(elong[m]):6.3f} "
              f"{(cos_a[m]>0.707).mean()*100:5.1f}% {m.sum():>8,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
