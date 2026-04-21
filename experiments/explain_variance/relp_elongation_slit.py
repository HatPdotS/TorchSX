#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Measure spot elongation by comparing radial vs tangential slit integrations.

No fitting — just sum pixel intensities in perpendicular slits through
the spot center. If the spot is radially elongated, the radial slit
captures more signal than the tangential slit.

Method:
  For each patch, define the radial direction (beam center → spot)
  in panel-local (fs, ss) coordinates using the detector geometry.

  Radial slit:    pixels within ±W of the radial line through patch center
  Tangential slit: pixels within ±W of the tangential line

  Metric: R = Σ(signal in radial slit) / Σ(signal in tangential slit)
    R > 1 → radially elongated
    R = 1 → isotropic
    R < 1 → tangentially elongated
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binned_statistic
import os, sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

ROI_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
STREAM_DIR = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"


def load_all_streams():
    from torchsx.io._safetensors import load_sfx
    names = ["delay_-45", "delay_-91", "delay_-136",
             "delay_-182", "delay_-227", "delay_-273"]
    all_I, all_fs, all_ss, all_pid = [], [], [], []
    all_hkl, all_cid, all_astar, all_bstar, all_cstar = [], [], [], [], []
    co = 0
    geo = None
    for name in names:
        c = load_sfx(str(STREAM_DIR / f"{name}.sfx"))
        all_I.append(c.I); all_fs.append(c.fs); all_ss.append(c.ss)
        all_pid.append(c.panel_id); all_hkl.append(c.hkl)
        all_cid.append(c.crystal_id + co)
        all_astar.append(c.astar); all_bstar.append(c.bstar); all_cstar.append(c.cstar)
        co += c.astar.shape[0]
        if geo is None: geo = c.geometry
        print(f"  {name}: {c.hkl.shape[0]:,}")
    return (torch.cat(all_hkl), torch.cat(all_I), torch.cat(all_fs),
            torch.cat(all_ss), torch.cat(all_pid), torch.cat(all_cid),
            torch.cat(all_astar), torch.cat(all_bstar), torch.cat(all_cstar), geo)


def compute_radial_dir(fs, ss, panel_id, geometry):
    """Radial unit vector in panel-local (fs, ss) coordinates."""
    lab = geometry.pixel_to_lab(fs, ss, panel_id)  # (N, 3) meters
    lab_xy = lab[:, :2]
    lab_r = lab_xy.norm(dim=1, keepdim=True).clamp(min=1e-6)
    rad_lab = lab_xy / lab_r  # (N, 2) unit radial in lab x,y

    panels = geometry.panels
    fs_dirs = torch.zeros(len(panels), 2)
    ss_dirs = torch.zeros(len(panels), 2)
    for i, p in enumerate(panels):
        fs_dirs[i] = torch.tensor([p.fs_vec[0], p.fs_vec[1]])
        ss_dirs[i] = torch.tensor([p.ss_vec[0], p.ss_vec[1]])

    pid_long = panel_id.long()
    fs_hat = fs_dirs[pid_long]  # (N, 2)
    ss_hat = ss_dirs[pid_long]

    rad_fs = (rad_lab * fs_hat).sum(dim=1)
    rad_ss = (rad_lab * ss_hat).sum(dim=1)
    norm = (rad_fs**2 + rad_ss**2).clamp(min=1e-6).sqrt()
    return rad_fs / norm, rad_ss / norm


def slit_elongation(patches, valid, rad_fs, rad_ss, slit_hw=1.5):
    """
    Compare radial vs tangential slit integrations.

    For each patch, compute displacement of every pixel from center,
    project onto radial and tangential, and select pixels within ±slit_hw
    of each axis.

    Returns ratio R = radial_sum / tangential_sum, and individual sums.
    """
    N, H, W = patches.shape
    cy, cx = H // 2, W // 2

    # Pixel displacement grid from patch center
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) - cy,
        torch.arange(W, dtype=torch.float32) - cx,
        indexing="ij",
    )
    # xx = fs direction, yy = ss direction in patch

    # For each pixel, compute distance to radial line and tangential line
    # Radial line direction: (rad_fs, rad_ss)
    # Distance of point (xx, yy) from the radial line through origin:
    #   d_to_radial = |xx * tan_fs + yy * tan_ss|
    # where (tan_fs, tan_ss) = (-rad_ss, rad_fs) is the tangential direction
    tan_fs = -rad_ss  # (N,)
    tan_ss = rad_fs

    # Distance of each pixel to the RADIAL line = projection onto tangential
    # This is the perpendicular distance to the radial axis
    dist_to_radial = (xx.unsqueeze(0) * tan_fs[:, None, None] +
                      yy.unsqueeze(0) * tan_ss[:, None, None]).abs()  # (N, H, W)

    # Distance of each pixel to the TANGENTIAL line = projection onto radial
    dist_to_tangential = (xx.unsqueeze(0) * rad_fs[:, None, None] +
                          yy.unsqueeze(0) * rad_ss[:, None, None]).abs()  # (N, H, W)

    # Slit masks
    radial_slit = (dist_to_radial <= slit_hw) & valid.bool()   # pixels near radial axis
    tangential_slit = (dist_to_tangential <= slit_hw) & valid.bool()  # pixels near tangential axis

    # Exclude overlap region (central ±slit_hw × ±slit_hw box)
    # Actually keep it simple — the overlap is the same for both slits

    # Sum signal in each slit
    sum_radial = (patches * radial_slit.float()).sum(dim=(1, 2))
    sum_tangential = (patches * tangential_slit.float()).sum(dim=(1, 2))
    n_radial = radial_slit.float().sum(dim=(1, 2))
    n_tangential = tangential_slit.float().sum(dim=(1, 2))

    # Background-normalized: mean intensity per pixel in each slit
    mean_radial = sum_radial / n_radial.clamp(min=1)
    mean_tangential = sum_tangential / n_tangential.clamp(min=1)

    # Ratio of mean intensities in the two slits
    ratio = mean_radial / mean_tangential.clamp(min=0.1)

    # Also compute total patch signal for quality filtering
    total = (patches * valid.bool().float()).sum(dim=(1, 2))
    peak = patches[:, cy, cx]

    return ratio, mean_radial, mean_tangential, total, peak


def main():
    print("Loading streams...")
    hkl, I, fs, ss, pid, cid, astar, bstar, cstar, geo = load_all_streams()
    print(f"  Total: {I.shape[0]:,} obs")

    print("Computing d-spacing...")
    from torchsx.merge._neural_model import compute_excitation_error
    _, q_lab = compute_excitation_error(hkl, astar, bstar, cstar, cid, 0.9567)
    d = 1.0 / q_lab.norm(dim=1).clamp(min=1e-6)

    print("Computing radial directions...")
    rad_fs, rad_ss = compute_radial_dir(fs, ss, pid, geo)

    # Process all ROI parts
    n_parts = 126
    bs = 10
    slit_widths = [0.8, 1.5, 2.5]  # try multiple slit half-widths

    results = {w: {"ratio": [], "d": [], "I": []} for w in slit_widths}

    obs_off = 0
    for b0 in range(0, n_parts, bs):
        b1 = min(b0 + bs, n_parts)
        print(f"  Parts {b0}–{b1-1}...", end=" ", flush=True)

        from safetensors import safe_open
        pl, vl = [], []
        for idx in range(b0, b1):
            f = safe_open(str(ROI_DIR / f"rois_part_{idx:03d}.safetensors"), framework="pt")
            pl.append(f.get_tensor("patches"))
            vl.append(f.get_tensor("valid"))
        patches = torch.cat(pl)
        valid = torch.cat(vl)
        nb = patches.shape[0]

        rf = rad_fs[obs_off:obs_off + nb]
        rs = rad_ss[obs_off:obs_off + nb]
        d_b = d[obs_off:obs_off + nb]
        I_b = I[obs_off:obs_off + nb]

        for w in slit_widths:
            ratio, mr, mt, total, peak = slit_elongation(patches, valid, rf, rs, slit_hw=w)

            # Quality: strong spots only
            good = (I_b > 50) & (peak > 10) & (total > 200) & \
                   (mt > 1) & (ratio > 0.3) & (ratio < 5)

            results[w]["ratio"].append(ratio[good].numpy())
            results[w]["d"].append(d_b[good].numpy())
            results[w]["I"].append(I_b[good].numpy())

        ng = good.sum().item()
        print(f"{nb} patches, {ng} good")
        obs_off += nb
        del patches, valid

    # Concatenate
    for w in slit_widths:
        for k in results[w]:
            results[w][k] = np.concatenate(results[w][k])

    # ── Figures ────────────────────────────────────────────────

    bins_d = np.linspace(0.05, 1.15, 35)

    # Fig 1: Slit ratio vs resolution for different slit widths
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Radial/tangential slit intensity ratio vs resolution\n"
        "R = ⟨I⟩_radial_slit / ⟨I⟩_tangential_slit  (R > 1 → radially elongated)",
        fontsize=12, fontweight="bold",
    )

    colors = ["#534AB7", "#D85A30", "#1D9E75"]
    for ax_i, (w, col) in enumerate(zip(slit_widths, colors)):
        ax = axes[ax_i]
        r = results[w]["ratio"]
        inv_d = 1.0 / results[w]["d"]
        n = len(r)

        med, edges, _ = binned_statistic(inv_d, r, statistic="median", bins=bins_d)
        q25, _, _ = binned_statistic(inv_d, r, statistic=lambda x: np.percentile(x, 25), bins=bins_d)
        q75, _, _ = binned_statistic(inv_d, r, statistic=lambda x: np.percentile(x, 75), bins=bins_d)
        c = 0.5 * (edges[:-1] + edges[1:])

        ax.fill_between(c, q25, q75, alpha=0.2, color=col)
        ax.plot(c, med, "o-", color=col, lw=2, ms=4)
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("1/d (Å⁻¹)")
        ax.set_ylabel("R = radial / tangential")
        ax.set_title(f"Slit half-width = {w} px (N={n:,})")
        ax.set_ylim(0.85, 1.25)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(OUTDIR / "fig_slit_ratio_vs_resolution.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_slit_ratio_vs_resolution.png'}")
    plt.close(fig)

    # Fig 2: Intensity-stratified slit ratio (use w=1.5)
    w = 1.5
    r = results[w]["ratio"]
    d_arr = results[w]["d"]
    I_arr = results[w]["I"]
    inv_d = 1.0 / d_arr

    I_thresholds = [50, 200, 500, 2000]
    fig, axes = plt.subplots(1, len(I_thresholds), figsize=(16, 4.5), sharey=True)
    fig.suptitle(f"Slit ratio vs resolution at different intensity thresholds (slit={w} px)",
                 fontsize=12, fontweight="bold")

    for ax, I_thr in zip(axes, I_thresholds):
        mask = I_arr > I_thr
        if mask.sum() < 200:
            ax.set_title(f"I > {I_thr} (too few)")
            continue
        med, edges, _ = binned_statistic(inv_d[mask], r[mask], statistic="median", bins=bins_d)
        q25, _, _ = binned_statistic(inv_d[mask], r[mask],
                                      statistic=lambda x: np.percentile(x, 25), bins=bins_d)
        q75, _, _ = binned_statistic(inv_d[mask], r[mask],
                                      statistic=lambda x: np.percentile(x, 75), bins=bins_d)
        c = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(c, q25, q75, alpha=0.2, color="#534AB7")
        ax.plot(c, med, "o-", color="#534AB7", lw=2, ms=4)
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("1/d (Å⁻¹)")
        ax.set_title(f"I > {I_thr} (N={mask.sum():,})")
        if ax == axes[0]:
            ax.set_ylabel("R")
    axes[0].set_ylim(0.85, 1.25)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUTDIR / "fig_slit_ratio_by_intensity.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_slit_ratio_by_intensity.png'}")
    plt.close(fig)

    # Fig 3: Distribution of slit ratios
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Slit ratio distribution (slit={w} px)", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.hist(r, bins=100, range=(0.5, 2.0), density=True, color="#534AB7", alpha=0.7)
    ax.axvline(np.median(r), color="red", ls="--", lw=2, label=f"median={np.median(r):.4f}")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("R = radial / tangential")
    ax.set_ylabel("Density")
    ax.set_title("All strong spots")
    ax.legend()

    ax = axes[1]
    # Split by resolution
    for d_lo, d_hi, col, label in [
        (0.9, 1.2, "#534AB7", "d = 0.9–1.2 Å"),
        (1.5, 2.5, "#D85A30", "d = 1.5–2.5 Å"),
        (3.0, 10.0, "#1D9E75", "d = 3–10 Å"),
    ]:
        mask = (d_arr >= d_lo) & (d_arr < d_hi)
        if mask.sum() > 100:
            ax.hist(r[mask], bins=80, range=(0.5, 2.0), density=True,
                    alpha=0.4, color=col, label=f"{label} (med={np.median(r[mask]):.3f})")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("R")
    ax.legend(fontsize=9)
    ax.set_title("By resolution shell")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig_slit_ratio_distribution.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_slit_ratio_distribution.png'}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 65)
    print(f"SLIT ELONGATION SUMMARY (slit half-width = {w} px)")
    print("=" * 65)
    print(f"  N strong spots: {len(r):,}")
    print(f"  Median R: {np.median(r):.4f}  (1.0 = isotropic)")
    d_bins = [0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    print(f"\n  {'d range':>12} {'med R':>7} {'mean R':>7} {'N':>8}")
    for i in range(len(d_bins) - 1):
        m = (d_arr >= d_bins[i]) & (d_arr < d_bins[i + 1])
        if m.sum() < 50: continue
        print(f"  {d_bins[i]:5.1f}–{d_bins[i+1]:4.1f} Å "
              f"{np.median(r[m]):7.4f} {np.mean(r[m]):7.4f} {m.sum():>8,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
