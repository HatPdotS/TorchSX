#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
What geometric descriptors predict where ρ(I, |r|) breaks down?

For each HKL we have a Spearman ρ value.  We also know geometric
properties of its observations: detector position, panel, q-vector
direction, excitation error spread, mosaicity proxy, etc.

If ρ ≈ 0 or ρ > 0 for some reflections, the calculated excitation
error is not predicting intensity variation → something is wrong
with the geometry for those observations.

Descriptors per HKL:
  - d-spacing (resolution)
  - mean |excitation_error|  (how far from Ewald sphere on average)
  - std |excitation_error|   (spread in offsets)
  - mean detector position (fs, ss)
  - dominant panel_id
  - number of distinct panels
  - mean profile_rad (mosaicity proxy of contributing crystals)
  - q_lab direction (azimuthal angle, polar angle)
  - multiplicity
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
import os, sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)


# ── Load & compute ────────────────────────────────────────────────

def load_data():
    from torchsx.io._safetensors import load_sfx
    from torchsx.merge._neural_model import compute_excitation_error
    from torchsx.merge._hkl import compute_hkl_index

    sfx_dir = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"
    files = sorted(sfx_dir.glob("delay_*.sfx"))
    # Use largest stream
    cores = []
    for f in files:
        c = load_sfx(str(f))
        cores.append(c)
    core = max(cores, key=lambda c: c.hkl.shape[0])
    print(f"Loaded: {core.hkl.shape[0]:,} obs, {core.astar.shape[0]:,} crystals")

    wavelength = 12398.42 / core.beam_energy.median().item()
    exc_err, q_lab = compute_excitation_error(
        core.hkl, core.astar, core.bstar, core.cstar,
        core.crystal_id, wavelength,
    )

    hkl_idx = compute_hkl_index(core.hkl)

    return core, exc_err.numpy(), q_lab.numpy(), hkl_idx


def compute_per_hkl_descriptors(core, exc_err, q_lab, hkl_idx,
                                 I_threshold=50, min_strong=10):
    """
    For each qualifying HKL, compute:
      - Spearman ρ(I, |r|)
      - Geometric descriptors
    """
    I = core.I.numpy()
    abs_r = np.abs(exc_err)
    hkl_id = hkl_idx.hkl_id.numpy()
    unique_hkl = hkl_idx.unique_hkl.numpy()
    n_unique = hkl_idx.n_unique

    fs = core.fs.numpy()
    ss = core.ss.numpy()
    panel = core.panel_id.numpy()
    cid = core.crystal_id.numpy()
    profile_rad = core.profile_rad.numpy()

    # q_lab spherical coordinates
    q_mag = np.sqrt(np.sum(q_lab**2, axis=1))
    q_theta = np.arccos(np.clip(q_lab[:, 2] / np.maximum(q_mag, 1e-10), -1, 1))  # polar
    q_phi = np.arctan2(q_lab[:, 1], q_lab[:, 0])  # azimuthal

    # Pre-sort by hkl_id
    order = np.argsort(hkl_id)
    boundaries = np.searchsorted(hkl_id[order], np.arange(n_unique + 1))

    # Reorder all arrays
    I_s = I[order]
    r_s = abs_r[order]
    fs_s = fs[order]
    ss_s = ss[order]
    panel_s = panel[order]
    cid_s = cid[order]
    q_mag_s = q_mag[order]
    q_theta_s = q_theta[order]
    q_phi_s = q_phi[order]

    results = []
    for uid in range(n_unique):
        lo, hi = boundaries[uid], boundaries[uid + 1]
        if hi - lo < min_strong:
            continue

        I_h = I_s[lo:hi]
        r_h = r_s[lo:hi]

        # Apply intensity threshold
        keep = I_h > I_threshold
        if keep.sum() < min_strong:
            continue

        I_filt = I_h[keep]
        r_filt = r_h[keep]

        if np.ptp(I_filt) < 1e-10 or np.ptp(r_filt) < 1e-10:
            continue

        rho, pval = spearmanr(I_filt, r_filt)

        # Descriptors from ALL observations (not just filtered)
        fs_h = fs_s[lo:hi]
        ss_h = ss_s[lo:hi]
        panel_h = panel_s[lo:hi]
        cid_h = cid_s[lo:hi]

        # Descriptors from filtered observations
        fs_filt = fs_h[keep]
        ss_filt = ss_h[keep]
        panel_filt = panel_h[keep]

        h, k, l = unique_hkl[uid]
        d = 1.0 / np.median(q_mag_s[lo:hi]) if np.median(q_mag_s[lo:hi]) > 0 else 99

        results.append({
            "uid": uid,
            "h": h, "k": k, "l": l,
            "rho": rho,
            "pval": pval,
            "mult": int(keep.sum()),
            "d": d,
            # Excitation error stats
            "mean_abs_r": np.mean(r_filt),
            "median_abs_r": np.median(r_filt),
            "std_abs_r": np.std(r_filt),
            "min_abs_r": np.min(r_filt),
            # Detector position
            "mean_fs": np.mean(fs_filt),
            "mean_ss": np.mean(ss_filt),
            "std_fs": np.std(fs_filt),
            "std_ss": np.std(ss_filt),
            # Panel distribution
            "n_panels": len(np.unique(panel_filt)),
            "dominant_panel": int(np.bincount(panel_filt.astype(int)).argmax()),
            # Mosaicity proxy
            "mean_profile_rad": np.mean(profile_rad[np.unique(cid_h)]),
            # q-vector direction
            "q_theta": np.median(q_theta_s[lo:hi]),
            "q_phi": np.median(q_phi_s[lo:hi]),
            # Intensity stats
            "mean_I": np.mean(I_filt),
            "std_I": np.std(I_filt),
            "cv_I": np.std(I_filt) / np.mean(I_filt) if np.mean(I_filt) > 0 else 0,
        })

    print(f"  {len(results)} HKLs with descriptors")
    return results


# ── Figures ───────────────────────────────────────────────────────

def make_figures(desc):
    """Scatter ρ against every geometric descriptor."""
    rho = np.array([d["rho"] for d in desc])

    descriptors = [
        ("d", "d-spacing (Å)", False),
        ("mean_abs_r", "mean |excitation error| (Å⁻¹)", False),
        ("std_abs_r", "std |excitation error| (Å⁻¹)", False),
        ("min_abs_r", "min |excitation error| (Å⁻¹)", False),
        ("mean_fs", "mean fs (px)", False),
        ("mean_ss", "mean ss (px)", False),
        ("std_fs", "σ(fs) across crystals (px)", False),
        ("std_ss", "σ(ss) across crystals (px)", False),
        ("n_panels", "number of distinct panels", False),
        ("dominant_panel", "dominant panel_id", True),
        ("mean_profile_rad", "mean profile_rad (mosaicity)", False),
        ("q_theta", "q polar angle θ (rad)", False),
        ("q_phi", "q azimuthal angle φ (rad)", False),
        ("mult", "multiplicity (strong obs)", False),
        ("mean_I", "mean I (strong obs)", False),
        ("cv_I", "CV(I) = σ(I)/⟨I⟩", False),
    ]

    # ── Fig 1: Big scatter matrix ──
    n_desc = len(descriptors)
    ncols = 4
    nrows = (n_desc + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    fig.suptitle(
        f"ρ(I, |r|) vs geometric descriptors — {len(desc)} strong HKLs (I > 50, mult ≥ 10)",
        fontsize=13, fontweight="bold",
    )

    for ax, (key, label, is_discrete) in zip(axes.flat, descriptors):
        vals = np.array([d[key] for d in desc])

        if is_discrete:
            # Box-plot by category
            categories = np.unique(vals)
            if len(categories) > 32:
                categories = categories[:32]
            data = [rho[vals == c] for c in categories if (vals == c).sum() >= 5]
            cats_filt = [c for c in categories if (vals == c).sum() >= 5]
            if len(data) > 0:
                bp = ax.boxplot(data, positions=range(len(data)), widths=0.6,
                                patch_artist=True, showfliers=False)
                for patch in bp["boxes"]:
                    patch.set_facecolor("#534AB7")
                    patch.set_alpha(0.5)
                ax.set_xticks(range(len(cats_filt)))
                ax.set_xticklabels([str(int(c)) for c in cats_filt], fontsize=6, rotation=90)
        else:
            ax.scatter(vals, rho, s=2, alpha=0.15, c="#534AB7", rasterized=True)
            # Binned mean
            n_bins = 15
            finite = np.isfinite(vals)
            if finite.sum() > 50:
                bins = np.percentile(vals[finite], np.linspace(0, 100, n_bins + 1))
                bins = np.unique(bins)
                if len(bins) > 2:
                    bin_idx = np.digitize(vals, bins) - 1
                    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
                    bin_means = [np.mean(rho[bin_idx == b]) for b in range(len(bins) - 1)
                                 if (bin_idx == b).sum() >= 5]
                    bin_centers = [np.mean(vals[bin_idx == b]) for b in range(len(bins) - 1)
                                   if (bin_idx == b).sum() >= 5]
                    ax.plot(bin_centers, bin_means, "o-", color="#D85A30", lw=2, ms=5, zorder=5)

            # Spearman of the descriptor vs ρ
            finite_both = finite & np.isfinite(rho)
            if finite_both.sum() > 20:
                rho_desc, p_desc = spearmanr(vals[finite_both], rho[finite_both])
                ax.set_title(f"{label}\nρ_desc = {rho_desc:+.3f} (p={p_desc:.1e})", fontsize=9)
            else:
                ax.set_title(label, fontsize=9)

        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_ylabel("ρ(I, |r|)", fontsize=8)

    # Hide unused axes
    for ax in axes.flat[n_desc:]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / "fig_geo_descriptors.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_geo_descriptors.png'}")
    plt.close(fig)

    # ── Fig 2: Panel-by-panel ρ distribution ──
    panels = np.array([d["dominant_panel"] for d in desc])
    unique_panels = np.unique(panels)

    fig, ax = plt.subplots(figsize=(12, 5))
    panel_medians = []
    panel_counts = []
    for p in unique_panels:
        mask = panels == p
        if mask.sum() < 5:
            panel_medians.append(np.nan)
            panel_counts.append(0)
            continue
        panel_medians.append(np.median(rho[mask]))
        panel_counts.append(mask.sum())

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(unique_panels)))
    # Sort by median ρ
    sort_idx = np.argsort(panel_medians)
    ax.bar(range(len(unique_panels)),
           [panel_medians[i] for i in sort_idx],
           color=[colors[i] for i in sort_idx], edgecolor="gray", lw=0.5)
    ax.set_xticks(range(len(unique_panels)))
    ax.set_xticklabels([str(int(unique_panels[i])) for i in sort_idx], fontsize=7, rotation=90)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axhline(np.median(rho), color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"overall median = {np.median(rho):.3f}")
    ax.set_xlabel("Panel ID (sorted by median ρ)")
    ax.set_ylabel("Median ρ(I, |r|)")
    ax.set_title("Per-panel median Spearman ρ — are some panels systematically worse?",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_panel_rho.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_panel_rho.png'}")
    plt.close(fig)

    # ── Fig 3: Detector map of ρ ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Detector-space maps of ρ(I, |r|)", fontsize=12, fontweight="bold")

    mean_fs_arr = np.array([d["mean_fs"] for d in desc])
    mean_ss_arr = np.array([d["mean_ss"] for d in desc])

    # Left: scatter colored by ρ
    ax = axes[0]
    vmin, vmax = -0.5, 0.1
    sc = ax.scatter(mean_fs_arr, mean_ss_arr, c=rho, s=3, alpha=0.3,
                    cmap="RdBu_r", vmin=vmin, vmax=vmax, rasterized=True)
    fig.colorbar(sc, ax=ax, label="ρ(I, |r|)")
    ax.set_xlabel("mean fs (px)")
    ax.set_ylabel("mean ss (px)")
    ax.set_title("ρ in detector space")

    # Right: binned 2D map
    ax = axes[1]
    n_bins_2d = 30
    finite = np.isfinite(mean_fs_arr) & np.isfinite(mean_ss_arr)
    from scipy.stats import binned_statistic_2d
    stat, xe, ye, _ = binned_statistic_2d(
        mean_fs_arr[finite], mean_ss_arr[finite], rho[finite],
        statistic="median", bins=n_bins_2d,
    )
    im = ax.pcolormesh(xe, ye, stat.T, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label="median ρ")
    ax.set_xlabel("fs (px)")
    ax.set_ylabel("ss (px)")
    ax.set_title("Binned median ρ")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / "fig_detector_map_rho.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_detector_map_rho.png'}")
    plt.close(fig)

    # ── Fig 4: q-space map ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Reciprocal-space maps of ρ(I, |r|)", fontsize=12, fontweight="bold")

    q_theta_arr = np.array([d["q_theta"] for d in desc])
    q_phi_arr = np.array([d["q_phi"] for d in desc])
    d_arr = np.array([d["d"] for d in desc])

    # Left: ρ vs (θ, φ)
    ax = axes[0]
    sc = ax.scatter(np.degrees(q_phi_arr), np.degrees(q_theta_arr),
                    c=rho, s=3, alpha=0.3, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                    rasterized=True)
    fig.colorbar(sc, ax=ax, label="ρ(I, |r|)")
    ax.set_xlabel("q azimuthal φ (deg)")
    ax.set_ylabel("q polar θ (deg)")
    ax.set_title("ρ in q-direction space")

    # Right: ρ vs 1/d² colored by ρ
    ax = axes[1]
    inv_d2 = 1.0 / d_arr**2
    sc = ax.scatter(inv_d2, rho, s=3, alpha=0.15, c="#534AB7", rasterized=True)
    # Binned trend
    bins_d = np.percentile(inv_d2, np.linspace(0, 100, 16))
    bins_d = np.unique(bins_d)
    if len(bins_d) > 2:
        bid = np.clip(np.digitize(inv_d2, bins_d) - 1, 0, len(bins_d) - 2)
        bm = [np.mean(rho[bid == b]) for b in range(len(bins_d) - 1) if (bid == b).sum() >= 5]
        bc = [np.mean(inv_d2[bid == b]) for b in range(len(bins_d) - 1) if (bid == b).sum() >= 5]
        ax.plot(bc, bm, "o-", color="#D85A30", lw=2, ms=5)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("1/d² (Å⁻²)")
    ax.set_ylabel("ρ(I, |r|)")
    ax.set_title("ρ vs resolution")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / "fig_qspace_rho.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_qspace_rho.png'}")
    plt.close(fig)

    # ── Print descriptor correlations with ρ ──
    print("\n" + "=" * 70)
    print("DESCRIPTOR CORRELATIONS WITH ρ(I, |r|)")
    print("=" * 70)
    print(f"{'descriptor':>25} {'ρ_desc':>8} {'p-value':>10}")
    print("─" * 50)

    for key, label, is_discrete in [
        ("d", "d-spacing", False),
        ("mean_abs_r", "mean |r|", False),
        ("std_abs_r", "std |r|", False),
        ("min_abs_r", "min |r|", False),
        ("mean_fs", "mean fs", False),
        ("mean_ss", "mean ss", False),
        ("std_fs", "σ(fs)", False),
        ("std_ss", "σ(ss)", False),
        ("n_panels", "N panels", False),
        ("mean_profile_rad", "profile_rad", False),
        ("q_theta", "q θ", False),
        ("q_phi", "q φ", False),
        ("mult", "multiplicity", False),
        ("mean_I", "mean I", False),
        ("cv_I", "CV(I)", False),
    ]:
        vals = np.array([d[key] for d in desc])
        fin = np.isfinite(vals) & np.isfinite(rho)
        if fin.sum() > 20:
            r, p = spearmanr(vals[fin], rho[fin])
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{label:>25} {r:>+8.4f} {p:>10.2e} {marker}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("Loading data...")
    core, exc_err, q_lab, hkl_idx = load_data()

    print("Computing per-HKL descriptors...")
    desc = compute_per_hkl_descriptors(core, exc_err, q_lab, hkl_idx,
                                        I_threshold=50, min_strong=10)

    print("Generating figures...")
    make_figures(desc)

    print(f"\nDone. Figures in {OUTDIR}/")
