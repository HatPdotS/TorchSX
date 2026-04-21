#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Distribution of Ewald sphere offsets (excitation errors).

What does the empirical distribution of |r| look like?
How does it depend on resolution?
Is it uniform (as assumed in the partiality simulation) or structured?
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

STREAM_DIR = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"


def load_all():
    from torchsx.io._safetensors import load_sfx
    from torchsx.merge._neural_model import compute_excitation_error

    names = ["delay_-45", "delay_-91", "delay_-136",
             "delay_-182", "delay_-227", "delay_-273"]
    all_hkl, all_I, all_cid = [], [], []
    all_astar, all_bstar, all_cstar = [], [], []
    co = 0
    for name in names:
        c = load_sfx(str(STREAM_DIR / f"{name}.sfx"))
        all_hkl.append(c.hkl); all_I.append(c.I)
        all_cid.append(c.crystal_id + co)
        all_astar.append(c.astar); all_bstar.append(c.bstar); all_cstar.append(c.cstar)
        co += c.astar.shape[0]
        print(f"  {name}: {c.hkl.shape[0]:,}")

    hkl = torch.cat(all_hkl)
    I = torch.cat(all_I)
    cid = torch.cat(all_cid)
    astar = torch.cat(all_astar)
    bstar = torch.cat(all_bstar)
    cstar = torch.cat(all_cstar)

    print(f"  Total: {hkl.shape[0]:,} obs")

    exc_err, q_lab = compute_excitation_error(hkl, astar, bstar, cstar, cid, 0.9567)
    q_mag = q_lab.norm(dim=1)
    d = 1.0 / q_mag.clamp(min=1e-6)

    return (exc_err.numpy(), q_lab.numpy(), d.numpy(),
            I.numpy(), hkl.numpy(), cid.numpy())


def main():
    print("Loading data...")
    exc_err, q_lab, d, I, hkl, cid = load_all()
    abs_r = np.abs(exc_err)

    print(f"\n  |r| range: {abs_r.min():.6f} – {abs_r.max():.6f} Å⁻¹")
    print(f"  |r| median: {np.median(abs_r):.6f}")
    print(f"  |r| mean:   {np.mean(abs_r):.6f}")

    inv_d = 1.0 / d

    # ── Fig 1: Overall distribution ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Ewald sphere offset distribution — {len(exc_err):,} observations",
                 fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(exc_err * 1e3, bins=200, range=(-7, 7), density=True,
            color="#534AB7", alpha=0.7, edgecolor="#3B3580", lw=0.3)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Excitation error r (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Density")
    ax.set_title("Signed excitation error")

    ax = axes[0, 1]
    ax.hist(abs_r * 1e3, bins=150, range=(0, 7), density=True,
            color="#D85A30", alpha=0.7, edgecolor="#993C1D", lw=0.3)
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Density")
    ax.set_title(f"|excitation error|  (median = {np.median(abs_r)*1e3:.3f}×10⁻³)")

    # Compare to uniform distribution
    # If reflections are uniformly distributed in offset, p(|r|) = const for |r| < cutoff
    # The actual distribution should reflect the detection probability × offset density

    ax = axes[1, 0]
    # CDF of |r|
    sorted_r = np.sort(abs_r)
    cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    # Subsample for plotting
    idx = np.linspace(0, len(sorted_r) - 1, 5000).astype(int)
    ax.plot(sorted_r[idx] * 1e3, cdf[idx], color="#534AB7", lw=2)
    # Uniform CDF for reference
    r_max = np.percentile(abs_r, 99)
    r_lin = np.linspace(0, r_max, 100)
    ax.plot(r_lin * 1e3, r_lin / r_max, "r--", lw=1.5, label="uniform")
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of |excitation error|")
    ax.legend()

    ax = axes[1, 1]
    # |r| vs resolution scatter (subsample)
    n_sub = min(500_000, len(abs_r))
    idx = np.random.choice(len(abs_r), n_sub, replace=False)
    ax.scatter(inv_d[idx], abs_r[idx] * 1e3, s=0.5, alpha=0.02,
               c="#534AB7", rasterized=True)
    # Binned median
    bins_d = np.linspace(0.05, 1.15, 30)
    med_r, edges, _ = binned_statistic(inv_d, abs_r * 1e3, statistic="median", bins=bins_d)
    q95, _, _ = binned_statistic(inv_d, abs_r * 1e3,
                                  statistic=lambda x: np.percentile(x, 95), bins=bins_d)
    c = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(c, med_r, "o-", color="#D85A30", lw=2, ms=4, label="median")
    ax.plot(c, q95, "s--", color="#D85A30", lw=1.5, ms=3, alpha=0.7, label="95th %ile")
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_title("|r| vs resolution")
    ax.legend()
    ax.set_ylim(0, 7)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTDIR / "fig_ewald_offset_overall.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_ewald_offset_overall.png'}")
    plt.close(fig)

    # ── Fig 2: Per-resolution-shell distributions ──
    d_shells = [(0.9, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0),
                (2.0, 3.0), (3.0, 5.0), (5.0, 10.0)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Excitation error distribution by resolution shell",
                 fontsize=12, fontweight="bold")

    for ax, (d_lo, d_hi) in zip(axes.flat, d_shells):
        mask = (d >= d_lo) & (d < d_hi)
        r_shell = exc_err[mask] * 1e3
        n = mask.sum()

        ax.hist(r_shell, bins=100, range=(-7, 7), density=True,
                color="#534AB7", alpha=0.7, edgecolor="#3B3580", lw=0.3)
        ax.axvline(0, color="gray", ls=":", lw=1)

        std_r = np.std(r_shell)
        ax.set_title(f"d = {d_lo}–{d_hi} Å\nσ = {std_r:.3f}×10⁻³, N = {n:,}",
                     fontsize=10)
        ax.set_xlabel("r (×10⁻³ Å⁻¹)")

    # Last panel: overlay σ(r) vs resolution
    ax = axes.flat[-1]
    d_mids = []
    stds = []
    for d_lo, d_hi in d_shells:
        mask = (d >= d_lo) & (d < d_hi)
        stds.append(np.std(exc_err[mask]) * 1e3)
        d_mids.append(0.5 * (1.0 / d_lo + 1.0 / d_hi))
    ax.plot(d_mids, stds, "o-", color="#D85A30", lw=2, ms=6)
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("σ(r) (×10⁻³ Å⁻¹)")
    ax.set_title("Width of offset distribution")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / "fig_ewald_offset_by_resolution.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_ewald_offset_by_resolution.png'}")
    plt.close(fig)

    # ── Fig 3: Conditioned on intensity ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Ewald offset conditioned on intensity",
                 fontsize=12, fontweight="bold")

    I_bins = [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, 100000)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(I_bins)))

    ax = axes[0]
    for (I_lo, I_hi), col in zip(I_bins, colors):
        mask = (I >= I_lo) & (I < I_hi)
        if mask.sum() < 100: continue
        ax.hist(abs_r[mask] * 1e3, bins=100, range=(0, 7), density=True,
                alpha=0.4, color=col, label=f"I={I_lo}–{I_hi}")
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Density")
    ax.set_title("|r| distribution by intensity")
    ax.legend(fontsize=7)

    ax = axes[1]
    # Median |r| vs intensity
    I_edges = np.logspace(-1, 5, 40)
    I_centers = np.sqrt(I_edges[:-1] * I_edges[1:])
    pos = I > 0
    med_r_I, _, _ = binned_statistic(I[pos], abs_r[pos] * 1e3,
                                      statistic="median", bins=I_edges)
    ax.plot(I_centers, med_r_I, "o-", color="#534AB7", lw=2, ms=3)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("median |r| (×10⁻³ Å⁻¹)")
    ax.set_xscale("log")
    ax.set_title("Median offset vs intensity")

    ax = axes[2]
    # Per-crystal: is there a crystal-level offset pattern?
    n_crystals = cid.max() + 1
    crystal_mean_r = np.zeros(n_crystals)
    crystal_std_r = np.zeros(n_crystals)
    crystal_n = np.zeros(n_crystals)
    for ci in range(n_crystals):
        mask = cid == ci
        if mask.sum() < 5: continue
        crystal_mean_r[ci] = np.mean(abs_r[mask])
        crystal_std_r[ci] = np.std(exc_err[mask])
        crystal_n[ci] = mask.sum()

    valid_c = crystal_n >= 10
    ax.hist(crystal_mean_r[valid_c] * 1e3, bins=80, range=(0, 5),
            density=True, color="#1D9E75", alpha=0.7)
    ax.set_xlabel("Per-crystal mean |r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Density")
    ax.set_title(f"Crystal-level offset (N={valid_c.sum():,} crystals)")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig_ewald_offset_conditioned.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_ewald_offset_conditioned.png'}")
    plt.close(fig)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("EWALD OFFSET DISTRIBUTION SUMMARY")
    print("=" * 65)
    print(f"  N observations:  {len(exc_err):,}")
    print(f"  Signed r:   mean = {np.mean(exc_err)*1e3:+.4f}×10⁻³ Å⁻¹")
    print(f"              std  = {np.std(exc_err)*1e3:.4f}×10⁻³")
    print(f"  |r|:        median = {np.median(abs_r)*1e3:.4f}×10⁻³")
    print(f"              mean   = {np.mean(abs_r)*1e3:.4f}×10⁻³")
    print(f"              95th   = {np.percentile(abs_r, 95)*1e3:.4f}×10⁻³")
    print(f"\n  By resolution shell:")
    print(f"  {'d range':>12} {'σ(r)':>10} {'med |r|':>10} {'95th |r|':>10} {'N':>9}")
    for d_lo, d_hi in d_shells:
        mask = (d >= d_lo) & (d < d_hi)
        print(f"  {d_lo:5.1f}–{d_hi:4.1f} Å "
              f"{np.std(exc_err[mask])*1e3:10.4f} "
              f"{np.median(abs_r[mask])*1e3:10.4f} "
              f"{np.percentile(abs_r[mask], 95)*1e3:10.4f} "
              f"{mask.sum():>9,}")
    print(f"\n  By intensity:")
    for I_lo, I_hi in I_bins:
        mask = (I >= I_lo) & (I < I_hi)
        if mask.sum() < 10: continue
        print(f"  I={I_lo:>5}–{I_hi:<5}: "
              f"med |r| = {np.median(abs_r[mask])*1e3:.4f}×10⁻³  "
              f"N = {mask.sum():>9,}")
    print("=" * 65)


if __name__ == "__main__":
    np.random.seed(42)
    main()
