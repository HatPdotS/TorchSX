#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
log(I_obs / I_merged) vs Ewald sphere offset.

For each observation j of reflection h:
   c_j = log(I_j / <I_h>)

This log-correction factor absorbs partiality, crystal scale, and noise.
If the excitation error r_j is accurate, c_j should negatively correlate
with |r_j| — reflections far from the Ewald sphere are weaker.

We normalize per-crystal to remove crystal-level scale:
   c_j = log(I_j) - <log(I)>_crystal - (log(<I_h>) - <log(<I_h>)>_crystal)

Simpler: just use log(I_j / I_merged_h).

Then bin by |r| and look at the trend. The slope/shape of <c> vs |r|
is the empirical partiality curve. The scatter around it quantifies
how much of the intensity variation is NOT explained by the offset.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binned_statistic, spearmanr
import os, sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

STREAM_DIR = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"


def main():
    from torchsx.io._safetensors import load_sfx
    from torchsx.merge._neural_model import compute_excitation_error
    from torchsx.merge._hkl import compute_hkl_index

    # Use one large stream for speed
    print("Loading stream...")
    core = load_sfx(str(STREAM_DIR / "delay_-91.sfx"))
    N = core.hkl.shape[0]
    print(f"  {N:,} obs, {core.astar.shape[0]:,} crystals")

    # Excitation error
    print("Computing excitation error...")
    wavelength = 0.9567
    exc_err, q_lab = compute_excitation_error(
        core.hkl, core.astar, core.bstar, core.cstar,
        core.crystal_id, wavelength,
    )
    r = exc_err.numpy()
    abs_r = np.abs(r)
    q_mag = q_lab.norm(dim=1).numpy()
    d = 1.0 / np.clip(q_mag, 1e-6, None)

    # HKL index → merged intensity
    print("Computing merged intensities...")
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id = hkl_idx.hkl_id.numpy()
    I = core.I.numpy()
    cid = core.crystal_id.numpy()

    # Merged I per HKL (mean of positive observations)
    n_unique = hkl_idx.n_unique
    I_sum = np.zeros(n_unique)
    I_count = np.zeros(n_unique)
    pos = I > 0
    np.add.at(I_sum, hkl_id[pos], I[pos])
    np.add.at(I_count, hkl_id[pos], 1)
    I_merged = I_sum / np.clip(I_count, 1, None)

    # Per-observation log correction — strong reflections only
    I_merged_obs = I_merged[hkl_id]  # (N,) merged I for each obs's HKL
    valid = (I > 50) & (I_merged_obs > 50) & (I_count[hkl_id] >= 5)
    log_corr = np.full(N, np.nan)
    log_corr[valid] = np.log(I[valid] / I_merged_obs[valid])

    print(f"  {valid.sum():,} valid observations ({valid.mean()*100:.1f}%)")
    print(f"  log(I/I_merged): mean={np.nanmean(log_corr[valid]):.4f}, "
          f"std={np.nanstd(log_corr[valid]):.4f}")

    # ── Fig 1: log(I/I_merged) vs |r| ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"log(I_obs / I_merged) vs Ewald offset — {valid.sum():,} observations\n"
        f"Negative trend = partiality signal in r",
        fontsize=12, fontweight="bold",
    )

    lc = log_corr[valid]
    ar = abs_r[valid]
    d_v = d[valid]
    I_v = I[valid]

    # Top-left: 2D histogram
    ax = axes[0, 0]
    h, xe, ye = np.histogram2d(ar * 1e3, lc, bins=[60, 80],
                                range=[[0, 7], [-5, 5]])
    ax.pcolormesh(xe, ye, h.T, cmap="magma", vmin=0)
    # Binned median
    bins_r = np.linspace(0, 0.007, 40)
    med_lc, edges, _ = binned_statistic(ar, lc, statistic="median", bins=bins_r)
    c = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(c * 1e3, med_lc, "c-", lw=2.5, label="median")
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("log(I/I_merged)")
    ax.set_title("2D density + median trend")
    ax.legend()

    # Top-right: binned statistics
    ax = axes[0, 1]
    mean_lc, _, _ = binned_statistic(ar, lc, statistic="mean", bins=bins_r)
    std_lc, _, _ = binned_statistic(ar, lc, statistic="std", bins=bins_r)
    ax.plot(c * 1e3, med_lc, "o-", color="#534AB7", lw=2, ms=4, label="median")
    ax.plot(c * 1e3, mean_lc, "s--", color="#D85A30", lw=2, ms=4, label="mean")
    ax.fill_between(c * 1e3, med_lc - std_lc, med_lc + std_lc,
                     alpha=0.15, color="#534AB7")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("log(I/I_merged)")
    ax.set_title("Binned mean/median ± std")
    ax.legend()

    # Bottom-left: by resolution shell
    ax = axes[1, 0]
    d_shells = [(0.9, 1.2, "#534AB7"), (1.2, 2.0, "#D85A30"),
                (2.0, 4.0, "#1D9E75"), (4.0, 15.0, "#B5338A")]
    for d_lo, d_hi, col in d_shells:
        mask = (d_v >= d_lo) & (d_v < d_hi)
        if mask.sum() < 1000: continue
        med, edges, _ = binned_statistic(ar[mask], lc[mask], statistic="median", bins=bins_r)
        c2 = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(c2 * 1e3, med, "o-", color=col, lw=2, ms=3,
                label=f"d={d_lo}–{d_hi}")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("median log(I/I_merged)")
    ax.set_title("By resolution shell")
    ax.legend(fontsize=9)

    # Bottom-right: by intensity
    ax = axes[1, 1]
    I_shells = [(10, 50, "#534AB7"), (50, 200, "#D85A30"),
                (200, 1000, "#1D9E75"), (1000, 100000, "#B5338A")]
    for I_lo, I_hi, col in I_shells:
        mask = (I_v >= I_lo) & (I_v < I_hi)
        if mask.sum() < 500: continue
        med, edges, _ = binned_statistic(ar[mask], lc[mask], statistic="median", bins=bins_r)
        c2 = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(c2 * 1e3, med, "o-", color=col, lw=2, ms=3,
                label=f"I={I_lo}–{I_hi}")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("median log(I/I_merged)")
    ax.set_title("By intensity range")
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUTDIR / "fig_logcorr_vs_ewald.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_logcorr_vs_ewald.png'}")
    plt.close(fig)

    # ── Fig 2: Per-crystal analysis ──
    print("\nPer-crystal correlation analysis...")
    n_crystals = cid.max() + 1
    crystal_rho = []
    crystal_n = []
    crystal_slope = []

    for ci in range(n_crystals):
        mask = (cid == ci) & valid
        n = mask.sum()
        if n < 20:
            continue
        lc_c = log_corr[mask]
        ar_c = abs_r[mask]
        if np.std(lc_c) < 1e-10 or np.std(ar_c) < 1e-10:
            continue
        rho, _ = spearmanr(lc_c, ar_c)
        crystal_rho.append(rho)
        crystal_n.append(n)
        # Simple linear slope: d(log_corr)/d(|r|)
        slope = np.polyfit(ar_c, lc_c, 1)[0]
        crystal_slope.append(slope)

    crystal_rho = np.array(crystal_rho)
    crystal_n = np.array(crystal_n)
    crystal_slope = np.array(crystal_slope)
    print(f"  {len(crystal_rho):,} crystals with ≥20 valid obs")
    print(f"  Per-crystal ρ(log_corr, |r|): "
          f"mean={np.mean(crystal_rho):.4f}, "
          f"median={np.median(crystal_rho):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Per-crystal ρ(log(I/I_merged), |r|) — {len(crystal_rho):,} crystals",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.hist(crystal_rho, bins=80, range=(-1, 0.5), density=True,
            color="#534AB7", alpha=0.7, edgecolor="#3B3580", lw=0.5)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.axvline(np.median(crystal_rho), color="red", ls="--", lw=2,
               label=f"median={np.median(crystal_rho):.3f}")
    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of per-crystal ρ")
    ax.legend()

    ax = axes[1]
    ax.hist(crystal_slope, bins=80, range=(-1000, 200), density=True,
            color="#D85A30", alpha=0.7)
    ax.axvline(np.median(crystal_slope), color="red", ls="--", lw=2,
               label=f"median={np.median(crystal_slope):.0f}")
    ax.set_xlabel("Slope d(log_corr)/d(|r|) (Å)")
    ax.set_ylabel("Density")
    ax.set_title("Per-crystal slope")
    ax.legend()

    ax = axes[2]
    ax.scatter(crystal_n, crystal_rho, s=3, alpha=0.2, color="#534AB7", rasterized=True)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("N observations per crystal")
    ax.set_ylabel("ρ(log_corr, |r|)")
    ax.set_title("ρ vs crystal size")
    ax.set_xscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUTDIR / "fig_logcorr_per_crystal.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_logcorr_per_crystal.png'}")
    plt.close(fig)

    # ── Fig 3: Variance decomposition ──
    # How much of var(log_corr) is explained by |r|?
    print("\nVariance decomposition...")
    total_var = np.var(lc)
    # Explained: variance of the binned median (systematic trend)
    bin_id = np.clip(np.digitize(ar, bins_r) - 1, 0, len(bins_r) - 2)
    predicted = np.array([np.median(lc[bin_id == b]) if (bin_id == b).sum() > 0
                          else 0 for b in range(len(bins_r) - 1)])
    lc_predicted = predicted[bin_id]
    explained_var = np.var(lc_predicted)
    residual_var = np.var(lc - lc_predicted)

    print(f"  Total var(log_corr):         {total_var:.4f}")
    print(f"  Explained by |r| bins:       {explained_var:.4f} ({explained_var/total_var*100:.1f}%)")
    print(f"  Residual (unexplained):      {residual_var:.4f} ({residual_var/total_var*100:.1f}%)")
    print(f"  Residual σ:                  {np.sqrt(residual_var):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Variance decomposition of log(I/I_merged)", fontsize=12, fontweight="bold")

    ax = axes[0]
    labels = [f"|r| trend\n{explained_var/total_var*100:.1f}%",
              f"Residual\n{residual_var/total_var*100:.1f}%"]
    sizes = [explained_var, residual_var]
    ax.pie(sizes, labels=labels, colors=["#534AB7", "#DDDDDD"],
           autopct='', startangle=90, textprops={"fontsize": 11})
    ax.set_title(f"Total var = {total_var:.3f}")

    ax = axes[1]
    residual = lc - lc_predicted
    ax.hist(residual, bins=100, range=(-5, 5), density=True,
            color="#D85A30", alpha=0.7, edgecolor="#993C1D", lw=0.3)
    from scipy.stats import norm
    x = np.linspace(-5, 5, 200)
    ax.plot(x, norm.pdf(x, 0, np.std(residual)), "k--", lw=1.5,
            label=f"Gaussian σ={np.std(residual):.3f}")
    ax.set_xlabel("Residual log(I/I_merged) - trend(|r|)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig_logcorr_variance_decomp.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_logcorr_variance_decomp.png'}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Global ρ(log_corr, |r|): {spearmanr(lc, ar)[0]:.4f}")
    print(f"  Per-crystal median ρ:    {np.median(crystal_rho):.4f}")
    print(f"  Var explained by |r|:    {explained_var/total_var*100:.1f}%")
    print(f"  Residual σ:              {np.sqrt(residual_var):.3f}")
    print(f"  (cf. observed σ_LN = 1.56)")
    print("=" * 65)


if __name__ == "__main__":
    np.random.seed(42)
    main()
