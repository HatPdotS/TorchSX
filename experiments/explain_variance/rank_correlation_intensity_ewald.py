#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Per-reflection rank correlation between intensity and Ewald sphere offset.

For each unique HKL with sufficient multiplicity, compute the Spearman
rank correlation between I_obs and |excitation_error|.

If partiality matters: observations closer to the Ewald sphere (small |r|)
should have higher intensity → expect negative ρ(I, |r|).

If partiality is negligible: no correlation → ρ ≈ 0.

v2: Filter out weak observations (Poisson-dominated) and require a
    minimum number of strong observations per HKL before computing ρ.
    Scan over intensity thresholds to show the signal emerging.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm
from pathlib import Path
import os, sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────

def load_all_streams():
    """Load and concatenate all delay streams."""
    from torchsx.io._safetensors import load_sfx

    sfx_dir = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"
    files = sorted(sfx_dir.glob("delay_*.sfx"))
    if not files:
        # Fall back to single file
        alt = Path(__file__).resolve().parent.parent / "merged_368.sfx"
        if alt.exists():
            print(f"Loading {alt} ...")
            return load_sfx(str(alt))
        raise FileNotFoundError("No .sfx data found")

    print(f"Found {len(files)} stream files")
    cores = []
    for f in files:
        print(f"  Loading {f.name} ...", end=" ", flush=True)
        c = load_sfx(str(f))
        print(f"{c.hkl.shape[0]} obs, {c.astar.shape[0]} crystals")
        cores.append(c)

    # Use first file as base (they all have same structure)
    return cores[0] if len(cores) == 1 else _concat_cores(cores)


def _concat_cores(cores):
    """Concatenate multiple _SFXStreamCore objects."""
    # Simple approach: just use the first one for now
    # For a proper concat we'd need to offset crystal_id and image_id
    # But for this analysis, using one stream is fine
    # Let's use the largest one or just the first
    biggest = max(cores, key=lambda c: c.hkl.shape[0])
    print(f"Using largest stream: {biggest.hkl.shape[0]} observations")
    return biggest


# ── Compute excitation error ──────────────────────────────────────

def compute_excitation_error(core, wavelength=0.9567):
    """Compute Ewald sphere offset for all observations."""
    from torchsx.merge._neural_model import compute_excitation_error as _compute
    exc_err, q_lab = _compute(
        core.hkl, core.astar, core.bstar, core.cstar,
        core.crystal_id, wavelength,
    )
    return exc_err.numpy(), q_lab.numpy()


# ── Per-HKL rank correlation ─────────────────────────────────────

def pregroup_by_hkl(I, abs_exc_err, hkl_id):
    """
    Pre-sort observations by HKL into a dict of arrays.
    Call once, then use fast_rank_correlation() for each threshold.
    """
    n_unique = hkl_id.max() + 1
    # argsort by hkl_id for contiguous access
    order = np.argsort(hkl_id)
    I_sorted = I[order]
    r_sorted = abs_exc_err[order]
    id_sorted = hkl_id[order]

    # Find group boundaries
    boundaries = np.searchsorted(id_sorted, np.arange(n_unique + 1))
    return I_sorted, r_sorted, boundaries


def per_hkl_rank_correlation(I, abs_exc_err, hkl_id, min_mult=10,
                              I_threshold=0.0, min_strong=0,
                              _pregroup=None):
    """
    Spearman ρ(I, |r|) for each unique HKL.

    Parameters
    ----------
    I_threshold : float — discard observations with I ≤ this value
    min_strong  : int   — require at least this many observations above
                          threshold per HKL (0 = use min_mult on filtered set)
    _pregroup   : tuple — (I_sorted, r_sorted, boundaries) from pregroup_by_hkl

    Returns arrays: rho, pval, multiplicity, hkl_indices (of qualifying HKLs)
    """
    if _pregroup is not None:
        I_sorted, r_sorted, boundaries = _pregroup
        n_unique = len(boundaries) - 1
    else:
        I_sorted, r_sorted, boundaries = pregroup_by_hkl(I, abs_exc_err, hkl_id)
        n_unique = len(boundaries) - 1

    rhos = []
    pvals = []
    mults = []
    hkl_ids_out = []

    min_n = max(min_mult, min_strong)

    for uid in range(n_unique):
        lo, hi = boundaries[uid], boundaries[uid + 1]
        if hi - lo < min_n:
            continue

        I_h = I_sorted[lo:hi]
        r_h = r_sorted[lo:hi]

        # Apply intensity threshold
        if I_threshold > 0:
            keep = I_h > I_threshold
            if keep.sum() < min_n:
                continue
            I_h = I_h[keep]
            r_h = r_h[keep]

        n = len(I_h)

        # Skip if no variance in either
        if np.ptp(I_h) < 1e-10 or np.ptp(r_h) < 1e-10:
            continue

        rho, pval = spearmanr(I_h, r_h)
        rhos.append(rho)
        pvals.append(pval)
        mults.append(n)
        hkl_ids_out.append(uid)

    return (np.array(rhos), np.array(pvals),
            np.array(mults), np.array(hkl_ids_out))


# ── Resolution binned analysis ────────────────────────────────────

def resolution_binned_correlation(I, abs_exc_err, hkl_id, d_spacing,
                                   n_bins=10, min_mult=10,
                                   I_threshold=0.0, min_strong=0,
                                   _pregroup=None):
    """
    Compute per-HKL rank correlations, then bin by resolution.
    """
    rhos, pvals, mults, hkl_ids = per_hkl_rank_correlation(
        I, abs_exc_err, hkl_id, min_mult,
        I_threshold=I_threshold, min_strong=min_strong,
        _pregroup=_pregroup,
    )

    # Get d-spacing for each qualifying HKL (use median d of its observations)
    d_per_hkl = np.array([np.median(d_spacing[hkl_id == uid]) for uid in hkl_ids])

    # Bin by resolution (equal-count bins)
    sort_idx = np.argsort(d_per_hkl)
    bin_size = len(sort_idx) // n_bins

    bin_results = []
    for b in range(n_bins):
        lo = b * bin_size
        hi = (b + 1) * bin_size if b < n_bins - 1 else len(sort_idx)
        idx = sort_idx[lo:hi]
        bin_results.append({
            "d_lo": d_per_hkl[idx].min(),
            "d_hi": d_per_hkl[idx].max(),
            "d_mean": d_per_hkl[idx].mean(),
            "mean_rho": rhos[idx].mean(),
            "median_rho": np.median(rhos[idx]),
            "std_rho": rhos[idx].std(),
            "n_hkl": len(idx),
            "mean_mult": mults[idx].mean(),
            "frac_negative": (rhos[idx] < 0).mean(),
            "frac_sig_neg": ((rhos[idx] < 0) & (pvals[idx] < 0.05)).mean(),
        })

    return rhos, pvals, mults, hkl_ids, d_per_hkl, bin_results


# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════

def make_figures(I, abs_exc_err, exc_err, hkl_id, d_spacing, unique_hkl):
    """Generate all analysis figures."""

    # Pre-sort once for fast per-threshold scanning
    print("\nPre-grouping observations by HKL...")
    pg = pregroup_by_hkl(I, abs_exc_err, hkl_id)
    print(f"  {len(pg[2])-1} HKL groups prepared")

    # ── Fig 1: Threshold scan — how ρ changes as we remove weak obs ──
    print("\nScanning intensity thresholds...")
    I_thresholds = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000]
    min_strong_obs = 10  # require at least this many above-threshold obs per HKL

    scan_results = []
    for thr in I_thresholds:
        above = (I > thr).sum()
        rhos_t, pvals_t, mults_t, _ = per_hkl_rank_correlation(
            I, abs_exc_err, hkl_id, min_mult=min_strong_obs,
            I_threshold=thr, min_strong=min_strong_obs,
            _pregroup=pg,
        )
        if len(rhos_t) < 10:
            scan_results.append(None)
            print(f"  I > {thr:>5}: {above:>8,} obs → too few HKLs")
            continue

        r = {
            "thr": thr,
            "n_obs_above": int(above),
            "n_hkl": len(rhos_t),
            "mean_rho": np.mean(rhos_t),
            "median_rho": np.median(rhos_t),
            "std_rho": np.std(rhos_t),
            "frac_neg": (rhos_t < 0).mean(),
            "frac_sig_neg": ((rhos_t < 0) & (pvals_t < 0.05)).mean(),
            "frac_sig_pos": ((rhos_t > 0) & (pvals_t < 0.05)).mean(),
            "median_mult": np.median(mults_t),
        }
        scan_results.append(r)
        print(f"  I > {thr:>5}: {above:>8,} obs, {r['n_hkl']:>5} HKLs  "
              f"⟨ρ⟩ = {r['mean_rho']:+.4f}  "
              f"frac_neg = {r['frac_neg']*100:.1f}%  "
              f"sig_neg = {r['frac_sig_neg']*100:.1f}%")

    # Filter out None entries
    valid = [(t, r) for t, r in zip(I_thresholds, scan_results) if r is not None]
    thrs_v = [v[0] for v in valid]
    res_v = [v[1] for v in valid]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Effect of intensity threshold on ρ(I, |Ewald offset|)\n"
        f"min {min_strong_obs} strong observations per HKL",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(thrs_v, [r["mean_rho"] for r in res_v], "o-", color="#534AB7", lw=2, ms=6, label="mean ρ")
    ax.plot(thrs_v, [r["median_rho"] for r in res_v], "s--", color="#D85A30", lw=2, ms=6, label="median ρ")
    ax.fill_between(thrs_v,
                     [r["mean_rho"] - r["std_rho"] for r in res_v],
                     [r["mean_rho"] + r["std_rho"] for r in res_v],
                     alpha=0.15, color="#534AB7")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Intensity threshold")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Mean/median ρ vs threshold")
    ax.legend(fontsize=9)
    ax.set_xscale("symlog", linthresh=10)

    ax = axes[0, 1]
    ax.plot(thrs_v, [r["frac_neg"] * 100 for r in res_v], "o-", color="#D85A30", lw=2, ms=6, label="ρ < 0")
    ax.plot(thrs_v, [r["frac_sig_neg"] * 100 for r in res_v], "s-", color="#534AB7", lw=2, ms=6, label="ρ < 0 & p<0.05")
    ax.plot(thrs_v, [r["frac_sig_pos"] * 100 for r in res_v], "^-", color="#1D9E75", lw=2, ms=6, label="ρ > 0 & p<0.05")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.set_xlabel("Intensity threshold")
    ax.set_ylabel("% of HKLs")
    ax.set_title("Fraction with negative ρ")
    ax.legend(fontsize=8)
    ax.set_xscale("symlog", linthresh=10)

    ax = axes[1, 0]
    ax.plot(thrs_v, [r["n_hkl"] for r in res_v], "o-", color="#534AB7", lw=2, ms=6)
    ax.set_xlabel("Intensity threshold")
    ax.set_ylabel("N qualifying HKLs")
    ax.set_title("Number of HKLs surviving filter")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("log")

    ax = axes[1, 1]
    ax.plot(thrs_v, [r["median_mult"] for r in res_v], "o-", color="#D85A30", lw=2, ms=6)
    ax.set_xlabel("Intensity threshold")
    ax.set_ylabel("Median multiplicity (strong obs)")
    ax.set_title("Multiplicity of strong observations")
    ax.set_xscale("symlog", linthresh=10)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUTDIR / "fig1_threshold_scan.png", dpi=150)
    print(f"  → {OUTDIR / 'fig1_threshold_scan.png'}")
    plt.close(fig)

    # ── Fig 2: Filtered ρ distribution — pick a good threshold ──
    # Use I > 100 and I > 1000 as representative
    for I_thr, tag in [(100, "I>100"), (1000, "I>1000")]:
        print(f"\nDetailed analysis with {tag}, min_strong={min_strong_obs}...")
        rhos_f, pvals_f, mults_f, hkl_ids_f, d_per_hkl_f, bins_f = \
            resolution_binned_correlation(
                I, abs_exc_err, hkl_id, d_spacing,
                n_bins=8, min_mult=min_strong_obs,
                I_threshold=I_thr, min_strong=min_strong_obs,
                _pregroup=pg,
            )
        if len(rhos_f) < 20:
            print(f"  Too few HKLs ({len(rhos_f)}) — skipping")
            continue
        print(f"  {len(rhos_f)} HKLs, median mult = {np.median(mults_f):.0f}")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"Per-HKL ρ(I, |r|) — {tag}, min {min_strong_obs} strong obs\n"
            f"N_hkl = {len(rhos_f)}, median mult = {np.median(mults_f):.0f}",
            fontsize=12, fontweight="bold",
        )

        ax = axes[0]
        ax.hist(rhos_f, bins=60, range=(-1, 1), density=True, color="#534AB7", alpha=0.7, edgecolor="#3B3580", lw=0.5)
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.axvline(np.mean(rhos_f), color="red", ls="--", lw=2, label=f"mean = {np.mean(rhos_f):.4f}")
        ax.axvline(np.median(rhos_f), color="orange", ls="--", lw=2, label=f"median = {np.median(rhos_f):.4f}")
        null_sig = 1.0 / np.sqrt(np.median(mults_f) - 1)
        x = np.linspace(-1, 1, 200)
        ax.plot(x, norm.pdf(x, 0, null_sig), "k--", lw=1.5, label=f"null (σ={null_sig:.3f})")
        ax.set_xlabel("Spearman ρ(I, |r|)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution")
        ax.legend(fontsize=8)

        ax = axes[1]
        d_m = [b["d_mean"] for b in bins_f]
        ax.errorbar(d_m, [b["mean_rho"] for b in bins_f],
                     yerr=[b["std_rho"] for b in bins_f],
                     fmt="o-", color="#534AB7", lw=2, ms=6, capsize=4, label="mean ± std")
        ax.plot(d_m, [b["median_rho"] for b in bins_f], "s--", color="#D85A30", lw=2, ms=6, label="median")
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("d-spacing (Å)")
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Resolution dependence")
        ax.legend(fontsize=9)

        ax = axes[2]
        ax.plot(d_m, [b["frac_negative"] * 100 for b in bins_f], "o-", color="#D85A30", lw=2, ms=6, label="ρ < 0")
        ax.plot(d_m, [b["frac_sig_neg"] * 100 for b in bins_f], "s-", color="#534AB7", lw=2, ms=6, label="ρ < 0 & p<0.05")
        ax.axhline(50, color="gray", ls=":", lw=1)
        ax.set_xlabel("d-spacing (Å)")
        ax.set_ylabel("% of HKLs")
        ax.set_title("Fraction with negative ρ")
        ax.legend(fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.86])
        fname = f"fig2_filtered_{tag.replace('>', '')}.png"
        fig.savefig(OUTDIR / fname, dpi=150)
        print(f"  → {OUTDIR / fname}")
        plt.close(fig)

        # Print resolution table
        print(f"\n  Resolution-binned ⟨ρ⟩ ({tag}):")
        for b in bins_f:
            print(f"    d = {b['d_lo']:.2f}–{b['d_hi']:.2f} Å: "
                  f"⟨ρ⟩ = {b['mean_rho']:+.4f}  "
                  f"median = {b['median_rho']:+.4f}  "
                  f"frac_neg = {b['frac_negative']*100:.1f}%  "
                  f"sig_neg = {b['frac_sig_neg']*100:.1f}%  "
                  f"N = {b['n_hkl']}  "
                  f"⟨mult⟩ = {b['mean_mult']:.0f}")

    # ── Fig 3: Scatter plot of log(I) vs |r| for strong reflections ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Global scatter: log(I) vs |Ewald offset| at different thresholds",
                 fontsize=12, fontweight="bold")

    for ax, (thr, label) in zip(axes, [(0, "All obs"), (100, "I > 100"), (1000, "I > 1000")]):
        mask = I > thr
        I_m = I[mask]
        r_m = abs_exc_err[mask]
        pos = I_m > 0
        log_I_m = np.log(I_m[pos])
        r_m_pos = r_m[pos]
        n_plot = min(200_000, len(log_I_m))
        idx = np.random.choice(len(log_I_m), n_plot, replace=False)
        ax.scatter(r_m_pos[idx] * 1e3, log_I_m[idx], s=1, alpha=0.03, c="#534AB7", rasterized=True)
        ax.set_xlabel("|excitation error| (×10⁻³ Å⁻¹)")
        ax.set_ylabel("log(I)")
        ax.set_title(f"{label} (N={mask.sum():,})")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig3_scatter_filtered.png", dpi=150)
    print(f"  → {OUTDIR / 'fig3_scatter_filtered.png'}")
    plt.close(fig)

    # ── Fig 4: Min-strong-obs scan at fixed I threshold ──
    print("\nScanning min strong obs at I > 200...")
    I_thr_fixed = 200
    min_strong_range = [5, 8, 10, 15, 20, 30, 50]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Effect of min strong observations per HKL (I > {I_thr_fixed})",
                 fontsize=12, fontweight="bold")

    mean_rhos_ms = []
    n_hkls_ms = []
    for ms in min_strong_range:
        rhos_ms, pvals_ms, mults_ms, _ = per_hkl_rank_correlation(
            I, abs_exc_err, hkl_id, min_mult=ms,
            I_threshold=I_thr_fixed, min_strong=ms,
            _pregroup=pg,
        )
        if len(rhos_ms) < 10:
            mean_rhos_ms.append(np.nan)
            n_hkls_ms.append(0)
            continue
        mean_rhos_ms.append(np.mean(rhos_ms))
        n_hkls_ms.append(len(rhos_ms))
        print(f"  min_strong={ms:>3}: {len(rhos_ms):>5} HKLs, "
              f"⟨ρ⟩ = {np.mean(rhos_ms):+.4f}, "
              f"frac_neg = {(rhos_ms < 0).mean()*100:.1f}%")

    ax = axes[0]
    ax.plot(min_strong_range, mean_rhos_ms, "o-", color="#534AB7", lw=2, ms=6)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Min strong obs per HKL")
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_title("Mean ρ vs min strong obs")

    ax = axes[1]
    ax.plot(min_strong_range, n_hkls_ms, "o-", color="#D85A30", lw=2, ms=6)
    ax.set_xlabel("Min strong obs per HKL")
    ax.set_ylabel("N qualifying HKLs")
    ax.set_title("HKLs surviving filter")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig4_min_strong_scan.png", dpi=150)
    print(f"  → {OUTDIR / 'fig4_min_strong_scan.png'}")
    plt.close(fig)

    # ── Summary ──
    print("\n" + "=" * 75)
    print("THRESHOLD SCAN SUMMARY")
    print("=" * 75)
    print(f"{'I_thr':>7} {'N_obs':>9} {'N_hkl':>6} {'med_mult':>8} "
          f"{'⟨ρ⟩':>7} {'med_ρ':>7} {'%neg':>6} {'%sig_neg':>8}")
    print("─" * 75)
    for r in res_v:
        print(f"{r['thr']:>7} {r['n_obs_above']:>9,} {r['n_hkl']:>6} "
              f"{r['median_mult']:>8.0f} "
              f"{r['mean_rho']:>+7.4f} {r['median_rho']:>+7.4f} "
              f"{r['frac_neg']*100:>5.1f}% {r['frac_sig_neg']*100:>7.1f}%")
    print("=" * 75)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("Loading data...")
    core = load_all_streams()
    print(f"  {core.hkl.shape[0]:,} observations, {core.astar.shape[0]:,} crystals")

    # Compute excitation error
    print("Computing excitation error...")
    wavelength = 12398.42 / core.beam_energy.median().item()
    print(f"  Wavelength: {wavelength:.4f} Å ({core.beam_energy.median().item():.0f} eV)")
    exc_err, q_lab = compute_excitation_error(core, wavelength)
    abs_exc_err = np.abs(exc_err)
    print(f"  |r| range: {abs_exc_err.min():.6f} – {abs_exc_err.max():.4f} Å⁻¹")
    print(f"  |r| median: {np.median(abs_exc_err):.6f} Å⁻¹")

    # HKL indexing
    print("Computing HKL index...")
    from torchsx.merge._hkl import compute_hkl_index
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id = hkl_idx.hkl_id.numpy()
    print(f"  {hkl_idx.n_unique} unique HKLs")

    # d-spacing per observation
    d_spacing = 1.0 / (2.0 * np.sqrt(np.sum(q_lab**2, axis=1) / 4.0))
    # More robust: use |q|/2 = 1/(2d) → d = 1/|q| ... wait, |q| = 2sin(θ)/λ = 1/d
    q_mag = np.sqrt(np.sum(q_lab**2, axis=1))
    d_spacing = 1.0 / q_mag
    d_spacing = np.clip(d_spacing, 0.5, 50.0)
    print(f"  d-spacing range: {d_spacing.min():.2f} – {d_spacing.max():.2f} Å")

    # Intensities
    I = core.I.numpy()
    print(f"  I range: {I.min():.1f} – {I.max():.1f}")

    # Run analysis
    make_figures(I, abs_exc_err, exc_err, hkl_id, d_spacing, hkl_idx.unique_hkl.numpy())

    # ── Ranking of strong reflections by Spearman ρ ──
    print("\n\nRanking strong reflections by Spearman ρ(I, |r|)...")
    I_THR = 50
    MIN_STRONG = 10
    pg = pregroup_by_hkl(I, abs_exc_err, hkl_id)
    rhos, pvals, mults, hkl_ids = per_hkl_rank_correlation(
        I, abs_exc_err, hkl_id, min_mult=MIN_STRONG,
        I_threshold=I_THR, min_strong=MIN_STRONG, _pregroup=pg,
    )
    unique_hkl = hkl_idx.unique_hkl.numpy()

    # d-spacing per qualifying HKL
    I_sorted, r_sorted, boundaries = pg
    d_per_hkl = np.array([
        np.median(d_spacing[hkl_id == uid]) for uid in hkl_ids
    ])

    # Mean intensity per HKL (above threshold)
    mean_I = np.array([
        np.mean(I_sorted[boundaries[uid]:boundaries[uid+1]][
            I_sorted[boundaries[uid]:boundaries[uid+1]] > I_THR
        ]) for uid in hkl_ids
    ])

    # Sort by ρ (most negative first)
    order = np.argsort(rhos)

    # Print top/bottom tables
    for label, idx_slice in [
        ("STRONGEST partiality signal (most negative ρ)", order[:40]),
        ("WEAKEST / reversed (most positive ρ)", order[-20:][::-1]),
    ]:
        print(f"\n{'=' * 95}")
        print(f"  {label}")
        print(f"  I > {I_THR}, min {MIN_STRONG} strong obs per HKL, N = {len(rhos)} HKLs")
        print(f"{'=' * 95}")
        hdr = f"{'rank':>5} {'h':>4} {'k':>4} {'l':>4} {'d(Å)':>6} {'mult':>5} {'⟨I⟩':>9} {'ρ':>7} {'p-val':>9}"
        print(hdr)
        print("─" * len(hdr))
        for rank_i, i in enumerate(idx_slice):
            uid = hkl_ids[i]
            h, k, l = unique_hkl[uid]
            print(f"{rank_i+1:>5} {h:>4} {k:>4} {l:>4} "
                  f"{d_per_hkl[i]:>6.2f} {mults[i]:>5} {mean_I[i]:>9.1f} "
                  f"{rhos[i]:>+7.4f} {pvals[i]:>9.2e}")
        print(f"{'=' * 95}")

    # Save full ranking to CSV
    csv_path = OUTDIR / "ranking_rho_strong.csv"
    with open(csv_path, "w") as f:
        f.write("rank,h,k,l,d_A,multiplicity,mean_I,spearman_rho,p_value\n")
        for rank_i, i in enumerate(order):
            uid = hkl_ids[i]
            h, k, l = unique_hkl[uid]
            f.write(f"{rank_i+1},{h},{k},{l},{d_per_hkl[i]:.4f},"
                    f"{mults[i]},{mean_I[i]:.2f},"
                    f"{rhos[i]:.6f},{pvals[i]:.4e}\n")
    print(f"\nFull ranking ({len(rhos)} HKLs) saved to {csv_path}")

    print(f"\nDone. Figures in {OUTDIR}/")
