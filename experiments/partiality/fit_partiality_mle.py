#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
MLE estimation of partiality per (resolution, |r|) bin.

Instead of median(log(I/I₀)) which requires positive strong obs,
compute the MLE of the partiality multiplier p using ALL observations:

    I_obs ~ N(I₀ × p, σ²_poi + (I₀ × p)² × σ²_LN)

This handles negatives, uses weak obs appropriately (weighted by
their information content), and naturally incorporates the noise model.

For each bin: 1D optimisation over log(p) — fast.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import binned_statistic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = Path(__file__).resolve().parent / "results"
OUTDIR.mkdir(exist_ok=True)


def bin_nll(log_p, I_obs, I0, sigma2_poi, sigma2_LN):
    """
    Negative log-likelihood of observations in one bin given log(p).

    Model: I_obs ~ N(I₀ × p, σ²_poi + (I₀ × p)² × σ²_LN)

    Uses ALL observations — no threshold needed.
    """
    p = np.exp(log_p)
    I_pred = I0 * p
    var = sigma2_poi + I_pred**2 * sigma2_LN
    var = np.clip(var, 1e-6, None)
    resid2 = (I_obs - I_pred)**2
    nll = 0.5 * (resid2 / var + np.log(var))
    return np.sum(nll)


def fit_bin_partiality(I_obs, I0, sigma2_poi, sigma2_LN):
    """
    MLE of log(p) for one (resolution, |r|) bin.

    Returns log(p), with bounds [-10, 2].
    """
    if len(I_obs) < 5:
        return np.nan

    result = minimize_scalar(
        bin_nll,
        bounds=(-10, 2),
        method="bounded",
        args=(I_obs, I0, sigma2_poi, sigma2_LN),
    )
    return result.x if result.success else np.nan


def main():
    from partiality._data import load_and_prepare
    from partiality._excitation import ExcitationErrorModel

    stream_dir = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"
    sfx_path = [
        stream_dir / "delay_-91.sfx",
        stream_dir / "delay_-136.sfx",
        stream_dir / "delay_-182.sfx",
        stream_dir / "delay_-227.sfx",
    ]

    pair_data = load_and_prepare(sfx_path, I_threshold=0, min_mult=2)

    # Load refined model
    model = ExcitationErrorModel(
        pair_data.n_crystals, pair_data.UB, pair_data.lambda_0)
    result_path = OUTDIR / "refinement_result.pt"
    if result_path.exists():
        saved = torch.load(result_path, weights_only=True)
        model.delta_omega_xy.data.copy_(saved["delta_omega_xy"])
        model.delta_lambda.data.copy_(saved["delta_lambda"])
        print("Loaded refined parameters")

    with torch.no_grad():
        abs_r = model.abs_excitation_error(
            pair_data.hkl_obs, pair_data.crystal_id).numpy()

    I_obs = pair_data.I_corr.numpy()  # B/G corrected
    hkl_id = pair_data.hkl_id.numpy()
    s_sq = pair_data.s_sq.numpy()
    d = 1.0 / (2 * np.sqrt(s_sq.clip(min=1e-10)))
    inv_d = 1.0 / d
    N = len(I_obs)
    print(f"Using ALL {N:,} observations (no threshold)")

    # Merged I₀ per HKL (use all positive obs, not just strong)
    n_unique = hkl_id.max() + 1
    I_sum = np.zeros(n_unique)
    I_cnt = np.zeros(n_unique)
    pos = I_obs > 0
    np.add.at(I_sum, hkl_id[pos], I_obs[pos])
    np.add.at(I_cnt, hkl_id[pos], 1)
    I0_merged = I_sum / np.clip(I_cnt, 1, None)
    I0_per_obs = I0_merged[hkl_id]

    # Poisson noise estimate (simple: σ²_poi ≈ |I_obs| + background)
    # Use a rough estimate based on gain and counts
    sigma2_poi = np.abs(I_obs) + 100  # floor at ~100 counts²

    # Per-shell σ_LN from the data (resolution-dependent lognormal width)
    shell_edges = [0.9, 1.1, 1.4, 1.8, 2.5, 4.0, 10.0]
    n_shells = len(shell_edges) - 1
    shell_sigma2_LN = np.zeros(n_shells)
    shell_id = np.digitize(d, shell_edges) - 1
    shell_id = np.clip(shell_id, 0, n_shells - 1)

    for si in range(n_shells):
        mask = (shell_id == si) & pos & (I0_per_obs > 10)
        if mask.sum() < 100:
            shell_sigma2_LN[si] = 1.0
            continue
        log_ratio = np.log(I_obs[mask] / I0_per_obs[mask])
        # Robust variance estimate (MAD-based)
        mad = np.median(np.abs(log_ratio - np.median(log_ratio)))
        shell_sigma2_LN[si] = (mad * 1.4826)**2  # MAD to σ conversion
        print(f"  Shell {shell_edges[si]:.1f}–{shell_edges[si+1]:.1f} Å: "
              f"σ_LN = {np.sqrt(shell_sigma2_LN[si]):.3f}")

    sigma2_LN_per_obs = shell_sigma2_LN[shell_id]

    # ── Bin and fit ──
    n_r_bins = 25
    r_edges = np.linspace(0, 0.007, n_r_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    print(f"\nFitting partiality MLE in {n_shells} × {n_r_bins} = {n_shells * n_r_bins} bins...")

    # Also compute median-based estimate for comparison
    log_p_mle = np.full((n_shells, n_r_bins), np.nan)
    log_p_median = np.full((n_shells, n_r_bins), np.nan)
    n_obs_per_bin = np.zeros((n_shells, n_r_bins), dtype=int)

    for si in range(n_shells):
        d_lo, d_hi = shell_edges[si], shell_edges[si + 1]
        shell_mask = (d >= d_lo) & (d < d_hi)

        for ri in range(n_r_bins):
            r_lo, r_hi = r_edges[ri], r_edges[ri + 1]
            mask = shell_mask & (abs_r >= r_lo) & (abs_r < r_hi)
            n = mask.sum()
            n_obs_per_bin[si, ri] = n

            if n < 10:
                continue

            I_bin = I_obs[mask]
            I0_bin = I0_per_obs[mask]
            sp_bin = sigma2_poi[mask]
            sln_bin = sigma2_LN_per_obs[mask]

            # MLE fit
            log_p_mle[si, ri] = fit_bin_partiality(I_bin, I0_bin, sp_bin, sln_bin)

            # Median-based (for comparison, only positive strong)
            strong = (I_bin > 50) & (I0_bin > 50)
            if strong.sum() >= 5:
                log_p_median[si, ri] = np.median(np.log(I_bin[strong] / I0_bin[strong]))

    # ── Figures ──

    # Fig 1: MLE vs median comparison per shell
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Partiality estimation: MLE (all obs) vs median (strong only)\n"
        "MLE uses full noise model; median requires I > 50",
        fontsize=13, fontweight="bold",
    )

    for ax_i, si in enumerate(range(n_shells)):
        ax = axes.flat[ax_i]
        d_lo, d_hi = shell_edges[si], shell_edges[si + 1]

        good_mle = np.isfinite(log_p_mle[si])
        good_med = np.isfinite(log_p_median[si])

        if good_mle.any():
            ax.plot(r_centers[good_mle] * 1e3, log_p_mle[si, good_mle],
                    "o-", color="#D85A30", lw=2, ms=5, label="MLE (all obs)")
        if good_med.any():
            ax.plot(r_centers[good_med] * 1e3, log_p_median[si, good_med],
                    "s--", color="#534AB7", lw=1.5, ms=4, alpha=0.7,
                    label="Median (I>50)")

        # N obs per bin on secondary axis
        ax2 = ax.twinx()
        ax2.bar(r_centers * 1e3, n_obs_per_bin[si],
                width=0.28 * 1e3 / n_r_bins, alpha=0.15, color="gray")
        ax2.set_ylabel("N obs", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
        ax.set_ylabel("log(p)")
        n_total = n_obs_per_bin[si].sum()
        ax.set_title(f"d = {d_lo}–{d_hi} Å  (N={n_total:,})")
        ax.legend(fontsize=8)
        ax.set_ylim(-3, 1.5)

    # Last panel: N obs used by each method
    ax = axes.flat[-1]
    shell_labels = [f"{a}–{b}" for a, b in zip(shell_edges[:-1], shell_edges[1:])]
    n_mle_total = [n_obs_per_bin[si].sum() for si in range(n_shells)]
    # Count strong obs per shell
    n_strong = []
    for si in range(n_shells):
        d_lo, d_hi = shell_edges[si], shell_edges[si + 1]
        mask = (d >= d_lo) & (d < d_hi) & (I_obs > 50) & (I0_per_obs > 50)
        n_strong.append(mask.sum())
    x = np.arange(n_shells)
    ax.bar(x - 0.15, n_mle_total, 0.3, color="#D85A30", label="MLE (all)")
    ax.bar(x + 0.15, n_strong, 0.3, color="#534AB7", label="Median (I>50)")
    ax.set_xticks(x)
    ax.set_xticklabels(shell_labels, fontsize=8)
    ax.set_ylabel("N observations")
    ax.set_title("Data utilisation")
    ax.legend()
    ax.set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(OUTDIR / "fig_partiality_mle.png", dpi=150)
    print(f"\n  → {OUTDIR / 'fig_partiality_mle.png'}")
    plt.close(fig)

    # Fig 2: Fit generalised Lorentzian to MLE curve
    from scipy.optimize import curve_fit

    def gen_lorentz(r, a0, gamma, alpha):
        return a0 - np.log1p((r / gamma) ** alpha)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Generalised Lorentzian fit to MLE partiality\n"
        "log(p) = a₀ - log(1 + (|r|/γ)^α)  — fitted on all observations",
        fontsize=13, fontweight="bold",
    )

    shell_gammas, shell_alphas = [], []

    for ax_i, si in enumerate(range(n_shells)):
        ax = axes.flat[ax_i]
        d_lo, d_hi = shell_edges[si], shell_edges[si + 1]
        good = np.isfinite(log_p_mle[si]) & (n_obs_per_bin[si] >= 50)

        if good.sum() < 5:
            ax.set_title(f"d = {d_lo}–{d_hi} Å (too few bins)")
            shell_gammas.append(np.nan)
            shell_alphas.append(np.nan)
            continue

        r_fit = r_centers[good]
        y_fit = log_p_mle[si, good]

        ax.plot(r_fit * 1e3, y_fit, "o", color="#534AB7", ms=6, label="MLE data")

        r_fine = np.linspace(1e-5, 0.007, 200)

        try:
            popt, _ = curve_fit(gen_lorentz, r_fit, y_fit,
                                 p0=[0.5, 0.003, 1.0],
                                 bounds=([-5, 1e-5, 0.2], [5, 0.05, 4.0]))
            rss = np.mean((y_fit - gen_lorentz(r_fit, *popt))**2)
            ax.plot(r_fine * 1e3, gen_lorentz(r_fine, *popt),
                    "-", color="#D85A30", lw=2.5,
                    label=f"GenL γ={popt[1]*1e3:.3f} α={popt[2]:.2f} (rss={rss:.5f})")
            shell_gammas.append(popt[1])
            shell_alphas.append(popt[2])
        except Exception as e:
            print(f"  Fit failed for shell {si}: {e}")
            shell_gammas.append(np.nan)
            shell_alphas.append(np.nan)

        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
        ax.set_ylabel("log(p) MLE")
        ax.set_title(f"d = {d_lo}–{d_hi} Å  (N={n_obs_per_bin[si].sum():,})")
        ax.legend(fontsize=8)
        ax.set_ylim(-3, 1.5)

    # Last panel: γ and α vs resolution
    ax = axes.flat[-1]
    sg = np.array(shell_gammas)
    sa = np.array(shell_alphas)
    inv_d_mids = np.array([0.5 * (1/a + 1/b) for a, b in zip(shell_edges[:-1], shell_edges[1:])])
    fin = np.isfinite(sg)
    if fin.any():
        ax.plot(inv_d_mids[fin], sg[fin] * 1e3, "o-", color="#D85A30", lw=2, ms=6, label="γ ×10⁻³")
        ax2 = ax.twinx()
        ax2.plot(inv_d_mids[fin], sa[fin], "s-", color="#534AB7", lw=2, ms=6, label="α")
        ax2.set_ylabel("α", color="#534AB7")
        ax2.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("γ (×10⁻³ Å⁻¹)", color="#D85A30")
    ax.set_title("Resolution dependence")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(OUTDIR / "fig_partiality_mle_genlorentz.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_partiality_mle_genlorentz.png'}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 70)
    print("MLE PARTIALITY ESTIMATION SUMMARY")
    print("=" * 70)
    print(f"  Total observations used: {N:,} (ALL, no threshold)")
    print(f"  Bins: {n_shells} shells × {n_r_bins} |r| bins = {n_shells * n_r_bins}")
    print(f"\n  Per-shell GenLorentz parameters:")
    print(f"  {'Shell':>10} {'N_obs':>10} {'γ×10³':>8} {'α':>6}")
    for si, (d_lo, d_hi) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        g = f"{shell_gammas[si]*1e3:.4f}" if np.isfinite(shell_gammas[si]) else "N/A"
        a = f"{shell_alphas[si]:.3f}" if np.isfinite(shell_alphas[si]) else "N/A"
        print(f"  {d_lo}–{d_hi:>4} Å {n_obs_per_bin[si].sum():>10,} {g:>8} {a:>6}")
    print("=" * 70)


if __name__ == "__main__":
    main()
