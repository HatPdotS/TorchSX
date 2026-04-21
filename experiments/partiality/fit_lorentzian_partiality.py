#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Fit generalised Lorentzian partiality model to the empirical data.

    log(p) = -log(1 + (|r| / γ(1/d))^α(1/d))

where γ and α are smooth functions of resolution.

Compare to: Gaussian, exponential, pure Lorentzian, and the empirical bins.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import binned_statistic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = Path(__file__).resolve().parent / "results"
OUTDIR.mkdir(exist_ok=True)


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

    pair_data = load_and_prepare(sfx_path, I_threshold=20.0, min_mult=3)

    # Load refined model
    result_path = OUTDIR / "refinement_result.pt"
    model = ExcitationErrorModel(
        pair_data.n_crystals, pair_data.UB, pair_data.lambda_0)
    if result_path.exists():
        saved = torch.load(result_path, weights_only=True)
        model.delta_omega_xy.data.copy_(saved["delta_omega_xy"])
        model.delta_lambda.data.copy_(saved["delta_lambda"])
        print("Loaded refined parameters")

    with torch.no_grad():
        abs_r = model.abs_excitation_error(
            pair_data.hkl_obs, pair_data.crystal_id).numpy()

    I_corr = pair_data.I_corr.numpy()
    hkl_id = pair_data.hkl_id.numpy()
    s_sq = pair_data.s_sq.numpy()
    d = 1.0 / (2 * np.sqrt(s_sq.clip(min=1e-10)))
    inv_d = 1.0 / d

    # Merged I
    n_unique = hkl_id.max() + 1
    I_sum = np.zeros(n_unique)
    I_cnt = np.zeros(n_unique)
    pos = I_corr > 0
    np.add.at(I_sum, hkl_id[pos], I_corr[pos])
    np.add.at(I_cnt, hkl_id[pos], 1)
    I_merged = I_sum / np.clip(I_cnt, 1, None)
    I_merged_obs = I_merged[hkl_id]

    valid = (I_corr > 50) & (I_merged_obs > 50)
    log_corr = np.full(len(I_corr), np.nan)
    log_corr[valid] = np.log(I_corr[valid] / I_merged_obs[valid])

    lc = log_corr[valid]
    ar = abs_r[valid]
    invd = inv_d[valid]
    print(f"{valid.sum():,} valid observations")

    # ── Models ──

    def gen_lorentz(r, a0, gamma, alpha):
        """log(p) = a0 - log(1 + (r/γ)^α)"""
        return a0 - np.log1p((r / gamma) ** alpha)

    def gaussian(r, a0, sigma):
        return a0 - r**2 / (2 * sigma**2)

    def exponential(r, a0, scale):
        return a0 - r / scale

    def lorentz_fixed(r, a0, gamma):
        """Standard Lorentzian (α=2)"""
        return a0 - np.log1p((r / gamma) ** 2)

    # ── Per-shell fits ──
    shell_edges = [0.9, 1.1, 1.4, 1.8, 2.5, 4.0, 10.0]
    n_r_bins = 30
    r_edges = np.linspace(0, 0.007, n_r_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Partiality model fits: Generalised Lorentzian vs alternatives\n"
        "log(p) = a₀ - log(1 + (|r|/γ)^α)",
        fontsize=13, fontweight="bold",
    )

    shell_gammas = []
    shell_alphas = []
    shell_invd_mids = []
    shell_rss = {m: [] for m in ["GenLorentz", "Lorentz(α=2)", "Gaussian", "Exponential"]}

    for ax_i, (d_lo, d_hi) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        ax = axes.flat[ax_i]
        mask = valid & (d >= d_lo) & (d < d_hi)
        n = mask.sum()

        lc_s = log_corr[mask]
        ar_s = abs_r[mask]

        # Binned medians
        med, edges, _ = binned_statistic(ar_s, lc_s, statistic="median", bins=r_edges)
        q25, _, _ = binned_statistic(ar_s, lc_s, statistic=lambda x: np.percentile(x, 25), bins=r_edges)
        q75, _, _ = binned_statistic(ar_s, lc_s, statistic=lambda x: np.percentile(x, 75), bins=r_edges)
        cnt, _, _ = binned_statistic(ar_s, lc_s, statistic="count", bins=r_edges)

        good = (cnt > 20) & np.isfinite(med)
        if good.sum() < 5:
            ax.set_title(f"d = {d_lo}–{d_hi} Å (N={n})")
            shell_gammas.append(np.nan)
            shell_alphas.append(np.nan)
            shell_invd_mids.append(0.5 * (1/d_lo + 1/d_hi))
            continue

        r_fit = r_centers[good]
        y_fit = med[good]

        # Plot data
        ax.fill_between(r_centers[good] * 1e3, q25[good], q75[good],
                         alpha=0.15, color="#534AB7")
        ax.plot(r_centers[good] * 1e3, med[good], "o",
                color="#534AB7", ms=5, alpha=0.8, label="data (median)")

        r_fine = np.linspace(1e-5, 0.007, 200)

        # Fit each model
        fits = {}

        # Generalised Lorentzian
        try:
            popt, _ = curve_fit(gen_lorentz, r_fit, y_fit,
                                 p0=[0.5, 0.003, 1.5],
                                 bounds=([-3, 1e-5, 0.3], [3, 0.05, 4.0]))
            fits["GenLorentz"] = popt
            rss = np.mean((y_fit - gen_lorentz(r_fit, *popt))**2)
            shell_rss["GenLorentz"].append(rss)
            ax.plot(r_fine * 1e3, gen_lorentz(r_fine, *popt),
                    "-", color="#D85A30", lw=2.5,
                    label=f"GenL γ={popt[1]*1e3:.2f} α={popt[2]:.2f} (rss={rss:.4f})")
            shell_gammas.append(popt[1])
            shell_alphas.append(popt[2])
        except Exception as e:
            print(f"  GenLorentz fit failed for d={d_lo}–{d_hi}: {e}")
            shell_gammas.append(np.nan)
            shell_alphas.append(np.nan)
            shell_rss["GenLorentz"].append(np.nan)

        # Standard Lorentzian (α=2)
        try:
            popt, _ = curve_fit(lorentz_fixed, r_fit, y_fit,
                                 p0=[0.5, 0.003], bounds=([-3, 1e-5], [3, 0.05]))
            fits["Lorentz"] = popt
            rss = np.mean((y_fit - lorentz_fixed(r_fit, *popt))**2)
            shell_rss["Lorentz(α=2)"].append(rss)
            ax.plot(r_fine * 1e3, lorentz_fixed(r_fine, *popt),
                    "--", color="#1D9E75", lw=2,
                    label=f"Lorentz γ={popt[1]*1e3:.2f} (rss={rss:.4f})")
        except Exception:
            shell_rss["Lorentz(α=2)"].append(np.nan)

        # Gaussian
        try:
            popt, _ = curve_fit(gaussian, r_fit, y_fit,
                                 p0=[0.5, 0.003], bounds=([-3, 1e-5], [3, 0.05]))
            rss = np.mean((y_fit - gaussian(r_fit, *popt))**2)
            shell_rss["Gaussian"].append(rss)
            ax.plot(r_fine * 1e3, gaussian(r_fine, *popt),
                    ":", color="#999999", lw=2,
                    label=f"Gauss σ={popt[1]*1e3:.2f} (rss={rss:.4f})")
        except Exception:
            shell_rss["Gaussian"].append(np.nan)

        # Exponential
        try:
            popt, _ = curve_fit(exponential, r_fit, y_fit,
                                 p0=[0.5, 0.003], bounds=([-3, 1e-5], [3, 0.05]))
            rss = np.mean((y_fit - exponential(r_fit, *popt))**2)
            shell_rss["Exponential"].append(rss)
            ax.plot(r_fine * 1e3, exponential(r_fine, *popt),
                    "-.", color="#B5338A", lw=2,
                    label=f"Exp s={popt[1]*1e3:.2f} (rss={rss:.4f})")
        except Exception:
            shell_rss["Exponential"].append(np.nan)

        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
        ax.set_ylabel("median log(I/I_merged)")
        ax.set_title(f"d = {d_lo}–{d_hi} Å  (N={n:,})")
        ax.legend(fontsize=6.5, loc="lower left")
        ax.set_ylim(-2.5, 1.5)

        shell_invd_mids.append(0.5 * (1/d_lo + 1/d_hi))

    # Last panel: γ and α vs resolution
    ax = axes.flat[-1]
    shell_gammas = np.array(shell_gammas)
    shell_alphas = np.array(shell_alphas)
    shell_invd_mids = np.array(shell_invd_mids)
    fin = np.isfinite(shell_gammas)

    ax2 = ax.twinx()
    if fin.any():
        ax.plot(shell_invd_mids[fin], shell_gammas[fin] * 1e3, "o-",
                color="#D85A30", lw=2, ms=6, label="γ (×10⁻³)")
        ax2.plot(shell_invd_mids[fin], shell_alphas[fin], "s-",
                 color="#534AB7", lw=2, ms=6, label="α")
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("γ (×10⁻³ Å⁻¹)", color="#D85A30")
    ax2.set_ylabel("α (tail exponent)", color="#534AB7")
    ax.set_title("Resolution dependence of γ and α")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUTDIR / "fig_gen_lorentz_fit.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_gen_lorentz_fit.png'}")
    plt.close(fig)

    # ── Summary: model comparison ──
    print("\n" + "=" * 75)
    print("MODEL COMPARISON (mean RSS per shell)")
    print("=" * 75)
    for name in ["GenLorentz", "Lorentz(α=2)", "Gaussian", "Exponential"]:
        vals = [v for v in shell_rss[name] if np.isfinite(v)]
        if vals:
            print(f"  {name:>15}: mean RSS = {np.mean(vals):.6f}")
    print()

    print("Per-shell parameters:")
    print(f"  {'Shell':>10} {'γ×10³':>8} {'α':>6}")
    print("  " + "─" * 30)
    for i, (d_lo, d_hi) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        if i < len(shell_gammas):
            g = f"{shell_gammas[i]*1e3:.3f}" if np.isfinite(shell_gammas[i]) else "N/A"
            a = f"{shell_alphas[i]:.3f}" if np.isfinite(shell_alphas[i]) else "N/A"
            print(f"  {d_lo}–{d_hi:>4} Å {g:>8} {a:>6}")

    # ── Global 2D fit: γ(1/d) and α(1/d) as polynomials ──
    if fin.sum() >= 3:
        print("\nGlobal parameterisation:")
        # Fit γ(1/d) = g0 + g1*(1/d)
        coeffs_g = np.polyfit(shell_invd_mids[fin], shell_gammas[fin], 1)
        print(f"  γ(1/d) = {coeffs_g[1]*1e3:.3f} + {coeffs_g[0]*1e3:.3f} × (1/d)  [×10⁻³ Å⁻¹]")

        # Fit α(1/d) = a0 + a1*(1/d)
        coeffs_a = np.polyfit(shell_invd_mids[fin], shell_alphas[fin], 1)
        print(f"  α(1/d) = {coeffs_a[1]:.3f} + {coeffs_a[0]:.3f} × (1/d)")

        # Evaluate residual with 2D model
        print(f"\n  At d=1.5 Å (1/d=0.67): γ={np.polyval(coeffs_g, 0.67)*1e3:.3f}×10⁻³, α={np.polyval(coeffs_a, 0.67):.2f}")
        print(f"  At d=3.0 Å (1/d=0.33): γ={np.polyval(coeffs_g, 0.33)*1e3:.3f}×10⁻³, α={np.polyval(coeffs_a, 0.33):.2f}")
        print(f"  At d=7.0 Å (1/d=0.14): γ={np.polyval(coeffs_g, 0.14)*1e3:.3f}×10⁻³, α={np.polyval(coeffs_a, 0.14):.2f}")

    print("=" * 75)


if __name__ == "__main__":
    main()
