#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Estimate the empirical partiality function from refined excitation errors.

After rank-concordance refinement (model-free), we have improved |r| values.
The relationship log(I_corr / I_merged) vs |r| traces out the partiality
function. Doing this per resolution shell reveals σ_relp(d).

No circularity: the rank refinement never assumed p(r)'s shape.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

    # Load data with B/G correction
    pair_data = load_and_prepare(sfx_path, I_threshold=20.0, min_mult=3)

    # Load refined model
    result_path = OUTDIR / "refinement_result.pt"
    if result_path.exists():
        print("Loading refined parameters...")
        saved = torch.load(result_path, weights_only=True)
        model = ExcitationErrorModel(
            pair_data.n_crystals, pair_data.UB, pair_data.lambda_0,
        )
        model.delta_omega_xy.data.copy_(saved["delta_omega_xy"])
        model.delta_lambda.data.copy_(saved["delta_lambda"])
        label_refined = "refined"
    else:
        print("No refined model found — using initial orientations")
        model = ExcitationErrorModel(
            pair_data.n_crystals, pair_data.UB, pair_data.lambda_0,
        )
        label_refined = "initial"

    # Compute |r|
    with torch.no_grad():
        abs_r = model.abs_excitation_error(
            pair_data.hkl_obs, pair_data.crystal_id,
        ).numpy()

    I_corr = pair_data.I_corr.numpy()
    hkl_id = pair_data.hkl_id.numpy()
    s_sq = pair_data.s_sq.numpy()
    d = 1.0 / (2 * np.sqrt(s_sq.clip(min=1e-10)))

    # Merged I per HKL (from corrected intensities)
    n_unique = hkl_id.max() + 1
    I_sum = np.zeros(n_unique)
    I_cnt = np.zeros(n_unique)
    pos = I_corr > 0
    np.add.at(I_sum, hkl_id[pos], I_corr[pos])
    np.add.at(I_cnt, hkl_id[pos], 1)
    I_merged = I_sum / np.clip(I_cnt, 1, None)
    I_merged_obs = I_merged[hkl_id]

    # log correction factor
    valid = (I_corr > 50) & (I_merged_obs > 50)
    log_corr = np.full(len(I_corr), np.nan)
    log_corr[valid] = np.log(I_corr[valid] / I_merged_obs[valid])

    print(f"\n{valid.sum():,} valid obs (I > 50, I_merged > 50)")

    # ── Resolution shells ──
    d_shells = [
        (0.9, 1.1, "0.9–1.1"),
        (1.1, 1.4, "1.1–1.4"),
        (1.4, 1.8, "1.4–1.8"),
        (1.8, 2.5, "1.8–2.5"),
        (2.5, 4.0, "2.5–4.0"),
        (4.0, 10.0, "4.0–10.0"),
    ]

    # ── Fig 1: Empirical partiality curves per resolution shell ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Empirical partiality from {label_refined} |r|\n"
        f"log(I_corr / I_merged) vs |r| per resolution shell",
        fontsize=13, fontweight="bold",
    )

    # Partiality models to fit
    # log(p) vs |r| — different functional forms
    def gaussian_model(r, log_p0, sigma):
        """log(p) = log_p0 - r²/(2σ²)  — concave, steepens at large r"""
        return log_p0 - r**2 / (2 * sigma**2)

    def lorentzian_model(r, log_p0, gamma):
        """log(p) = log_p0 - log(1 + (r/γ)²) — heavy tails"""
        return log_p0 - np.log(1 + (r / gamma)**2)

    def exponential_model(r, log_p0, scale):
        """log(p) = log_p0 - r/scale — LINEAR in |r|"""
        return log_p0 - r / scale

    def gen_gaussian_model(r, log_p0, alpha, beta):
        """log(p) = log_p0 - |r/α|^β — flexible shape
        β=2: Gaussian, β=1: exponential, β<1: sub-exponential (heavy tail)"""
        return log_p0 - np.abs(r / alpha)**beta

    fitted_sigmas = []
    fitted_gammas = []
    shell_d_mids = []
    shell_s_sq_mids = []

    bins_r = np.linspace(0, 0.006, 40)
    bin_centers = 0.5 * (bins_r[:-1] + bins_r[1:])

    for ax, (d_lo, d_hi, shell_label) in zip(axes.flat, d_shells):
        mask = valid & (d >= d_lo) & (d < d_hi)
        n = mask.sum()
        if n < 500:
            ax.set_title(f"d = {shell_label} Å (N={n})")
            continue

        lc = log_corr[mask]
        ar = abs_r[mask]

        # Binned median
        med, edges, _ = binned_statistic(ar, lc, statistic="median", bins=bins_r)
        q25, _, _ = binned_statistic(ar, lc, statistic=lambda x: np.percentile(x, 25), bins=bins_r)
        q75, _, _ = binned_statistic(ar, lc, statistic=lambda x: np.percentile(x, 75), bins=bins_r)
        cnt, _, _ = binned_statistic(ar, lc, statistic="count", bins=bins_r)

        # Plot data
        good_bins = cnt > 20
        ax.fill_between(bin_centers[good_bins] * 1e3,
                         q25[good_bins], q75[good_bins],
                         alpha=0.15, color="#534AB7")
        ax.plot(bin_centers[good_bins] * 1e3, med[good_bins],
                "o", color="#534AB7", ms=4, alpha=0.8)

        # Fit Gaussian: log(p) = log_p0 - r²/(2σ²)
        fit_mask = good_bins & np.isfinite(med)
        r_fit = bin_centers[fit_mask]
        y_fit = med[fit_mask]

        r_fine = np.linspace(0, 0.006, 200)
        sigma_gauss, gamma_lor, scale_exp, alpha_gg, beta_gg = [np.nan] * 5
        residuals = {}

        # Gaussian: log(p) = a - r²/(2σ²)
        try:
            popt_g, _ = curve_fit(gaussian_model, r_fit, y_fit,
                                   p0=[0.5, 0.002], bounds=([-5, 1e-5], [5, 0.1]))
            sigma_gauss = popt_g[1]
            y_pred = gaussian_model(r_fit, *popt_g)
            residuals["Gauss"] = np.mean((y_fit - y_pred)**2)
            ax.plot(r_fine * 1e3, gaussian_model(r_fine, *popt_g),
                    "-", color="#D85A30", lw=2,
                    label=f"Gauss σ={sigma_gauss*1e3:.2f}")
        except Exception:
            pass

        # Lorentzian: log(p) = a - log(1 + (r/γ)²)
        try:
            popt_l, _ = curve_fit(lorentzian_model, r_fit, y_fit,
                                   p0=[0.5, 0.002], bounds=([-5, 1e-5], [5, 0.1]))
            gamma_lor = popt_l[1]
            y_pred = lorentzian_model(r_fit, *popt_l)
            residuals["Lorentz"] = np.mean((y_fit - y_pred)**2)
            ax.plot(r_fine * 1e3, lorentzian_model(r_fine, *popt_l),
                    "--", color="#1D9E75", lw=2,
                    label=f"Lorentz γ={gamma_lor*1e3:.2f}")
        except Exception:
            pass

        # Exponential: log(p) = a - r/s  (LINEAR in |r|)
        try:
            popt_e, _ = curve_fit(exponential_model, r_fit, y_fit,
                                   p0=[0.5, 0.003], bounds=([-5, 1e-5], [5, 0.1]))
            scale_exp = popt_e[1]
            y_pred = exponential_model(r_fit, *popt_e)
            residuals["Exp"] = np.mean((y_fit - y_pred)**2)
            ax.plot(r_fine * 1e3, exponential_model(r_fine, *popt_e),
                    "-.", color="#B5338A", lw=2,
                    label=f"Exp s={scale_exp*1e3:.2f}")
        except Exception:
            pass

        # Generalized Gaussian: log(p) = a - |r/α|^β
        try:
            popt_gg, _ = curve_fit(gen_gaussian_model, r_fit, y_fit,
                                    p0=[0.5, 0.003, 1.0],
                                    bounds=([-5, 1e-5, 0.3], [5, 0.1, 3.0]))
            alpha_gg, beta_gg = popt_gg[1], popt_gg[2]
            y_pred = gen_gaussian_model(r_fit, *popt_gg)
            residuals["GenGauss"] = np.mean((y_fit - y_pred)**2)
            ax.plot(r_fine * 1e3, gen_gaussian_model(r_fine, *popt_gg),
                    ":", color="#333333", lw=2.5,
                    label=f"GG α={alpha_gg*1e3:.2f} β={beta_gg:.2f}")
        except Exception:
            pass

        # Print best model
        if residuals:
            best = min(residuals, key=residuals.get)
            for name, rss in sorted(residuals.items(), key=lambda x: x[1]):
                marker = " ← best" if name == best else ""
                # Just store for later printing

        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
        ax.set_ylabel("median log(I/I_merged)")
        ax.set_title(f"d = {shell_label} Å  (N={n:,})")
        ax.legend(fontsize=8)
        ax.set_ylim(-2.5, 1.5)

        d_mid = 0.5 * (d_lo + d_hi)
        s_sq_mid = 1.0 / (4 * d_mid**2)
        fitted_sigmas.append(sigma_gauss)
        fitted_gammas.append(gamma_lor)
        shell_d_mids.append(d_mid)
        shell_s_sq_mids.append(s_sq_mid)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUTDIR / "fig_empirical_partiality.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_empirical_partiality.png'}")
    plt.close(fig)

    # ── Fig 2: σ(d) and γ(d) — resolution dependence of relp width ──
    fitted_sigmas = np.array(fitted_sigmas)
    fitted_gammas = np.array(fitted_gammas)
    shell_d_mids = np.array(shell_d_mids)
    shell_s_sq_mids = np.array(shell_s_sq_mids)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Resolution dependence of relp width", fontsize=12, fontweight="bold")

    # vs d
    ax = axes[0]
    finite_g = np.isfinite(fitted_sigmas)
    finite_l = np.isfinite(fitted_gammas)
    if finite_g.any():
        ax.plot(shell_d_mids[finite_g], fitted_sigmas[finite_g] * 1e3,
                "o-", color="#D85A30", lw=2, ms=6, label="Gaussian σ")
    if finite_l.any():
        ax.plot(shell_d_mids[finite_l], fitted_gammas[finite_l] * 1e3,
                "s-", color="#1D9E75", lw=2, ms=6, label="Lorentzian γ")
    ax.set_xlabel("d-spacing (Å)")
    ax.set_ylabel("Width (×10⁻³ Å⁻¹)")
    ax.set_title("Relp width vs resolution")
    ax.legend()

    # vs s² = 1/(4d²) — should be linear if σ² = σ₀² + B_eff × s²
    ax = axes[1]
    if finite_g.any():
        ax.plot(shell_s_sq_mids[finite_g], (fitted_sigmas[finite_g] * 1e3)**2,
                "o-", color="#D85A30", lw=2, ms=6, label="σ²")
        # Fit σ² = σ₀² + B_eff × s²
        try:
            s2 = shell_s_sq_mids[finite_g]
            sig2 = (fitted_sigmas[finite_g])**2
            # Linear fit: sig2 = a + b*s2
            coeffs = np.polyfit(s2, sig2, 1)
            B_eff = coeffs[0]
            sigma0_sq = coeffs[1]
            sigma0 = np.sqrt(max(sigma0_sq, 0))
            s2_fine = np.linspace(0, s2.max() * 1.1, 100)
            ax.plot(s2_fine, (np.sqrt(sigma0_sq + B_eff * s2_fine) * 1e3)**2,
                    "--", color="#D85A30", lw=1.5, alpha=0.7,
                    label=f"fit: σ₀={sigma0*1e3:.3f}, B={B_eff:.2e}")
        except Exception:
            pass
    ax.set_xlabel("s² = (sinθ/λ)² (Å⁻²)")
    ax.set_ylabel("σ² (×10⁻⁶ Å⁻²)")
    ax.set_title("σ² vs s² (should be linear for disorder model)")
    ax.legend(fontsize=9)

    # vs 1/d
    ax = axes[2]
    inv_d = 1.0 / shell_d_mids
    if finite_g.any():
        ax.plot(inv_d[finite_g], fitted_sigmas[finite_g] * 1e3,
                "o-", color="#D85A30", lw=2, ms=6, label="σ")
    if finite_l.any():
        ax.plot(inv_d[finite_l], fitted_gammas[finite_l] * 1e3,
                "s-", color="#1D9E75", lw=2, ms=6, label="γ")
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("Width (×10⁻³ Å⁻¹)")
    ax.set_title("Relp width vs 1/d")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig_relp_width_vs_resolution.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_relp_width_vs_resolution.png'}")
    plt.close(fig)

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("EMPIRICAL PARTIALITY MODEL")
    print("=" * 70)
    print(f"  {'Shell':>12} {'d_mid':>6} {'1/d':>6} {'σ_G×10³':>8} {'γ_L×10³':>8}")
    print("  " + "─" * 55)
    for i, (d_lo, d_hi, lbl) in enumerate(d_shells):
        if i < len(fitted_sigmas):
            sg = f"{fitted_sigmas[i]*1e3:.4f}" if np.isfinite(fitted_sigmas[i]) else "N/A"
            gl = f"{fitted_gammas[i]*1e3:.4f}" if np.isfinite(fitted_gammas[i]) else "N/A"
            print(f"  {lbl:>12} {shell_d_mids[i]:6.2f} {1/shell_d_mids[i]:6.3f} {sg:>8} {gl:>8}")

    if finite_g.sum() >= 3:
        s2 = shell_s_sq_mids[finite_g]
        sig2 = fitted_sigmas[finite_g]**2
        coeffs = np.polyfit(s2, sig2, 1)
        B_eff = coeffs[0]
        sigma0 = np.sqrt(max(coeffs[1], 0))
        print(f"\n  Disorder model fit: σ²(s) = σ₀² + B_eff × s²")
        print(f"    σ₀ = {sigma0*1e3:.4f} ×10⁻³ Å⁻¹")
        print(f"    B_eff = {B_eff:.4e} Å²")
        print(f"    σ₀ corresponds to relp FWHM = {sigma0 * 2.355 * 1e3:.4f} ×10⁻³ Å⁻¹")

    # Gaussian vs Lorentzian comparison
    if finite_g.any() and finite_l.any():
        print(f"\n  Gaussian σ / Lorentzian γ ratio:")
        both = finite_g & finite_l
        if both.any():
            ratio = fitted_sigmas[both] / fitted_gammas[both]
            print(f"    mean ratio = {np.mean(ratio):.3f}")
            print(f"    (ratio = 1.18 for equivalent FWHM)")

    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # Fig 3: Empirical per-bin correction table (model-free partiality)
    # ══════════════════════════════════════════════════════════════════

    print("\nBuilding empirical per-bin partiality correction table...")

    # Resolution shells × |r| bins → median log(I/I_merged) = empirical log(p)
    n_r_bins = 25
    r_edges = np.linspace(0, 0.007, n_r_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    correction_table = {}  # (shell_idx, r_bin_idx) → median log correction
    shell_edges = [0.9, 1.1, 1.4, 1.8, 2.5, 4.0, 10.0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Empirical partiality correction table (model-free)\n"
        "median log(I_corr / I_merged) per (resolution, |r|) bin",
        fontsize=13, fontweight="bold",
    )

    all_shell_corrections = []

    for ax_i, (d_lo, d_hi) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        ax = axes.flat[ax_i]
        mask = valid & (d >= d_lo) & (d < d_hi)
        n = mask.sum()
        if n < 200:
            ax.set_title(f"d = {d_lo}–{d_hi} Å (N={n})")
            all_shell_corrections.append(np.full(n_r_bins, np.nan))
            continue

        lc = log_corr[mask]
        ar = abs_r[mask]

        med_corr = np.full(n_r_bins, np.nan)
        q25_corr = np.full(n_r_bins, np.nan)
        q75_corr = np.full(n_r_bins, np.nan)
        std_corr = np.full(n_r_bins, np.nan)
        n_per_bin = np.zeros(n_r_bins)

        for bi in range(n_r_bins):
            r_mask = (ar >= r_edges[bi]) & (ar < r_edges[bi + 1])
            vals = lc[r_mask]
            finite_vals = vals[np.isfinite(vals)]
            n_per_bin[bi] = len(finite_vals)
            if len(finite_vals) >= 20:
                med_corr[bi] = np.median(finite_vals)
                q25_corr[bi] = np.percentile(finite_vals, 25)
                q75_corr[bi] = np.percentile(finite_vals, 75)
                std_corr[bi] = np.std(finite_vals)

        all_shell_corrections.append(med_corr)
        good = np.isfinite(med_corr)

        # Overall σ for this shell (before binned correction)
        finite_lc = lc[np.isfinite(lc)]
        shell_overall_std = np.std(finite_lc) if len(finite_lc) > 10 else np.nan

        ax.fill_between(r_centers[good] * 1e3, q25_corr[good], q75_corr[good],
                         alpha=0.2, color="#534AB7")
        ax.plot(r_centers[good] * 1e3, med_corr[good], "o-",
                color="#534AB7", lw=2, ms=4)
        # Secondary axis: within-bin σ
        ax2 = ax.twinx()
        ax2.plot(r_centers[good] * 1e3, std_corr[good], "s-",
                 color="#D85A30", lw=1.5, ms=3, alpha=0.7)
        ax2.axhline(shell_overall_std, color="#D85A30", ls="--", lw=1, alpha=0.5)
        ax2.set_ylabel("within-bin σ", color="#D85A30", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#D85A30", labelsize=7)
        ax2.set_ylim(0, 3)

        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
        ax.set_ylabel("Empirical log(p)")
        ax.set_title(f"d = {d_lo}–{d_hi} Å  (N={n:,})\noverall σ={shell_overall_std:.2f}")
        ax.set_ylim(-2.5, 1.5)

        # Also show the parametric fits for comparison
        if good.sum() >= 3:
            r_fit = r_centers[good]
            y_fit = med_corr[good]
            try:
                popt, _ = curve_fit(exponential_model, r_fit, y_fit,
                                     p0=[0.5, 0.003], bounds=([-5, 1e-5], [5, 0.1]))
                ax.plot(np.linspace(0, 0.007, 200) * 1e3,
                        exponential_model(np.linspace(0, 0.007, 200), *popt),
                        "--", color="#D85A30", lw=1.5, alpha=0.7, label="exp fit")
            except Exception:
                pass

            try:
                popt, _ = curve_fit(gen_gaussian_model, r_fit, y_fit,
                                     p0=[0.5, 0.003, 1.0],
                                     bounds=([-5, 1e-5, 0.3], [5, 0.1, 3.0]))
                ax.plot(np.linspace(0, 0.007, 200) * 1e3,
                        gen_gaussian_model(np.linspace(0, 0.007, 200), *popt),
                        ":", color="#1D9E75", lw=2, label=f"GG β={popt[2]:.2f}")
            except Exception:
                pass
        ax.legend(fontsize=7)

    # Last panel: 2D heatmap of the correction table
    ax = axes.flat[-1]
    table = np.array(all_shell_corrections)  # (n_shells, n_r_bins)
    shell_labels = [f"{d_lo}–{d_hi}" for d_lo, d_hi in zip(shell_edges[:-1], shell_edges[1:])]
    im = ax.pcolormesh(r_edges * 1e3,
                        np.arange(len(shell_labels) + 1),
                        table, cmap="RdBu_r", vmin=-2, vmax=1, shading="flat")
    ax.set_yticks(np.arange(len(shell_labels)) + 0.5)
    ax.set_yticklabels(shell_labels, fontsize=9)
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("d-spacing (Å)")
    ax.set_title("Correction table heatmap")
    fig.colorbar(im, ax=ax, label="log(p) empirical")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUTDIR / "fig_empirical_correction_table.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_empirical_correction_table.png'}")
    plt.close(fig)

    # ── Fig 4: Variance tracking — does binning help? ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Variance tracking: within-bin σ vs overall σ\n"
        f"If within-bin σ ≈ overall σ, the binned correction adds nothing",
        fontsize=12, fontweight="bold",
    )

    # Collect per-shell variance stats
    shell_overall_stds = []
    shell_mean_within_stds = []
    shell_residual_stds = []

    for si, (d_lo, d_hi) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        mask = valid & (d >= d_lo) & (d < d_hi)
        lc = log_corr[mask]
        ar = abs_r[mask]
        finite_lc = lc[np.isfinite(lc)]

        if len(finite_lc) < 100:
            shell_overall_stds.append(np.nan)
            shell_mean_within_stds.append(np.nan)
            shell_residual_stds.append(np.nan)
            continue

        overall_std = np.std(finite_lc)
        shell_overall_stds.append(overall_std)

        # Within-bin σ (mean across bins)
        corr = all_shell_corrections[si]
        within_stds = []
        residuals_all = []
        for bi in range(n_r_bins):
            r_mask = (ar >= r_edges[bi]) & (ar < r_edges[bi + 1])
            vals = lc[r_mask]
            fin = vals[np.isfinite(vals)]
            if len(fin) >= 20 and np.isfinite(corr[bi]):
                within_stds.append(np.std(fin))
                residuals_all.append(fin - corr[bi])

        shell_mean_within_stds.append(np.mean(within_stds) if within_stds else np.nan)
        if residuals_all:
            all_resid = np.concatenate(residuals_all)
            shell_residual_stds.append(np.std(all_resid))
        else:
            shell_residual_stds.append(np.nan)

    shell_overall_stds = np.array(shell_overall_stds)
    shell_mean_within_stds = np.array(shell_mean_within_stds)
    shell_residual_stds = np.array(shell_residual_stds)
    shell_mids = np.array([0.5 * (a + b) for a, b in zip(shell_edges[:-1], shell_edges[1:])])

    ax = axes[0]
    fin = np.isfinite(shell_overall_stds)
    ax.plot(shell_mids[fin], shell_overall_stds[fin], "o-", color="#534AB7", lw=2, ms=6,
            label="Overall σ (no correction)")
    ax.plot(shell_mids[fin], shell_mean_within_stds[fin], "s-", color="#D85A30", lw=2, ms=6,
            label="Mean within-bin σ")
    ax.plot(shell_mids[fin], shell_residual_stds[fin], "^-", color="#1D9E75", lw=2, ms=6,
            label="Residual σ (after subtraction)")
    ax.axhline(1.56, color="red", ls="--", lw=1.5, alpha=0.5, label="observed σ_LN = 1.56")
    ax.set_xlabel("d-spacing (Å)")
    ax.set_ylabel("σ(log(I/I_merged))")
    ax.set_title("Variance per resolution shell")
    ax.legend(fontsize=8)

    ax = axes[1]
    # Variance reduction = 1 - (residual_var / overall_var)
    var_reduction = 1 - (shell_residual_stds**2 / shell_overall_stds**2)
    ax.plot(shell_mids[fin], var_reduction[fin] * 100, "o-", color="#534AB7", lw=2, ms=6)
    ax.set_xlabel("d-spacing (Å)")
    ax.set_ylabel("Variance reduction (%)")
    ax.set_title("% variance explained by |r| binning")
    ax.axhline(0, color="gray", ls=":", lw=1)

    ax = axes[2]
    # Within-bin σ vs |r| for one representative shell
    rep_si = 3  # d = 1.8–2.5
    d_lo, d_hi = shell_edges[rep_si], shell_edges[rep_si + 1]
    mask = valid & (d >= d_lo) & (d < d_hi)
    lc = log_corr[mask]
    ar = abs_r[mask]
    bin_stds = []
    bin_rs = []
    for bi in range(n_r_bins):
        r_mask = (ar >= r_edges[bi]) & (ar < r_edges[bi + 1])
        vals = lc[r_mask]
        fin_v = vals[np.isfinite(vals)]
        if len(fin_v) >= 20:
            bin_stds.append(np.std(fin_v))
            bin_rs.append(r_centers[bi])
    ax.plot(np.array(bin_rs) * 1e3, bin_stds, "o-", color="#534AB7", lw=2, ms=5)
    overall_this = np.std(lc[np.isfinite(lc)])
    ax.axhline(overall_this, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"overall σ = {overall_this:.2f}")
    ax.axhline(1.56, color="gray", ls=":", lw=1, alpha=0.5, label="σ_LN = 1.56")
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Within-bin σ")
    ax.set_title(f"Within-bin σ vs |r| (d = {d_lo}–{d_hi} Å)")
    ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUTDIR / "fig_variance_tracking.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_variance_tracking.png'}")
    plt.close(fig)

    # Print variance summary
    print("\n  Variance tracking:")
    print(f"  {'Shell':>10} {'overall_σ':>10} {'within_σ':>10} {'resid_σ':>10} {'%reduced':>10}")
    for si, (d_lo, d_hi) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        lbl = f"{d_lo}–{d_hi}"
        ov = shell_overall_stds[si]
        wi = shell_mean_within_stds[si]
        re = shell_residual_stds[si]
        if np.isfinite(ov) and np.isfinite(re):
            pct = (1 - re**2 / ov**2) * 100
            print(f"  {lbl:>10} {ov:10.3f} {wi:10.3f} {re:10.3f} {pct:9.1f}%")
        else:
            print(f"  {lbl:>10}    N/A")

    # Save the correction table
    table_data = {
        "r_edges": r_edges,
        "r_centers": r_centers,
        "shell_edges": np.array(shell_edges),
        "correction_table": table,  # (n_shells, n_r_bins) median log(p)
    }
    np.savez(OUTDIR / "empirical_correction_table.npz", **table_data)

    # Print table
    print("\n  Empirical correction table (median log(p)):")
    print(f"  {'|r|×10³':>8}", end="")
    for lbl in shell_labels:
        print(f"  {lbl:>8}", end="")
    print()
    print("  " + "─" * (10 + 10 * len(shell_labels)))
    for bi in range(0, n_r_bins, 3):
        print(f"  {r_centers[bi]*1e3:8.3f}", end="")
        for si in range(len(shell_labels)):
            v = table[si, bi]
            print(f"  {v:+8.3f}" if np.isfinite(v) else f"  {'N/A':>8}", end="")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
