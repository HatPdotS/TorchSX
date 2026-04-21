#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""Scan restraint strength to find optimal regularisation."""

import sys
from pathlib import Path
import torch
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from partiality._data import load_and_prepare
from partiality._excitation import ExcitationErrorModel
from partiality._refine import refine_orientations, RefinementConfig

OUTDIR = Path(__file__).resolve().parent / "results"
OUTDIR.mkdir(exist_ok=True)


def per_hkl_rho(model, core_hkl, core_I, core_crystal_id, hkl_id, n_unique,
                I_threshold=50.0, min_mult=10):
    """Compute per-HKL ρ(I, |r|) — the GLOBAL ranking metric."""
    with torch.no_grad():
        abs_r = model.abs_excitation_error(core_hkl, core_crystal_id).numpy()
    I = core_I.numpy()
    hid = hkl_id

    order = np.argsort(hid)
    boundaries = np.searchsorted(hid[order], np.arange(n_unique + 1))

    rhos = []
    for uid in range(n_unique):
        lo, hi = boundaries[uid], boundaries[uid + 1]
        idx = order[lo:hi]
        I_h = I[idx]
        r_h = abs_r[idx]
        keep = I_h > I_threshold
        if keep.sum() < min_mult:
            continue
        I_f, r_f = I_h[keep], r_h[keep]
        if np.ptp(I_f) < 1e-10 or np.ptp(r_f) < 1e-10:
            continue
        rho, _ = spearmanr(I_f, r_f)
        rhos.append(rho)
    return np.array(rhos)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stream_dir = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"
    sfx_paths = [
        stream_dir / "delay_-91.sfx",
        stream_dir / "delay_-136.sfx",
        stream_dir / "delay_-182.sfx",
        stream_dir / "delay_-227.sfx",
    ]
    device = "cpu"

    pair_data = load_and_prepare(sfx_paths, device=device, I_threshold=20.0, min_mult=3)

    # Precompute merged I for ρ evaluation
    I = pair_data.I_corr.numpy()
    hkl_id = pair_data.hkl_id.numpy()
    n_unique = hkl_id.max() + 1
    I_sum = np.zeros(n_unique)
    I_cnt = np.zeros(n_unique)
    pos = I > 0
    np.add.at(I_sum, hkl_id[pos], I[pos])
    np.add.at(I_cnt, hkl_id[pos], 1)
    I_merged = I_sum / np.clip(I_cnt, 1, None)
    I_merged_obs = I_merged[hkl_id]

    n_unique = hkl_id.max() + 1

    # Baseline (no refinement)
    model_init = ExcitationErrorModel(
        pair_data.n_crystals, pair_data.UB, pair_data.lambda_0)
    rho_baseline = per_hkl_rho(
        model_init, pair_data.hkl_obs, pair_data.I_corr,
        pair_data.crystal_id, hkl_id, n_unique)
    print(f"Baseline: median per-HKL ρ = {np.median(rho_baseline):.4f} "
          f"({len(rho_baseline)} HKLs, {(rho_baseline < 0).mean()*100:.1f}% negative)")

    # Scan restraint strengths — extended range
    restraint_values = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

    results = []
    for lam_omega in restraint_values:
        print(f"\n{'='*60}")
        print(f"Restraint λ_ω = {lam_omega}")
        config = RefinementConfig(
            n_steps=200,
            lr_omega=1e-4,
            lr_lambda=1e-5,
            beta_start=5.0,
            beta_end=25.0,
            restraint_omega=lam_omega,
            restraint_lambda=2e4,  # keep wavelength frozen for this scan
            log_every=100,
            device=device,
        )
        result = refine_orientations(pair_data, config)

        omega_xy = result.delta_omega_xy.numpy()
        omega_mag = np.sqrt((omega_xy ** 2).sum(axis=1))
        omega_mdeg = np.degrees(omega_mag) * 1000

        rho_after_hkl = per_hkl_rho(
            result.model, pair_data.hkl_obs, pair_data.I_corr,
            pair_data.crystal_id, hkl_id, n_unique)

        # Per-HKL improvement
        # Match HKLs between baseline and after (both use same qualifying set)
        n_hkl = min(len(rho_baseline), len(rho_after_hkl))
        delta_rho_hkl = rho_after_hkl[:n_hkl] - rho_baseline[:n_hkl]
        improved_hkl = (delta_rho_hkl < 0).mean() * 100

        med_rho = np.median(rho_after_hkl)
        med_omega = np.median(omega_mdeg)

        # Also compute global Spearman
        with torch.no_grad():
            abs_r_after = result.model.abs_excitation_error(
                pair_data.hkl_obs, pair_data.crystal_id).numpy()
        strong = I > 50
        if strong.sum() > 1000:
            global_rho, _ = spearmanr(I[strong], abs_r_after[strong])
        else:
            global_rho = np.nan

        r = {
            "lam": lam_omega,
            "med_rho_hkl": med_rho,
            "delta_rho_hkl_med": np.median(delta_rho_hkl),
            "improved_hkl_pct": improved_hkl,
            "frac_neg": (rho_after_hkl < 0).mean() * 100,
            "global_rho": global_rho,
            "med_omega_mdeg": med_omega,
            "concordance": result.concordance_history[-1] if result.concordance_history else np.nan,
            "omega_mdeg": omega_mdeg,
        }
        results.append(r)

        print(f"  per-HKL median ρ = {med_rho:.4f}, Δρ = {r['delta_rho_hkl_med']:.4f}, "
              f"improved = {improved_hkl:.0f}%, neg = {r['frac_neg']:.0f}%, "
              f"global ρ = {global_rho:.4f}, "
              f"|δω| med = {med_omega:.1f} mdeg")

    # ── Figure: restraint scan ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Restraint strength scan: λ_ω × ||δω||²",
                 fontsize=13, fontweight="bold")

    lams = [r["lam"] for r in results]
    baseline_med = np.median(rho_baseline)

    ax = axes[0, 0]
    ax.plot(lams, [r["med_rho_hkl"] for r in results], "o-", color="#534AB7", lw=2, ms=6)
    ax.axhline(baseline_med, color="red", ls="--", lw=1.5,
               label=f"baseline = {baseline_med:.4f}")
    ax.set_xlabel("λ_ω (restraint strength)")
    ax.set_ylabel("Median per-HKL ρ")
    ax.set_title("Per-HKL ρ(I, |r|)")
    ax.set_xscale("log")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(lams, [r["delta_rho_hkl_med"] for r in results], "o-", color="#D85A30", lw=2, ms=6)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("λ_ω")
    ax.set_ylabel("Median Δρ (per-HKL)")
    ax.set_title("Per-HKL ρ change")
    ax.set_xscale("log")

    ax = axes[0, 2]
    ax.plot(lams, [r["improved_hkl_pct"] for r in results], "o-", color="#1D9E75", lw=2, ms=6)
    ax.plot(lams, [r["frac_neg"] for r in results], "s--", color="#534AB7", lw=1.5, ms=5, label="% ρ<0")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.axhline((rho_baseline < 0).mean() * 100, color="red", ls="--", lw=1, alpha=0.5, label="baseline %neg")
    ax.set_xlabel("λ_ω")
    ax.set_ylabel("%")
    ax.set_title("HKLs improved / negative")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(lams, [r["med_omega_mdeg"] for r in results], "o-", color="#534AB7", lw=2, ms=6)
    ax.set_xlabel("λ_ω")
    ax.set_ylabel("Median |δω| (mdeg)")
    ax.set_title("Correction magnitude")
    ax.set_xscale("log")

    ax = axes[1, 1]
    ax.plot(lams, [r["global_rho"] for r in results], "o-", color="#D85A30", lw=2, ms=6)
    with torch.no_grad():
        abs_r_init = model_init.abs_excitation_error(
            pair_data.hkl_obs, pair_data.crystal_id).numpy()
    strong = I > 50
    global_rho_base = spearmanr(I[strong], abs_r_init[strong])[0] if strong.sum() > 100 else np.nan
    ax.axhline(global_rho_base, color="red", ls="--", lw=1.5,
               label=f"baseline = {global_rho_base:.4f}")
    ax.set_xlabel("λ_ω")
    ax.set_ylabel("Global Spearman ρ(I, |r|)")
    ax.set_title("Global ρ")
    ax.set_xscale("log")
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    ax.plot(lams, [r["concordance"] for r in results], "o-", color="#1D9E75", lw=2, ms=6)
    ax.set_xlabel("λ_ω")
    ax.set_ylabel("Concordance fraction")
    ax.set_title("Hard concordance")
    ax.set_xscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / "fig_restraint_scan.png", dpi=150)
    print(f"\n  → {OUTDIR / 'fig_restraint_scan.png'}")
    plt.close(fig)

    # Summary table
    print("\n" + "=" * 85)
    print("RESTRAINT SCAN SUMMARY")
    print("=" * 85)
    print(f"{'λ_ω':>8} {'HKL_ρ':>7} {'Δρ':>7} {'%imp':>5} {'%neg':>5} {'glob_ρ':>7} {'|δω|':>7} {'conc':>6}")
    print("─" * 85)
    for r in results:
        print(f"{r['lam']:8.1f} {r['med_rho_hkl']:+7.4f} {r['delta_rho_hkl_med']:+7.4f} "
              f"{r['improved_hkl_pct']:5.0f}% {r['frac_neg']:5.0f}% "
              f"{r['global_rho']:+7.4f} {r['med_omega_mdeg']:7.1f} {r['concordance']:6.4f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
