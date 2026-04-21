"""Before/after diagnostics for rank-concordance refinement."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, binned_statistic
from pathlib import Path

from partiality._excitation import ExcitationErrorModel
from partiality._data import PairData
from partiality._refine import RefinementResult


def evaluate_concordance(
    abs_r: np.ndarray,
    I_corr: np.ndarray,
    hkl_id: np.ndarray,
    crystal_id: np.ndarray,
    I_threshold: float = 50.0,
    min_mult: int = 10,
) -> dict:
    """Compute per-HKL and per-crystal Spearman ρ(I, |r|)."""
    n_unique = hkl_id.max() + 1
    order = np.argsort(hkl_id)
    boundaries = np.searchsorted(hkl_id[order], np.arange(n_unique + 1))

    rhos = []
    for uid in range(n_unique):
        lo, hi = boundaries[uid], boundaries[uid + 1]
        idx = order[lo:hi]
        I_h = I_corr[idx]
        r_h = abs_r[idx]
        keep = I_h > I_threshold
        if keep.sum() < min_mult:
            continue
        I_f, r_f = I_h[keep], r_h[keep]
        if np.ptp(I_f) < 1e-10 or np.ptp(r_f) < 1e-10:
            continue
        rho, _ = spearmanr(I_f, r_f)
        rhos.append(rho)

    rhos = np.array(rhos)
    return {
        "median_rho": np.median(rhos) if len(rhos) > 0 else np.nan,
        "mean_rho": np.mean(rhos) if len(rhos) > 0 else np.nan,
        "frac_negative": (rhos < 0).mean() if len(rhos) > 0 else np.nan,
        "n_hkl": len(rhos),
        "rhos": rhos,
    }


def plot_diagnostics(
    result: RefinementResult,
    pair_data: PairData,
    abs_r_before: np.ndarray,
    abs_r_after: np.ndarray,
    output_dir: str | Path,
):
    """Generate diagnostic figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    I_corr = pair_data.I_corr.cpu().numpy()
    hkl_id = pair_data.hkl_id.cpu().numpy()
    s_sq = pair_data.s_sq.cpu().numpy()
    d = 1.0 / (2 * np.sqrt(s_sq.clip(min=1e-10)))

    # ── Fig 1: Convergence ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Rank concordance refinement convergence", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(result.loss_history, color="#534AB7", lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss convergence")

    ax = axes[1]
    steps = np.linspace(0, len(result.loss_history) - 1, len(result.concordance_history))
    ax.plot(steps, result.concordance_history, "o-", color="#1D9E75", lw=2, ms=5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Concordance fraction")
    ax.set_title("Hard concordance")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "fig_convergence.png", dpi=150)
    plt.close(fig)

    # ── Fig 2: Before/after ρ distributions ──
    stats_before = evaluate_concordance(abs_r_before, I_corr, hkl_id, pair_data.crystal_id.cpu().numpy())
    stats_after = evaluate_concordance(abs_r_after, I_corr, hkl_id, pair_data.crystal_id.cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Per-HKL ρ(I, |r|) before and after refinement", fontsize=12, fontweight="bold")

    ax = axes[0]
    if len(stats_before["rhos"]) > 0:
        ax.hist(stats_before["rhos"], bins=60, range=(-1, 0.5), density=True,
                alpha=0.5, color="#534AB7", label=f"before (med={stats_before['median_rho']:.3f})")
    if len(stats_after["rhos"]) > 0:
        ax.hist(stats_after["rhos"], bins=60, range=(-1, 0.5), density=True,
                alpha=0.5, color="#D85A30", label=f"after (med={stats_after['median_rho']:.3f})")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Spearman ρ")
    ax.legend()
    ax.set_title("Per-HKL ρ distribution")

    ax = axes[1]
    # Resolution-binned ρ
    bins_d = np.linspace(0.05, 1.15, 15)
    inv_d = 1.0 / d
    for abs_r_arr, label, col in [
        (abs_r_before, "before", "#534AB7"),
        (abs_r_after, "after", "#D85A30"),
    ]:
        # Global binned ρ: for each 1/d bin, compute ρ(I, |r|) over all obs
        meds = []
        for i in range(len(bins_d) - 1):
            mask = (inv_d >= bins_d[i]) & (inv_d < bins_d[i + 1]) & (I_corr > 50)
            if mask.sum() < 100:
                meds.append(np.nan)
                continue
            rho, _ = spearmanr(I_corr[mask], abs_r_arr[mask])
            meds.append(rho)
        centers = 0.5 * (bins_d[:-1] + bins_d[1:])
        ax.plot(centers, meds, "o-", color=col, lw=2, ms=4, label=label)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("1/d (Å⁻¹)")
    ax.set_ylabel("Spearman ρ(I, |r|)")
    ax.set_title("Resolution-binned global ρ")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "fig_rho_before_after.png", dpi=150)
    plt.close(fig)

    # ── Fig 3: Parameter distributions ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Refined parameters", fontsize=12, fontweight="bold")

    omega = result.delta_omega_xy.numpy()
    omega_mag = np.sqrt((omega ** 2).sum(axis=1))

    ax = axes[0]
    omega_mdeg = np.degrees(omega_mag) * 1000
    ax.hist(omega_mdeg, bins=80, color="#534AB7", alpha=0.7)
    ax.set_xlabel("|δω| (millidegrees)")
    ax.set_title(f"Orientation correction\nmedian = {np.median(omega_mdeg):.1f} mdeg")
    # Mark potential bimodality
    from scipy.signal import find_peaks
    h, edges = np.histogram(omega_mdeg, bins=80)
    centers_h = 0.5 * (edges[:-1] + edges[1:])
    peaks, _ = find_peaks(h, height=h.max() * 0.3, distance=5)
    for pk in peaks:
        ax.axvline(centers_h[pk], color="red", ls="--", lw=1, alpha=0.7)

    ax = axes[1]
    dlam = result.delta_lambda.numpy()
    ax.hist(dlam * 1e4, bins=60, color="#D85A30", alpha=0.7)
    ax.set_xlabel("δλ (×10⁻⁴ Å)")
    ax.set_title(f"Wavelength correction\nmedian = {np.median(np.abs(dlam))*1e4:.2f}×10⁻⁴ Å")

    ax = axes[2]
    ax.scatter(np.degrees(omega[:, 0]) * 1000, np.degrees(omega[:, 1]) * 1000,
               s=1, alpha=0.2, color="#534AB7", rasterized=True)
    ax.set_xlabel("δωx (mdeg)")
    ax.set_ylabel("δωy (mdeg)")
    ax.set_title("Orientation corrections (2D)")
    ax.set_aspect("equal")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_dir / "fig_parameters.png", dpi=150)
    plt.close(fig)

    # ── Fig 4: |r| distribution shift ──
    fig, ax = plt.subplots(figsize=(8, 5))
    strong = I_corr > 50
    ax.hist(abs_r_before[strong] * 1e3, bins=100, range=(0, 7), density=True,
            alpha=0.5, color="#534AB7", label="before")
    ax.hist(abs_r_after[strong] * 1e3, bins=100, range=(0, 7), density=True,
            alpha=0.5, color="#D85A30", label="after")
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Density")
    ax.set_title("Excitation error distribution (strong reflections)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_r_distribution.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to {output_dir}/")
    return stats_before, stats_after
