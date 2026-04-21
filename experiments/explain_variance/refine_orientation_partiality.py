#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Refine crystal orientation by maximising the partiality–intensity correlation.

For each crystal, find a small rotation δω = (ωx, ωy, ωz) that minimises:

    L = Σ_j  w_j × r_j(ω)²

where w_j = I_j / I_merged(h_j) weights bright observations more heavily,
pulling them onto the Ewald sphere.

Linear approximation: r(ω) ≈ r₀ + J·ω  where  J_ji = ∂r_j/∂ω_i.
Solution: ω* = -(Jᵀ W J)⁻¹ Jᵀ W r₀   (3×3 solve per crystal).

The Jacobian:  ∂r/∂ωi = k̂_f · (ê_i × q)
where k_f = q + s₀, s₀ = (0, 0, 1/λ), k̂_f = k_f/|k_f|.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, binned_statistic
import os, sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

STREAM_DIR = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"


def compute_r_and_jacobian(hkl, astar, bstar, cstar, crystal_id, wavelength):
    """
    Compute excitation error and its Jacobian w.r.t. small rotations.

    Returns
    -------
    r : (N,) excitation error
    J : (N, 3) Jacobian dr/d(ωx, ωy, ωz)
    q_lab : (N, 3) scattering vector
    """
    h = hkl[:, 0].float()
    k = hkl[:, 1].float()
    l = hkl[:, 2].float()
    cid = crystal_id.long()

    q = (h.unsqueeze(1) * astar[cid]
         + k.unsqueeze(1) * bstar[cid]
         + l.unsqueeze(1) * cstar[cid])

    k0 = 1.0 / wavelength
    s0 = torch.zeros_like(q)
    s0[:, 2] = k0

    kf = q + s0
    kf_norm = kf.norm(dim=1, keepdim=True).clamp(min=1e-10)
    kf_hat = kf / kf_norm  # (N, 3)

    r = kf_norm.squeeze(1) - k0  # (N,)

    # Jacobian: dr/dωi = k̂_f · (ê_i × q)
    # ê_x × q = (0, -qz, qy)
    # ê_y × q = (qz, 0, -qx)
    # ê_z × q = (-qy, qx, 0)
    J = torch.zeros(q.shape[0], 3)
    J[:, 0] = kf_hat[:, 1] * (-q[:, 2]) + kf_hat[:, 2] * q[:, 1]   # k̂·(ê_x × q)
    J[:, 1] = kf_hat[:, 0] * q[:, 2] + kf_hat[:, 2] * (-q[:, 0])   # k̂·(ê_y × q)
    J[:, 2] = kf_hat[:, 0] * (-q[:, 1]) + kf_hat[:, 1] * q[:, 0]   # k̂·(ê_z × q)

    return r, J, q


def refine_one_crystal(r, J, w):
    """
    Solve for ωx, ωy only (rotation around z doesn't change |r|).

    ω* = -(Jᵀ W J)⁻¹ Jᵀ W r₀   using only x,y columns of J.

    r : (n,) excitation errors
    J : (n, 3) Jacobian (but we only use columns 0,1)
    w : (n,) weights (I/I_merged)

    Returns ω (3,) with ωz=0, and r_corrected (n,)
    """
    J2 = J[:, :2]  # (n, 2) — only ωx, ωy
    Jw = J2 * w.unsqueeze(1)
    A = Jw.T @ J2   # (2, 2)
    b = Jw.T @ r     # (2,)

    try:
        omega2 = -torch.linalg.solve(A, b)  # (2,)
    except Exception:
        omega2 = torch.zeros(2)

    omega = torch.zeros(3)
    omega[:2] = omega2
    r_corrected = r + J @ omega
    return omega, r_corrected


def main():
    from torchsx.io._safetensors import load_sfx
    from torchsx.merge._hkl import compute_hkl_index

    print("Loading stream...")
    core = load_sfx(str(STREAM_DIR / "delay_-91.sfx"))
    N = core.hkl.shape[0]
    n_cryst = core.astar.shape[0]
    print(f"  {N:,} obs, {n_cryst:,} crystals")

    # Compute r and Jacobian
    print("Computing excitation error + Jacobian...")
    wavelength = 0.9567
    r, J, q_lab = compute_r_and_jacobian(
        core.hkl, core.astar, core.bstar, core.cstar,
        core.crystal_id, wavelength,
    )
    abs_r = r.abs()

    # HKL indexing → merged I
    print("Computing merged intensities...")
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id = hkl_idx.hkl_id
    I = core.I
    cid = core.crystal_id

    n_unique = hkl_idx.n_unique
    I_sum = torch.zeros(n_unique)
    I_count = torch.zeros(n_unique)
    pos = I > 0
    I_sum.scatter_add_(0, hkl_id[pos], I[pos])
    I_count.scatter_add_(0, hkl_id[pos], torch.ones_like(I[pos]))
    I_merged = I_sum / I_count.clamp(min=1)
    I_merged_obs = I_merged[hkl_id]

    # Weights: I / I_merged — all observations contribute, but bright ones more
    w = (I / I_merged_obs.clamp(min=1)).clamp(min=0, max=20)
    # Zero out negative-I observations only
    w[I <= 0] = 0.0

    # Refine per crystal
    print(f"Refining {n_cryst:,} crystals...")
    r_corrected = r.clone()
    omegas = torch.zeros(n_cryst, 3)
    rho_before = []
    rho_after = []
    n_obs_per = []
    delta_r_rms = []

    # Sort by crystal_id for fast slicing
    order = cid.argsort()
    boundaries = torch.searchsorted(cid[order], torch.arange(n_cryst + 1))

    for ci in range(n_cryst):
        lo, hi = boundaries[ci].item(), boundaries[ci + 1].item()
        idx = order[lo:hi]
        n_obs = hi - lo

        if n_obs < 20:
            continue

        r_c = r[idx]
        J_c = J[idx]
        w_c = w[idx]
        I_c = I[idx]
        Im_c = I_merged_obs[idx]

        # Need some observations with non-zero weight
        if (w_c > 0.1).sum() < 3:
            continue

        omega, r_new = refine_one_crystal(r_c, J_c, w_c)

        # Clip unreasonable corrections (> 2° = 0.035 rad)
        if omega.norm() > 0.035:
            continue

        omegas[ci] = omega
        r_corrected[idx] = r_new

        # Compute correlation before/after for strong obs
        valid = (I_c > 20) & (Im_c > 20)
        if valid.sum() < 10:
            continue

        lc = torch.log(I_c[valid] / Im_c[valid]).numpy()
        r_before_v = r_c[valid].abs().numpy()
        r_after_v = r_new[valid].abs().numpy()

        if np.std(lc) < 1e-10 or np.std(r_before_v) < 1e-10:
            continue

        rho_b, _ = spearmanr(lc, r_before_v)
        rho_a, _ = spearmanr(lc, r_after_v)
        rho_before.append(rho_b)
        rho_after.append(rho_a)
        n_obs_per.append(valid.sum().item())
        delta_r_rms.append(np.sqrt(np.mean((r_after_v - r_before_v)**2)))

    rho_before = np.array(rho_before)
    rho_after = np.array(rho_after)
    n_obs_per = np.array(n_obs_per)
    delta_r_rms = np.array(delta_r_rms)
    omega_norms = omegas.norm(dim=1).numpy()

    print(f"  {len(rho_before):,} crystals refined")
    print(f"  ρ before: median = {np.median(rho_before):.4f}")
    print(f"  ρ after:  median = {np.median(rho_after):.4f}")
    print(f"  Δρ:       median = {np.median(rho_after - rho_before):.4f}")

    # ── Figures ────────────────────────────────────────────────

    # Fig 1: Before vs after correlation
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Orientation refinement: maximise ρ(log(I/I_merged), |r|)\n"
        f"{len(rho_before):,} crystals, weighted r² minimisation",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.hist(rho_before, bins=60, range=(-1, 0.5), density=True,
            alpha=0.5, color="#534AB7", label=f"before (med={np.median(rho_before):.3f})")
    ax.hist(rho_after, bins=60, range=(-1, 0.5), density=True,
            alpha=0.5, color="#D85A30", label=f"after (med={np.median(rho_after):.3f})")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Spearman ρ(log_corr, |r|)")
    ax.set_ylabel("Density")
    ax.set_title("Per-crystal ρ distribution")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(rho_before, rho_after, s=3, alpha=0.2, color="#534AB7", rasterized=True)
    ax.plot([-1, 0.5], [-1, 0.5], "k:", lw=1)
    ax.set_xlabel("ρ before")
    ax.set_ylabel("ρ after")
    ax.set_title("Per-crystal improvement")
    ax.set_xlim(-1, 0.5)
    ax.set_ylim(-1, 0.5)
    improved = (rho_after < rho_before).mean() * 100
    ax.text(0.05, 0.95, f"{improved:.0f}% improved", transform=ax.transAxes,
            fontsize=11, va="top")

    ax = axes[1, 0]
    delta_rho = rho_after - rho_before
    ax.hist(delta_rho, bins=80, range=(-0.5, 0.3), density=True,
            color="#1D9E75", alpha=0.7)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.axvline(np.median(delta_rho), color="red", ls="--", lw=2,
               label=f"median Δρ = {np.median(delta_rho):.4f}")
    ax.set_xlabel("Δρ (after − before)")
    ax.set_ylabel("Density")
    ax.set_title("Change in correlation")
    ax.legend()

    ax = axes[1, 1]
    valid_omega = omega_norms[omega_norms > 0]
    ax.hist(np.degrees(valid_omega[valid_omega < 0.01]) * 1000,
            bins=60, color="#B5338A", alpha=0.7)
    ax.set_xlabel("Rotation correction |δω| (millidegrees)")
    ax.set_ylabel("Count")
    ax.set_title(f"Correction magnitude\nmedian = {np.median(np.degrees(valid_omega[valid_omega < 0.01])*1000):.1f} mdeg")

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUTDIR / "fig_orientation_refinement.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_orientation_refinement.png'}")
    plt.close(fig)

    # Fig 2: Global log_corr vs |r| before and after
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("log(I/I_merged) vs |r| — before and after refinement",
                 fontsize=12, fontweight="bold")

    valid_global = (I > 50) & (I_merged_obs > 50)
    lc_global = torch.log(I[valid_global] / I_merged_obs[valid_global]).numpy()

    bins_r = np.linspace(0, 0.007, 40)

    for ax, (r_vals, label) in zip(axes, [
        (abs_r[valid_global].numpy(), "Before refinement"),
        (r_corrected[valid_global].abs().numpy(), "After refinement"),
    ]):
        med, edges, _ = binned_statistic(r_vals, lc_global, statistic="median", bins=bins_r)
        std, _, _ = binned_statistic(r_vals, lc_global, statistic="std", bins=bins_r)
        c = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(c * 1e3, med - std, med + std, alpha=0.15, color="#534AB7")
        ax.plot(c * 1e3, med, "o-", color="#534AB7", lw=2, ms=4)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
        ax.set_ylabel("median log(I/I_merged)")

        rho_g, _ = spearmanr(lc_global, r_vals)
        ax.set_title(f"{label}\nglobal ρ = {rho_g:.4f}")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUTDIR / "fig_refinement_logcorr.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_refinement_logcorr.png'}")
    plt.close(fig)

    # Fig 3: |r| distribution before/after
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("|r| distribution before and after refinement", fontsize=12, fontweight="bold")

    strong = valid_global
    ax = axes[0]
    ax.hist(abs_r[strong].numpy() * 1e3, bins=100, range=(0, 7), density=True,
            alpha=0.5, color="#534AB7", label="before")
    ax.hist(r_corrected[strong].abs().numpy() * 1e3, bins=100, range=(0, 7), density=True,
            alpha=0.5, color="#D85A30", label="after")
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Density")
    ax.set_title("Strong reflections (I > 50)")
    ax.legend()

    ax = axes[1]
    # CDF comparison
    for r_vals, col, label in [
        (abs_r[strong].numpy(), "#534AB7", "before"),
        (r_corrected[strong].abs().numpy(), "#D85A30", "after"),
    ]:
        sr = np.sort(r_vals)
        cdf = np.arange(1, len(sr) + 1) / len(sr)
        idx = np.linspace(0, len(sr) - 1, 2000).astype(int)
        ax.plot(sr[idx] * 1e3, cdf[idx], color=col, lw=2, label=label)
    ax.set_xlabel("|r| (×10⁻³ Å⁻¹)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of |r|")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig_refinement_r_distribution.png", dpi=150)
    print(f"  → {OUTDIR / 'fig_refinement_r_distribution.png'}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 65)
    print("ORIENTATION REFINEMENT SUMMARY")
    print("=" * 65)
    print(f"  Crystals refined:   {len(rho_before):,}")
    print(f"  ρ before:  median = {np.median(rho_before):.4f}")
    print(f"  ρ after:   median = {np.median(rho_after):.4f}")
    print(f"  Δρ:        median = {np.median(delta_rho):.4f}")
    print(f"  Improved:  {improved:.0f}% of crystals")
    valid_om = omega_norms[(omega_norms > 0) & (omega_norms < 0.01)]
    print(f"  |δω|:      median = {np.median(np.degrees(valid_om)*1000):.1f} millideg")
    print(f"             95th   = {np.percentile(np.degrees(valid_om)*1000, 95):.1f} millideg")
    r_before_strong = abs_r[valid_global].numpy()
    r_after_strong = r_corrected[valid_global].abs().numpy()
    print(f"  Median |r| before: {np.median(r_before_strong)*1e3:.4f}×10⁻³")
    print(f"  Median |r| after:  {np.median(r_after_strong)*1e3:.4f}×10⁻³")
    rho_g_before, _ = spearmanr(lc_global, r_before_strong)
    rho_g_after, _ = spearmanr(lc_global, r_after_strong)
    print(f"  Global ρ before:   {rho_g_before:.4f}")
    print(f"  Global ρ after:    {rho_g_after:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
