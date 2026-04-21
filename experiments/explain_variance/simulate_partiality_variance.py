#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""
Partiality variance simulation for SFX — isotropic crystal.

For a spherical crystal of diameter D, the relp is isotropic:
    σ_relp = 1 / (2.355 × D)     [Gaussian approximation, Å⁻¹]

The Ewald shell has radial width set by bandwidth:
    half-width = s × BW/2 = BW / (2 × d)     [top-hat, Å⁻¹]

Partiality p(δ) = fraction of relp captured at Ewald offset δ.
Reflections sample δ uniformly within ±cutoff.

Question: can this explain σ_LN ≈ 1.56 (var = 2.04)?

Dataset:  SwissFEL dark-SFX, E = 12 960 eV, λ = 0.9567 Å
Cell:     a ≈ 14, b ≈ 18, c ≈ 18 Å  (P-1)
"""

import numpy as np
from scipy.special import erf
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

# ── Observed targets ──────────────────────────────────────────────
OBSERVED_SIGMA_LN = 1.56
OBSERVED_VAR = 2.04
SIGMA_LN_FIT = lambda d: 2.25 - 1.16 / d**2  # empirical resolution dependence


# ══════════════════════════════════════════════════════════════════
# Partiality functions
# ══════════════════════════════════════════════════════════════════


def partiality_gauss_tophat(delta, sigma_relp, shell_hw):
    """Gaussian relp × top-hat Ewald shell."""
    sq2s = np.sqrt(2) * sigma_relp
    return 0.5 * (erf((shell_hw - delta) / sq2s) - erf((-shell_hw - delta) / sq2s))


def partiality_gauss_gauss(delta, sigma_relp, sigma_bw):
    """Gaussian relp × Gaussian bandwidth."""
    sigma_eff = np.sqrt(sigma_relp**2 + sigma_bw**2)
    return (sigma_relp / sigma_eff) * np.exp(-0.5 * delta**2 / sigma_eff**2)


def sinc2_profile(x, L):
    """sinc²(π L x) — true shape transform."""
    arg = np.pi * L * x
    out = np.ones_like(arg, dtype=float)
    mask = np.abs(arg) > 1e-10
    out[mask] = (np.sin(arg[mask]) / arg[mask]) ** 2
    return out


def partiality_sinc2_tophat(delta, L, shell_hw, n_quad=800):
    """sinc² relp × top-hat Ewald shell via quadrature."""
    span = max(5.0 / L, 2 * shell_hw)
    x = np.linspace(-span, span, n_quad)
    dx = x[1] - x[0]
    profile = sinc2_profile(x, L)
    norm_val = np.sum(profile) * dx

    delta = np.atleast_1d(delta)
    p = np.zeros_like(delta, dtype=float)
    for i, d in enumerate(delta):
        mask = (x >= d - shell_hw) & (x <= d + shell_hw)
        p[i] = np.sum(profile[mask]) * dx / norm_val if norm_val > 0 else 0
    return np.clip(p, 0, 1)


# ══════════════════════════════════════════════════════════════════
# Parameter helpers
# ══════════════════════════════════════════════════════════════════


def relp_sigma(D_um):
    """Relp σ (Å⁻¹) for isotropic crystal of diameter D (µm)."""
    return 1.0 / (2.355 * D_um * 1e4)


def shell_params(bw_pct, d_spacing):
    """Ewald shell half-width (top-hat) and σ (Gaussian), in Å⁻¹."""
    hw = 0.5 * (bw_pct / 100.0) / d_spacing
    sigma = hw / (2.355 / 2)  # same FWHM as top-hat
    return hw, sigma


# ══════════════════════════════════════════════════════════════════
# Monte Carlo sampler
# ══════════════════════════════════════════════════════════════════


def mc_partiality(
    D_um, bw_pct, d_spacing,
    beam_type="tophat", relp_model="gaussian",
    n_mc=200_000, threshold=0.01,
):
    """
    Sample random Ewald offsets, compute partiality, return detected subset.
    """
    sig_r = relp_sigma(D_um)
    hw, sig_bw = shell_params(bw_pct, d_spacing)
    L = D_um * 1e4  # crystal size in Å (for sinc²)

    # Offset cutoff: where partiality drops below threshold
    cutoff = 4 * sig_r + 2 * max(hw, 3 * sig_bw)
    delta = np.random.uniform(-cutoff, cutoff, n_mc)

    if relp_model == "gaussian":
        if beam_type == "tophat":
            p = partiality_gauss_tophat(delta, sig_r, hw)
        else:
            p = partiality_gauss_gauss(delta, sig_r, sig_bw)
    else:  # sinc²
        if beam_type == "tophat":
            p = partiality_sinc2_tophat(delta, L, hw)
        else:
            # sinc² × Gaussian — skip for speed, just use Gaussian approx
            p = partiality_gauss_gauss(delta, sig_r, sig_bw)

    mask = p > threshold
    p_det = p[mask]
    info = {
        "det_frac": mask.sum() / n_mc,
        "sig_relp": sig_r,
        "shell_hw": hw,
        "ratio": 2 * hw / (2.355 * sig_r),  # shell FWHM / relp FWHM
    }
    return p_det, info


def sigma_logp(p):
    """σ(log p) for array of partialities."""
    return np.std(np.log(p)) if len(p) >= 50 else np.nan


# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════


def fig1_partiality_curves():
    """Partiality vs offset for representative conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Partiality vs Ewald offset — isotropic crystal, d = 3 Å", fontsize=12, fontweight="bold")

    # Vary crystal size
    ax = axes[0]
    bw = 0.15
    d = 3.0
    for D in [0.1, 0.2, 0.5, 1.0, 2.0]:
        sig_r = relp_sigma(D)
        hw, _ = shell_params(bw, d)
        cut = 4 * sig_r + 2 * hw
        delta = np.linspace(-cut, cut, 500)
        p = partiality_gauss_tophat(delta, sig_r, hw)
        ax.plot(delta * 1e3, p, lw=1.5, label=f"D={D} µm")
    ax.axhline(0.01, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Offset (×10⁻³ Å⁻¹)")
    ax.set_ylabel("Partiality")
    ax.set_title(f"Vary crystal size (BW={bw}%)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)

    # Vary bandwidth
    ax = axes[1]
    D = 0.5
    for bw in [0.05, 0.1, 0.2, 0.5, 1.0]:
        sig_r = relp_sigma(D)
        hw, _ = shell_params(bw, d)
        cut = 4 * sig_r + 2 * hw
        delta = np.linspace(-cut, cut, 500)
        p = partiality_gauss_tophat(delta, sig_r, hw)
        ax.plot(delta * 1e3, p, lw=1.5, label=f"BW={bw}%")
    ax.axhline(0.01, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Offset (×10⁻³ Å⁻¹)")
    ax.set_title(f"Vary bandwidth (D={D} µm)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)

    # Top-hat vs Gaussian beam
    ax = axes[2]
    D = 0.5
    bw = 0.2
    sig_r = relp_sigma(D)
    hw, sig_bw = shell_params(bw, d)
    cut = 4 * sig_r + 2 * hw
    delta = np.linspace(-cut, cut, 500)
    ax.plot(delta * 1e3, partiality_gauss_tophat(delta, sig_r, hw), lw=2, label="top-hat")
    ax.plot(delta * 1e3, partiality_gauss_gauss(delta, sig_r, sig_bw), lw=2, ls="--", label="Gaussian")
    ax.axhline(0.01, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Offset (×10⁻³ Å⁻¹)")
    ax.set_title(f"Top-hat vs Gaussian (D={D}, BW={bw}%)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig1_partiality_curves.png", dpi=150)
    print(f"  → {OUTDIR / 'fig1_partiality_curves.png'}")
    plt.close(fig)


def fig2_sigma_vs_crystal_size():
    """σ(log p) vs crystal size for several bandwidths."""
    sizes = np.array([0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    bws = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
    d_sp = 3.0
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bws)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"σ(log p) vs crystal size  —  d = {d_sp} Å, Gaussian relp\n"
        f"target: σ_LN = {OBSERVED_SIGMA_LN}",
        fontsize=12, fontweight="bold",
    )

    # Left: top-hat
    ax = axes[0]
    for bw, col in zip(bws, colors):
        sigs = [sigma_logp(mc_partiality(sz, bw, d_sp, "tophat")[0]) for sz in sizes]
        ax.plot(sizes, sigs, "o-", color=col, lw=2, ms=4, label=f"BW={bw}%")
    ax.axhline(OBSERVED_SIGMA_LN, color="red", ls="--", lw=2, alpha=0.7, label=f"observed = {OBSERVED_SIGMA_LN}")
    ax.set_xlabel("Crystal diameter D (µm)")
    ax.set_ylabel("σ(log p)")
    ax.set_title("Top-hat beam")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 2.5)
    ax.set_xscale("log")

    # Right: Gaussian
    ax = axes[1]
    for bw, col in zip(bws, colors):
        sigs = [sigma_logp(mc_partiality(sz, bw, d_sp, "gauss")[0]) for sz in sizes]
        ax.plot(sizes, sigs, "o-", color=col, lw=2, ms=4, label=f"BW={bw}%")
    ax.axhline(OBSERVED_SIGMA_LN, color="red", ls="--", lw=2, alpha=0.7, label=f"observed = {OBSERVED_SIGMA_LN}")
    ax.set_xlabel("Crystal diameter D (µm)")
    ax.set_ylabel("σ(log p)")
    ax.set_title("Gaussian beam")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 2.5)
    ax.set_xscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUTDIR / "fig2_sigma_vs_size.png", dpi=150)
    print(f"  → {OUTDIR / 'fig2_sigma_vs_size.png'}")
    plt.close(fig)


def fig3_resolution_dependence():
    """σ(log p) vs d-spacing, compare to empirical fit."""
    d_range = np.array([1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    sizes = [0.1, 0.2, 0.5, 1.0, 2.0]
    bw = 0.15
    colors = ["#1D9E75", "#534AB7", "#D85A30", "#B5338A", "#666666"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Resolution dependence — BW = {bw}%, top-hat", fontsize=12, fontweight="bold")

    # Left: vs d
    ax = axes[0]
    for sz, col in zip(sizes, colors):
        sigs = [sigma_logp(mc_partiality(sz, bw, d)[0]) for d in d_range]
        ax.plot(d_range, sigs, "o-", color=col, lw=2, ms=4, label=f"D={sz} µm")
    ax.axhline(OBSERVED_SIGMA_LN, color="red", ls="--", lw=2, alpha=0.7)
    ax.set_xlabel("d-spacing (Å)")
    ax.set_ylabel("σ(log p)")
    ax.set_title("σ(log p) vs resolution")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 2.5)

    # Right: vs 1/d², compare to empirical
    ax = axes[1]
    inv_d2 = 1.0 / d_range**2
    for sz, col in zip(sizes, colors):
        sigs = [sigma_logp(mc_partiality(sz, bw, d)[0]) for d in d_range]
        ax.plot(inv_d2, sigs, "o-", color=col, lw=2, ms=4, label=f"D={sz} µm")
    x = np.linspace(0, 1.2, 100)
    ax.plot(x, np.clip(2.25 - 1.16 * x, 0, None), "r--", lw=2, alpha=0.7, label="data: 2.25 − 1.16/d²")
    ax.set_xlabel("1/d² (Å⁻²)")
    ax.set_ylabel("σ(log p)")
    ax.set_title("Compare to observed σ_LN(d)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 2.5)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTDIR / "fig3_resolution.png", dpi=150)
    print(f"  → {OUTDIR / 'fig3_resolution.png'}")
    plt.close(fig)


def fig4_distributions():
    """log(p) histograms for representative conditions."""
    conditions = [
        ("D=0.1µm\nBW=0.15%", 0.1, 0.15),
        ("D=0.2µm\nBW=0.15%", 0.2, 0.15),
        ("D=0.5µm\nBW=0.15%", 0.5, 0.15),
        ("D=0.1µm\nBW=0.50%", 0.1, 0.50),
        ("D=0.5µm\nBW=0.50%", 0.5, 0.50),
        ("D=1.0µm\nBW=0.50%", 1.0, 0.50),
    ]
    d_sp = 3.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"log(partiality) distributions — d = {d_sp} Å, top-hat\n"
        f"observed σ_LN = {OBSERVED_SIGMA_LN}",
        fontsize=12, fontweight="bold",
    )

    for ax, (label, sz, bw) in zip(axes.flat, conditions):
        p, info = mc_partiality(sz, bw, d_sp)
        if len(p) < 50:
            ax.text(0.5, 0.5, "Too few", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=10)
            continue

        log_p = np.log(p)
        sig = np.std(log_p)
        lo = max(-10, log_p.min() - 0.3)
        ax.hist(log_p, bins=80, range=(lo, 0.5), density=True, color="#D85A30", alpha=0.6, edgecolor="#993C1D", lw=0.5)
        x_fit = np.linspace(lo, 0.5, 200)
        ax.plot(x_fit, norm.pdf(x_fit, np.mean(log_p), sig), color="#534AB7", lw=2)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("log(p)")
        txt = (f"σ(log p) = {sig:.3f}\n⟨p⟩ = {np.mean(p):.4f}\n"
               f"det = {info['det_frac']*100:.1f}%\nshell/relp = {info['ratio']:.3f}")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUTDIR / "fig4_distributions.png", dpi=150)
    print(f"  → {OUTDIR / 'fig4_distributions.png'}")
    plt.close(fig)


def fig5_variance_budget():
    """How much of var(log c) = 2.04 can partiality explain?"""
    sizes = np.arange(0.05, 5.05, 0.1)
    bws = [0.10, 0.15, 0.20, 0.30, 0.50]
    d_sp = 3.0
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bws)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"Variance budget — d = {d_sp} Å, top-hat\n"
        f"observed var(log c) = {OBSERVED_VAR}",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    for bw, col in zip(bws, colors):
        vp = []
        for sz in sizes:
            s = sigma_logp(mc_partiality(sz, bw, d_sp, n_mc=80_000)[0])
            vp.append(s**2 if not np.isnan(s) else np.nan)
        ax.plot(sizes, vp, "-", color=col, lw=2, label=f"BW={bw}%")
    ax.axhline(OBSERVED_VAR, color="red", ls="--", lw=2, alpha=0.7, label=f"observed = {OBSERVED_VAR}")
    ax.set_xlabel("Crystal diameter D (µm)")
    ax.set_ylabel("var(log p)")
    ax.set_title("Partiality variance")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 3.0)

    ax = axes[1]
    for bw, col in zip(bws, colors):
        pcts = []
        for sz in sizes:
            s = sigma_logp(mc_partiality(sz, bw, d_sp, n_mc=80_000)[0])
            pcts.append(s**2 / OBSERVED_VAR * 100 if not np.isnan(s) else np.nan)
        ax.plot(sizes, pcts, "-", color=col, lw=2, label=f"BW={bw}%")
    ax.axhline(100, color="red", ls="--", lw=2, alpha=0.7, label="100%")
    ax.set_xlabel("Crystal diameter D (µm)")
    ax.set_ylabel("% of observed variance")
    ax.set_title("Fraction explained by partiality")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 150)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUTDIR / "fig5_variance_budget.png", dpi=150)
    print(f"  → {OUTDIR / 'fig5_variance_budget.png'}")
    plt.close(fig)


def fig6_heatmap():
    """2-D heatmap: σ(log p) over (BW, crystal size)."""
    bw_list = np.array([0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5])
    sz_list = np.array([0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    d_sp = 3.0

    grid = np.full((len(sz_list), len(bw_list)), np.nan)
    for i, sz in enumerate(sz_list):
        for j, bw in enumerate(bw_list):
            p, _ = mc_partiality(sz, bw, d_sp, n_mc=80_000)
            grid[i, j] = sigma_logp(p)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(bw_list, sz_list, grid, cmap="magma_r", vmin=0, vmax=2.0, shading="nearest")
    try:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((sz_list, bw_list), grid, method="linear", bounds_error=False, fill_value=np.nan)
        bw_f = np.linspace(bw_list[0], bw_list[-1], 100)
        sz_f = np.linspace(sz_list[0], sz_list[-1], 100)
        BW, SZ = np.meshgrid(bw_f, sz_f)
        Z = interp(np.column_stack([SZ.ravel(), BW.ravel()])).reshape(BW.shape)
        cs = ax.contour(bw_f, sz_f, Z, levels=[0.3, 0.5, 1.0, OBSERVED_SIGMA_LN],
                        colors=["white", "cyan", "lime", "red"], linewidths=[1, 1, 1.5, 2.5])
        ax.clabel(cs, fmt="%.2f", fontsize=9)
    except Exception:
        pass
    ax.set_xlabel("Bandwidth (%)")
    ax.set_ylabel("Crystal diameter D (µm)")
    ax.set_yscale("log")
    ax.set_title(
        f"σ(log p) — d = {d_sp} Å, Gaussian relp, top-hat beam\n"
        f"Red contour = observed σ_LN = {OBSERVED_SIGMA_LN}",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="σ(log p)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig6_heatmap.png", dpi=150)
    print(f"  → {OUTDIR / 'fig6_heatmap.png'}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════


def print_summary():
    print("\n" + "=" * 90)
    print("PARTIALITY VARIANCE — ISOTROPIC CRYSTAL")
    print(f"Target: σ_LN = {OBSERVED_SIGMA_LN},  var(log c) = {OBSERVED_VAR}")
    print("=" * 90)

    conditions = [
        # (D_µm, BW%, d_Å, beam)
        (0.05, 0.15, 3.0, "tophat"),
        (0.1,  0.15, 3.0, "tophat"),
        (0.2,  0.15, 3.0, "tophat"),
        (0.5,  0.15, 3.0, "tophat"),
        (1.0,  0.15, 3.0, "tophat"),
        (2.0,  0.15, 3.0, "tophat"),
        (5.0,  0.15, 3.0, "tophat"),
        (0.1,  0.30, 3.0, "tophat"),
        (0.2,  0.30, 3.0, "tophat"),
        (0.5,  0.30, 3.0, "tophat"),
        (0.1,  0.50, 3.0, "tophat"),
        (0.5,  0.50, 3.0, "tophat"),
        (0.5,  0.15, 1.5, "tophat"),
        (0.5,  0.15, 5.0, "tophat"),
        (0.5,  0.15, 10., "tophat"),
        (0.5,  0.15, 3.0, "gauss"),
    ]

    hdr = f"{'D(µm)':>6} {'BW%':>5} {'d(Å)':>5} {'beam':>7} │ {'σ_relp':>10} {'shell_hw':>10} {'ratio':>6} │ {'σ(logp)':>8} {'var':>6} {'%obs':>6} {'⟨p⟩':>8} {'det%':>5}"
    print(hdr)
    print("─" * len(hdr))

    for sz, bw, d, beam in conditions:
        p, info = mc_partiality(sz, bw, d, beam_type=beam, n_mc=300_000)
        sig_r = info["sig_relp"]
        hw = info["shell_hw"]
        ratio = info["ratio"]
        if len(p) < 50:
            print(f"{sz:6.2f} {bw:5.2f} {d:5.1f} {beam:>7} │ {sig_r:10.2e} {hw:10.2e} {ratio:6.3f} │ {'N/A':>8}")
            continue
        sig = sigma_logp(p)
        var = sig**2
        pct = var / OBSERVED_VAR * 100
        print(
            f"{sz:6.2f} {bw:5.2f} {d:5.1f} {beam:>7} │ "
            f"{sig_r:10.2e} {hw:10.2e} {ratio:6.3f} │ "
            f"{sig:8.3f} {var:6.3f} {pct:5.1f}% {np.mean(p):8.5f} {info['det_frac']*100:5.1f}"
        )

    print("=" * 90)
    print()
    print("Key physics:")
    print(f"  σ_relp = 1/(2.355 × D)    shell_hw = BW/(2d)")
    print(f"  ratio = shell_FWHM / relp_FWHM")
    print(f"    ratio >> 1: shell much wider than relp → p ≈ 1 for all → σ(log p) ≈ 0")
    print(f"    ratio << 1: shell much thinner than relp → p varies hugely → σ(log p) large")
    print(f"    ratio ~ 1: crossover regime")
    print()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("Partiality variance simulation — isotropic crystal")
    print("=" * 50)

    print_summary()

    print("\nGenerating figures...")
    fig1_partiality_curves()
    fig2_sigma_vs_crystal_size()
    fig3_resolution_dependence()
    fig4_distributions()
    fig5_variance_budget()
    fig6_heatmap()

    print(f"\nDone. Figures in {OUTDIR}/")
