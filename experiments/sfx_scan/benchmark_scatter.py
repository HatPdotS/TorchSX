#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 4:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/benchmark_scatter.out

"""Scatter plots: half-vs-half self-consistency and cross-method comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsx.stream import SFXStream
from torchsx.merge import (
    match_halves, cc_half, r_split,
    fit_em, EMConfig, ScaleBCorrection,
)
from torchsx.merge._distribution import fit_gengauss
from torchsx.merge._stats import spearman_cc

# ── Configuration ────────────────────────────────────────────
STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
PART_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/partialator")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/benchmark_final")
OUTDIR.mkdir(parents=True, exist_ok=True)

DELAYS = [-91, -136, -182, -227]
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ── Helpers ──────────────────────────────────────────────────
def parse_crystfel_hkl(path):
    h, k, l, I_ = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("CrystFEL", "Symmetry", "h ", "End")):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                h.append(int(parts[0])); k.append(int(parts[1]))
                l.append(int(parts[2])); I_.append(float(parts[3]))
            except ValueError:
                continue
    hkl = torch.tensor(list(zip(h, k, l)), dtype=torch.int16)
    return hkl, torch.tensor(I_, dtype=torch.float32)


def load_partialator_halves(half1_path, half2_path):
    from torchref.symmetry import canonicalize_hkl
    hkl1, I1 = parse_crystfel_hkl(half1_path)
    hkl2, I2 = parse_crystfel_hkl(half2_path)
    hkl1, _, _, s1 = canonicalize_hkl(hkl1.int(), "P -1", include_friedel=True)
    hkl2, _, _, s2 = canonicalize_hkl(hkl2.int(), "P -1", include_friedel=True)
    F1 = I1[s1].clamp(min=0).sqrt()
    F2 = I2[s2].clamp(min=0).sqrt()
    return match_halves(hkl1.to(torch.int16), F1,
                        hkl2.to(torch.int16), F2, CELL)


def to_F(I):
    return I.clamp(min=0).sqrt()


def density_scatter(ax, x, y, **kwargs):
    """Scatter plot colored by point density."""
    from scipy.stats import gaussian_kde
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
    except Exception:
        z = np.ones_like(x)
    idx = z.argsort()
    ax.scatter(x[idx], y[idx], c=z[idx], s=2, alpha=0.6,
               cmap="viridis", rasterized=True, **kwargs)


# ══════════════════════════════════════════════════════════════
# 1. Load data and merge
# ══════════════════════════════════════════════════════════════
print("── Loading streams ──")
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"Total: {s.n_crystals} crystals, split: {h1.n_crystals}/{h2.n_crystals}")

# Partialator unity
print("── Partialator unity ──")
m_pu = load_partialator_halves(PART_DIR / "ind_half1.hkl", PART_DIR / "ind_half2.hkl")
print(f"  CC1/2={cc_half(m_pu):.4f}  R={r_split(m_pu):.4f}")

# GenGauss (log-space)
print("── fit_gengauss ──")
FIT_KW = dict(min_obs=20, n_smooth_iter=15, smooth_lambda=0.3)
fg1 = fit_gengauss(h1, n_em_cycles=10, scale_crystals=True, **FIT_KW)
fg2 = fit_gengauss(h2, n_em_cycles=10, scale_crystals=True, **FIT_KW)
m_gg = match_halves(fg1.hkl, to_F(fg1.I0), fg2.hkl, to_F(fg2.I0), CELL)
print(f"  CC1/2={cc_half(m_gg):.4f}  R={r_split(m_gg):.4f}")

# EM Poisson
print("── fit_em ──")
cfg = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)
de1 = fit_em(h1, config=cfg, corrections=[ScaleBCorrection()], device=device)
de2 = fit_em(h2, config=cfg, corrections=[ScaleBCorrection()], device=device)
m_em = match_halves(de1.hkl, to_F(de1.I0), de2.hkl, to_F(de2.I0), CELL)
print(f"  CC1/2={cc_half(m_em):.4f}  R={r_split(m_em):.4f}")

# Cross-method: EM full-dataset vs partialator full merge
print("── Cross-method comparisons ──")
# EM on full dataset
de_full = fit_em(s, config=cfg, corrections=[ScaleBCorrection()], device=device)
# Partialator full merge
hkl_pu, I_pu = parse_crystfel_hkl(PART_DIR / "neg.hkl")
from torchref.symmetry import canonicalize_hkl
hkl_pu, _, _, s_pu = canonicalize_hkl(hkl_pu.int(), "P -1", include_friedel=True)
F_pu = I_pu[s_pu].clamp(min=0).sqrt()
m_em_vs_pu = match_halves(de_full.hkl, to_F(de_full.I0),
                           hkl_pu.to(torch.int16), F_pu, CELL)
print(f"  EM vs Part: CC={cc_half(m_em_vs_pu):.4f}")

# GenGauss full
fg_full = fit_gengauss(s, n_em_cycles=10, scale_crystals=True, **FIT_KW)
m_gg_vs_pu = match_halves(fg_full.hkl, to_F(fg_full.I0),
                           hkl_pu.to(torch.int16), F_pu, CELL)
print(f"  GG vs Part: CC={cc_half(m_gg_vs_pu):.4f}")

m_em_vs_gg = match_halves(de_full.hkl, to_F(de_full.I0),
                           fg_full.hkl, to_F(fg_full.I0), CELL)
print(f"  EM vs GG:   CC={cc_half(m_em_vs_gg):.4f}")


# ══════════════════════════════════════════════════════════════
# 2. Self-consistency scatter plots (half vs half)
# ══════════════════════════════════════════════════════════════
print("\n── Plotting ──")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, (m, label) in zip(axes, [
    (m_pu,  f"Partialator unity\nCC1/2={cc_half(m_pu):.4f}  R={r_split(m_pu):.4f}"),
    (m_gg,  f"GenGauss (log-space)\nCC1/2={cc_half(m_gg):.4f}  R={r_split(m_gg):.4f}"),
    (m_em,  f"EM Poisson\nCC1/2={cc_half(m_em):.4f}  R={r_split(m_em):.4f}"),
]):
    F1 = m.F1.numpy()
    F2 = m.F2.numpy()
    hi = max(F1.max(), F2.max())
    density_scatter(ax, F1, F2)
    ax.plot([0, hi], [0, hi], "r-", lw=1, alpha=0.7)
    ax.set_xlabel("F half 1")
    ax.set_ylabel("F half 2")
    ax.set_title(label, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, hi * 1.05)
    ax.set_ylim(0, hi * 1.05)

fig.suptitle("Split-Half Self-Consistency", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "scatter_self_consistency.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR / 'scatter_self_consistency.png'}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 3. Cross-method scatter plots (full dataset)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, (m, xlabel, ylabel, label) in zip(axes, [
    (m_em_vs_pu, "Partialator unity", "EM Poisson",
     f"EM vs Partialator\nCC={cc_half(m_em_vs_pu):.4f}"),
    (m_gg_vs_pu, "Partialator unity", "GenGauss",
     f"GenGauss vs Partialator\nCC={cc_half(m_gg_vs_pu):.4f}"),
    (m_em_vs_gg, "GenGauss", "EM Poisson",
     f"EM vs GenGauss\nCC={cc_half(m_em_vs_gg):.4f}"),
]):
    F1 = m.F1.numpy()
    F2 = m.F2.numpy()
    hi = max(F1.max(), F2.max())
    density_scatter(ax, F1, F2)
    ax.plot([0, hi], [0, hi], "r-", lw=1, alpha=0.7)
    ax.set_xlabel(f"F {xlabel}")
    ax.set_ylabel(f"F {ylabel}")
    ax.set_title(label, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, hi * 1.05)
    ax.set_ylim(0, hi * 1.05)

fig.suptitle("Cross-Method Comparison (full dataset)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "scatter_cross_method.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTDIR / 'scatter_cross_method.png'}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 4. Ratio plots (relative difference vs intensity)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (m, label) in zip(axes, [
    (m_pu,  "Partialator unity"),
    (m_gg,  "GenGauss"),
    (m_em,  "EM Poisson"),
]):
    F1 = m.F1.numpy()
    F2 = m.F2.numpy()
    F_mean = 0.5 * (F1 + F2)
    rel_diff = 2.0 * np.abs(F1 - F2) / (F1 + F2).clip(1e-8)

    ax.scatter(np.log10(F_mean.clip(0.01)), rel_diff, s=1, alpha=0.15,
               c="C0", rasterized=True)

    # Binned median
    log_F = np.log10(F_mean.clip(0.01))
    edges = np.linspace(log_F.min(), log_F.max(), 30)
    centers, medians = [], []
    for i in range(len(edges) - 1):
        mask = (log_F >= edges[i]) & (log_F < edges[i + 1])
        if mask.sum() > 5:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            medians.append(np.median(rel_diff[mask]))
    ax.plot(centers, medians, "r-o", ms=3, lw=2, label="median")

    ax.set_xlabel("log₁₀(F_mean)")
    ax.set_ylabel("2|F₁−F₂| / (F₁+F₂)")
    ax.set_title(f"{label}\nR_split={r_split(m):.4f}")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Split-Half Relative Difference vs Intensity", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "scatter_ratio.png", dpi=150)
print(f"Saved {OUTDIR / 'scatter_ratio.png'}")
plt.close(fig)

print("\nDone.")
