#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Evaluate: shell statistics, scatter plots, alpha diagnostics.

Reads .pt files saved by fit.py — no GPU needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsx.merge import match_halves, shell_stats, cc_half, cc_star, r_split

OUTDIR = Path(__file__).parent
CELL = torch.load(OUTDIR / "cell.pt", weights_only=True)
N_SHELLS = 20
D_MIN, D_MAX = 0.9, 10.0  # strict resolution cutoff for all comparisons

to_F = lambda I: I.clamp(min=0).sqrt()


def load_pt(name):
    return torch.load(OUTDIR / name, weights_only=True)


def density_scatter(ax, x, y, **kwargs):
    from scipy.stats import gaussian_kde
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
    except Exception:
        z = np.ones_like(x)
    idx = z.argsort()
    ax.scatter(x[idx], y[idx], c=z[idx], s=2, alpha=0.6,
               cmap="viridis", rasterized=True, **kwargs)


def d_to_x(d):
    return 1.0 / np.asarray(d) ** 2


def setup_res_axis(ax):
    tick_d = np.array([10, 5, 3, 2, 1.5, 1.2, 1.0, 0.9])
    ax.set_xticks(d_to_x(tick_d))
    ax.set_xticklabels([f"{d:g}" for d in tick_d])
    ax.set_xlim(d_to_x(D_MAX), d_to_x(D_MIN))
    ax.set_xlabel("Resolution (Å)")


# ══════════════════════════════════════════════════════════════
# Load saved results
# ══════════════════════════════════════════════════════════════
print("Loading results...")

pu_h1 = load_pt("part_unity_half1.pt")
pu_h2 = load_pt("part_unity_half2.pt")
pu_m = load_pt("part_unity_merge.pt")

gg_h1 = load_pt("gg_half1.pt")
gg_h2 = load_pt("gg_half2.pt")
gg_full = load_pt("gg_full.pt")

em_h1 = load_pt("em_half1.pt")
em_h2 = load_pt("em_half2.pt")
em_full = load_pt("em_full.pt")

has_xsphere = (OUTDIR / "part_xsphere_half1.pt").exists()
if has_xsphere:
    xs_h1 = load_pt("part_xsphere_half1.pt")
    xs_h2 = load_pt("part_xsphere_half2.pt")

# ══════════════════════════════════════════════════════════════
# Build MatchedHalves with strict resolution cutoff
# ══════════════════════════════════════════════════════════════
print(f"Matching halves (d={D_MIN}–{D_MAX} Å)...")

m_pu = match_halves(pu_h1["hkl"], pu_h1["F"], pu_h2["hkl"], pu_h2["F"],
                     CELL, d_min=D_MIN, d_max=D_MAX)
m_gg = match_halves(gg_h1["hkl"], to_F(gg_h1["I0"]),
                     gg_h2["hkl"], to_F(gg_h2["I0"]),
                     CELL, d_min=D_MIN, d_max=D_MAX)
m_em = match_halves(em_h1["hkl"], to_F(em_h1["I0"]),
                     em_h2["hkl"], to_F(em_h2["I0"]),
                     CELL, d_min=D_MIN, d_max=D_MAX)

if has_xsphere:
    m_xs = match_halves(xs_h1["hkl"], xs_h1["F"], xs_h2["hkl"], xs_h2["F"],
                         CELL, d_min=D_MIN, d_max=D_MAX)

# Cross-method (full dataset)
m_em_pu = match_halves(em_full["hkl"], to_F(em_full["I0"]),
                        pu_m["hkl"], pu_m["F"],
                        CELL, d_min=D_MIN, d_max=D_MAX)
m_gg_pu = match_halves(gg_full["hkl"], to_F(gg_full["I0"]),
                        pu_m["hkl"], pu_m["F"],
                        CELL, d_min=D_MIN, d_max=D_MAX)
m_em_gg = match_halves(em_full["hkl"], to_F(em_full["I0"]),
                        gg_full["hkl"], to_F(gg_full["I0"]),
                        CELL, d_min=D_MIN, d_max=D_MAX)

# ══════════════════════════════════════════════════════════════
# Global metrics table
# ══════════════════════════════════════════════════════════════
methods_self = [("Part. unity", m_pu)]
if has_xsphere:
    methods_self.append(("Part. xsphere", m_xs))
methods_self += [("GenGauss", m_gg), ("EM Poisson", m_em)]

print(f"\n{'='*65}")
print(f"SELF-CONSISTENCY (split-half, {D_MIN}–{D_MAX} Å)")
print(f"{'='*65}")
print(f"  {'Method':>20s}  {'CC1/2':>7s}  {'CC*':>7s}  {'R_split':>8s}  {'N':>6s}")
print(f"  {'-'*55}")
for name, m in methods_self:
    cc = cc_half(m)
    print(f"  {name:>20s}  {cc:>7.4f}  {cc_star(cc):>7.4f}  {r_split(m):>8.4f}  {m.F1.numel():>6d}")

print(f"\n{'='*65}")
print("CROSS-METHOD (full dataset)")
print(f"{'='*65}")
for name, m in [("EM vs Part.", m_em_pu), ("GG vs Part.", m_gg_pu), ("EM vs GG", m_em_gg)]:
    print(f"  {name:>20s}  CC={cc_half(m):.4f}  N={m.F1.numel()}")

# ══════════════════════════════════════════════════════════════
# Shell statistics
# ══════════════════════════════════════════════════════════════
shell_data = [(name, shell_stats(m, n_shells=N_SHELLS)) for name, m in methods_self]

for metric_name, getter in [("CC1/2", lambda s: s.cc_half), ("R_SPLIT", lambda s: s.r_split)]:
    print(f"\n{'='*100}")
    print(f"PER-SHELL {metric_name}")
    print(f"{'='*100}")
    header = f"  {'d_max':>7s} {'d_min':>7s}"
    for name, _ in shell_data:
        header += f"  {name:>14s}"
    print(header)
    print(f"  {'-'*90}")
    for i in range(N_SHELLS):
        d_max = d_min = None
        for _, shells in shell_data:
            if i < len(shells):
                d_max, d_min = shells[i].d_max, shells[i].d_min
                break
        if d_max is None:
            continue
        row = f"  {d_max:>7.2f} {d_min:>7.2f}"
        for _, shells in shell_data:
            if i < len(shells):
                row += f"  {getter(shells[i]):>14.4f}"
            else:
                row += f"  {'---':>14s}"
        print(row)

# CSV
csv_path = OUTDIR / "benchmark_final_shells.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "d_max", "d_min", "n_matched", "cc_half", "cc_star", "r_split"])
    for name, shells in shell_data:
        for sh in shells:
            w.writerow([name, f"{sh.d_max:.4f}", f"{sh.d_min:.4f}",
                        sh.n_matched, f"{sh.cc_half:.6f}", f"{sh.cc_star:.6f}",
                        f"{sh.r_split:.6f}"])
print(f"\nWrote {csv_path}")

# ══════════════════════════════════════════════════════════════
# Plot 1: CC1/2 and R_split vs resolution
# ══════════════════════════════════════════════════════════════
colors = {"Part. unity": "gray", "Part. xsphere": "silver",
          "GenGauss": "C1", "EM Poisson": "C0"}
markers = {"Part. unity": "s", "Part. xsphere": "D",
           "GenGauss": "^", "EM Poisson": "o"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for name, shells in shell_data:
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    kw = dict(color=colors.get(name, "k"), marker=markers.get(name, "o"),
              ms=5, lw=1.8, label=name)
    ax1.plot(d_to_x(d_mid), [s.cc_half for s in shells], "-", **kw)
    ax2.plot(d_to_x(d_mid), [s.r_split for s in shells], "-", **kw)

ax1.set_ylabel("CC1/2"); ax1.set_title("CC1/2 vs Resolution")
ax1.legend(loc="lower left"); ax1.set_ylim(-0.05, 1.05); ax1.grid(True, alpha=0.3)
setup_res_axis(ax1)
ax2.set_ylabel("R_split"); ax2.set_title("R_split vs Resolution")
ax2.legend(loc="upper left"); ax2.set_ylim(0, None); ax2.grid(True, alpha=0.3)
setup_res_axis(ax2)
fig.suptitle(f"Precision: EM Poisson vs GenGauss vs Partialator ({D_MIN}–{D_MAX} Å)", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_final.png", dpi=150)
print(f"Saved benchmark_final.png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
# Plot 2: Self-consistency scatter (linear F)
# ══════════════════════════════════════════════════════════════

# Detection limit: F_floor = √(σ²_bg / n_obs) where σ²_bg ≈ bg_total.
# Compute range from the EM full-dataset n_obs distribution.
n_obs_em = em_full.get("n_obs", em_h1.get("n_obs", None))
if n_obs_em is None:
    # Approximate from obs_counts if n_obs not saved
    n_obs_em = torch.tensor([385.0])  # fallback median
bg_total = 92.0  # bg_per_pixel(0.6) × n_peak(154)
I0_floor_median = float((bg_total / n_obs_em.float().clamp(min=1).median()).sqrt())
I0_floor_p25 = float((bg_total / n_obs_em.float().clamp(min=1).quantile(0.75)).sqrt())  # high mult → low floor
I0_floor_p75 = float((bg_total / n_obs_em.float().clamp(min=1).quantile(0.25)).sqrt())  # low mult → high floor
F_floor_median = I0_floor_median ** 0.5
F_floor_lo = I0_floor_p25 ** 0.5
F_floor_hi = I0_floor_p75 ** 0.5
print(f"\nDetection limit: I0_floor={I0_floor_median:.3f} (range {I0_floor_p25:.3f}–{I0_floor_p75:.3f})")
print(f"  F_floor={F_floor_median:.3f} (range {F_floor_lo:.3f}–{F_floor_hi:.3f})")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, (m, label) in zip(axes, [
    (m_pu, f"Partialator unity\nCC1/2={cc_half(m_pu):.4f}  R={r_split(m_pu):.4f}"),
    (m_gg, f"GenGauss (log-space)\nCC1/2={cc_half(m_gg):.4f}  R={r_split(m_gg):.4f}"),
    (m_em, f"EM Poisson\nCC1/2={cc_half(m_em):.4f}  R={r_split(m_em):.4f}"),
]):
    F1, F2 = m.F1.numpy(), m.F2.numpy()
    hi = max(F1.max(), F2.max())
    density_scatter(ax, F1, F2)
    ax.plot([0, hi], [0, hi], "r-", lw=1, alpha=0.7)

    # Detection limit band
    ax.axvline(F_floor_median, color="red", ls="--", lw=1, alpha=0.6)
    ax.axhline(F_floor_median, color="red", ls="--", lw=1, alpha=0.6)
    ax.axvspan(0, F_floor_hi, alpha=0.06, color="red")
    ax.axhspan(0, F_floor_hi, alpha=0.06, color="red")
    ax.text(F_floor_median + 0.15, hi * 0.92, f"detection\nlimit",
            color="red", fontsize=7, va="top", alpha=0.8)

    ax.set_xlabel("F half 1"); ax.set_ylabel("F half 2")
    ax.set_title(label, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, hi * 1.05); ax.set_ylim(0, hi * 1.05)

fig.suptitle("Split-Half Self-Consistency", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "scatter_self_consistency.png", dpi=150, bbox_inches="tight")
print(f"Saved scatter_self_consistency.png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
# Plot 3: Cross-method scatter (linear F)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# m_em_pu: F1=EM, F2=Part
# m_gg_pu: F1=GG, F2=Part
# m_em_gg: F1=EM, F2=GG
for ax, (m, xlabel, ylabel, label) in zip(axes, [
    (m_em_pu, "F (EM Poisson)", "F (Partialator unity)",
     f"EM vs Partialator\nCC={cc_half(m_em_pu):.4f}"),
    (m_gg_pu, "F (GenGauss)", "F (Partialator unity)",
     f"GenGauss vs Partialator\nCC={cc_half(m_gg_pu):.4f}"),
    (m_em_gg, "F (EM Poisson)", "F (GenGauss)",
     f"EM vs GenGauss\nCC={cc_half(m_em_gg):.4f}"),
]):
    F1, F2 = m.F1.numpy(), m.F2.numpy()
    hi = max(F1.max(), F2.max())
    density_scatter(ax, F1, F2)
    ax.plot([0, hi], [0, hi], "r-", lw=1, alpha=0.7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(label, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, hi * 1.05); ax.set_ylim(0, hi * 1.05)

fig.suptitle("Cross-Method Comparison (full dataset)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "scatter_cross_method.png", dpi=150, bbox_inches="tight")
print(f"Saved scatter_cross_method.png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
# Plot 4: Ratio plots (relative difference vs intensity)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (m, label) in zip(axes, [
    (m_pu, "Partialator unity"),
    (m_gg, "GenGauss"),
    (m_em, "EM Poisson"),
]):
    F1, F2 = m.F1.numpy(), m.F2.numpy()
    F_mean = 0.5 * (F1 + F2)
    rel_diff = 2.0 * np.abs(F1 - F2) / (F1 + F2).clip(1e-8)
    ax.scatter(np.log10(F_mean.clip(0.01)), rel_diff, s=1, alpha=0.15,
               c="C0", rasterized=True)
    log_F = np.log10(F_mean.clip(0.01))
    edges = np.linspace(log_F.min(), log_F.max(), 30)
    centers, medians = [], []
    for i in range(len(edges) - 1):
        mask = (log_F >= edges[i]) & (log_F < edges[i + 1])
        if mask.sum() > 5:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            medians.append(np.median(rel_diff[mask]))
    ax.plot(centers, medians, "r-o", ms=3, lw=2, label="median")
    ax.set_xlabel("log₁₀(F_mean)"); ax.set_ylabel("2|F₁−F₂| / (F₁+F₂)")
    ax.set_title(f"{label}\nR_split={r_split(m):.4f}")
    ax.set_ylim(0, 1.0); ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle("Split-Half Relative Difference vs Intensity", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "scatter_ratio.png", dpi=150)
print(f"Saved scatter_ratio.png")
plt.close(fig)

# ══════════════════════════════════════════════════════════════
# Plot 5: Alpha scatter (half1 vs half2 and vs resolution)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Top row: alpha half1 vs half2
for ax, (d1, d2, label, color) in zip(axes[0], [
    (gg_h1, gg_h2, f"GenGauss  med(α)={gg_h1['alpha'].median():.3f}", "C1"),
    (em_h1, em_h2, f"EM Poisson  med(α)={em_h1['alpha'].median():.3f}", "C0"),
]):
    # Match HKLs between halves
    m_alpha = match_halves(d1["hkl"], d1["alpha"], d2["hkl"], d2["alpha"],
                            CELL, d_min=D_MIN, d_max=D_MAX)
    a1, a2 = m_alpha.F1.numpy(), m_alpha.F2.numpy()
    hi = max(np.percentile(a1, 99), np.percentile(a2, 99))
    density_scatter(ax, a1, a2)
    ax.plot([0, hi], [0, hi], "r-", lw=1, alpha=0.7)
    ax.set_xlabel("α half 1"); ax.set_ylabel("α half 2")
    ax.set_title(label, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, hi * 1.05); ax.set_ylim(0, hi * 1.05)

# Bottom row: alpha vs resolution (full dataset)
for ax, (d, label, color) in zip(axes[1], [
    (gg_full, "GenGauss (full)", "C1"),
    (em_full, "EM Poisson (full)", "C0"),
]):
    d_sp = d["d_spacing"].numpy()
    alpha = d["alpha"].numpy()
    I0 = d["I0"].numpy() if "I0" in d else np.ones_like(alpha)

    # Color by log(I0)
    sc = ax.scatter(d_to_x(d_sp), alpha, c=np.log10(I0.clip(0.1)),
                    s=3, alpha=0.4, cmap="viridis", vmin=-1, vmax=4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="log₁₀(I₀)", shrink=0.8)

    # Binned median
    edges = np.percentile(d_sp, np.linspace(0, 100, N_SHELLS + 1))
    d_mid, a_med, a_med_strong = [], [], []
    for i in range(N_SHELLS):
        mask = (d_sp >= edges[i]) & (d_sp < edges[i + 1])
        if mask.sum() < 2:
            continue
        d_mid.append(0.5 * (edges[i] + edges[i + 1]))
        a_med.append(np.median(alpha[mask]))
        strong = mask & (I0 > 30)
        a_med_strong.append(np.median(alpha[strong]) if strong.sum() > 5 else np.nan)
    ax.plot(d_to_x(d_mid), a_med, "r-o", ms=4, lw=2, label="median (all)", zorder=5)
    ax.plot(d_to_x(d_mid), a_med_strong, "k--s", ms=3, lw=1.5,
            label="median (I₀>30)", zorder=5)

    ax.set_ylabel("α"); ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(3.0, np.percentile(alpha, 99)))
    setup_res_axis(ax)

fig.suptitle("Alpha: Self-Consistency and Resolution Dependence", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "scatter_alpha.png", dpi=150)
print(f"Saved scatter_alpha.png")
plt.close(fig)

print("\nDone.")
