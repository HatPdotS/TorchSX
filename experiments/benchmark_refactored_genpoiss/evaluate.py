#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Evaluate: shell statistics, scatter plots, alpha diagnostics.

Ported from benchmark_final/evaluate.py and refactored to iterate over a
METHODS list so adding new methods (e.g. "EM Unified") is one line.

Methods come in two flavours (same `.pt` schema, different I vs F field):
  - "I0" → we compute F = sqrt(clamp(I0, min=0))
  - "F"  → we use F directly (e.g. partialator)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
D_MIN, D_MAX = 0.9, 10.0

to_F = lambda I: I.clamp(min=0).sqrt()


def load_pt(name):
    p = OUTDIR / name
    if not p.exists():
        return None
    return torch.load(p, weights_only=True)


# ── Methods registry ─────────────────────────────────────────
@dataclass
class MethodSpec:
    name: str         # display name
    prefix: str       # file prefix ({prefix}_half1.pt etc.)
    field: str        # "I0" or "F"
    color: str
    marker: str


METHODS = [
    MethodSpec("EM Unified",    "unified",      "I0", "C2", "*"),
    MethodSpec("EM Poisson",    "em",           "I0", "C0", "o"),
    MethodSpec("GenGauss",      "gg",           "I0", "C1", "^"),
    MethodSpec("Part. unity",   "part_unity",   "F",  "gray", "s"),
    MethodSpec("Part. xsphere", "part_xsphere", "F",  "silver", "D"),
]


# ── Load every method's data ─────────────────────────────────
@dataclass
class MethodData:
    spec: MethodSpec
    h1: dict
    h2: dict
    full_or_merge: dict | None  # "merge" for partialator, "full" for fit_* methods

    def F_half(self, tag: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.h1 if tag == "half1" else self.h2
        if self.spec.field == "F":
            return d["hkl"], d["F"]
        return d["hkl"], to_F(d["I0"])

    def F_full(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.full_or_merge is None:
            return None
        if self.spec.field == "F":
            return self.full_or_merge["hkl"], self.full_or_merge["F"]
        return self.full_or_merge["hkl"], to_F(self.full_or_merge["I0"])


def load_method(spec: MethodSpec) -> MethodData | None:
    h1 = load_pt(f"{spec.prefix}_half1.pt")
    h2 = load_pt(f"{spec.prefix}_half2.pt")
    if h1 is None or h2 is None:
        return None
    full = load_pt(f"{spec.prefix}_full.pt")
    if full is None:
        full = load_pt(f"{spec.prefix}_merge.pt")
    return MethodData(spec, h1, h2, full)


print("Loading results...", flush=True)
methods: list[MethodData] = []
for spec in METHODS:
    md = load_method(spec)
    if md is None:
        print(f"  skip {spec.name} (missing files for prefix '{spec.prefix}')")
        continue
    methods.append(md)


# ── Split-half matching ──────────────────────────────────────
print(f"Matching halves (d={D_MIN}–{D_MAX} Å)...", flush=True)
halves_matched = []
for md in methods:
    hkl1, F1 = md.F_half("half1")
    hkl2, F2 = md.F_half("half2")
    m = match_halves(hkl1, F1, hkl2, F2, CELL, d_min=D_MIN, d_max=D_MAX)
    halves_matched.append((md, m))


# ── Global metrics table ─────────────────────────────────────
print(f"\n{'='*70}")
print(f"SELF-CONSISTENCY (split-half, {D_MIN}–{D_MAX} Å)")
print(f"{'='*70}")
print(f"  {'Method':>20s}  {'CC1/2':>7s}  {'CC*':>7s}  {'R_split':>8s}  {'N':>6s}")
print(f"  {'-'*60}")
for md, m in halves_matched:
    cc = cc_half(m)
    print(f"  {md.spec.name:>20s}  {cc:>7.4f}  {cc_star(cc):>7.4f}  "
          f"{r_split(m):>8.4f}  {m.F1.numel():>6d}")


# ── Cross-method (vs EM Unified, if present) ─────────────────
ref = next((md for md in methods if md.spec.prefix == "unified"), None)
if ref is None:
    ref = next((md for md in methods if md.spec.prefix == "em"), None)

if ref is not None:
    ref_full = ref.F_full()
    if ref_full is not None:
        hkl_ref, F_ref = ref_full
        print(f"\n{'='*70}")
        print(f"CROSS-METHOD (full dataset, reference = {ref.spec.name})")
        print(f"{'='*70}")
        for md in methods:
            if md is ref:
                continue
            full = md.F_full()
            if full is None:
                continue
            hkl_o, F_o = full
            mx = match_halves(hkl_ref, F_ref, hkl_o, F_o, CELL, d_min=D_MIN, d_max=D_MAX)
            print(f"  {ref.spec.name:>15s} vs {md.spec.name:<15s}  "
                  f"CC={cc_half(mx):.4f}  N={mx.F1.numel()}")


# ── Per-shell statistics ─────────────────────────────────────
shell_data = [(md.spec.name, shell_stats(m, n_shells=N_SHELLS)) for md, m in halves_matched]

for metric_name, getter in [("CC1/2", lambda s: s.cc_half),
                             ("R_SPLIT", lambda s: s.r_split)]:
    print(f"\n{'='*110}")
    print(f"PER-SHELL {metric_name}")
    print(f"{'='*110}")
    header = f"  {'d_max':>7s} {'d_min':>7s}"
    for name, _ in shell_data:
        header += f"  {name:>14s}"
    print(header)
    print(f"  {'-'*100}")
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


# ── CSV ──────────────────────────────────────────────────────
csv_path = OUTDIR / "benchmark_refactored_genpoiss_shells.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "d_max", "d_min", "n_matched", "cc_half", "cc_star", "r_split"])
    for name, shells in shell_data:
        for sh in shells:
            w.writerow([name, f"{sh.d_max:.4f}", f"{sh.d_min:.4f}",
                        sh.n_matched, f"{sh.cc_half:.6f}",
                        f"{sh.cc_star:.6f}", f"{sh.r_split:.6f}"])
print(f"\nWrote {csv_path}")


# ── Plots ────────────────────────────────────────────────────
def d_to_x(d):
    return 1.0 / np.asarray(d) ** 2


def setup_res_axis(ax):
    tick_d = np.array([10, 5, 3, 2, 1.5, 1.2, 1.0, 0.9])
    ax.set_xticks(d_to_x(tick_d))
    ax.set_xticklabels([f"{d:g}" for d in tick_d])
    ax.set_xlim(d_to_x(D_MAX), d_to_x(D_MIN))
    ax.set_xlabel("Resolution (Å)")


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


# Plot 1: CC1/2 and R_split vs resolution (all methods)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for (md, _), (name, shells) in zip(halves_matched, shell_data):
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    kw = dict(color=md.spec.color, marker=md.spec.marker, ms=5, lw=1.8, label=name)
    ax1.plot(d_to_x(d_mid), [s.cc_half for s in shells], "-", **kw)
    ax2.plot(d_to_x(d_mid), [s.r_split for s in shells], "-", **kw)

ax1.set_ylabel("CC1/2"); ax1.set_title("CC1/2 vs Resolution")
ax1.legend(loc="lower left"); ax1.set_ylim(-0.05, 1.05); ax1.grid(True, alpha=0.3)
setup_res_axis(ax1)
ax2.set_ylabel("R_split"); ax2.set_title("R_split vs Resolution")
ax2.legend(loc="upper left"); ax2.set_ylim(0, None); ax2.grid(True, alpha=0.3)
setup_res_axis(ax2)
fig.suptitle(f"Split-Half Precision ({D_MIN}–{D_MAX} Å)", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_refactored_genpoiss.png", dpi=150)
print(f"Saved benchmark_refactored_genpoiss.png")
plt.close(fig)


# Plot 2: Self-consistency F-scatter (one subplot per method)
n = len(halves_matched)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
if n == 1:
    axes = [axes]
for ax, (md, m) in zip(axes, halves_matched):
    F1, F2 = m.F1.numpy(), m.F2.numpy()
    hi = max(F1.max(), F2.max())
    density_scatter(ax, F1, F2)
    ax.plot([0, hi], [0, hi], "r-", lw=1, alpha=0.7)
    ax.set_xlabel("F half 1"); ax.set_ylabel("F half 2")
    ax.set_title(f"{md.spec.name}\nCC1/2={cc_half(m):.4f}  R={r_split(m):.4f}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, hi * 1.05); ax.set_ylim(0, hi * 1.05)
fig.suptitle("Split-Half Self-Consistency", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "scatter_self_consistency.png", dpi=150, bbox_inches="tight")
print(f"Saved scatter_self_consistency.png")
plt.close(fig)


# Plot 3: Alpha vs resolution, full dataset (only for I0-field methods)
fig_rows = [md for md in methods if md.spec.field == "I0" and md.full_or_merge is not None]
if fig_rows:
    fig, axes = plt.subplots(1, len(fig_rows), figsize=(6 * len(fig_rows), 5), squeeze=False)
    for ax, md in zip(axes[0], fig_rows):
        d = md.full_or_merge
        d_sp = d["d_spacing"].numpy()
        alpha = d["alpha"].numpy()
        I0 = d["I0"].numpy()
        sc = ax.scatter(d_to_x(d_sp), alpha, c=np.log10(I0.clip(0.1)),
                        s=3, alpha=0.4, cmap="viridis", vmin=-1, vmax=4, rasterized=True)
        plt.colorbar(sc, ax=ax, label="log₁₀(I₀)", shrink=0.8)
        edges = np.percentile(d_sp, np.linspace(0, 100, N_SHELLS + 1))
        d_mid, a_med = [], []
        for i in range(N_SHELLS):
            mask = (d_sp >= edges[i]) & (d_sp < edges[i + 1])
            if mask.sum() < 2:
                continue
            d_mid.append(0.5 * (edges[i] + edges[i + 1]))
            a_med.append(np.median(alpha[mask]))
        ax.plot(d_to_x(d_mid), a_med, "r-o", ms=4, lw=2, label="median", zorder=5)
        ax.set_ylabel("α"); ax.set_title(md.spec.name, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(3.0, np.percentile(alpha, 99)))
        setup_res_axis(ax)
    fig.suptitle("Alpha vs Resolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "scatter_alpha.png", dpi=150)
    print(f"Saved scatter_alpha.png")
    plt.close(fig)


print("\nDone.")
