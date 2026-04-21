#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 4:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/benchmark_final.out

"""Final precision benchmark: EM poisson_log vs fit_gengauss vs Partialator."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsx.stream import SFXStream
from torchsx.merge import (
    match_halves, shell_stats, cc_half, cc_star, r_split,
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
N_SHELLS = 20

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


def d_to_x(d):
    return 1.0 / np.asarray(d) ** 2


def setup_res_axis(ax):
    tick_d = np.array([10, 5, 3, 2, 1.5, 1.2, 1.0, 0.9])
    ax.set_xticks(d_to_x(tick_d))
    ax.set_xticklabels([f"{d:g}" for d in tick_d])
    ax.set_xlim(d_to_x(10.0), d_to_x(0.9))
    ax.set_xlabel("Resolution (Å)")


# ══════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════
print("\n── Loading streams ──")
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
print(f"Total: {s.n_crystals} crystals, {s.n_observations} observations")

h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"Split: {h1.n_crystals} / {h2.n_crystals}")

# ══════════════════════════════════════════════════════════════
# 2. Partialator references
# ══════════════════════════════════════════════════════════════
print("\n── Partialator unity ──")
m_pu = load_partialator_halves(PART_DIR / "ind_half1.hkl", PART_DIR / "ind_half2.hkl")
sh_pu = shell_stats(m_pu, n_shells=N_SHELLS)
print(f"  CC1/2={cc_half(m_pu):.4f}  R={r_split(m_pu):.4f}  n={m_pu.F1.numel()}")

m_xsph = None
sh_xsph = []
if (PART_DIR / "xsphere.hkl1").exists():
    print("\n── Partialator xsphere ──")
    m_xsph = load_partialator_halves(PART_DIR / "xsphere.hkl1", PART_DIR / "xsphere.hkl2")
    sh_xsph = shell_stats(m_xsph, n_shells=N_SHELLS)
    print(f"  CC1/2={cc_half(m_xsph):.4f}  R={r_split(m_xsph):.4f}  n={m_xsph.F1.numel()}")

# ══════════════════════════════════════════════════════════════
# 3. fit_gengauss (log-space IRLS)
# ══════════════════════════════════════════════════════════════
print("\n── fit_gengauss + EM×10 + ScaleB ──")
FIT_KW = dict(min_obs=20, n_smooth_iter=15, smooth_lambda=0.3)
t0 = time.perf_counter()
fg1 = fit_gengauss(h1, n_em_cycles=10, scale_crystals=True, **FIT_KW)
fg2 = fit_gengauss(h2, n_em_cycles=10, scale_crystals=True, **FIT_KW)
t_gg = time.perf_counter() - t0
m_gg = match_halves(fg1.hkl, to_F(fg1.I0), fg2.hkl, to_F(fg2.I0), CELL)
sh_gg = shell_stats(m_gg, n_shells=N_SHELLS)
print(f"  CC1/2={cc_half(m_gg):.4f}  R={r_split(m_gg):.4f}  n={m_gg.F1.numel()}  ({t_gg:.0f}s)")

# ══════════════════════════════════════════════════════════════
# 4. fit_em with poisson_log (corrected bg)
# ══════════════════════════════════════════════════════════════
print("\n── fit_em poisson_log + ScaleB ──")
cfg = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)
t0 = time.perf_counter()
de1 = fit_em(h1, config=cfg, corrections=[ScaleBCorrection()], device=device)
de2 = fit_em(h2, config=cfg, corrections=[ScaleBCorrection()], device=device)
t_em = time.perf_counter() - t0
m_em = match_halves(de1.hkl, to_F(de1.I0), de2.hkl, to_F(de2.I0), CELL)
sh_em = shell_stats(m_em, n_shells=N_SHELLS)
print(f"  CC1/2={cc_half(m_em):.4f}  R={r_split(m_em):.4f}  n={m_em.F1.numel()}  ({t_em:.0f}s)")

# ══════════════════════════════════════════════════════════════
# 5. Tables
# ══════════════════════════════════════════════════════════════
methods = [
    ("Part. unity",  m_pu,   sh_pu),
]
if m_xsph is not None:
    methods.append(("Part. xsphere", m_xsph, sh_xsph))
methods += [
    ("GenGauss",     m_gg,   sh_gg),
    ("EM poisson",   m_em,   sh_em),
]

print(f"\n{'='*70}")
print("GLOBAL METRICS")
print(f"{'='*70}")
print(f"  {'Method':>20s}  {'CC1/2':>7s}  {'CC*':>7s}  {'R_split':>8s}  {'N':>6s}")
print(f"  {'-'*55}")
for name, m, _ in methods:
    cc = cc_half(m)
    print(f"  {name:>20s}  {cc:>7.4f}  {cc_star(cc):>7.4f}  {r_split(m):>8.4f}  {m.F1.numel():>6d}")

# Per-shell CC1/2
print(f"\n{'='*100}")
print("PER-SHELL CC1/2")
print(f"{'='*100}")
header = f"  {'d_max':>7s} {'d_min':>7s}"
for name, _, _ in methods:
    header += f"  {name:>14s}"
print(header)
print(f"  {'-'*90}")
for i in range(N_SHELLS):
    row = ""
    d_max = d_min = None
    for _, _, shells in methods:
        if i < len(shells):
            d_max = shells[i].d_max
            d_min = shells[i].d_min
            break
    if d_max is None:
        continue
    row = f"  {d_max:>7.2f} {d_min:>7.2f}"
    for _, _, shells in methods:
        if i < len(shells):
            row += f"  {shells[i].cc_half:>14.4f}"
        else:
            row += f"  {'---':>14s}"
    print(row)

# Per-shell R_split
print(f"\n{'='*100}")
print("PER-SHELL R_SPLIT")
print(f"{'='*100}")
header = f"  {'d_max':>7s} {'d_min':>7s}"
for name, _, _ in methods:
    header += f"  {name:>14s}"
print(header)
print(f"  {'-'*90}")
for i in range(N_SHELLS):
    d_max = d_min = None
    for _, _, shells in methods:
        if i < len(shells):
            d_max = shells[i].d_max
            d_min = shells[i].d_min
            break
    if d_max is None:
        continue
    row = f"  {d_max:>7.2f} {d_min:>7.2f}"
    for _, _, shells in methods:
        if i < len(shells):
            row += f"  {shells[i].r_split:>14.4f}"
        else:
            row += f"  {'---':>14s}"
    print(row)

# ══════════════════════════════════════════════════════════════
# 6. CSV
# ══════════════════════════════════════════════════════════════
csv_path = OUTDIR / "benchmark_final_shells.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "d_max", "d_min", "n_matched", "cc_half", "cc_star", "r_split"])
    for name, _, shells in methods:
        for sh in shells:
            w.writerow([name, f"{sh.d_max:.4f}", f"{sh.d_min:.4f}",
                        sh.n_matched, f"{sh.cc_half:.6f}", f"{sh.cc_star:.6f}",
                        f"{sh.r_split:.6f}"])
print(f"\nWrote {csv_path}")

# ══════════════════════════════════════════════════════════════
# 7. Plots
# ══════════════════════════════════════════════════════════════
colors = {
    "Part. unity": "gray",
    "Part. xsphere": "silver",
    "GenGauss": "C1",
    "EM poisson": "C0",
}
markers = {"Part. unity": "s", "Part. xsphere": "D", "GenGauss": "^", "EM poisson": "o"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for name, _, shells in methods:
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    cc_vals = [s.cc_half for s in shells]
    r_vals = [s.r_split for s in shells]
    kw = dict(color=colors.get(name, "k"), marker=markers.get(name, "o"),
              ms=5, lw=1.8, label=name)
    ax1.plot(d_to_x(d_mid), cc_vals, "-", **kw)
    ax2.plot(d_to_x(d_mid), r_vals, "-", **kw)

ax1.set_ylabel("CC1/2")
ax1.set_title("Split-Half CC1/2 vs Resolution")
ax1.legend(loc="lower left")
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)
setup_res_axis(ax1)

ax2.set_ylabel("R_split")
ax2.set_title("Split-Half R_split vs Resolution")
ax2.legend(loc="upper left")
ax2.set_ylim(0, None)
ax2.grid(True, alpha=0.3)
setup_res_axis(ax2)

fig.suptitle("Precision Benchmark: EM Poisson-GenGauss vs Partialator vs GenGauss", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_final.png", dpi=150)
print(f"Saved {OUTDIR / 'benchmark_final.png'}")
plt.close(fig)

print("\nDone.")
