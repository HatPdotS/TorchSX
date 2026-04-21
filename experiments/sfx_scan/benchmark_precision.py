#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 4:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/benchmark_precision.out

"""Split-half precision benchmark: Poisson+GenGauss EM vs Partialator.

Compares CC1/2 and R_split per resolution shell for:
  - Partialator unity (pre-computed)
  - Partialator xsphere (pre-computed)
  - Median merge (baseline)
  - EM + ScaleB with resolution-dependent alpha prior (new)
  - EM + ScaleB with Laplacian alpha smoothing (old)
"""

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
    compute_hkl_index, merge_median,
    match_halves, shell_stats, cc_half, cc_star, r_split,
    format_shell_table, write_shell_csv,
    fit_em, EMConfig, ScaleBCorrection,
)
from torchsx.merge._stats import spearman_cc

# Patch quadrature to 51 points (matches run_em_201q.py)
import torchsx.merge._dstep as dstep
_orig_quad = dstep._prepare_quadrature
def _patched_quad(alpha, beta, obs_counts, sigma2_poisson, n_quad=21, device=None):
    return _orig_quad(alpha, beta, obs_counts, sigma2_poisson, n_quad=51, device=device)
dstep._prepare_quadrature = _patched_quad


# ── Configuration ────────────────────────────────────────────
STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
PART_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/partialator")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/benchmark_precision")
OUTDIR.mkdir(parents=True, exist_ok=True)

DELAYS = [-91, -136, -182, -227]  # same 4 as partialator reference
SPACEGROUP = "P-1"
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SEED = 42
N_SHELLS = 20
D_MIN, D_MAX = 0.9, 10.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ── Helpers ──────────────────────────────────────────────────
def parse_crystfel_hkl(path):
    """Parse CrystFEL .hkl file -> (hkl, I) tensors."""
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
    """Load partialator split-half .hkl files -> MatchedHalves."""
    from torchref.symmetry import canonicalize_hkl

    hkl1, I1 = parse_crystfel_hkl(half1_path)
    hkl2, I2 = parse_crystfel_hkl(half2_path)
    hkl1, _, _, s1 = canonicalize_hkl(hkl1.int(), "P -1", include_friedel=True)
    hkl2, _, _, s2 = canonicalize_hkl(hkl2.int(), "P -1", include_friedel=True)
    F1 = I1[s1].clamp(min=0).sqrt()
    F2 = I2[s2].clamp(min=0).sqrt()
    return match_halves(hkl1.to(torch.int16), F1,
                        hkl2.to(torch.int16), F2, CELL, d_min=D_MIN, d_max=D_MAX)


def to_F(I):
    return I.clamp(min=0).sqrt()


def d_to_x(d):
    return 1.0 / np.asarray(d) ** 2


def setup_res_axis(ax, xlabel=True):
    tick_d = np.array([10, 5, 3, 2, 1.5, 1.2, 1.0, 0.9])
    tick_d = tick_d[(tick_d >= D_MIN) & (tick_d <= D_MAX)]
    ax.set_xticks(d_to_x(tick_d))
    ax.set_xticklabels([f"{d:g}" for d in tick_d])
    ax.set_xlim(d_to_x(D_MAX), d_to_x(D_MIN))
    if xlabel:
        ax.set_xlabel("Resolution (Å)")


# ══════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════
print("\n── Loading streams ──")
streams = []
for d in DELAYS:
    p = STREAM_DIR / f"delay_{d}.sfx"
    sx = SFXStream.load(p)
    print(f"  {p.name}: {sx.n_crystals} crystals, {sx.n_observations} obs")
    streams.append(sx)

s = SFXStream.concat(streams)
print(f"Total: {s.n_crystals} crystals, {s.n_observations} observations")

# ══════════════════════════════════════════════════════════════
# 2. Split into halves
# ══════════════════════════════════════════════════════════════
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"Split: half1={h1.n_crystals} crystals, half2={h2.n_crystals} crystals")

# Also need HKL index for median merge
core = s._core
hkl_idx = compute_hkl_index(core.hkl)
n_c = core.astar.shape[0]
gen = torch.Generator().manual_seed(SEED)
assignment = torch.randint(2, (n_c,), generator=gen)
mask_A = assignment == 0
obs_A = mask_A[core.crystal_id.long()]
obs_B = ~obs_A

# ══════════════════════════════════════════════════════════════
# 3. Partialator references
# ══════════════════════════════════════════════════════════════
print("\n── Partialator unity (independent halves) ──")
m_part_unity = load_partialator_halves(PART_DIR / "ind_half1.hkl", PART_DIR / "ind_half2.hkl")
cc_pu = cc_half(m_part_unity)
r_pu = r_split(m_part_unity)
sp_pu = spearman_cc(m_part_unity)
sh_pu = shell_stats(m_part_unity, n_shells=N_SHELLS)
print(f"  {m_part_unity.F1.numel()} matched, CC1/2={cc_pu:.4f}, Sp={sp_pu:.4f}, R={r_pu:.4f}")

m_part_xsph = None
sh_xsph = []
if (PART_DIR / "xsphere.hkl1").exists():
    print("\n── Partialator xsphere (internal split) ──")
    m_part_xsph = load_partialator_halves(PART_DIR / "xsphere.hkl1", PART_DIR / "xsphere.hkl2")
    cc_xsph = cc_half(m_part_xsph)
    r_xsph = r_split(m_part_xsph)
    sp_xsph = spearman_cc(m_part_xsph)
    sh_xsph = shell_stats(m_part_xsph, n_shells=N_SHELLS)
    print(f"  {m_part_xsph.F1.numel()} matched, CC1/2={cc_xsph:.4f}, Sp={sp_xsph:.4f}, R={r_xsph:.4f}")
else:
    print("  (xsphere results not available)")

# ══════════════════════════════════════════════════════════════
# 4. Median merge
# ══════════════════════════════════════════════════════════════
print("\n── Median merge ──")
t0 = time.perf_counter()
I_A_md, _, _ = merge_median(hkl_idx.hkl_id[obs_A], core.I[obs_A], hkl_idx.n_unique, min_mult=2)
I_B_md, _, _ = merge_median(hkl_idx.hkl_id[obs_B], core.I[obs_B], hkl_idx.n_unique, min_mult=2)
v = torch.isfinite(I_A_md) & torch.isfinite(I_B_md) & (I_A_md > 0) & (I_B_md > 0)
idx_v = torch.where(v)[0]
m_med = match_halves(hkl_idx.unique_hkl[idx_v], to_F(I_A_md[idx_v]),
                     hkl_idx.unique_hkl[idx_v], to_F(I_B_md[idx_v]),
                     CELL, d_min=D_MIN, d_max=D_MAX)
cc_med = cc_half(m_med)
r_med = r_split(m_med)
sp_med = spearman_cc(m_med)
sh_med = shell_stats(m_med, n_shells=N_SHELLS)
print(f"  {m_med.F1.numel()} matched, CC1/2={cc_med:.4f}, Sp={sp_med:.4f}, R={r_med:.4f}  "
      f"({time.perf_counter() - t0:.1f}s)")

# ══════════════════════════════════════════════════════════════
# 5. EM with alpha prior (new, default lambda=1.0)
# ══════════════════════════════════════════════════════════════
print("\n── EM + ScaleB + alpha prior (lambda=1.0) ──")
config_prior = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=1.0)

t0 = time.perf_counter()
print("  Half 1...")
dist_A_prior = fit_em(h1, config=config_prior, corrections=[ScaleBCorrection()], device=device)
print("  Half 2...")
dist_B_prior = fit_em(h2, config=config_prior, corrections=[ScaleBCorrection()], device=device)
t_prior = time.perf_counter() - t0
print(f"  Total: {t_prior:.0f}s")

m_prior = match_halves(dist_A_prior.hkl, to_F(dist_A_prior.I0),
                       dist_B_prior.hkl, to_F(dist_B_prior.I0),
                       CELL, d_min=D_MIN, d_max=D_MAX)
cc_pr = cc_half(m_prior)
r_pr = r_split(m_prior)
sp_pr = spearman_cc(m_prior)
sh_pr = shell_stats(m_prior, n_shells=N_SHELLS)
print(f"  {m_prior.F1.numel()} matched, CC1/2={cc_pr:.4f}, Sp={sp_pr:.4f}, R={r_pr:.4f}")

# ══════════════════════════════════════════════════════════════
# 6. EM with Laplacian smoothing (old, lambda=0)
# ══════════════════════════════════════════════════════════════
print("\n── EM + ScaleB + Laplacian alpha (lambda=0) ──")
config_lapl = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)

t0 = time.perf_counter()
print("  Half 1...")
dist_A_lapl = fit_em(h1, config=config_lapl, corrections=[ScaleBCorrection()], device=device)
print("  Half 2...")
dist_B_lapl = fit_em(h2, config=config_lapl, corrections=[ScaleBCorrection()], device=device)
t_lapl = time.perf_counter() - t0
print(f"  Total: {t_lapl:.0f}s")

m_lapl = match_halves(dist_A_lapl.hkl, to_F(dist_A_lapl.I0),
                      dist_B_lapl.hkl, to_F(dist_B_lapl.I0),
                      CELL, d_min=D_MIN, d_max=D_MAX)
cc_lp = cc_half(m_lapl)
r_lp = r_split(m_lapl)
sp_lp = spearman_cc(m_lapl)
sh_lp = shell_stats(m_lapl, n_shells=N_SHELLS)
print(f"  {m_lapl.F1.numel()} matched, CC1/2={cc_lp:.4f}, Sp={sp_lp:.4f}, R={r_lp:.4f}")


# ══════════════════════════════════════════════════════════════
# 7. Summary tables
# ══════════════════════════════════════════════════════════════
methods = [
    ("Part. unity",   cc_pu,   sp_pu,   r_pu,   sh_pu),
]
if m_part_xsph is not None:
    methods.append(("Part. xsphere", cc_xsph, sp_xsph, r_xsph, sh_xsph))
methods += [
    ("Median",          cc_med, sp_med, r_med, sh_med),
    ("EM alpha-prior",  cc_pr,  sp_pr,  r_pr,  sh_pr),
    ("EM Laplacian",    cc_lp,  sp_lp,  r_lp,  sh_lp),
]

print(f"\n{'='*70}")
print("GLOBAL METRICS")
print(f"{'='*70}")
print(f"  {'Method':>20s}  {'CC1/2':>7s}  {'Spear':>7s}  {'CC*':>7s}  {'R_split':>8s}")
print(f"  {'-'*55}")
for name, cc, sp, rs, _ in methods:
    ccs = cc_star(cc) if cc > 0 else 0.0
    print(f"  {name:>20s}  {cc:>7.4f}  {sp:>7.4f}  {ccs:>7.4f}  {rs:>8.4f}")

# Per-shell table
ref_shells = sh_pr  # use alpha-prior as reference for shell boundaries
print(f"\n{'='*100}")
print("PER-SHELL CC1/2")
print(f"{'='*100}")
header = f"  {'d_max':>7s} {'d_min':>7s}"
for name, _, _, _, _ in methods:
    header += f"  {name:>14s}"
print(header)
print(f"  {'-'*95}")
for i in range(N_SHELLS):
    s_ref = ref_shells[i] if i < len(ref_shells) else None
    if s_ref is None:
        continue
    row = f"  {s_ref.d_max:>7.2f} {s_ref.d_min:>7.2f}"
    for _, _, _, _, shells in methods:
        if i < len(shells):
            row += f"  {shells[i].cc_half:>14.4f}"
        else:
            row += f"  {'---':>14s}"
    print(row)

print(f"\n{'='*100}")
print("PER-SHELL R_SPLIT")
print(f"{'='*100}")
header = f"  {'d_max':>7s} {'d_min':>7s}"
for name, _, _, _, _ in methods:
    header += f"  {name:>14s}"
print(header)
print(f"  {'-'*95}")
for i in range(N_SHELLS):
    s_ref = ref_shells[i] if i < len(ref_shells) else None
    if s_ref is None:
        continue
    row = f"  {s_ref.d_max:>7.2f} {s_ref.d_min:>7.2f}"
    for _, _, _, _, shells in methods:
        if i < len(shells):
            row += f"  {shells[i].r_split:>14.4f}"
        else:
            row += f"  {'---':>14s}"
    print(row)


# ══════════════════════════════════════════════════════════════
# 8. CSV output
# ══════════════════════════════════════════════════════════════
csv_path = OUTDIR / "benchmark_precision_shells.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "d_max", "d_min", "n_matched", "cc_half", "cc_star", "r_split"])
    for name, _, _, _, shells in methods:
        for sh in shells:
            w.writerow([name, f"{sh.d_max:.4f}", f"{sh.d_min:.4f}",
                        sh.n_matched, f"{sh.cc_half:.6f}", f"{sh.cc_star:.6f}",
                        f"{sh.r_split:.6f}"])
print(f"\nWrote {csv_path}")


# ══════════════════════════════════════════════════════════════
# 9. Plots
# ══════════════════════════════════════════════════════════════
colors = {
    "Part. unity": "gray",
    "Part. xsphere": "dimgray",
    "Median": "C2",
    "EM alpha-prior": "C0",
    "EM Laplacian": "C1",
}
markers = {
    "Part. unity": "s",
    "Part. xsphere": "D",
    "Median": "^",
    "EM alpha-prior": "o",
    "EM Laplacian": "v",
}

# ── Plot 1: CC1/2 vs resolution ──
fig, ax = plt.subplots(figsize=(10, 6))
for name, _, _, _, shells in methods:
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    cc_vals = [s.cc_half for s in shells]
    ax.plot(d_to_x(d_mid), cc_vals, "-", color=colors.get(name, "k"),
            marker=markers.get(name, "o"), ms=5, lw=1.5, label=name)
ax.set_ylabel("CC1/2")
ax.set_title("Split-Half Precision: CC1/2 vs Resolution")
ax.legend(loc="lower left")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
setup_res_axis(ax)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_precision_cc.png", dpi=150)
print(f"Saved {OUTDIR / 'benchmark_precision_cc.png'}")
plt.close(fig)

# ── Plot 2: R_split vs resolution ──
fig, ax = plt.subplots(figsize=(10, 6))
for name, _, _, _, shells in methods:
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    r_vals = [s.r_split for s in shells]
    ax.plot(d_to_x(d_mid), r_vals, "-", color=colors.get(name, "k"),
            marker=markers.get(name, "o"), ms=5, lw=1.5, label=name)
ax.set_ylabel("R_split")
ax.set_title("Split-Half Precision: R_split vs Resolution")
ax.legend(loc="upper left")
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
setup_res_axis(ax)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_precision_rsplit.png", dpi=150)
print(f"Saved {OUTDIR / 'benchmark_precision_rsplit.png'}")
plt.close(fig)

# ── Plot 3: Alpha vs resolution diagnostic ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (dist, label) in zip(axes, [
    (dist_A_prior, "EM alpha-prior (half 1)"),
    (dist_A_lapl, "EM Laplacian (half 1)"),
]):
    d = dist.d_spacing.numpy()
    a = dist.alpha.numpy()
    I0 = dist.I0.numpy()

    # Color by log(I0)
    sc = ax.scatter(d_to_x(d), a, c=np.log10(I0.clip(0.1)), s=3, alpha=0.4,
                    cmap="viridis", vmin=-1, vmax=4)
    ax.set_ylabel("alpha")
    ax.set_title(label)
    ax.set_ylim(0, max(3.0, np.percentile(a, 99)))
    ax.grid(True, alpha=0.3)
    setup_res_axis(ax)

fig.colorbar(sc, ax=axes, label="log10(I0)", shrink=0.8)
fig.suptitle("Alpha vs Resolution (colored by intensity)", fontsize=12)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_precision_alpha.png", dpi=150)
print(f"Saved {OUTDIR / 'benchmark_precision_alpha.png'}")
plt.close(fig)

# ── Plot 4: Alpha shell means ──
fig, ax = plt.subplots(figsize=(10, 5))
for dist, label, color in [
    (dist_A_prior, "alpha-prior (half 1)", "C0"),
    (dist_A_lapl, "Laplacian (half 1)", "C1"),
]:
    d = dist.d_spacing.numpy()
    a = dist.alpha.numpy()
    I0 = dist.I0.numpy()
    # Bin by resolution
    edges = np.percentile(d, np.linspace(0, 100, N_SHELLS + 1))
    d_mid, a_mean, a_mean_strong = [], [], []
    for i in range(N_SHELLS):
        mask = (d >= edges[i]) & (d < edges[i + 1])
        if mask.sum() < 2:
            continue
        d_mid.append(0.5 * (edges[i] + edges[i + 1]))
        a_mean.append(np.median(a[mask]))
        strong = mask & (I0 > 30)
        a_mean_strong.append(np.median(a[strong]) if strong.sum() > 5 else np.nan)
    ax.plot(d_to_x(d_mid), a_mean, "-o", color=color, ms=5, lw=1.5, label=f"{label} (all)")
    ax.plot(d_to_x(d_mid), a_mean_strong, "--s", color=color, ms=4, lw=1, alpha=0.7,
            label=f"{label} (I0>30)")

ax.set_ylabel("Median alpha")
ax.set_title("Alpha vs Resolution: Prior vs Laplacian")
ax.legend()
ax.grid(True, alpha=0.3)
setup_res_axis(ax)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_precision_alpha_shells.png", dpi=150)
print(f"Saved {OUTDIR / 'benchmark_precision_alpha_shells.png'}")
plt.close(fig)

print("\nDone.")
