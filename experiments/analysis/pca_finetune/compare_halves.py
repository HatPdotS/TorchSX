#!/usr/bin/env python
"""Compare F² from independent half-dataset refinements.

Loads checkpoint pairs (joint or decoupled), matches by unique hkl,
and computes CC, R-factor, ratio statistics.

Usage:
    python compare_halves.py                  # joint_refine_state_{A,B}.pt
    python compare_halves.py --prefix decoupled_refine_state  # decoupled
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, default="joint_refine_state",
                    help="Checkpoint prefix (default: joint_refine_state)")
args = parser.parse_args()

OUT_DIR = Path(__file__).resolve().parent

# Load both checkpoints
state_A = torch.load(OUT_DIR / f"{args.prefix}_A.pt", map_location="cpu", weights_only=True)
state_B = torch.load(OUT_DIR / f"{args.prefix}_B.pt", map_location="cpu", weights_only=True)
print(f"Loaded: {args.prefix}_A.pt, {args.prefix}_B.pt")

hkl_A = state_A["unique_hkl"]  # (N_A, 3)
hkl_B = state_B["unique_hkl"]  # (N_B, 3)

# F² — support both checkpoint formats
if "log_global_scale" in state_A:
    F_sq_A = torch.exp(state_A["log_global_scale"] + 2.0 * state_A["log_F"])
    F_sq_B = torch.exp(state_B["log_global_scale"] + 2.0 * state_B["log_F"])
elif "F_sq_merged" in state_A:
    F_sq_A = state_A["F_sq_merged"]
    F_sq_B = state_B["F_sq_merged"]
else:
    raise KeyError("Unknown checkpoint format")

print(f"Half A: {len(F_sq_A):,} reflections, F² median={F_sq_A.median():.2f}")
print(f"Half B: {len(F_sq_B):,} reflections, F² median={F_sq_B.median():.2f}")

# Match hkl between halves using a dict
hkl_to_idx_B = {}
for i, hkl in enumerate(hkl_B.tolist()):
    hkl_to_idx_B[tuple(hkl)] = i

matched_A, matched_B = [], []
for i, hkl in enumerate(hkl_A.tolist()):
    j = hkl_to_idx_B.get(tuple(hkl))
    if j is not None:
        matched_A.append(i)
        matched_B.append(j)

matched_A = torch.tensor(matched_A)
matched_B = torch.tensor(matched_B)
n_matched = len(matched_A)
print(f"Matched: {n_matched:,} reflections")

fA = F_sq_A[matched_A]
fB = F_sq_B[matched_B]
hkl_matched = hkl_A[matched_A]

# Filter out floor values
good = (fA > 0.02) & (fB > 0.02)
fA_g = fA[good]
fB_g = fB[good]
print(f"Above floor: {good.sum():,} reflections")

# ── Statistics ──
# Pearson CC
mean_A = fA_g.mean()
mean_B = fB_g.mean()
cov = ((fA_g - mean_A) * (fB_g - mean_B)).sum()
std_A = (fA_g - mean_A).pow(2).sum().sqrt()
std_B = (fB_g - mean_B).pow(2).sum().sqrt()
CC = (cov / (std_A * std_B)).item()

# CC on log scale
log_A = torch.log(fA_g)
log_B = torch.log(fB_g)
mean_lA = log_A.mean()
mean_lB = log_B.mean()
cov_l = ((log_A - mean_lA) * (log_B - mean_lB)).sum()
std_lA = (log_A - mean_lA).pow(2).sum().sqrt()
std_lB = (log_B - mean_lB).pow(2).sum().sqrt()
CC_log = (cov_l / (std_lA * std_lB)).item()

# R-factor: Σ|F²_A - F²_B| / Σ(F²_A + F²_B) * 2
R_iso = (2 * (fA_g - fB_g).abs().sum() / (fA_g + fB_g).sum()).item()

# Ratio statistics
ratio = fA_g / fB_g
log_ratio = torch.log(ratio)

print(f"\n--- Agreement statistics ---")
print(f"  CC(F²):       {CC:.4f}")
print(f"  CC(log F²):   {CC_log:.4f}")
print(f"  R_iso:         {R_iso:.4f}")
print(f"  Ratio A/B:     median={ratio.median():.4f}, mean={ratio.mean():.4f}")
print(f"  log(A/B):      mean={log_ratio.mean():.4f}, std={log_ratio.std():.4f}")

# ── Resolution-dependent CC ──
# Compute d-spacing for matched reflections
# Use a representative cell (from state_A's unique_hkl)
# We'll bin by |h|² as a proxy for 1/d²
h_sq = (hkl_matched.float() ** 2).sum(dim=1)  # rough proxy
n_bins = 10
sorted_idx = h_sq.argsort()
bin_size = len(sorted_idx) // n_bins

print(f"\n--- Resolution-dependent CC (binned by |h|²) ---")
print(f"  {'Bin':>4s}  {'|h|²_range':>15s}  {'N_refl':>7s}  {'CC':>8s}  {'CC_log':>8s}  {'R_iso':>8s}")
bin_ccs = []
bin_centers = []
for b in range(n_bins):
    start = b * bin_size
    end = (b + 1) * bin_size if b < n_bins - 1 else len(sorted_idx)
    idx = sorted_idx[start:end]
    idx_good = idx[good[idx]]
    if len(idx_good) < 5:
        continue
    a = fA[idx_good]
    bb = fB[idx_good]
    mA, mB = a.mean(), bb.mean()
    c = ((a - mA) * (bb - mB)).sum()
    sA = (a - mA).pow(2).sum().sqrt()
    sB = (bb - mB).pow(2).sum().sqrt()
    cc = (c / (sA * sB + 1e-12)).item()
    la, lb = torch.log(a), torch.log(bb)
    mlA, mlB = la.mean(), lb.mean()
    cl = ((la - mlA) * (lb - mlB)).sum()
    slA = (la - mlA).pow(2).sum().sqrt()
    slB = (lb - mlB).pow(2).sum().sqrt()
    cc_l = (cl / (slA * slB + 1e-12)).item()
    r = (2 * (a - bb).abs().sum() / (a + bb).sum()).item()
    h2_min = h_sq[idx].min().item()
    h2_max = h_sq[idx].max().item()
    print(f"  {b:4d}  {h2_min:6.0f} — {h2_max:6.0f}  {len(idx_good):7d}  {cc:8.4f}  {cc_l:8.4f}  {r:8.4f}")
    bin_ccs.append(cc_l)
    bin_centers.append((h2_min + h2_max) / 2)

# ── Visualization ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: F² scatter (linear)
ax = axes[0, 0]
ax.scatter(fA_g.numpy(), fB_g.numpy(), s=1, alpha=0.2, rasterized=True)
lim = max(fA_g.quantile(0.99).item(), fB_g.quantile(0.99).item())
ax.plot([0, lim], [0, lim], "r-", lw=0.8)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("F² (Half A)")
ax.set_ylabel("F² (Half B)")
ax.set_title(f"F² agreement (CC={CC:.4f})")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Plot 2: F² scatter (log)
ax = axes[0, 1]
ax.scatter(log_A.numpy(), log_B.numpy(), s=1, alpha=0.2, rasterized=True)
lims = [log_A.min().item(), log_A.max().item()]
ax.plot(lims, lims, "r-", lw=0.8)
ax.set_xlabel("log(F²) Half A")
ax.set_ylabel("log(F²) Half B")
ax.set_title(f"log(F²) agreement (CC={CC_log:.4f})")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Plot 3: Ratio histogram
ax = axes[0, 2]
lr_np = log_ratio.numpy()
ax.hist(lr_np, bins=100, range=(-2, 2), alpha=0.7, color="steelblue")
ax.axvline(0, color="r", ls="--", lw=0.8)
ax.set_xlabel("log(F²_A / F²_B)")
ax.set_ylabel("Reflections")
ax.set_title(f"Ratio distribution (σ={log_ratio.std():.3f})")
ax.grid(True, alpha=0.3)

# Plot 4: Resolution-dependent CC
ax = axes[1, 0]
if bin_ccs:
    ax.plot(bin_centers, bin_ccs, "bo-", lw=1.5)
    ax.set_xlabel("|h|² (proxy for 1/d²)")
    ax.set_ylabel("CC(log F²)")
    ax.set_title("Resolution-dependent CC")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

# Plot 5: F² distribution comparison
ax = axes[1, 1]
bins = np.linspace(-3, 8, 100)
ax.hist(log_A.numpy(), bins=bins, alpha=0.5, label="Half A", density=True)
ax.hist(log_B.numpy(), bins=bins, alpha=0.5, label="Half B", density=True)
ax.set_xlabel("log(F²)")
ax.set_ylabel("Density")
ax.set_title("F² distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 6: Scale distribution comparison (if available)
ax = axes[1, 2]
if "log_G" in state_A and "log_G" in state_B:
    log_G_A = state_A["log_G"].numpy()
    log_G_B = state_B["log_G"].numpy()
    bins_s = np.linspace(min(log_G_A.min(), log_G_B.min()),
                         max(log_G_A.max(), log_G_B.max()), 100)
    ax.hist(log_G_A, bins=bins_s, alpha=0.5, label=f"Half A (σ={np.std(log_G_A):.3f})", density=True)
    ax.hist(log_G_B, bins=bins_s, alpha=0.5, label=f"Half B (σ={np.std(log_G_B):.3f})", density=True)
    ax.set_xlabel("log(G) — crystal scale")
    ax.set_ylabel("Density")
    ax.set_title("Crystal scale distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, "No scale data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("(no crystal scale in checkpoint)")

fig.suptitle(
    f"Half-dataset comparison: {n_matched:,} matched reflections\n"
    f"CC(F²)={CC:.4f}, CC(log F²)={CC_log:.4f}, R_iso={R_iso:.4f}",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
out_path = OUT_DIR / f"half_dataset_comparison_{args.prefix}.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")
plt.close(fig)

print("\nDone.")
