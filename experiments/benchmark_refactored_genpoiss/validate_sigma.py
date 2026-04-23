#!/das/work/units/LBR-FEL/p17490/Peter/torchsx_bg_anchor/.venv/bin/python -u
"""Validate the analytical σ(I₀) formula against half-dataset differences.

If σ correctly describes the sampling distribution of I_h, then for every HKL
present in both halves the normalised deviation

    Z_h = (I_h^(1) − I_h^(2)) / √(σ_h^(1)² + σ_h^(2)²)

should be approximately N(0, 1).  Deviations from that (globally or in bins)
tell us where the noise model is miscalibrated.

Caveat: below the detection floor, both halves' I_h are determined by
noise/prior rather than data — expect fat tails / bias there, not a
model bug.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUTDIR = Path(__file__).parent
D_MIN, D_MAX = 0.9, 10.0

h1 = torch.load(OUTDIR / "unified_half1.pt", weights_only=True)
h2 = torch.load(OUTDIR / "unified_half2.pt", weights_only=True)

# ── Match HKLs between halves ────────────────────────────────
hkl1 = h1["hkl"].long()
hkl2 = h2["hkl"].long()
lookup = {(int(hkl2[i, 0]), int(hkl2[i, 1]), int(hkl2[i, 2])): i
          for i in range(hkl2.shape[0])}

idx1, idx2 = [], []
for i in range(hkl1.shape[0]):
    key = (int(hkl1[i, 0]), int(hkl1[i, 1]), int(hkl1[i, 2]))
    if key in lookup:
        idx1.append(i)
        idx2.append(lookup[key])

idx1 = torch.tensor(idx1)
idx2 = torch.tensor(idx2)
print(f"Matched {len(idx1)} HKLs between halves.", flush=True)

I1 = h1["I0"][idx1]; s1 = h1["sigma_I0"][idx1]
I2 = h2["I0"][idx2]; s2 = h2["sigma_I0"][idx2]
d  = 0.5 * (h1["d_spacing"][idx1] + h2["d_spacing"][idx2])
n  = 0.5 * (h1["n_obs"][idx1].float() + h2["n_obs"][idx2].float())

# ── Filter by resolution ─────────────────────────────────────
res_mask = (d >= D_MIN) & (d <= D_MAX)
I1, I2, s1, s2, d, n = (t[res_mask] for t in (I1, I2, s1, s2, d, n))
print(f"After d ∈ [{D_MIN}, {D_MAX}] cut: {len(I1)} HKLs.", flush=True)

# ── Z statistic ───────────────────────────────────────────────
I_mean = 0.5 * (I1 + I2)
s_comb = torch.sqrt(s1 ** 2 + s2 ** 2)
Z = (I1 - I2) / s_comb.clamp(min=1e-12)

# I/σ on each half
isn1 = I1 / s1.clamp(min=1e-12)
isn2 = I2 / s2.clamp(min=1e-12)

print(f"\nZ statistic overall:")
print(f"  mean(Z) = {Z.mean():.4f}   (expect 0)")
print(f"  std(Z)  = {Z.std():.4f}   (expect 1)")
print(f"  median(Z) = {Z.median():.4f}")
print(f"  fraction |Z|>3 = {(Z.abs() > 3).float().mean():.4f}   (expect 0.003 for N(0,1))")
print(f"  fraction |Z|>5 = {(Z.abs() > 5).float().mean():.4f}   (expect ~3e-7)")

# ── Binned std(Z) vs log(I_mean), log(n), d ──────────────────
def binned_std_z(x, Z, x_label, n_bins=15):
    # equal-count bins
    order = x.argsort()
    chunk = len(x) // n_bins
    centers, stds, counts, iqrs, med_z = [], [], [], [], []
    for b in range(n_bins):
        i0 = b * chunk
        i1 = (b + 1) * chunk if b < n_bins - 1 else len(x)
        sel = order[i0:i1]
        if len(sel) < 5:
            continue
        centers.append(x[sel].median().item())
        stds.append(Z[sel].std().item())
        counts.append(len(sel))
        iqrs.append((Z[sel].quantile(0.75) - Z[sel].quantile(0.25)).item() / 1.349)  # robust σ
        med_z.append(Z[sel].median().item())
    return centers, stds, counts, iqrs, med_z


def print_bin_table(x_label, centers, stds, counts, iqrs, med_z):
    print(f"\n{'='*80}\nBinned std(Z) by {x_label}\n{'='*80}")
    print(f"  {'center':>12s}  {'std(Z)':>7s}  {'robust σ':>9s}  {'med(Z)':>7s}  {'N':>5s}")
    for c, s, n_, iqr, m in zip(centers, stds, counts, iqrs, med_z):
        print(f"  {c:>12.3g}  {s:>7.3f}  {iqr:>9.3f}  {m:>7.3f}  {n_:>5d}")


for xlab, x in [("log10(I_mean)", torch.log10(I_mean.clamp(min=1e-6))),
                ("log10(n_obs)", torch.log10(n.clamp(min=1))),
                ("d-spacing (Å)", d)]:
    ctrs, stds, counts, iqrs, med_z = binned_std_z(x, Z, xlab)
    print_bin_table(xlab, ctrs, stds, counts, iqrs, med_z)

# ── Plot 1: histogram of Z vs N(0,1) ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
zz = Z.numpy()
zz_clipped = np.clip(zz, -8, 8)
ax.hist(zz_clipped, bins=100, density=True, alpha=0.6, label="observed Z")
x_ref = np.linspace(-5, 5, 400)
ax.plot(x_ref, np.exp(-0.5 * x_ref ** 2) / np.sqrt(2 * np.pi), "r-", lw=1.5,
         label="N(0, 1)")
ax.set_xlabel("Z = (I₁ − I₂) / √(σ₁² + σ₂²)")
ax.set_ylabel("density")
ax.set_title(f"Z overall: std={Z.std():.3f}, mean={Z.mean():.3f}")
ax.set_xlim(-6, 6)
ax.legend()

# ── Plot 2: std(Z) binned by log(I_mean) ──────────────────────
ax = axes[1]
x = torch.log10(I_mean.clamp(min=1e-6))
ctrs, stds, counts, iqrs, _ = binned_std_z(x, Z, "logI")
ax.plot(ctrs, stds, "bo-", label="sd(Z)")
ax.plot(ctrs, iqrs, "g^-", label="robust σ (IQR/1.349)")
ax.axhline(1.0, color="r", ls="--", alpha=0.5, label="N(0,1)")
ax.set_xlabel("log₁₀(I_mean)")
ax.set_ylabel("sd(Z) per bin")
ax.set_title("Z dispersion vs intensity")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(2.5, max(stds) * 1.1))

# ── Plot 3: std(Z) binned by d-spacing ────────────────────────
ax = axes[2]
ctrs, stds, counts, iqrs, _ = binned_std_z(d, Z, "d")
ax.plot(ctrs, stds, "bo-", label="sd(Z)")
ax.plot(ctrs, iqrs, "g^-", label="robust σ")
ax.axhline(1.0, color="r", ls="--", alpha=0.5, label="N(0,1)")
ax.invert_xaxis()
ax.set_xlabel("d-spacing (Å)")
ax.set_ylabel("sd(Z) per bin")
ax.set_title("Z dispersion vs resolution")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(2.5, max(stds) * 1.1))

fig.tight_layout()
out = OUTDIR / "validate_sigma.png"
fig.savefig(out, dpi=150)
print(f"\nSaved plot: {out}")
