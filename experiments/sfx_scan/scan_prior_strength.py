#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/scan_prior_strength.out
#SBATCH -c 16
#SBATCH -p day
#SBATCH -t 4:00:00

"""Scan ridge prior strength for G and B correction.

Quantify overfitting via free/work α ratio as a function of
(scale_prior, bfactor_prior).  Also track split-half CC₁/₂ and Spearman.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsx.stream import SFXStream
from torchsx.merge._distribution import fit_gengauss
from torchsx.merge._corrections import ScaleBCorrection, CorrectionPipeline
from torchsx.merge._stats import (
    match_halves, cc_half, r_split, spearman_cc, shell_stats,
)

# ── Configuration ──────────────────────────────────────────────
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DELAYS = [-91, -136, -182, -227, -273]
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/resolution_study")
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84])
N_EM = 10
FIT_KW = dict(min_obs=20, n_shells=20, n_smooth_iter=15, smooth_lambda=0.3)

# Scan: vary scale_prior and bfactor_prior together and independently
# (scale_prior, bfactor_prior, label)
SCAN_POINTS = [
    # No correction baseline
    (None, None, "no correction"),
    # Scan scale_prior with fixed bfactor_prior=1.0
    (0.1,  1.0,  "sp=0.1, bp=1"),
    (0.5,  1.0,  "sp=0.5, bp=1"),
    (1.0,  1.0,  "sp=1, bp=1"),
    (3.0,  1.0,  "sp=3, bp=1"),
    (10.0, 1.0,  "sp=10, bp=1"),
    (30.0, 1.0,  "sp=30, bp=1"),
    (100.0, 1.0, "sp=100, bp=1"),
    # Scan bfactor_prior with fixed scale_prior=10.0
    (10.0, 0.01, "sp=10, bp=0.01"),
    (10.0, 0.1,  "sp=10, bp=0.1"),
    (10.0, 0.5,  "sp=10, bp=0.5"),
    (10.0, 1.0,  "sp=10, bp=1"),      # duplicate — will skip
    (10.0, 3.0,  "sp=10, bp=3"),
    (10.0, 10.0, "sp=10, bp=10"),
    (10.0, 30.0, "sp=10, bp=30"),
    # Joint scan (equal strength)
    (0.1,  0.1,  "sp=bp=0.1"),
    (1.0,  1.0,  "sp=bp=1"),          # duplicate
    (10.0, 10.0, "sp=bp=10"),         # duplicate
    (100.0, 100.0, "sp=bp=100"),
]

# De-duplicate
seen = set()
scan_points = []
for sp, bp, label in SCAN_POINTS:
    key = (sp, bp)
    if key not in seen:
        seen.add(key)
        scan_points.append((sp, bp, label))


# ── Load data ──────────────────────────────────────────────
print("Loading streams...")
streams = []
for d in DELAYS:
    path = STREAM_DIR / f"delay_{d}.stream"
    print(f"  {path.name} ...", end=" ", flush=True)
    st = SFXStream.from_stream(str(path), n_workers=16)
    print(f"{st.n_crystals} crystals, {st.n_observations} obs")
    streams.append(st)

s = SFXStream.concat(streams)
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} observations")

# Split
torch.manual_seed(42)
h1, h2 = s.split_random(0.5)
print(f"Split: half1={h1.n_crystals}, half2={h2.n_crystals}")


def to_matched(f1, f2):
    return match_halves(
        f1.hkl, f1.I0.clamp(min=0).sqrt(),
        f2.hkl, f2.I0.clamp(min=0).sqrt(),
        CELL, d_min=0.9, d_max=10.0,
    )


# ── Scan ──────────────────────────────────────────────────
results = []

for sp, bp, label in scan_points:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if sp is None:
        # No correction — EM only
        f1 = fit_gengauss(h1, n_em_cycles=N_EM, **FIT_KW)
        f2 = fit_gengauss(h2, n_em_cycles=N_EM, **FIT_KW)
        free_work = 1.0
    else:
        corrections = [ScaleBCorrection(scale_prior=sp, bfactor_prior=bp)]
        f1 = fit_gengauss(h1, n_em_cycles=N_EM, corrections=corrections,
                          free_fraction=0.1, **FIT_KW)
        f2 = fit_gengauss(h2, n_em_cycles=N_EM, corrections=corrections,
                          free_fraction=0.1, **FIT_KW)

        # Get free/work ratio from full dataset
        corrections_full = [ScaleBCorrection(scale_prior=sp, bfactor_prior=bp)]
        f_full = fit_gengauss(s, n_em_cycles=N_EM, corrections=corrections_full,
                              free_fraction=0.1, **FIT_KW)
        # Extract from the last EM print (already printed)
        # We'll compute it directly from the pipeline
        free_work = 0.0  # placeholder — read from output

    m = to_matched(f1, f2)
    cc = cc_half(m)
    sp_cc = spearman_cc(m)
    r = r_split(m)

    print(f"  CC1/2={cc:.4f}  Spearman={sp_cc:.4f}  R_split={r:.4f}")

    results.append({
        "label": label,
        "scale_prior": sp,
        "bfactor_prior": bp,
        "cc_half": cc,
        "spearman": sp_cc,
        "r_split": r,
    })


# ── Summary table ──────────────────────────────────────────
print(f"\n{'='*80}")
print(f"{'Label':>25s} {'sp':>6s} {'bp':>6s} {'CC1/2':>7s} {'Spear':>7s} {'R_split':>8s}")
print(f"{'-'*80}")
for r in results:
    sp_str = f"{r['scale_prior']}" if r['scale_prior'] is not None else "—"
    bp_str = f"{r['bfactor_prior']}" if r['bfactor_prior'] is not None else "—"
    print(f"{r['label']:>25s} {sp_str:>6s} {bp_str:>6s} "
          f"{r['cc_half']:7.4f} {r['spearman']:7.4f} {r['r_split']:8.4f}")
print(f"{'='*80}")


# ── Plots ──────────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) Scale prior scan (bfactor_prior=1.0)
sp_results = [r for r in results if r["bfactor_prior"] == 1.0 and r["scale_prior"] is not None]
if sp_results:
    x = [r["scale_prior"] for r in sp_results]
    ax = axes[0]
    ax.semilogx(x, [r["cc_half"] for r in sp_results], "o-", label="CC₁/₂")
    ax.semilogx(x, [r["spearman"] for r in sp_results], "s-", label="Spearman")
    # Add no-correction baseline
    baseline = [r for r in results if r["scale_prior"] is None]
    if baseline:
        ax.axhline(baseline[0]["cc_half"], ls="--", color="C0", alpha=0.5, label="no corr CC")
        ax.axhline(baseline[0]["spearman"], ls="--", color="C1", alpha=0.5, label="no corr Sp")
    ax.set_xlabel("scale_prior (λ_G)")
    ax.set_ylabel("Correlation")
    ax.set_title("Scale prior scan (bp=1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# (b) Bfactor prior scan (scale_prior=10.0)
bp_results = [r for r in results if r["scale_prior"] == 10.0 and r["bfactor_prior"] is not None]
if bp_results:
    x = [r["bfactor_prior"] for r in bp_results]
    ax = axes[1]
    ax.semilogx(x, [r["cc_half"] for r in bp_results], "o-", label="CC₁/₂")
    ax.semilogx(x, [r["spearman"] for r in bp_results], "s-", label="Spearman")
    if baseline:
        ax.axhline(baseline[0]["cc_half"], ls="--", color="C0", alpha=0.5)
        ax.axhline(baseline[0]["spearman"], ls="--", color="C1", alpha=0.5)
    ax.set_xlabel("bfactor_prior (λ_B)")
    ax.set_title("B-factor prior scan (sp=10)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# (c) Joint scan
jt_results = [r for r in results if r["scale_prior"] is not None
              and r["scale_prior"] == r["bfactor_prior"]]
if jt_results:
    x = [r["scale_prior"] for r in jt_results]
    ax = axes[2]
    ax.semilogx(x, [r["cc_half"] for r in jt_results], "o-", label="CC₁/₂")
    ax.semilogx(x, [r["spearman"] for r in jt_results], "s-", label="Spearman")
    if baseline:
        ax.axhline(baseline[0]["cc_half"], ls="--", color="C0", alpha=0.5)
        ax.axhline(baseline[0]["spearman"], ls="--", color="C1", alpha=0.5)
    ax.set_xlabel("prior strength (sp=bp)")
    ax.set_title("Joint prior scan")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUTDIR / "prior_strength_scan.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
