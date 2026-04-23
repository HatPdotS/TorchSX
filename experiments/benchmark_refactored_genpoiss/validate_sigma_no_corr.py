#!/das/work/units/LBR-FEL/p17490/Peter/torchsx_bg_anchor/.venv/bin/python -u
"""Diagnostic: fit halves with NO corrections at all.

With no corrections, observations are fit with pure per-observation Poisson
+ multiplicative noise.  α, β absorb only the genuine per-obs scatter, not
correction misfit.  If σ formula is correct, Z = (I1 − I2)/√(σ1² + σ2²)
should be N(0,1).

If std(Z) ≈ 1 here, the σ formula is right for the pure-noise regime, and
the ~1.19 we saw with free corrections reflects genuine correction-fit
uncertainty that the Fisher-info formulation doesn't capture.  If std(Z)
is still off, something more fundamental is miscalibrated.
"""

from __future__ import annotations

from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.merge import match_halves, cc_half, r_split
from torchsx.merging_clean import Merger, MergeConfig

STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
OUTDIR = Path(__file__).parent
DELAYS = [-91, -136, -182, -227]
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
D_MIN, D_MAX = 0.9, 10.0
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"{s.n_crystals} crystals → halves {h1.n_crystals}/{h2.n_crystals}", flush=True)

cfg = MergeConfig(min_obs=5, max_iter=150, history_size=10)

# ── Full fit (for seeding halves) ───────────────────────────
print("\n── full fit (no corrections) ──", flush=True)
r_full = Merger(s, config=cfg, corrections=[], device=device).merge()
print(
    f"  full: {len(r_full.I0)} HKLs  med(α)={r_full.alpha.median():.4f}  "
    f"med(β)={r_full.beta.median():.4f}",
    flush=True,
)

# ── Halves (no corrections, seeded from full) ───────────────
results = {}
for label, stream in [("half1", h1), ("half2", h2)]:
    print(f"\n── {label} (no corrections, seeded) ──", flush=True)
    r = Merger(stream, config=cfg, corrections=[], init_from=r_full, device=device).merge()
    results[label] = r
    print(
        f"  {label}: {len(r.I0)} HKLs  med(α)={r.alpha.median():.4f}  "
        f"med(β)={r.beta.median():.4f}  "
        f"final_nll={r.diagnostics.get('final_data_nll', 'n/a')}",
        flush=True,
    )


# ── Match HKLs, compute Z ────────────────────────────────────
h1r = results["half1"]; h2r = results["half2"]

hkl1 = h1r.hkl.long(); hkl2 = h2r.hkl.long()
lookup = {(int(hkl2[i, 0]), int(hkl2[i, 1]), int(hkl2[i, 2])): i
          for i in range(hkl2.shape[0])}
idx1, idx2 = [], []
for i in range(hkl1.shape[0]):
    key = (int(hkl1[i, 0]), int(hkl1[i, 1]), int(hkl1[i, 2]))
    if key in lookup:
        idx1.append(i); idx2.append(lookup[key])

idx1 = torch.tensor(idx1); idx2 = torch.tensor(idx2)
I1 = h1r.I0[idx1]; s1 = h1r.sigma_I0[idx1]
I2 = h2r.I0[idx2]; s2 = h2r.sigma_I0[idx2]
d = 0.5 * (h1r.d_spacing[idx1] + h2r.d_spacing[idx2])

res_mask = (d >= D_MIN) & (d <= D_MAX)
I1, I2, s1, s2, d = (t[res_mask] for t in (I1, I2, s1, s2, d))

Z = (I1 - I2) / torch.sqrt(s1 ** 2 + s2 ** 2).clamp(min=1e-12)
I_mean = 0.5 * (I1 + I2)

print(f"\nZ statistic (NO corrections applied):")
print(f"  n = {len(I1)}")
print(f"  mean(Z) = {Z.mean():.4f}   (expect 0)")
print(f"  std(Z)  = {Z.std():.4f}   (expect 1)")
print(f"  median(Z) = {Z.median():.4f}")
print(f"  |Z|>3:   {(Z.abs() > 3).float().mean():.4f}   (expect 0.003)")

# Binned by I
print(f"\nBinned std(Z) by log10(I_mean):")
order = I_mean.argsort()
n_bins = 15
chunk = len(I_mean) // n_bins
for b in range(n_bins):
    sel = order[b * chunk : (b + 1) * chunk if b < n_bins - 1 else len(I_mean)]
    if len(sel) < 5:
        continue
    ctr = torch.log10(I_mean[sel].clamp(min=1e-6)).median().item()
    sZ = Z[sel].std().item()
    iqr = (Z[sel].quantile(0.75) - Z[sel].quantile(0.25)).item() / 1.349
    print(f"  log10(I) ≈ {ctr:>6.3f}:  std(Z)={sZ:.3f}  robust_σ={iqr:.3f}  ({len(sel)} HKLs)")

# CC½ (expected to be poor — no scale corrections)
F1 = I1.clamp(min=0).sqrt(); F2 = I2.clamp(min=0).sqrt()
match = match_halves(h1r.hkl[idx1][res_mask], F1,
                      h2r.hkl[idx2][res_mask], F2,
                      CELL, d_min=D_MIN, d_max=D_MAX)
print(f"\nWith NO corrections:  CC½={cc_half(match):.4f}  R_split={r_split(match):.4f}")
print("(Expected to be much worse than with corrections — per-crystal scale "
      "variation is large in SFX.  Test is about σ calibration, not quality.)")
