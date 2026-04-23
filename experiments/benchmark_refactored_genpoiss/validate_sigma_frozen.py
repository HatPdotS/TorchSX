#!/das/work/units/LBR-FEL/p17490/Peter/torchsx_bg_anchor/.venv/bin/python -u
"""Diagnostic: fit halves with corrections FROZEN to the full-fit values.

Hypothesis: the ~20% under-estimate of σ(I_h) comes from correction-parameter
uncertainty (log_G_c, B_c are co-estimated with I_h; their uncertainty
propagates to I_h via shared crystals).  If we pre-load the full-fit's
corrections and freeze them, each half's fit should depend only on its
observations (pure per-obs noise), and the σ formula should match the
observed half-to-half scatter → std(Z) → 1.

If std(Z) drops to ~1 in the mid-I regime under this protocol, we've
confirmed the correction-uncertainty hypothesis.  If it stays at 1.2,
the gap is from something else (probably heavier-than-Gaussian q tails).
"""

from __future__ import annotations

from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.merge import match_halves, cc_half, r_split
from torchsx.merging_clean import Merger, MergeConfig, ScaleBCorrection

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

# ── 1. Full fit ─────────────────────────────────────────────
print("\n── full fit (baseline for frozen corrections) ──", flush=True)
m_full = Merger(s, config=cfg, corrections=[ScaleBCorrection], device=device)
r_full = m_full.merge()
print(
    f"  full: {len(r_full.I0)} HKLs  med(α)={r_full.alpha.median():.4f}  "
    f"med(β)={r_full.beta.median():.4f}",
    flush=True,
)

# Extract trained ScaleBCorrection params
full_sb = m_full.loss.corrections[0]
log_G_full = full_sb.log_G_raw.detach().clone().cpu()
B_full = full_sb.B.detach().clone().cpu()
n_crys_full = full_sb.n_crystals
print(
    f"  extracted ScaleBCorrection:  n_crystals={n_crys_full}  "
    f"‖log_G‖_max={log_G_full.abs().max():.3f}  ‖B‖_max={B_full.abs().max():.3f}",
    flush=True,
)


def make_frozen_sb() -> ScaleBCorrection:
    """Construct a ScaleBCorrection pre-loaded and frozen at the full-fit values."""
    c = ScaleBCorrection(
        n_crystals=n_crys_full,
        scale_prior=full_sb.scale_prior,
        bfactor_prior=full_sb.bfactor_prior,
    )
    with torch.no_grad():
        c.log_G_raw.copy_(log_G_full)
        c.B.copy_(B_full)
    c.log_G_raw.requires_grad_(False)
    c.B.requires_grad_(False)
    return c


# ── 2. Halves with FROZEN corrections ───────────────────────
results = {}
for label, stream in [("half1", h1), ("half2", h2)]:
    print(f"\n── {label} with frozen corrections (seeded from full) ──", flush=True)
    frozen_sb = make_frozen_sb()
    m = Merger(
        stream, config=cfg,
        corrections=[frozen_sb],
        init_from=r_full,
        device=device,
    )
    r = m.merge()
    results[label] = r
    print(
        f"  {label}: {len(r.I0)} HKLs  med(α)={r.alpha.median():.4f}  "
        f"med(β)={r.beta.median():.4f}  "
        f"final_nll={r.diagnostics.get('final_data_nll', 'n/a')}",
        flush=True,
    )


# ── 3. Match HKLs, compute Z ────────────────────────────────
h1r = results["half1"]; h2r = results["half2"]

hkl1 = h1r.hkl.long()
hkl2 = h2r.hkl.long()
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
print(f"\nAfter d ∈ [{D_MIN}, {D_MAX}]: {len(I1)} HKLs", flush=True)

Z = (I1 - I2) / torch.sqrt(s1 ** 2 + s2 ** 2).clamp(min=1e-12)
I_mean = 0.5 * (I1 + I2)

print(f"\nZ statistic (corrections FROZEN at full-fit values):")
print(f"  mean(Z) = {Z.mean():.4f}   (expect 0)")
print(f"  std(Z)  = {Z.std():.4f}   (expect 1)")
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
    print(f"  log10(I) ≈ {ctr:>6.3f}:  std(Z)={sZ:.3f}  ({len(sel)} HKLs)")

# CC½ with frozen corrections
F1 = I1.clamp(min=0).sqrt(); F2 = I2.clamp(min=0).sqrt()
match = match_halves(h1r.hkl[idx1][res_mask], F1,
                      h2r.hkl[idx2][res_mask], F2,
                      CELL, d_min=D_MIN, d_max=D_MAX)
print(f"\nWith frozen corrections:  CC½={cc_half(match):.4f}  R_split={r_split(match):.4f}")
