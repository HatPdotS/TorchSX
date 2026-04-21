#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Check how many observations survive background filtering."""

import torch
from torchsx.stream import SFXStream

paths = [
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-91.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-136.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-182.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-227.sfx",
]
s = SFXStream.concat([SFXStream.load(p) for p in paths])
bg = s._core.background
I = s._core.I
n_total = len(bg)

print(f"Total observations: {n_total}")
print(f"\nSurvival by bg threshold:")
for thresh in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    keep = (bg <= thresh).sum().item()
    # Also check per-HKL: how many unique HKLs retain >= 5 obs?
    print(f"  bg <= {thresh:5.2f}: {keep:>8d} obs ({100*keep/n_total:.1f}%)")

# What are the bg=0 observations?
exact_zero = (bg == 0).sum().item()
near_zero = (bg.abs() < 0.01).sum().item()
print(f"\n  bg == 0:    {exact_zero:>8d} ({100*exact_zero/n_total:.1f}%)")
print(f"  |bg| < 0.01: {near_zero:>8d} ({100*near_zero/n_total:.1f}%)")

# Distribution of bg for JUNGFRAU — is it quantized?
print(f"\nTop 10 most common bg values:")
bg_round = (bg * 100).round() / 100  # round to 2 decimals
vals, counts = bg_round.unique(return_counts=True)
top = counts.argsort(descending=True)[:10]
for i in top:
    print(f"  bg={vals[i].item():8.2f}: {counts[i].item():>8d} ({100*counts[i].item()/n_total:.1f}%)")
