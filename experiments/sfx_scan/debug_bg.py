#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Check background vs signal magnitude."""

import torch
import numpy as np
from torchsx.stream import SFXStream
from torchsx.merge._em import EMConfig, prepare_observations
from torchsx.merge._noise import PoissonNoiseModel

paths = [
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-91.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-136.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-182.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-227.sfx",
]
device = "cuda"
s = SFXStream.concat([SFXStream.load(p) for p in paths])

cfg = EMConfig(n_em_cycles=1, min_obs=5)
noise = PoissonNoiseModel()
prep = prepare_observations(s, cfg, noise, device)

I = prep.I_all_sorted
bg = prep.bg_all
n_ph = I + bg  # total photon counts

print(f"Observations: {len(I)}")
print(f"\nI_obs:  min={I.min():.1f}  median={I.median():.1f}  max={I.max():.1f}")
print(f"bg:     min={bg.min():.1f}  median={bg.median():.1f}  max={bg.max():.1f}")
print(f"n_ph:   min={n_ph.min():.1f}  median={n_ph.median():.1f}  max={n_ph.max():.1f}")
print(f"bg/n_ph (where n_ph>1): median={(bg[n_ph>1]/n_ph[n_ph>1]).median():.4f}")

# Per-HKL: what fraction of signal is background?
I0_log = torch.exp(prep.mu_init)
print(f"\nI0 (log-space): median={I0_log.median():.2f}")
print(f"bg vs I0:")
print(f"  bg.median / I0.median = {bg.median() / I0_log.median():.2f}")

# Sigma2_poisson
s2 = prep.sigma2_poi_all
print(f"\nsigma2_poi: min={s2.min():.1f}  median={s2.median():.1f}  max={s2.max():.1f}")
print(f"sqrt(sigma2_poi) / I_obs (where I>1): median={(s2[I>1].sqrt() / I[I>1]).median():.4f}")

# How does PoissonNoiseModel compute sigma2?
print(f"\nPoissonNoiseModel: gain={noise.gain}, r_peak={noise.r_peak}, r_gap={noise.r_gap}, r_bg={noise.r_bg}")

# Check: is bg the TOTAL background or per-pixel?
# CrystFEL stores total integrated background
print(f"\nHistogram of bg values:")
for lo, hi in [(-10, 0), (0, 1), (1, 5), (5, 20), (20, 50), (50, 200)]:
    n = ((bg >= lo) & (bg < hi)).sum().item()
    print(f"  [{lo:4d}, {hi:4d}): {n:8d}  ({100*n/len(bg):.1f}%)")
