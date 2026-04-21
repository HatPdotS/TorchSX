#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare alpha estimated from log-space I0 vs poisson_log I0."""

import torch
from torchsx.stream import SFXStream
from torchsx.merge._em import EMConfig, prepare_observations
from torchsx.merge._noise import PoissonNoiseModel
from torchsx.merge._dstep import fit_I0_poisson_log, fit_alpha

paths = [
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-91.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-136.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-182.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-227.sfx",
]
device = "cuda"
s = SFXStream.concat([SFXStream.load(p) for p in paths])
h1, _ = s.split_random(n_parts=2, seed=42)

cfg = EMConfig(n_em_cycles=1, min_obs=5)
noise = PoissonNoiseModel()
prep = prepare_observations(h1, cfg, noise, device)

# 1. Log-space median (what fit_gengauss uses)
I0_log = torch.exp(prep.mu_init)
print(f"I0 from log-space IRLS: median={I0_log.median():.2f}")

# 2. Poisson marginal MLE
scale = torch.ones_like(prep.I_all_sorted)
I0_poi, _ = fit_I0_poisson_log(
    prep.I_all_sorted, prep.hkl_id_all, scale,
    prep.alpha_init, prep.beta_perhkl,
    I0_log, prep.obs_counts_all, prep.n_valid,
    bg=prep.bg_all, max_iter=50,
)
print(f"I0 from poisson_log:    median={I0_poi.median():.2f}")

ratio = I0_poi / I0_log.clamp(min=0.01)
print(f"I0 ratio (poi/log):     median={ratio.median():.4f}")

# 3. Alpha from log-space I0
alpha_log = fit_alpha(
    prep.I_pos_sorted, prep.hkl_id_pos, I0_log,
    prep.beta_perhkl, prep.obs_counts_pos, prep.n_valid,
)
print(f"\nalpha from log-space I0:   median={alpha_log.median():.4f}")

# 4. Alpha from poisson_log I0
alpha_poi = fit_alpha(
    prep.I_pos_sorted, prep.hkl_id_pos, I0_poi,
    prep.beta_perhkl, prep.obs_counts_pos, prep.n_valid,
)
print(f"alpha from poisson_log I0: median={alpha_poi.median():.4f}")

# 5. Breakdown by I0 strength
bins = [0, 1, 3, 10, 30, 100, 1000]
for label, I0_est, alpha in [("log-space", I0_log, alpha_log), ("poisson_log", I0_poi, alpha_poi)]:
    print(f"\n  alpha({label}) by I0 bin:")
    for i in range(len(bins) - 1):
        mask = (I0_log >= bins[i]) & (I0_log < bins[i + 1])
        if mask.sum() > 10:
            print(f"    I0=[{bins[i]:5},{bins[i+1]:5}): n={mask.sum().item():5d}"
                  f"  alpha={alpha[mask].median():.4f}"
                  f"  I0_ratio={ratio[mask].median():.4f}")
