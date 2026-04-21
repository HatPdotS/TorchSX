#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 0:30:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/test_sigma.out

"""Check sigma_I0 and detection limits."""

import torch
from torchsx.stream import SFXStream
from torchsx.merge import fit_em, EMConfig, ScaleBCorrection

paths = [
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-91.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-136.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-182.sfx",
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-227.sfx",
]
device = "cuda"
s = SFXStream.concat([SFXStream.load(p) for p in paths])

cfg = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)
dist = fit_em(s, config=cfg, corrections=[ScaleBCorrection()], device=device)

I0 = dist.I0
sigma = dist.sigma_I0
snr = I0 / sigma.clamp(min=1e-8)
n_obs = dist.n_obs.float()

print(f"I0:       min={I0.min():.4f}  p5={I0.quantile(0.05):.3f}  median={I0.median():.2f}")
print(f"sigma_I0: min={sigma.min():.4f}  p5={sigma.quantile(0.05):.3f}  median={sigma.median():.2f}")
print(f"I0/sigma: min={snr.min():.3f}  p5={snr.quantile(0.05):.3f}  median={snr.median():.2f}")
print(f"n_obs:    min={int(n_obs.min())}  median={int(n_obs.median())}  max={int(n_obs.max())}")

# Expected detection limit: sqrt(bg_total * (1 + n_peak/n_bg) / n_obs)
# bg_per_pixel ~ 0.6, n_peak=154, n_bg=126
bg_per_pixel = 0.6
n_peak = 154.0
n_bg = 126.0
sigma2_per_obs = bg_per_pixel * n_peak * (1 + n_peak / n_bg)  # ~205
sigma_theory = (sigma2_per_obs / n_obs.clamp(min=1)).sqrt()
print(f"\nPoisson floor: sigma2_per_obs = {sigma2_per_obs:.1f}")
print(f"Theoretical sigma(I0): min={sigma_theory.min():.3f}  median={sigma_theory.median():.3f}")

# Breakdown by I0
print(f"\n  {'I0 range':>16s}  {'N':>6s}  {'med sigma':>10s}  {'med SNR':>8s}  {'theory':>8s}  {'med n_obs':>10s}")
bins = [0, 0.1, 0.5, 1, 3, 10, 30, 100, 1000]
for i in range(len(bins) - 1):
    mask = (I0 >= bins[i]) & (I0 < bins[i + 1])
    if mask.sum() < 5:
        continue
    print(f"  [{bins[i]:6.1f}, {bins[i+1]:6.1f})"
          f"  {mask.sum().item():>6d}"
          f"  {sigma[mask].median():>10.4f}"
          f"  {snr[mask].median():>8.2f}"
          f"  {sigma_theory[mask].median():>8.4f}"
          f"  {n_obs[mask].median():>10.0f}")

# How many are below detection limit?
below_1sigma = (I0 < sigma).sum().item()
below_2sigma = (I0 < 2 * sigma).sum().item()
below_theory = (I0 < sigma_theory).sum().item()
print(f"\nI0 < 1*sigma_I0: {below_1sigma}/{len(I0)}")
print(f"I0 < 2*sigma_I0: {below_2sigma}/{len(I0)}")
print(f"I0 < sigma_theory (Poisson floor): {below_theory}/{len(I0)}")
