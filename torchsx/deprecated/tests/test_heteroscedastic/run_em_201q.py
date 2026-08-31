#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare Median vs EM(201 quadrature points) vs Partialator on 4 dark streams."""

import torch, csv, time
from torchsx.stream import SFXStream
from torchsx.deprecated.merge import (
    compute_hkl_index, merge_median,
    match_halves, shell_stats, cc_half, r_split,
    fit_em, EMConfig, ScaleBCorrection,
)
import torchsx.deprecated.merge._dstep as dstep
_orig = dstep._prepare_quadrature
def _patched(alpha, beta, obs_counts, sigma2_poisson, n_quad=21, device=None):
    return _orig(alpha, beta, obs_counts, sigma2_poisson, n_quad=51, device=device)
dstep._prepare_quadrature = _patched

paths = [
    '/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-91.sfx',
    '/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-136.sfx',
    '/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-182.sfx',
    '/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams/delay_-227.sfx',
]
streams = [SFXStream.load(p) for p in paths]
stream = SFXStream.concat(streams)
core = stream._core; cell = core.cell.median(dim=0).values
hkl_idx = compute_hkl_index(core.hkl)
print(f'{stream.n_observations:,} obs, {stream.n_crystals:,} crystals')

n_c = core.astar.shape[0]
perm = torch.randperm(n_c, generator=torch.Generator().manual_seed(42))
mask_A = torch.zeros(n_c, dtype=torch.bool); mask_A[perm[:n_c//2]] = True
obs_A = mask_A[core.crystal_id.long()]; obs_B = ~obs_A
def to_F(I): return I.clamp(min=0).sqrt()

# Median
I_A_md, _, _ = merge_median(hkl_idx.hkl_id[obs_A], core.I[obs_A], hkl_idx.n_unique, min_mult=2)
I_B_md, _, _ = merge_median(hkl_idx.hkl_id[obs_B], core.I[obs_B], hkl_idx.n_unique, min_mult=2)
v = torch.isfinite(I_A_md) & torch.isfinite(I_B_md) & (I_A_md > 0) & (I_B_md > 0)
idx = torch.where(v)[0]
m_md = match_halves(hkl_idx.unique_hkl[idx], to_F(I_A_md[idx]),
                     hkl_idx.unique_hkl[idx], to_F(I_B_md[idx]), cell)

# EM
t0 = time.perf_counter()
import sys
device = "cuda" if "--cuda" in sys.argv else "cpu"
print(f'Device: {device}')

print('\nEM half A (201 quad)...')
dist_A = fit_em(stream.filter_crystals(mask_A),
                config=EMConfig(n_em_cycles=10, min_obs=5),
                corrections=[ScaleBCorrection()], device=device)
print('\nEM half B (201 quad)...')
dist_B = fit_em(stream.filter_crystals(~mask_A),
                config=EMConfig(n_em_cycles=10, min_obs=5),
                corrections=[ScaleBCorrection()], device=device)
print(f'\nEM total: {time.perf_counter() - t0:.0f}s')
m_em = match_halves(dist_A.hkl, to_F(dist_A.I0),
                     dist_B.hkl, to_F(dist_B.I0), cell)

part = []
with open('/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/partialator_shell_stats.csv') as f:
    for row in csv.DictReader(f): part.append(row)

s_md = shell_stats(m_md, n_shells=20)
s_em = shell_stats(m_em, n_shells=20)
print()
print('=' * 80)
print('F=sqrt(I): Median vs EM(201q,ScaleB) vs Partialator')
print('=' * 80)
print(f'{"d_max":>7} {"d_min":>7} | {"CC_med":>7} {"CC_em":>7} {"CC_prt":>7} | {"R_med":>7} {"R_em":>7} {"R_prt":>7}')
print('-' * 80)
for i in range(min(20, len(s_em), len(s_md))):
    sm, se = s_md[i], s_em[i]
    best = min(part, key=lambda ps: abs(float(ps['d_min']) - se.d_min))
    print(f'{se.d_max:>7.2f} {se.d_min:>7.2f} | {sm.cc_half:>7.4f} {se.cc_half:>7.4f} {float(best["cc_half"]):>7.4f} | '
          f'{sm.r_split:>7.4f} {se.r_split:>7.4f} {float(best["r_split"]):>7.4f}')
print('-' * 80)
print(f'{"Overall":>15} | {cc_half(m_md):>7.4f} {cc_half(m_em):>7.4f}         | '
      f'{r_split(m_md):>7.4f} {r_split(m_em):>7.4f}')
