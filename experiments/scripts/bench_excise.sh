#!/bin/bash
#SBATCH --job-name=bench_excise
#SBATCH --partition=hour
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=21
#SBATCH --mem=128G
#SBATCH --output=bench_excise_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx

.venv/bin/python -W ignore -c "
import time
import torch
from torchsx.io._excise import excise_patches
from torchsx.stream import SFXStream

s = SFXStream.load('sfx_scan/streams/delay_-45.sfx', merge_friedel=False)
core = s._core

fnames = s.filenames
unique_files = list(dict.fromkeys(fnames))
print(f'Total: {s.n_observations} obs, {s.n_images} images, {len(unique_files)} files')
print()

for nw in [1, 4, 11, 21]:
    t0 = time.perf_counter()
    r = excise_patches(core, patch_size=21, n_workers=nw)
    t1 = time.perf_counter()
    print(f'n_workers={nw:2d}: {t1-t0:6.1f}s  valid={r.valid.float().mean():.4f}')
    del r
"
