#!/bin/bash
#SBATCH --job-name=cpu-scale
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=compress_scaling_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx

H5=h5files/run0121-s2_inj55_scan_0p5uJ/acq0001.JF06T32V07.h5

for NCPU in 1 2 4 8 16; do
    echo "=== Testing with $NCPU CPUs ==="
    OMP_NUM_THREADS=$NCPU \
    MKL_NUM_THREADS=$NCPU \
    OPENBLAS_NUM_THREADS=$NCPU \
    .venv/bin/python -u -c "
import os, time
os.environ['OMP_NUM_THREADS'] = '$NCPU'

import numpy as np
import bitshuffle.h5
import h5py
import torch
import torch.nn.functional as F

# Load mask
f = h5py.File('$H5', 'r')
good = np.asarray(f['data/JF06T32V07/meta/pixel_mask'][()], dtype=bool)
dset = f['data/JF06T32V07/data']
n_frames = min(200, dset.shape[0])

# Time just reading
t0 = time.time()
for i in range(n_frames):
    frame = dset[i]
dt_read = time.time() - t0
print(f'  Read only:      {n_frames/dt_read:.1f} fps  ({dt_read:.1f}s for {n_frames} frames)')

# Time read + numpy compute (background + z-score)
from sparse_analysis import build_radial_map, parse_geometry_from_stream
geom = parse_geometry_from_stream('merged_368.stream')
r_map = build_radial_map(geom)
r_good = r_map[good]
n_bins = 500
r_max = r_good.max()
bin_edges = np.linspace(0, r_max + 1, n_bins + 1)
bin_idx = np.clip(np.digitize(r_good, bin_edges) - 1, 0, n_bins - 1).astype(np.int64)
bin_counts = np.zeros(n_bins)
np.add.at(bin_counts, bin_idx, 1)
bin_counts = np.clip(bin_counts, 1, None)

t0 = time.time()
for i in range(n_frames):
    frame = dset[i]
    vals = frame[good].astype(np.float32)
    profile = np.zeros(n_bins, dtype=np.float64)
    np.add.at(profile, bin_idx, vals)
    profile /= bin_counts
    profile = np.clip(profile, 0, None)
    bg = profile[bin_idx].astype(np.float32)
    z = (np.sqrt(vals + 0.375) - np.sqrt(bg + 0.375)) / 0.5
dt_full = time.time() - t0
print(f'  Read + numpy:   {n_frames/dt_full:.1f} fps  ({dt_full:.1f}s)')

# Time read + torch CPU compute (with erosion)
bin_idx_t = torch.from_numpy(bin_idx)
bin_counts_t = torch.from_numpy(bin_counts.astype(np.float32))
good_t = torch.from_numpy(good)
kernel = torch.ones(1, 1, 3, 3)
kernel[0, 0, 1, 1] = 0

panels = geom['panels']
panel_slices = [(p['min_ss'], p['max_ss']+1, p['min_fs'], p['max_fs']+1) for p in panels]
total_ss = max(s[1] for s in panel_slices)
total_fs = max(s[3] for s in panel_slices)

t0 = time.time()
for i in range(n_frames):
    frame = dset[i]
    frame_t = torch.from_numpy(frame.astype(np.int32))
    vals = frame_t[good_t].float()
    profile = torch.zeros(n_bins)
    profile.scatter_add_(0, bin_idx_t, vals)
    profile /= bin_counts_t
    profile = profile.clamp(min=0)
    bg = profile[bin_idx_t]
    z = (torch.sqrt(vals + 0.375) - torch.sqrt(bg + 0.375)) / 0.5
    z_2d = torch.zeros(total_ss, total_fs)
    z_2d[good_t] = z
    signal_2d = z_2d > 2.0
    eroded = torch.zeros_like(signal_2d)
    for ss0, ss1, fs0, fs1 in panel_slices:
        panel = signal_2d[ss0:ss1, fs0:fs1].float().unsqueeze(0).unsqueeze(0)
        nb = F.conv2d(panel, kernel, padding=1)
        survived = (panel > 0.5) & (nb >= 1)
        eroded[ss0:ss1, fs0:fs1] = survived.squeeze()
dt_torch = time.time() - t0
print(f'  Read + torch:   {n_frames/dt_torch:.1f} fps  ({dt_torch:.1f}s)')

f.close()
print()
"
done
