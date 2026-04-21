#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction/roi_extract_%a.out
#SBATCH -c 16
#SBATCH -p hour
#SBATCH -t 1:00:00
#SBATCH --array=0-5

"""Extract ROIs for one delay stream. Submit as array job: sbatch --array=0-5 extract_rois.py"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, "/das/work/p17/p17490/Peter/torchsx")

from torchsx.stream import SFXStream

STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
OUT_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
DELAYS = [-91, -136, -182, -227, -273, -45]
SPACEGROUP = "P-1"
PATCH_SIZE = 21
N_WORKERS = 16

# Array task index
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
delay = DELAYS[task_id]

OUT_DIR.mkdir(parents=True, exist_ok=True)
roi_path = OUT_DIR / f"rois_delay_{delay}.safetensors"

if roi_path.exists():
    print(f"ROIs already cached: {roi_path}")
    sys.exit(0)

stream_path = STREAM_DIR / f"delay_{delay}.stream"
print(f"Task {task_id}: extracting ROIs from {stream_path.name}")

s = SFXStream.from_stream(stream_path, spacegroup=SPACEGROUP, n_workers=N_WORKERS)
print(f"  {s.n_crystals} crystals, {s.n_observations} observations")

print(f"  Excising {PATCH_SIZE}×{PATCH_SIZE} patches with {N_WORKERS} workers...")
roi_store = s.excise_rois(
    path=str(roi_path),
    patch_size=PATCH_SIZE,
    n_workers=N_WORKERS,
)

print(f"  Saved: {roi_path} ({roi_path.stat().st_size / 1e9:.1f} GB)")
print(f"  Shape: {roi_store['patches'].shape}")
print("Done.")
