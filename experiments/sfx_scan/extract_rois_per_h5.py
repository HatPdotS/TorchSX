#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction/roi_extract_%a.out
#SBATCH -c 4
#SBATCH -p hour
#SBATCH -t 0:30:00
#SBATCH --array=0-125

"""Extract ROIs for observations from one HDF5 file.

Submit: sbatch --array=0-125 extract_rois_per_h5.py

Each task:
1. Loads all streams (from .sfx cache — fast)
2. Filters to observations from one HDF5 file
3. Extracts 21×21 patches for those observations only
4. Saves to neural_correction/rois_part_NNN.safetensors

After all tasks complete, run concat_rois.py to merge.
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, "/das/work/p17/p17490/Peter/torchsx")

from torchsx.stream import SFXStream
from torchsx.io._roi_store import ROIStore

STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
OUT_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
H5_LIST = OUT_DIR / "h5_files.txt"
DELAYS = [-91, -136, -182, -227, -273, -45]
SPACEGROUP = "P-1"
PATCH_SIZE = 21

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Read H5 file list
with open(H5_LIST) as f:
    h5_files = [line.strip() for line in f if line.strip()]

if task_id >= len(h5_files):
    print(f"Task {task_id}: no work (only {len(h5_files)} files)")
    sys.exit(0)

h5_path = h5_files[task_id]
out_path = OUT_DIR / f"rois_part_{task_id:03d}.safetensors"

if out_path.exists():
    print(f"Task {task_id}: already cached at {out_path}")
    sys.exit(0)

print(f"Task {task_id}: extracting ROIs from {Path(h5_path).name}")

# Load all streams from .sfx cache (fast — no re-parsing)
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    s = SFXStream.from_stream(stem, spacegroup=SPACEGROUP, cache=cache, n_workers=1)
    streams.append(s)

s = SFXStream.concat(streams)
core = s._core
print(f"  Total: {core.n_crystals} crystals, {core.n_observations} obs")

# Find observations from this H5 file
# image_id per crystal → filename via metadata
meta = core.metadata
filenames = meta.get("image", {}).get("filename", [])
if not filenames:
    # Fallback: parse from source_path stored per image
    # The metadata stores filenames as a list indexed by image_id
    print(f"  ERROR: no image filenames in metadata")
    sys.exit(1)

# Map image_id → filename, find images from this H5
target_image_ids = []
for img_id, fname in enumerate(filenames):
    if fname == h5_path:
        target_image_ids.append(img_id)

if not target_image_ids:
    print(f"  No images from {Path(h5_path).name}, skipping")
    sys.exit(0)

target_image_ids = set(target_image_ids)
print(f"  {len(target_image_ids)} images from this file")

# Find crystals from these images
crystal_image_ids = core.image_id.numpy()
crystal_mask = torch.tensor([int(iid) in target_image_ids for iid in crystal_image_ids])
target_crystal_ids = set(torch.where(crystal_mask)[0].tolist())
print(f"  {len(target_crystal_ids)} crystals")

# Find observations from these crystals
obs_crystal_ids = core.crystal_id.numpy()
obs_mask = torch.tensor([int(cid) in target_crystal_ids for cid in obs_crystal_ids])
n_obs = obs_mask.sum().item()
print(f"  {n_obs} observations to extract")

if n_obs == 0:
    print(f"  No observations, skipping")
    sys.exit(0)

# Extract ROIs for these observations only
roi_store = s.excise_rois(
    path=str(out_path),
    patch_size=PATCH_SIZE,
    obs_mask=obs_mask,
    n_workers=1,  # single H5 file, no need for parallel
)

# Also save the observation indices so we can reconstruct the mapping
obs_indices = torch.where(obs_mask)[0].to(torch.int64)
torch.save(obs_indices, str(out_path).replace(".safetensors", "_indices.pt"))

print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB), {n_obs} patches")
print("Done.")
