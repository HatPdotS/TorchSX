#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction/concat_rois.out
#SBATCH -c 4
#SBATCH -p hour
#SBATCH -t 0:30:00
#SBATCH --mem=64G

"""Concatenate per-H5 ROI files into a single rois.safetensors.

Run after all extract_rois_per_h5.py tasks complete.
"""

import sys
import torch
from pathlib import Path
from safetensors.torch import save_file, load_file

sys.path.insert(0, "/das/work/p17/p17490/Peter/torchsx")

OUT_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
FINAL_PATH = OUT_DIR / "rois.safetensors"
N_OBS_TOTAL = 9757334  # from the concatenated stream

# Collect all partial files
parts = sorted(OUT_DIR.glob("rois_part_*.safetensors"))
print(f"Found {len(parts)} partial ROI files")

if not parts:
    print("ERROR: no partial files found")
    sys.exit(1)

# Load first to get patch dimensions
sample = load_file(str(parts[0]))
ph, pw = sample["patches"].shape[1], sample["patches"].shape[2]
print(f"Patch size: {ph}×{pw}")

# Allocate output arrays
patches_out = torch.zeros(N_OBS_TOTAL, ph, pw, dtype=torch.float32)
valid_out = torch.zeros(N_OBS_TOTAL, ph, pw, dtype=torch.uint8)
filled = torch.zeros(N_OBS_TOTAL, dtype=torch.bool)

total_loaded = 0
for part_path in parts:
    idx_path = str(part_path).replace(".safetensors", "_indices.pt")
    try:
        indices = torch.load(idx_path, weights_only=True)
        data = load_file(str(part_path))
    except Exception as e:
        print(f"  SKIP {part_path.name}: {e}")
        continue

    n = indices.shape[0]
    patches_out[indices] = data["patches"]
    if "valid" in data:
        valid_out[indices] = data["valid"]
    else:
        valid_out[indices] = 1
    filled[indices] = True
    total_loaded += n
    print(f"  {part_path.name}: {n} obs (total: {total_loaded})")

n_filled = filled.sum().item()
n_missing = N_OBS_TOTAL - n_filled
print(f"\nFilled: {n_filled}/{N_OBS_TOTAL} ({100*n_filled/N_OBS_TOTAL:.1f}%)")
if n_missing > 0:
    print(f"WARNING: {n_missing} observations missing ROIs")

# Save concatenated
print(f"\nSaving to {FINAL_PATH}...")
save_file({
    "patches": patches_out,
    "valid": valid_out,
}, str(FINAL_PATH))

size_gb = FINAL_PATH.stat().st_size / 1e9
print(f"Done: {FINAL_PATH} ({size_gb:.1f} GB)")
