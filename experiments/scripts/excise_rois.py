"""Pre-excise ROI patches for all streams (parallel-safe).

Run this before refine_e2e.py to cache ROI safetensors files.

Usage:
    .venv/bin/python excise_rois.py
"""

import time
from pathlib import Path

from torchsx import SFXStream
from torchsx.io._roi_store import ROIStore

STREAM_NAMES = ["delay_-45", "delay_-91", "delay_-136", "delay_-182"]
STREAM_DIR = "sfx_scan/streams"
H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"
PATCH_SIZE = 11
N_WORKERS = 4


def main():
    for name in STREAM_NAMES:
        roi_path = f"{name}_rois_{PATCH_SIZE}x{PATCH_SIZE}.safetensors"
        if Path(roi_path).exists():
            print(f"  {name}: cached ({roi_path})")
            continue

        sfx_path = f"{STREAM_DIR}/{name}.sfx"
        print(f"  {name}: loading stream...")
        s = SFXStream.load(sfx_path)
        s = s.remap_filenames(
            r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
            H5_ROOT + r"/\1/",
        )
        print(f"  {name}: excising {s._core.n_observations:,} patches with {N_WORKERS} workers...")
        t0 = time.time()
        s.excise_rois(roi_path, patch_size=PATCH_SIZE, n_workers=N_WORKERS)
        print(f"  {name}: done in {time.time()-t0:.1f}s")

    # Concatenate
    combined = f"combined_{len(STREAM_NAMES)}stream_rois_{PATCH_SIZE}x{PATCH_SIZE}.safetensors"
    if Path(combined).exists():
        print(f"  Combined ROI store already exists: {combined}")
    else:
        stores = [ROIStore(f"{n}_rois_{PATCH_SIZE}x{PATCH_SIZE}.safetensors") for n in STREAM_NAMES]
        print(f"  Concatenating {len(stores)} stores...")
        t0 = time.time()
        ROIStore.concat(stores, combined)
        print(f"  Done in {time.time()-t0:.1f}s → {combined}")


if __name__ == "__main__":
    main()
