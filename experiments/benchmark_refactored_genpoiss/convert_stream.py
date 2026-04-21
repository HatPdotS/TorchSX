#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --array=0-133
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -p hour
#SBATCH -t 0:30:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss/convert_logs/convert_%a.out

"""SLURM array job: convert one CrystFEL .stream file to .sfx cache.

Usage:
    sbatch convert_stream.py

The SLURM_ARRAY_TASK_ID selects which .stream file to convert from the
sorted listing of streams/*.stream.  Pre-existing .sfx files are skipped.

Adjust --array=0-N to match the stream count if it changes.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

STREAM_DIR = Path(
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams"
)
SPACEGROUP = "P -1"

files = sorted(STREAM_DIR.glob("*.stream"))
task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
if task_id_env is None:
    print("SLURM_ARRAY_TASK_ID not set — running all files sequentially", flush=True)
    indices = range(len(files))
else:
    indices = [int(task_id_env)]

from torchsx.stream import SFXStream

for idx in indices:
    if idx >= len(files):
        print(f"  skip idx={idx} (only {len(files)} streams)", flush=True)
        continue
    src = files[idx]
    dst = src.with_suffix(".sfx")
    if dst.exists() and dst.with_suffix(dst.suffix + ".json").exists():
        print(f"  skip {src.name} (cache exists)", flush=True)
        continue
    t0 = time.perf_counter()
    try:
        SFXStream.from_stream(src, spacegroup=SPACEGROUP, cache=dst, n_workers=2)
    except Exception as exc:
        print(f"  FAIL {src.name}: {exc}", flush=True)
        sys.exit(1)
    print(f"  done {src.name} in {time.perf_counter()-t0:.1f}s", flush=True)
