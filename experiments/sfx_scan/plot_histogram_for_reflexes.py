from torchsx.stream import SFXStream
from torchsx.scaling import AbsoluteScale, BScale, prepare_scaler_inputs
from torchsx.target._kl import KLScalingTarget
from random import sample
import os 
from pathlib import Path
import torch    
stream_path = '/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams/delay_-45.stream'
outdir = '/das/work/p17/p17490/Peter/torchsx/sfx_scan/hist'
cache = stream_path.replace('.stream', '.sfx')

STREAM_DIR = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan') / "streams"
DELAYS = [-91, -136, -182, -227, -273]
SPACEGROUP = "P-1"
OUTPUT_MTZ = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan') / "neg_delay_merged.mtz"

device = 'cpu'

print(f"Using device: {device}")

n_workers = 16

# -- 1. Load streams on CPU (with .sfx caching) --
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    print(f"Loading {stem.name} ...")
    s = SFXStream.from_stream(stem, spacegroup=SPACEGROUP, cache=cache, n_workers=n_workers)
    print(f"  {s.n_crystals} crystals, {s.n_observations} obs")
    streams.append(s)


s = SFXStream.concat(streams)

hkls = s.observations_by_hkl()


all_hkls = [hkl for hkl in hkls.keys()]


sample_hkls = sample(all_hkls, 10)

import matplotlib.pyplot as plt
os.makedirs(outdir, exist_ok=True)

for hkl in sample_hkls:
    plt.hist(hkls[hkl].cpu().numpy(), bins=50)
    plt.title(f"hkl={hkl}")
    plt.xlabel("I_obs")
    plt.ylabel("Count")
    plt.savefig(outdir + f"/hkl_{hkl}.png")
    plt.close()


