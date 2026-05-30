#!/usr/bin/env python
"""Build an indexed SFXStream from a run — the full ingestion → Stream path.

Selects crystal-containing frames (from the online-monitor ``.dap`` hits, else by
curation), indexes each against the lysozyme cell, and folds the indexed crystals
into a columnar :class:`~torchsx.stream.SFXStream` via the async ``StreamBuilder``.
The result is saved as ``.sfx`` (and optionally a CrystFEL ``.stream``) and is
directly mergeable by ``torchsx-merge``.

Example
-------
    python -m torchsx.datareduction.scripts.build_stream \
        --data-dir raw_data/run0032-lyso_13keV/data \
        --geom raw_data/16M_v1_distance_93.8.geom \
        --output /tmp/lyso_indexed.sfx --write-stream /tmp/lyso_indexed.stream
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from torchsx.geometry import Geometry
from torchsx.datareduction import (
    JungfrauReader,
    PeakFinder8Params,
    build_stream,
    curate_file,
    list_run,
    parse_dap,
)


def select_frames(reader, geom, args) -> list[int]:
    """Frames to index: .dap hits if available, else curated hits."""
    dap_path = args.dap
    if dap_path is None:
        p = Path(reader.path)
        cand = p.with_name(p.name[:-3] + ".dap") if p.name.endswith(".h5") else None
        if cand and cand.exists():
            dap_path = str(cand)
    if dap_path and Path(dap_path).exists():
        dap = parse_dap(dap_path)
        pid = np.asarray(reader._grp()["pulse_id"][:]).ravel()
        pid2idx = {int(p): i for i, p in enumerate(pid)}
        frames = sorted(pid2idx[p] for p, (flag, n) in dap.items()
                        if flag and p in pid2idx)
        if frames:
            print(f"[select] {len(frames)} .dap hit frames")
            return frames
        print("[select] .dap has no hits; falling back to curation")

    params = PeakFinder8Params(adc_thresh=args.adc_thresh, min_snr=args.min_snr)
    records, summary = curate_file(
        reader, geom, params=params, min_peaks=args.min_peaks,
        max_frames=args.curate_frames,
    )
    print(f"[curate] {summary}")
    return sorted(r.frame for r in records if r.is_hit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="raw_data/run0032-lyso_13keV/data")
    ap.add_argument("--geom", default="raw_data/16M_v1_distance_93.8.geom")
    ap.add_argument("--acq", type=int, default=0)
    ap.add_argument("--output", default="/tmp/lyso_indexed.sfx")
    ap.add_argument("--write-stream", default=None,
                    help="also write a CrystFEL .stream to this path")
    ap.add_argument("--spacegroup", default="P 43 21 2")
    ap.add_argument("--dap", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--adc-thresh", type=float, default=150.0)
    ap.add_argument("--min-snr", type=float, default=8.0)
    ap.add_argument("--min-peaks", type=int, default=15)
    ap.add_argument("--min-num-spots", type=int, default=8)
    ap.add_argument("--curate-frames", type=int, default=200)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    geom = Geometry.from_geom_file(args.geom)
    files = list_run(args.data_dir)
    if not files:
        raise SystemExit(f"no JUNGFRAU files in {args.data_dir}")
    reader = JungfrauReader(files[args.acq])
    print(f"[load] {Path(reader.path).name}: {reader.n_frames} frames, "
          f"λ={geom.wavelength_A:.4f} Å")

    frames = select_frames(reader, geom, args)
    if not frames:
        raise SystemExit("no candidate frames; loosen thresholds")

    params = PeakFinder8Params(adc_thresh=args.adc_thresh, min_snr=args.min_snr)
    # NB: leave source_path empty — it is the "original .stream to copy from"
    # for write_stream; ingested data has none, so the header is synthesized.
    # Frame provenance lives in the builder's (file, frame) row-range index.
    stream, summary = build_stream(
        reader, geom, frames, params=params, spacegroup=args.spacegroup,
        device=args.device, min_num_spots=args.min_num_spots,
        verbose=True,
    )
    print(f"[build] {summary}")

    stream.save(args.output)
    print(f"[save] {args.output}  "
          f"({stream.n_crystals} crystals, {stream.n_observations} obs, "
          f"{stream.n_images} images)")
    if args.write_stream:
        stream.write_stream(args.write_stream)
        print(f"[save] {args.write_stream} (CrystFEL stream)")
    reader.close()


if __name__ == "__main__":
    main()
