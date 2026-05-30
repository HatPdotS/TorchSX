#!/usr/bin/env python
"""End-to-end milestone: load a run, find a crystal, index it.

Curates an acquisition file (or uses the online-monitor ``.dap`` to pick the
strongest hit), finds spots on that frame, maps them to reciprocal space, and
indexes them against the lysozyme cell with the PyTorch TORO indexer — then
prints the refined cell, orientation, and how many spots were indexed.

Example
-------
    python -m torchsx.datareduction.scripts.index_one \
        --data-dir raw_data/run0032-lyso_13keV/data \
        --geom raw_data/16M_v1_distance_93.8.geom
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

from torchsx.geometry import Geometry
from torchsx.datareduction import (
    HEWL_CELL,
    JungfrauReader,
    PeakFinder8Params,
    curate_file,
    find_spots_on_frame,
    index_spots,
    list_run,
    parse_dap,
    spots_to_q,
)


def pick_frame(reader, geom, args) -> int:
    """Choose a frame to index: the strongest .dap hit, else by curation."""
    dap_path = args.dap
    if dap_path is None:
        # default: sibling .dap next to the chosen h5 file (e.g.
        # acq0001.JF06T32V07.h5 -> acq0001.JF06T32V07.dap). Note the double
        # ".JF06T32V07.h5" suffix, so swap the trailing ".h5" by name.
        p = Path(reader.path)
        cand = p.with_name(p.name[:-3] + ".dap") if p.name.endswith(".h5") else None
        if cand and cand.exists():
            dap_path = str(cand)
    if dap_path and Path(dap_path).exists():
        dap = parse_dap(dap_path)
        pid = reader._grp()["pulse_id"][:].ravel()
        pid2idx = {int(p): i for i, p in enumerate(pid)}
        hits = [
            (pid2idx[p], p, n)
            for p, (flag, n) in dap.items()
            if flag and p in pid2idx
        ]
        if hits:
            hits.sort(key=lambda t: -t[2])
            idx, pulse, n = hits[0]
            print(f"[pick] strongest .dap hit: frame {idx} (pulse {pulse}, "
                  f"{n} peaks reported)")
            return idx
        print("[pick] .dap has no flagged hits; falling back to curation")

    params = PeakFinder8Params(adc_thresh=args.adc_thresh, min_snr=args.min_snr)
    records, summary = curate_file(
        reader, geom, params=params, min_peaks=args.min_peaks,
        max_frames=args.curate_frames,
    )
    print(f"[curate] {summary}")
    hits = sorted((r for r in records if r.is_hit), key=lambda r: -r.n_peaks)
    if not hits:
        raise SystemExit("no hits found during curation; loosen --min-peaks")
    print(f"[pick] best curated hit: frame {hits[0].frame} "
          f"({hits[0].n_peaks} peaks)")
    return hits[0].frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="raw_data/run0032-lyso_13keV/data")
    ap.add_argument("--geom", default="raw_data/16M_v1_distance_93.8.geom")
    ap.add_argument("--acq", type=int, default=0,
                    help="index into the sorted acq*.h5 file list")
    ap.add_argument("--frame", type=int, default=None,
                    help="explicit frame index (skips .dap/curation)")
    ap.add_argument("--dap", default=None, help="path to a .dap file")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--adc-thresh", type=float, default=150.0)
    ap.add_argument("--min-snr", type=float, default=8.0)
    ap.add_argument("--min-peaks", type=int, default=15)
    ap.add_argument("--curate-frames", type=int, default=200)
    ap.add_argument("--min-num-spots", type=int, default=8)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    geom = Geometry.from_geom_file(args.geom)
    files = list_run(args.data_dir)
    if not files:
        raise SystemExit(f"no JUNGFRAU files in {args.data_dir}")
    reader = JungfrauReader(files[args.acq])
    print(f"[load] {Path(reader.path).name}: {reader.n_frames} frames, "
          f"detector {reader.detector_shape}, {reader.n_panels} panels, "
          f"λ={geom.wavelength_A:.4f} Å")

    frame = args.frame if args.frame is not None else pick_frame(reader, geom, args)

    params = PeakFinder8Params(adc_thresh=args.adc_thresh, min_snr=args.min_snr)
    spots = find_spots_on_frame(reader, frame, geom, params)
    q = spots_to_q(spots, geom)
    print(f"[spots] frame {frame}: {spots.shape[0]} peaks, "
          f"|q| {float(q.norm(dim=1).min()):.3f}–{float(q.norm(dim=1).max()):.3f} Å⁻¹")

    res = index_spots(q, HEWL_CELL, device=args.device,
                      min_num_spots=args.min_num_spots)

    print("\n========== INDEXING RESULT ==========")
    print(f"  success         : {res.success}")
    print(f"  spots indexed   : {res.n_indexed} / {res.n_spots} "
          f"({100 * res.n_indexed / max(res.n_spots, 1):.0f}%)")
    cl = [round(float(x), 2) for x in res.cell_lengths]
    ca = [round(float(x), 2) for x in res.cell_angles]
    print(f"  refined cell    : a,b,c = {cl} Å   α,β,γ = {ca}°")
    print(f"  (HEWL reference : a,b,c = [79.0, 79.0, 38.0] Å   90,90,90°)")
    print(f"  max error       : {res.max_error:.2e}")
    print(f"  orientation B   :\n{res.basis.numpy()}")
    reader.close()


if __name__ == "__main__":
    main()
