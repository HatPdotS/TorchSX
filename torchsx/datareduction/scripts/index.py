#!/usr/bin/env python
"""Run the torchsx indexing pipeline end-to-end and write an SFXStream.

frames → spots → TORO indexing → integration → :class:`~torchsx.stream.SFXStream`.

The spot finder is selected automatically by hardware:

* **CUDA available** → the fused **matched-filter** Bayes finder (GPU; the
  precomputed-operator path, ~7.5 ms/frame).
* **CPU only** → the **standard** Bayes finder (the conv2d reference path).

Both are the ``"bayes"`` backend — :func:`~torchsx.datareduction.build_stream`
builds the fused operator only on CUDA, so the same call gives the right finder on
either device (override with ``--device``).

Inputs
------
* ``input``  — an HDF5 detector cube (JUNGFRAU), or a ``.lst`` listing HDF5 files
  (optionally with CrystFEL ``//event`` specifiers; one event per line restricts
  that file to those frames, a bare filename indexes all its frames).
* ``cell``   — a CrystFEL unit-cell file (``.cell``); the lattice is used as the
  TORO initial cell.
* ``geom``   — a CrystFEL geometry file (``.geom``).
* ``output`` — the stream path. ``.sfx`` (native safetensors, mergeable by
  ``torchsx-merge``) unless the suffix is ``.stream`` (CrystFEL text).

Example
-------
    python -m torchsx.datareduction.scripts.index \
        raw_data/run0032-lyso_13keV/data/acq0001.JF06T32V07.h5 \
        crystfel_ref/hewl.cell raw_data/16M_v1_distance_93.8.geom \
        /tmp/lyso_indexed.sfx
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import torch

from torchsx.datareduction import JungfrauReader, build_stream
from torchsx.geometry import Geometry
from torchsx.stream import SFXStream


# ───────────────────────────── unit cell ─────────────────────────────
def parse_cell_file(path: str | Path) -> tuple[torch.Tensor, dict]:
    """Parse a CrystFEL ``.cell`` file into a real-space basis + lattice metadata.

    Returns ``(basis, info)`` where ``basis`` is a ``(3, 3)`` tensor whose rows are
    the real-space cell vectors a, b, c (Å) — the convention used throughout
    ``torchsx.datareduction`` (``HEWL_CELL``, ``reciprocal_vectors``). ``info`` holds
    the ``lattice_type``/``centering``/``unique_axis`` strings (a ``.cell`` file does
    **not** carry the space group — pass it with ``--spacegroup`` for downstream
    symmetry/merging).
    """
    vals: dict[str, float] = {}
    info = {"lattice_type": "", "centering": "", "unique_axis": ""}
    for raw in Path(path).read_text().splitlines():
        line = raw.split(";", 1)[0].strip()        # strip comments
        if "=" not in line:
            continue
        key, val = (x.strip() for x in line.split("=", 1))
        if key in ("a", "b", "c", "al", "be", "ga"):
            vals[key] = float(val.split()[0])      # drop the "A" / "deg" unit token
        elif key in info:
            info[key] = val
    missing = {"a", "b", "c", "al", "be", "ga"} - vals.keys()
    if missing:
        raise ValueError(f"{path}: missing cell parameters {sorted(missing)}")

    a, b, c = vals["a"], vals["b"], vals["c"]
    al, be, ga = (math.radians(vals[k]) for k in ("al", "be", "ga"))
    ca, cb, cg, sg = math.cos(al), math.cos(be), math.cos(ga), math.sin(ga)
    # Standard crystallographic fractional→Cartesian: a∥x, b in xy-plane.
    vol = math.sqrt(max(0.0, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg))
    basis = torch.tensor(
        [[a, 0.0, 0.0],
         [b * cg, b * sg, 0.0],
         [c * cb, c * (ca - cb * cg) / sg, c * vol / sg]],
        dtype=torch.float32,
    )
    return basis, info


# ───────────────────────────── inputs ─────────────────────────────
def collect_inputs(path: str | Path) -> list[tuple[str, list[int] | None]]:
    """Resolve ``input`` to ``[(h5_path, frames|None), ...]``.

    A single ``.h5`` → one entry indexing all frames (``frames=None``). A ``.lst``
    (or ``.txt``/``.list``) is read line by line: ``file.h5`` indexes every frame,
    ``file.h5 //N`` (or ``file.h5//N``) restricts that file to listed events. A file
    appearing with any bare (event-less) line is taken as "all frames".
    """
    p = Path(path)
    if p.suffix.lower() not in (".lst", ".txt", ".list"):
        return [(str(p), None)]

    groups: dict[str, set[int] | None] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        fn, ev = parts[0], None
        if "//" in fn:                              # file.h5//N
            fn, _, evs = fn.partition("//")
            ev = int(evs) if evs else None
        elif len(parts) > 1 and parts[1].startswith("//"):  # file.h5 //N
            evs = parts[1][2:]
            ev = int(evs) if evs else None
        if ev is None:
            groups[fn] = None                       # bare line ⇒ all frames
        else:
            cur = groups.get(fn, set())
            if cur is not None:                     # don't override an "all" mark
                cur.add(ev)
                groups[fn] = cur
            groups.setdefault(fn, cur)
    return [(fn, (sorted(evs) if evs else None)) for fn, evs in groups.items()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Index a run with the Bayes matched-filter finder and write an SFXStream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="HDF5 detector cube, or a .lst of HDF5 files")
    ap.add_argument("cell", help="CrystFEL unit-cell file (.cell)")
    ap.add_argument("geom", help="CrystFEL geometry file (.geom)")
    ap.add_argument("output", help="output stream (.sfx native, or .stream CrystFEL)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto: CUDA→fused matched-filter, CPU→standard bayes")
    ap.add_argument("--data-path", default=None,
                    help="HDF5 dataset path of the detector cube (default: reader default)")
    ap.add_argument("--spacegroup", default=None,
                    help="space group for the stream (a .cell has none); default 'P 1'")
    ap.add_argument("--write-stream", default=None,
                    help="also write a CrystFEL .stream alongside the main output")
    ap.add_argument("--min-num-spots", type=int, default=8,
                    help="minimum indexed spots for a viable crystal")
    ap.add_argument("--hit-min-sumprob", type=float, default=2.0,
                    help="skip indexing a frame when Σ peak-probability is below this")
    ap.add_argument("--index-only", action="store_true",
                    help="write only indexed crystals; drop hit frames that failed to "
                         "index (default: keep every hit, with its peak list)")
    ap.add_argument("--n-calib", type=int, default=150,
                    help="frames sampled per file for noise/null calibration")
    ap.add_argument("--sigma-radial", type=float, default=2.0,
                    help="matched-filter radial kernel σ (spectral dispersion; per-dataset)")
    ap.add_argument("--sigma-azimuthal", type=float, default=0.7,
                    help="matched-filter azimuthal kernel σ")
    ap.add_argument("--mask", default=None,
                    help="external detector mask (.h5/.npy/.pt); AND-ed with the auto-build "
                         "unless --no-auto-mask (then used wholesale)")
    ap.add_argument("--no-auto-mask", action="store_true",
                    help="disable the automatic beamstop/edge mask build (use --mask if given, "
                         "else only the detector pixel_mask)")
    ap.add_argument("--force-rebuild-mask", action="store_true",
                    help="rebuild the mask even if a sidecar already exists")
    ap.add_argument("--mask-out", default=None,
                    help="also write the mask to this sidecar h5 and reuse it on reruns "
                         "(the mask is always embedded in the output .sfx regardless)")
    ap.add_argument("--mask-edge-dilate", type=int, default=1,
                    help="grow the bad/gap region by N px (edge guard band)")
    ap.add_argument("--mask-border", type=int, default=1,
                    help="mask the outer N rows/cols of every module")
    ap.add_argument("--mask-persist-frac", type=float, default=0.30,
                    help="mask pixels that fire as a peak in > this fraction of calib frames")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames processed per file (debug)")
    ap.add_argument("--good-frames-only", action="store_true",
                    help="skip frames flagged is_good_frame=False")
    ap.add_argument("-q", "--quiet", action="store_true", help="less per-frame logging")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    finder_kind = "fused matched-filter (GPU)" if device == "cuda" else "standard conv path (CPU)"
    print(f"[device] {device} → Bayes {finder_kind}")

    geom = Geometry.from_geom_file(args.geom)
    basis, cell_info = parse_cell_file(args.cell)
    lengths = basis.norm(dim=1)
    print(f"[cell] {args.cell}: a,b,c = {[round(float(x), 2) for x in lengths]} Å  "
          f"{cell_info['lattice_type']} {cell_info['centering']}".rstrip())
    spacegroup = args.spacegroup
    if spacegroup is None:
        spacegroup = "P 1"
        print("[cell] no --spacegroup given; using 'P 1' for the stream "
              "(set it for correct symmetry/merging)")

    inputs = collect_inputs(args.input)
    print(f"[input] {args.input}: {len(inputs)} file(s)")

    streams, tried, hits, indexed, obs = [], 0, 0, 0, 0
    for h5path, frames in inputs:
        reader = (JungfrauReader(h5path, data_path=args.data_path)
                  if args.data_path else JungfrauReader(h5path))
        if frames is None:
            frames = list(range(reader.n_frames))
            if args.good_frames_only:
                frames = [i for i in frames if reader.is_good_frame(i)]
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        print(f"[load] {Path(h5path).name}: indexing {len(frames)} frames "
              f"(λ={geom.wavelength_A:.4f} Å)")

        # Mask is embedded in the output .sfx; only write a separate sidecar (and
        # reuse it across reruns) when --mask-out is given.
        mask_path = args.mask_out
        stream, summary = build_stream(
            reader, geom, frames,
            initial_cell=basis, spacegroup=spacegroup,
            device=device, backend="bayes",
            min_num_spots=args.min_num_spots,
            hit_min_sumprob=args.hit_min_sumprob,
            keep_unindexed_hits=not args.index_only,
            n_calib=args.n_calib,
            sigma_radial=args.sigma_radial, sigma_azimuthal=args.sigma_azimuthal,
            mask_detector=not args.no_auto_mask,
            external_mask=args.mask,
            mask_path=mask_path,
            force_rebuild_mask=args.force_rebuild_mask,
            edge_dilate=args.mask_edge_dilate,
            module_border=args.mask_border,
            persist_frac=args.mask_persist_frac,
            verbose=not args.quiet,
        )
        print(f"[build] {Path(h5path).name}: {summary}")
        streams.append(stream)
        tried += summary["frames_tried"]
        hits += summary["hits"]
        indexed += summary["crystals_indexed"]
        obs += summary["n_observations"]
        reader.close()

    stream = SFXStream.concat(streams)
    print(f"[total] {indexed}/{tried} frames indexed, {hits} hits, {obs} observations")

    out = Path(args.output)
    if out.suffix == ".stream":
        stream.write_stream(str(out))
        print(f"[save] {out} (CrystFEL stream)")
    else:
        stream.save(str(out))
        print(f"[save] {out} ({stream.n_crystals} crystals, "
              f"{stream.n_observations} obs, {stream.n_images} images)")
    if args.write_stream:
        stream.write_stream(args.write_stream)
        print(f"[save] {args.write_stream} (CrystFEL stream)")


if __name__ == "__main__":
    main()
