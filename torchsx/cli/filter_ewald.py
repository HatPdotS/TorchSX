"""Filter SFX stream observations by distance from the Ewald sphere.

Per-observation excitation error r = ‖q + s₀‖ − 1/λ (Å⁻¹) is computed from
crystal orientation matrices and beam wavelength.  By default observations
with |r| > max_offset are dropped (use --invert to keep them instead).

Orphan crystals and images that lose all their observations are removed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torchsx.stream import SFXStream


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="torchsx-filter-ewald",
        description="Filter SFX stream observations by Ewald sphere distance.",
    )
    p.add_argument("stream", type=Path, help="Input .sfx or .stream file")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--max-offset", type=float, default=None,
        help="Absolute |r| cutoff in Å⁻¹ (e.g. 0.003 keeps obs within "
        "3×10⁻³ Å⁻¹).  Mutually exclusive with --max-offset-rel.",
    )
    g.add_argument(
        "--max-offset-rel", type=float, default=None,
        help="Per-crystal cutoff as a fraction of profile_rad: keep |r| ≤ "
        "K·profile_rad[crystal].  K=0.1 means 10%% of each crystal's "
        "CrystFEL integration radius.  Mutually exclusive with --max-offset.",
    )
    p.add_argument(
        "--invert", action="store_true",
        help="Keep |r| > cutoff (off-Bragg tail) instead of dropping it.",
    )
    p.add_argument(
        "--output-sfx", type=Path, default=None,
        help="Write filtered stream as .sfx (binary safetensors).",
    )
    p.add_argument(
        "--output-stream", type=Path, default=None,
        help="Write filtered stream as CrystFEL .stream (text).",
    )
    p.add_argument(
        "--merge-friedel", action="store_true",
        help="Passed through to SFXStream.load().",
    )
    p.add_argument(
        "--n-threads", type=int, default=4,
        help="Parallel threads for write_stream output.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not (args.output_sfx or args.output_stream):
        print("ERROR: supply at least one of --output-sfx / --output-stream",
              file=sys.stderr)
        return 2

    print(f"[filter-ewald] load {args.stream}", flush=True)
    stream = SFXStream.load(args.stream, merge_friedel=args.merge_friedel)
    n_before = int(stream._core.hkl.shape[0])
    nc_before = int(stream._core.n_crystals)
    ni_before = int(stream._core.n_images)

    direction = "keep" if args.invert else "drop"
    if args.max_offset_rel is not None:
        print(
            f"[filter-ewald] {direction} |r| > {args.max_offset_rel:.6g}·profile_rad "
            f"(per-crystal)",
            flush=True,
        )
        filtered = stream.filter_ewald(
            invert=args.invert, max_offset_rel=args.max_offset_rel,
        )
    else:
        print(
            f"[filter-ewald] {direction} |r| > {args.max_offset:.6g} Å⁻¹",
            flush=True,
        )
        filtered = stream.filter_ewald(args.max_offset, invert=args.invert)
    n_after = int(filtered._core.hkl.shape[0])
    nc_after = int(filtered._core.n_crystals)
    ni_after = int(filtered._core.n_images)

    pct = 100.0 * n_after / max(n_before, 1)
    print(
        f"[filter-ewald] obs: {n_before:,} → {n_after:,} ({pct:.1f}% kept)\n"
        f"               crystals: {nc_before:,} → {nc_after:,}\n"
        f"               images:   {ni_before:,} → {ni_after:,}",
        flush=True,
    )

    if args.output_sfx:
        args.output_sfx.parent.mkdir(parents=True, exist_ok=True)
        filtered.save(args.output_sfx)
        print(f"[filter-ewald] wrote {args.output_sfx}", flush=True)
    if args.output_stream:
        args.output_stream.parent.mkdir(parents=True, exist_ok=True)
        filtered.write_stream(args.output_stream, n_threads=args.n_threads)
        print(f"[filter-ewald] wrote {args.output_stream}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
