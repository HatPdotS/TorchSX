"""``torchsx-merge`` — command-line interface for the streamlined chain.

Wraps ``stream.make_loss(...).refine(...).posterior(...).integrate()``
exactly as the Python API does — no extra knobs, no divergent code path.

Examples
--------
::

    torchsx-merge BR_sub.sfx \\
        --n-points 513 --gain auto --device cuda \\
        --prior flat --prior wilson \\
        --out unified
    # writes unified_flat.mtz and unified_wilson.mtz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchsx.stream import SFXStream


def _parse_gain(value: str) -> float | str:
    if value in ("auto", "stream", "empirical"):
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--gain must be a float or one of {{auto, stream, empirical}}, got {value!r}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="torchsx-merge",
        description="Streamlined log-space SFX merging.",
    )
    p.add_argument("stream", type=Path, help=".sfx file")
    p.add_argument("--n-points", type=int, default=513,
                   help="Log-S quadrature grid size (default 513)")
    p.add_argument("--S-min", type=float, default=1e-3,
                   help="Log-S grid lower bound (default 1e-3)")
    p.add_argument("--S-max", type=float, default=1e6,
                   help="Log-S grid upper bound (default 1e6)")
    p.add_argument("--gain", type=_parse_gain, default="auto",
                   help="ADU/photon: 'auto' | 'stream' | 'empirical' | float")
    p.add_argument("--min-obs", type=int, default=5,
                   help="Drop HKLs with fewer than this many positive obs")
    p.add_argument("--scale", default="b_per_crystal",
                   choices=["b_per_crystal", "panel"],
                   help="Scale model")
    p.add_argument("--smooth", type=float, default=500.0,
                   help="Miller-graph Dirichlet smoothing weight on log_α/log_β "
                        "(0 disables; 500 = legacy default)")
    p.add_argument("--wilson-aniso", type=float, default=0.0,
                   help="Anisotropic Wilson MAP prior weight on log_I0 "
                        "(0 disables; 5.0 = one pseudo-obs per HKL)")
    p.add_argument("--obs-chunk-size", type=int, default=0,
                   help="Per-obs chunk for the data target (0 = no chunking)")
    p.add_argument("--max-iter", type=int, default=150,
                   help="LBFGS max_iter")
    p.add_argument("--history-size", type=int, default=10,
                   help="LBFGS history size")
    p.add_argument("--prior", action="append",
                   choices=["flat", "wilson", "wilson_aniso"], default=None,
                   help="Posterior prior; pass multiple times for multiple priors")
    p.add_argument("--device", default="cpu",
                   help="'cpu' or 'cuda'")
    p.add_argument("--out", type=Path, required=True,
                   help="Output prefix (e.g. 'unified' → unified.mtz, "
                        "or unified_<prior>.mtz when multiple priors)")
    p.add_argument("--no-friedel", dest="merge_friedel",
                   action="store_false", default=True,
                   help="Don't merge Friedel mates on stream load")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    priors = args.prior or ["flat"]

    stream = SFXStream.load(args.stream, merge_friedel=args.merge_friedel)
    loss = stream.make_loss(
        n_points=args.n_points,
        S_min=args.S_min, S_max=args.S_max,
        gain=args.gain,
        min_obs=args.min_obs,
        scale=args.scale,
        smooth=args.smooth,
        wilson_aniso=args.wilson_aniso,
        obs_chunk_size=args.obs_chunk_size,
        device=args.device,
    ).refine(
        max_iter=args.max_iter,
        history_size=args.history_size,
    )

    out_prefix = args.out
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for prior in priors:
        if len(priors) > 1:
            target = out_prefix.with_name(f"{out_prefix.name}_{prior}.mtz")
        else:
            target = out_prefix if out_prefix.suffix == ".mtz" \
                     else out_prefix.with_suffix(".mtz")
        loss.posterior(prior=prior).integrate().write_mtz(str(target))
        print(f"wrote {target}", flush=True)

    print(
        f"final_nll={loss.final_nll:.6e}  n_hkl={loss.n_valid}  "
        f"gain={loss.stats.get('gain'):.4g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
