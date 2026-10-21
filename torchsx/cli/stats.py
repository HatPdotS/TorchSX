"""Plot raw observation distributions to motivate scaling and likelihood modeling."""
import argparse
import json
from pathlib import Path

import numpy as np

from torchsx.cli import path_arg
from torchsx.stats import PLOTS, digest, load_cache, read_inputs, render, save_cache


def _sample_size(value):
    if value == 'all':
        return None
    try:
        n = int(value)
        if n > 0:
            return n
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f'Invalid sample size {value!r}; expected a positive integer or all')


def _build_parser():
    p = argparse.ArgumentParser(prog='torchsx.observation-stats', description=__doc__,
        epilog='CPU only. Signed input ADU, before scaling or merging. '
               'Sampling keeps complete images and precedes custom-split selection.')
    p.add_argument('stream', nargs='*', type=path_arg, help='.stream or .sfx inputs')
    p.add_argument('--split', action='append', default=[], metavar='NAME=PATH', help='named input; repeatable')
    p.add_argument('--split-file', type=path_arg, help='CrystFEL filename/event/label custom splits')
    p.add_argument('--select', nargs='+', help='custom-split labels to plot; default all labels')
    p.add_argument('--spacegroup', help='required for .stream; defaults to metadata for .sfx')
    p.add_argument('--cell', type=float, nargs=6, metavar=('A', 'B', 'C', 'ALPHA', 'BETA', 'GAMMA'),
                   help='override the median input cell used for resolution (angstrom/degrees)')
    p.add_argument('--sample-crystals', type=_sample_size, default=argparse.SUPPRESS, metavar='N|all',
                   help='random complete images until the crystal-count target per input '
                        '(default 50000); all reads the full input. The last image may exceed N.')
    p.add_argument('--seed', type=int, default=None, help='sampling seed (default 42)')
    p.add_argument('--merge-friedel', action=argparse.BooleanOptionalAction, default=None,
                   help='group Friedel mates (default); --no-merge-friedel keeps acentric mates apart')
    p.add_argument('-o', '--out', type=path_arg, required=True, help='new figure directory; cache saved beside it as OUT-cache')
    p.add_argument('--from-cache', type=path_arg, help='replot a stats cache without original inputs')
    p.add_argument('--plots', nargs='+', choices=PLOTS, default=list(PLOTS), help='plot families; default all four')
    p.add_argument('--target-skew', type=float, nargs='+', default=[-1, 0, 1, 2, 4, 6],
                   help='choose distinct example HKLs nearest these skewnesses; default -1 0 1 2 4 6')
    p.add_argument('--min-observations', type=int, default=30, help='minimum observations per HKL for shape plots (default 30)')
    p.add_argument('--min-frames', type=int, default=20, help='minimum distinct images per HKL for shape plots (default 20)')
    return p


def main(argv=None):
    p = _build_parser()
    args = p.parse_args(argv)
    if args.min_observations < 3 or args.min_frames < 1 or not np.isfinite(args.target_skew).all():
        p.error('Require minimum observations >= 3, frames >= 1 and finite skew targets')
    cache = args.out.with_name(args.out.name + '-cache')
    if args.out.exists() or (not args.from_cache and cache.exists()):
        p.error('Output/cache already exists; choose a new --out directory')
    if args.from_cache:
        if (args.stream or args.split or args.split_file or args.select or args.spacegroup or args.cell
                or hasattr(args, 'sample_crystals') or args.merge_friedel is not None or args.seed is not None):
            p.error('--from-cache accepts plot/eligibility options, not input overrides')
        results = load_cache(args.from_cache)
        cache = args.from_cache
    else:
        if not (args.stream or args.split):
            p.error('Supply a stream, --split NAME=PATH, or --from-cache')
        if args.select and not args.split_file:
            p.error('--select requires --split-file')
        args.seed = 42 if args.seed is None else args.seed
        args.sample_crystals = getattr(args, 'sample_crystals', 50000)
        args.merge_friedel = True if args.merge_friedel is None else args.merge_friedel
        results, sources = read_inputs(args)
        settings = {k: ([str(p) for p in v] if k == 'stream' else str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()}
        save_cache(cache, results, dict(sources=sources, settings=settings,
            code_sha256=digest(Path(__file__).parents[1] / 'stats.py'),
            cli_sha256=digest(Path(__file__)),
            split_sha256=digest(args.split_file) if args.split_file else None,
            sampling='all supplied observations' if args.sample_crystals is None else
                     'random complete images before split selection; crystal-count stopping rule',
            caveat='Raw distribution descriptors combine counting noise, crystal scales and other '
                   'heterogeneity; they do not measure accuracy or isolated multiplicative scatter.'))
    intensity_unit = json.loads((cache / 'manifest.json').read_text()).get('settings', {}).get('intensity_unit') or 'ADU'
    render(results, args.out, plots=args.plots, targets=args.target_skew,
           min_obs=args.min_observations, min_frames=args.min_frames, intensity_unit=intensity_unit)
    # Retain the figure recipe beside the cached observations, leaving the figure folder PNG-only.
    cache.joinpath(args.out.name + '-plots.json').write_text(json.dumps(dict(
        output=str(args.out.resolve()), plots=args.plots, target_skew=args.target_skew,
        min_observations=args.min_observations, min_frames=args.min_frames), indent=2) + '\n')
    print(f'PNG directory: {args.out}\nReusable cache: {cache}')


if __name__ == '__main__':
    main()
