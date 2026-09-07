"""Generate raw-intensity, reflection-skew, mean–median-gap and example PNGs."""
import argparse
import json
from pathlib import Path

from torchsx.stats import PLOTS, digest, load_cache, read_inputs, render, save_cache


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog='CPU only. Raw signed ADU; no scaling or merging. Sampled chunks precede custom-split selection.')
    p.add_argument('stream', nargs='*', type=Path)
    p.add_argument('--split', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--split-file', type=Path)
    p.add_argument('--select', nargs='+', help='Custom-split labels to include')
    p.add_argument('--spacegroup')
    cell = p.add_mutually_exclusive_group()
    cell.add_argument('--cell', type=float, nargs=6, metavar=('A', 'B', 'C', 'ALPHA', 'BETA', 'GAMMA'))
    cell.add_argument('--reference-cell', type=Path, help='PDB/mmCIF; use unit cell only')
    sampling = p.add_mutually_exclusive_group()
    sampling.add_argument('--sample-crystals', type=int, help='Random complete chunks; may exceed this count')
    sampling.add_argument('--full', action='store_true', help='All supplied observations; can require substantial memory')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-workers', type=int, default=1)
    p.add_argument('--no-friedel', action='store_true')
    p.add_argument('--output', type=Path, required=True, help='New PNG-only directory')
    p.add_argument('--cache', type=Path, help='New cache directory; default OUTPUT-cache')
    p.add_argument('--from-cache', type=Path, help='Re-render without reading original streams')
    p.add_argument('--plots', nargs='+', choices=PLOTS, default=list(PLOTS))
    p.add_argument('--title', default='')
    p.add_argument('--intensity-unit', help='Axis label only, no conversion; default ADU or cached label')
    p.add_argument('--target-skew', type=float, nargs='+', default=[-1, 0, 1, 2, 4, 6])
    p.add_argument('--min-observations', type=int, default=30)
    p.add_argument('--min-frames', type=int, default=20)
    args = p.parse_args(argv)
    import importlib.util
    if importlib.util.find_spec('matplotlib') is None:
        p.error('Plotting requires matplotlib; install torchsx[stats]')
    import gemmi
    import numpy as np
    if args.min_observations < 3 or args.min_frames < 1 or args.n_workers < 1 or not np.isfinite(args.target_skew).all():
        p.error('Require minimum observations ≥ 3, frames/workers ≥ 1 and finite skew targets')
    cache = args.cache or args.output.with_name(args.output.name + '-cache')
    if args.output.exists() or (not args.from_cache and cache.exists()):
        p.error('Output/cache already exists; choose new directories')
    if not args.from_cache and (cache.resolve().is_relative_to(args.output.resolve()) or args.output.resolve().is_relative_to(cache.resolve())):
        p.error('PNG output and cache must be separate, non-nested directories')
    if args.from_cache:
        if args.stream or args.split or args.split_file or args.select or args.cache or args.spacegroup or args.cell or args.reference_cell or args.sample_crystals or args.full or args.no_friedel:
            p.error('--from-cache accepts presentation/eligibility options, not input overrides')
        results = load_cache(args.from_cache)
        cache = args.from_cache
        args.intensity_unit = args.intensity_unit or json.loads((cache / 'manifest.json').read_text()).get('settings', {}).get('intensity_unit')
    else:
        if not args.spacegroup or not (args.cell or args.reference_cell) or not (args.full or args.sample_crystals):
            p.error('Specify --spacegroup, --cell/--reference-cell and --full/--sample-crystals')
        if args.sample_crystals is not None and args.sample_crystals < 1:
            p.error('--sample-crystals must be positive')
        if args.select and not args.split_file:
            p.error('--select requires --split-file')
        if args.reference_cell:
            args.cell = list(gemmi.read_structure(str(args.reference_cell)).cell.parameters)
        unit_cell = gemmi.UnitCell(*args.cell)
        if not np.isfinite(args.cell).all() or min(args.cell[:3]) <= 0 or not all(0 < v < 180 for v in args.cell[3:]) or not np.isfinite(unit_cell.volume) or unit_cell.volume <= 0:
            p.error('Invalid reference unit cell')
        results, sources = read_inputs(args)
        save_cache(cache, results, dict(sources=sources, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'stream'},
            code_sha256=digest(Path(__file__).parents[1] / 'stats.py'),
            split_sha256=digest(args.split_file) if args.split_file else None,
            sampling='all supplied observations' if args.full else 'random complete chunks; crystal-count stopping rule',
            caveat='Raw-distribution descriptors, not accuracy or isolated multiplicative scatter. No byte-window eligibility gate.'))
    render(results, args.output, plots=args.plots, title=args.title, targets=args.target_skew,
           min_obs=args.min_observations, min_frames=args.min_frames, intensity_unit=args.intensity_unit or 'ADU')
    print(f'PNG directory: {args.output}\nReusable cache: {cache}')


if __name__ == '__main__':
    main()
