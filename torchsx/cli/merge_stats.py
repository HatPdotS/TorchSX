"""Statistics of merged reflection files: single merges, independent halves, and differences against a reference."""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from torchsx.merge_stats import (AMPLITUDES, FORMATS, LABEL_KEYS, difference_stats, file_format, halves_stats,
                                 load_merge, render, summary_stats)


def _labels(items):
    labels = {}
    for item in items or []:
        key, sep, col = item.partition('=')
        if not sep or key not in LABEL_KEYS or not col:
            raise argparse.ArgumentTypeError(f'--labels takes KEY=COLUMN with KEY one of {", ".join(LABEL_KEYS)}')
        labels[key] = col
    return labels


def _build_parser():
    p = argparse.ArgumentParser(prog='torchsx-merge-stats', description=__doc__,
        epilog='A dataset is one merge (summary statistics) or two independent halves (A/B statistics). '
               'Every dataset other than the reference is also scored as a difference against it, '
               'when both have two halves. Inputs: MTZ, CrystFEL .hkl/.hkl1/.hkl2, mmCIF structure factors. '
               'CPU only.')
    p.add_argument('files', nargs='*', type=Path, metavar='FILE',
                   help='the first dataset: one merge, or two independent halves')
    p.add_argument('--name', help='name of the first dataset (default: stem of its first file)')
    p.add_argument('--dataset', action='append', nargs='+', default=[], metavar=('NAME', 'FILE'),
                   help='an additional dataset: a name and one merge or two halves; repeatable')
    p.add_argument('--reference', metavar='NAME', help='dataset the differences are taken against (default: the first)')
    p.add_argument('--amplitudes', choices=AMPLITUDES, default='french-wilson',
                   help='amplitudes from intensity-only files: French-Wilson posterior (default) or sqrt(I) for I > 0')
    p.add_argument('--cell', type=float, nargs=6, metavar=('A', 'B', 'C', 'ALPHA', 'BETA', 'GAMMA'),
                   help='cell for CrystFEL .hkl inputs (default: from another input)')
    p.add_argument('--spacegroup', help='space group for CrystFEL .hkl inputs (default: from another input)')
    p.add_argument('--labels', nargs='+', metavar='KEY=COLUMN',
                   help='MTZ columns to use, keys F, SIGF, I, SIGI (default: TorchRef column priority)')
    p.add_argument('--shells', type=int, default=20, help='equal-count resolution shells (default 20)')
    p.add_argument('--sigma-bins', type=int, default=5,
                   help='quantile bins of sigma(dF) per shell for information in bits (default 5)')
    p.add_argument('--dmin', type=float, help='high-resolution cut (angstrom)')
    p.add_argument('--dmax', type=float, help='low-resolution cut (angstrom)')
    p.add_argument('--format', nargs='+', choices=FORMATS, default=['csv', 'plots'], dest='formats',
                   help='outputs to write (default: csv plots)')
    p.add_argument('-o', '--out', type=Path, required=True, help='new output directory')
    return p


def _datasets(p, args):
    datasets = []
    if args.files:
        datasets.append((args.name or args.files[0].name.split('.')[0], list(args.files)))
    elif args.name:
        p.error('--name names the positional dataset; give its files')
    for entry in args.dataset:
        datasets.append((entry[0], [Path(f) for f in entry[1:]]))
    if not datasets:
        p.error('Supply a dataset: FILE [FILE] and/or --dataset NAME FILE [FILE]')
    for name, files in datasets:
        if len(files) not in (1, 2):
            p.error(f'dataset {name!r} takes one merge or two halves, got {len(files)} files')
    names = [n for n, _ in datasets]
    if len(set(names)) != len(names):
        p.error(f'dataset names must be unique: {names}')
    return datasets


def _load(p, args, datasets, labels):
    """Every distinct file once; CrystFEL lists take the given symmetry, else the first other input's."""
    paths = list(dict.fromkeys(f for _, files in datasets for f in files))
    for f in paths:
        if not f.exists():
            p.error(f'{f}: no such file')
        try:
            file_format(f)
        except ValueError as e:
            p.error(str(e))
    merges = {}
    opts = dict(labels=labels, amplitudes=args.amplitudes, dmin=args.dmin, dmax=args.dmax)
    try:
        for f in (f for f in paths if file_format(f) != 'hkl'):
            merges[f] = load_merge(f, **opts)
        first = next(iter(merges.values()), None)
        cell = args.cell if args.cell is not None else (None if first is None else first.cell)
        spacegroup = args.spacegroup or (None if first is None else first.spacegroup)
        for f in (f for f in paths if file_format(f) == 'hkl'):
            merges[f] = load_merge(f, cell=cell, spacegroup=spacegroup, **opts)
    except ValueError as e:
        p.error(str(e))
    laue = {m.laue for m in merges.values()}
    if len(laue) > 1:
        p.error('inputs differ in Laue class: ' + ', '.join(f'{f} {m.spacegroup}' for f, m in merges.items()))
    return merges


def _num(v):
    if isinstance(v, (float, np.floating)):
        return '' if not np.isfinite(v) else f'{float(v):.6g}'
    return v


def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval='')
        w.writeheader()
        for r in rows:
            w.writerow({k: _num(v) for k, v in r.items()})


def write_csv(results, out):
    """``summary.csv`` (one headline row per dataset and difference) and the long per-shell tables."""
    summary, shells = [], {'summary': [], 'halves': [], 'difference': []}
    bins = []
    for r in results['datasets']:
        summary.append({'kind': r['kind'], 'name': r['name'], 'reference': '', 'files': ';'.join(r['files']),
                        **r['headline']})
        shells[r['kind']] += [{'name': r['name'], **s} for s in r['shells']]
    for r in results['differences']:
        summary.append({'kind': 'difference', 'name': r['name'], 'reference': r['reference'],
                        'files': ';'.join(r['files']), **r['headline']})
        shells['difference'] += [{'name': r['name'], 'reference': r['reference'], **s} for s in r['shells']]
        bins += [{'name': r['name'], 'reference': r['reference'], **b} for b in r['bins']]
    written = ['summary.csv']
    _write_csv(out / 'summary.csv', summary)
    for kind, rows in shells.items():
        if rows:
            _write_csv(out / f'shells_{kind}.csv', rows)
            written.append(f'shells_{kind}.csv')
    if bins:
        _write_csv(out / 'bins_difference.csv', bins)
        written.append('bins_difference.csv')
    return written


def _f(v, fmt):
    return '-' if v is None or not np.isfinite(v) else format(v, fmt)


def table(results):
    rows = []
    for r in results['datasets']:
        h = r['headline']
        if r['kind'] == 'summary':
            rows.append(f"summary     {r['name']:<16} n {h['n']:>8d}  d {h['d_max']:.2f}-{h['d_min']:.2f}  "
                        f"completeness {_f(h['completeness'], '.3f')}  I/sig {_f(h['mean_I_over_sigma'], '.2f')}  "
                        f"multiplicity {_f(h['multiplicity'], '.1f')}  Wilson B {_f(h['wilson_B'], '.1f')}")
        else:
            rows.append(f"halves      {r['name']:<16} n {h['n_common']:>8d}  cc_I {_f(h['cc_I'], '.4f')}  "
                        f"outer {_f(h['cc_I_outer'], '.4f')}  rsplit_I {_f(h['rsplit_I'], '.4f')}  "
                        f"rms_z {_f(h['rms_z'], '.3f')}  lnsr_I {_f(h['lnsr_I'], '.3f')}  "
                        f"lowres_drift {_f(h['lowres_drift'], '+.3f')}")
    for r in results['differences']:
        h = r['headline']
        rows.append(f"difference  {r['name'] + ' - ' + r['reference']:<16} n {h['n_common']:>8d}  "
                    f"bits {_f(h['bits'], '.0f')}  bits_res_only {_f(h['bits_res_only'], '.0f')}")
    return '\n'.join(rows)


def main(argv=None):
    p = _build_parser()
    args = p.parse_args(argv)
    try:
        labels = _labels(args.labels)
    except argparse.ArgumentTypeError as e:
        p.error(str(e))
    if args.shells < 1 or args.sigma_bins < 1:
        p.error('--shells and --sigma-bins must be >= 1')
    datasets = _datasets(p, args)
    reference = args.reference or datasets[0][0]
    if reference not in dict(datasets):
        p.error(f'--reference {reference!r} is not a dataset name: {[n for n, _ in datasets]}')
    if args.out.exists():
        p.error(f'{args.out} already exists; choose a new --out directory')
    if 'plots' in args.formats and importlib.util.find_spec('matplotlib') is None:
        p.error('Plotting requires matplotlib; install torchsx[stats] or drop plots from --format')
    merges = _load(p, args, datasets, labels)

    results = {'settings': {'reference': reference, 'amplitudes': args.amplitudes, 'shells': args.shells,
                            'sigma_bins': args.sigma_bins, 'dmin': args.dmin, 'dmax': args.dmax, 'labels': labels},
               'datasets': [], 'differences': []}
    for name, files in datasets:
        ms = [merges[f] for f in files]
        res = summary_stats(ms[0], args.shells, args.dmax) if len(ms) == 1 else halves_stats(*ms, n_shells=args.shells)
        results['datasets'].append({'name': name, 'kind': 'summary' if len(ms) == 1 else 'halves',
                                    'files': [str(f) for f in files], **res})
    ref_files = dict(datasets)[reference]
    for name, files in datasets:
        if name == reference:
            continue
        if len(files) != 2 or len(ref_files) != 2:
            print(f'torchsx-merge-stats: skipping difference {name} - {reference}: '
                  'both datasets need two independent halves', file=sys.stderr)
            continue
        res = difference_stats(tuple(merges[f] for f in ref_files), tuple(merges[f] for f in files),
                               n_shells=args.shells, n_sigma=args.sigma_bins)
        results['differences'].append({'name': name, 'reference': reference,
                                       'files': [str(f) for f in ref_files + files], **res})

    args.out.mkdir(parents=True)
    written = []
    if 'csv' in args.formats:
        written += write_csv(results, args.out)
    if 'json' in args.formats:
        (args.out / 'stats.json').write_text(json.dumps(results, indent=2) + '\n')
        written.append('stats.json')
    if 'plots' in args.formats:
        written += render(results, args.out)
    print(table(results))
    print(f'wrote {", ".join(written)} to {args.out}')


if __name__ == '__main__':
    main()
