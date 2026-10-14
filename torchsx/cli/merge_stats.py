"""Statistics of merged reflection files: single merges, independent halves, and differences against a reference."""
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from torchsx.merge_stats import (AMPLITUDES, FORMATS, LABEL_KEYS, difference_stats, file_format, load_merge, render,
                                 split_half_stats, summary_stats)


def _labels(items):
    labels = {}
    for item in items or []:
        key, sep, col = item.partition('=')
        if not sep or key not in LABEL_KEYS or not col:
            raise argparse.ArgumentTypeError(f'--labels takes KEY=COLUMN with KEY one of {", ".join(LABEL_KEYS)}')
        labels[key] = col
    return labels


@dataclass
class Dataset:
    """One merge, two independent halves, or both (a state of a torchsx export)."""
    name: str
    merge: Path | None = None
    halves: tuple[Path, Path] | None = None

    @property
    def files(self):
        return [f for f in (self.merge, *(self.halves or ())) if f is not None]


def _from_files(name, files):
    return Dataset(name, merge=files[0]) if len(files) == 1 else Dataset(name, halves=tuple(files))


def discover(folder):
    """The states exported to ``folder`` by torchsx.train or torchsx.merge, in state order.

    An export writes ``STEM.metrics.json``, keyed by the states it wrote, next to
    ``STEM.mtz`` (``STEM_STATE.mtz`` when the run has several states) and its
    ``_half1``/``_half2`` companions. Each state is one dataset with its merge and
    its two halves, named by the state (by ``STEM`` when unnamed), or by the
    merge's file stem when the folder holds several exports.
    """
    folder = Path(folder)
    exports = sorted(folder.glob('*.metrics.json'))
    datasets = []
    for metrics in exports:
        stem = metrics.name[:-len('.metrics.json')]
        try:
            states = list(json.loads(metrics.read_text()))
        except (OSError, ValueError):
            continue
        for state in states:
            bases = [folder / f'{stem}_{state}', folder / stem] if state else [folder / stem]
            for base in bases:
                merge, half1, half2 = (base.with_name(base.name + s + '.mtz') for s in ('', '_half1', '_half2'))
                halves = (half1, half2) if half1.is_file() and half2.is_file() else None
                if merge.is_file() or halves:
                    name = base.name if len(exports) > 1 else (state or stem)
                    datasets.append(Dataset(name, merge if merge.is_file() else None, halves))
                    break
    return datasets


def _build_parser():
    p = argparse.ArgumentParser(prog='torchsx.merge-stats', description=__doc__,
        epilog='A dataset is one merge (summary statistics), two independent halves (CC1/2 and R_split), '
               'or, for each state of a torchsx output directory, both. Every dataset other than the '
               'reference is also scored as a difference against it, when both have two halves; a run '
               'with one state has no differences. Inputs: MTZ, CrystFEL .hkl/.hkl1/.hkl2, mmCIF '
               'structure factors. CPU only.')
    p.add_argument('files', nargs='*', type=Path, metavar='PATH',
                   help='the first dataset: one merge or two independent halves; or a torchsx.train / '
                        'torchsx.merge output directory, one dataset per state. Default: the current '
                        'directory, when it holds a torchsx export')
    p.add_argument('--name', help='name of the first dataset (default: stem of its first file, '
                                    'without _half1)')
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
    p.add_argument('--format', nargs='+', choices=FORMATS, default=['tsv', 'plots'], dest='formats',
                   help='outputs to write (default: tsv plots)')
    p.add_argument('-o', '--out', type=Path,
                   help='new output directory (default for a torchsx output directory: DIR/merge_stats)')
    return p


def _datasets(p, args):
    """The datasets in order, and the directory they were discovered in (None when given as files)."""
    folder = None
    if any(f.is_dir() for f in args.files):
        if len(args.files) != 1:
            p.error('give one torchsx output directory, or one or two merge files')
        folder = args.files[0]
    elif not (args.files or args.dataset or args.name):
        folder = Path('.')
    datasets = []
    if folder is not None:
        if args.name:
            p.error('--name names a dataset given as files; a torchsx output directory names its states')
        datasets = discover(folder)
        if not datasets:
            p.error(f'{folder.resolve()}: no torchsx export (STEM.metrics.json beside its merged MTZs); '
                    'supply FILE [FILE] or --dataset NAME FILE [FILE]')
        print(f'{p.prog}: {folder.resolve()}: ' + ', '.join(
            f"{d.name} ({' + '.join(['merge'] * bool(d.merge) + ['halves'] * bool(d.halves))})" for d in datasets))
    elif args.files:
        if len(args.files) > 2:
            p.error(f'the first dataset takes one merge or two halves, got {len(args.files)} files')
        stem = args.files[0].name.split('.')[0]
        if len(args.files) == 2:
            stem = stem.removesuffix('_half1')
        datasets.append(_from_files(args.name or stem, list(args.files)))
    elif args.name:
        p.error('--name names the positional dataset; give its files')
    for entry in args.dataset:
        if len(entry) not in (2, 3):
            p.error(f'dataset {entry[0]!r} takes one merge or two halves, got {len(entry) - 1} files')
        datasets.append(_from_files(entry[0], [Path(f) for f in entry[1:]]))
    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        p.error(f'dataset names must be unique: {names}')
    return datasets, folder


def _load(p, args, datasets, labels):
    """Every distinct file once; CrystFEL lists take the given symmetry, else the first other input's."""
    paths = list(dict.fromkeys(f for d in datasets for f in d.files))
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


def _write_tsv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval='', delimiter='\t', lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: _num(v) for k, v in r.items()})


def write_tsv(results, out):
    """``summary.tsv`` (one headline row per merge, pair of halves and difference) and the long per-shell tables."""
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
    written = ['summary.tsv']
    _write_tsv(out / 'summary.tsv', summary)
    for kind, rows in shells.items():
        if rows:
            _write_tsv(out / f'shells_{kind}.tsv', rows)
            written.append(f'shells_{kind}.tsv')
    if bins:
        _write_tsv(out / 'bins_difference.tsv', bins)
        written.append('bins_difference.tsv')
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
            rows.append(f"halves      {r['name']:<16} n {h['n_common']:>8d}  CC1/2 {_f(h['cc_half'], '.4f')}  "
                        f"outer {_f(h['cc_half_outer'], '.4f')}  R_split {_f(h['r_split'], '.4f')}  "
                        f"outer {_f(h['r_split_outer'], '.4f')}")
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
    datasets, folder = _datasets(p, args)
    by_name = {d.name: d for d in datasets}
    reference = args.reference or datasets[0].name
    if reference not in by_name:
        p.error(f'--reference {reference!r} is not a dataset name: {list(by_name)}')
    if args.out is None:
        if folder is None:
            p.error('the following arguments are required: -o/--out')
        args.out = folder / 'merge_stats'
    if args.out.exists():
        p.error(f'{args.out} already exists; choose a new --out directory')
    merges = _load(p, args, datasets, labels)

    results = {'settings': {'reference': reference, 'amplitudes': args.amplitudes, 'shells': args.shells,
                            'sigma_bins': args.sigma_bins, 'dmin': args.dmin, 'dmax': args.dmax, 'labels': labels},
               'datasets': [], 'differences': []}
    for d in datasets:
        if d.merge is not None:
            results['datasets'].append({'name': d.name, 'kind': 'summary', 'files': [str(d.merge)],
                                        **summary_stats(merges[d.merge], args.shells, args.dmax)})
        if d.halves is not None:
            results['datasets'].append({'name': d.name, 'kind': 'halves', 'files': [str(f) for f in d.halves],
                                        **split_half_stats(*(merges[f] for f in d.halves), n_shells=args.shells)})
    ref = by_name[reference]
    for d in datasets:
        if d is ref:
            continue
        if d.halves is None or ref.halves is None:
            print(f'{p.prog}: skipping difference {d.name} - {reference}: '
                  'both datasets need two independent halves', file=sys.stderr)
            continue
        res = difference_stats(tuple(merges[f] for f in ref.halves), tuple(merges[f] for f in d.halves),
                               n_shells=args.shells, n_sigma=args.sigma_bins)
        results['differences'].append({'name': d.name, 'reference': reference,
                                       'files': [str(f) for f in ref.halves + d.halves], **res})

    args.out.mkdir(parents=True)
    written = []
    if 'tsv' in args.formats:
        written += write_tsv(results, args.out)
    if 'json' in args.formats:
        (args.out / 'stats.json').write_text(json.dumps(results, indent=2) + '\n')
        written.append('stats.json')
    if 'plots' in args.formats:
        written += render(results, args.out)
    print(table(results))
    print(f'wrote {", ".join(written)} to {args.out}')


if __name__ == '__main__':
    main()
