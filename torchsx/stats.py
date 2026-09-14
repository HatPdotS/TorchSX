"""CPU raw-distribution statistics. No merging, gain fitting or training.

Float64 arithmetic on the supplied intensities (stream parser storage is float32).
Cache retains finite signed observations, grouped by HKL, for exact re-rendering.
"""
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np

PLOTS = ('raw-intensity', 'reflection-skew', 'mean-median-gap', 'skew-examples')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def calculate(hkl, intensity, frames, cell, spacegroup, *, friedel=True, sigma=None):
    """Group symmetry equivalents; retain negatives, zeros and invalid-sigma rows."""
    hkl, intensity, frames = np.asarray(hkl), np.asarray(intensity, dtype=float), np.asarray(frames)
    if hkl.shape != (len(intensity), 3) or frames.shape != intensity.shape:
        raise ValueError('HKL, intensity and frame arrays must align')
    good = np.isfinite(intensity)
    if not good.any():
        raise ValueError('No finite intensities in selection')
    unique, inverse = np.unique(hkl[good], axis=0, return_inverse=True)
    sg = gemmi.SpaceGroup(spacegroup)
    asu, ops = gemmi.ReciprocalAsu(sg), sg.operations()
    mapped = []
    for row in unique:
        canonical, isym = asu.to_asu(row.tolist(), ops)
        minus = not friedel and isym % 2 == 0 and not ops.is_reflection_centric(canonical)
        mapped.append(np.asarray(canonical) * (-1 if minus else 1))
    unique, remap = np.unique(mapped, axis=0, return_inverse=True)
    ids, values = remap.reshape(-1)[inverse.reshape(-1)], intensity[good]
    n = np.bincount(ids, minlength=len(unique))
    mean = np.bincount(ids, weights=values) / n
    delta = values - mean[ids]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        m2, m3 = [np.bincount(ids, weights=delta**p) / n for p in (2, 3)]
        skew = np.sqrt(n * (n - 1)) / (n - 2) * m3 / m2**1.5
    skew[(n < 3) | (m2 <= 0) | ~np.isfinite(skew)] = np.nan
    order = np.lexsort((values, ids))
    values = values[order]
    start = np.cumsum(n) - n
    quantiles = []
    for q in (.25, .5, .75):
        position = (n - 1) * q
        lo, hi = np.floor(position).astype(int), np.ceil(position).astype(int)
        quantiles.append(values[start + lo] + (position - lo) * (values[start + hi] - values[start + lo]))
    q25, median, q75 = quantiles
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        gap = (mean - median) / (q75 - q25)
    gap[(q75 <= q25) | ~np.isfinite(gap)] = np.nan
    pairs = np.unique(np.column_stack((ids, frames[good])), axis=0)
    n_frames = np.bincount(pairs[:, 0].astype(int), minlength=len(n))
    cell = gemmi.UnitCell(*cell)
    d = np.array([cell.calculate_d(row.tolist()) for row in unique])
    absent = np.array([ops.is_systematically_absent(row.tolist()) for row in unique])
    if sigma is not None and np.asarray(sigma).shape != intensity.shape:
        raise ValueError('Sigma and intensity arrays must align')
    positive_sigma = np.asarray(sigma)[good] if sigma is not None else np.array([])
    positive_sigma = positive_sigma[np.isfinite(positive_sigma) & (positive_sigma > 0)]
    width = float(np.median(positive_sigma)) if len(positive_sigma) else max(float(np.median(abs(values))), 1.)
    return dict(hkl=unique, I=values, start=start, n=n, n_frames=n_frames, mean=mean,
                median=median, q25=q25, q75=q75, skew=skew, gap=gap, d=d, absent=absent,
                width=np.array(width), n_input=np.array(len(intensity)), n_nonfinite=np.array((~good).sum()))


def read_inputs(args):
    """Reuse production I/O, with explicit sampling and strict split assignment."""
    from torchsx.stream import SFXStream
    from torchsx.io.times import load_split_labels
    from torchsx.cli._pipeline import _workers
    inputs = []
    for entry in args.split:
        name, sep, path = entry.partition('=')
        if not sep or not name.strip() or not path.strip():
            raise ValueError('--split requires NAME=PATH')
        inputs.append((name.strip(), Path(path.strip())))
    inputs += [(p.stem, p) for p in args.stream]
    if not inputs or len({n for n, _ in inputs}) != len(inputs):
        raise ValueError('Supply inputs with unique names')
    if args.split_file and len(inputs) != 1:
        raise ValueError('--split-file requires exactly one input')
    lookup = None
    if args.split_file:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # Conflicting labels must not silently win.
            lookup = load_split_labels(args.split_file)
    results, sources = {}, []
    for name, path in inputs:
        stamp = path.stat()
        if path.suffix not in ('.stream', '.sfx'):
            raise ValueError('Expected .stream or .sfx input')
        if path.suffix == '.stream' and not args.spacegroup:
            raise ValueError('--spacegroup is required for .stream input')
        stream = SFXStream.read(path, spacegroup=args.spacegroup, merge_friedel=args.merge_friedel,
                               max_crystals=args.sample_crystals, seed=args.seed, n_workers=_workers())
        if path.suffix == '.sfx' and args.sample_crystals is not None and stream.n_crystals > args.sample_crystals:
            # Cached tensors load in full; selection still keeps complete images.
            import torch
            counts = np.bincount(stream.image_id.cpu().numpy(), minlength=stream.n_images)
            order = np.random.default_rng(args.seed).permutation(np.flatnonzero(counts))
            stop = np.searchsorted(np.cumsum(counts[order]), args.sample_crystals) + 1
            keep = np.zeros(stream.n_images, dtype=bool)
            keep[order[:stop]] = True
            stream = stream.filter_images(torch.from_numpy(keep))
        array = lambda t: t.detach().cpu().numpy()
        cells = array(stream.cell)
        valid_cells = cells[np.isfinite(cells).all(axis=1) & (cells[:, :3] > 0).all(axis=1)
                            & (cells[:, 3:] > 0).all(axis=1) & (cells[:, 3:] < 180).all(axis=1)]
        if args.cell is None and not len(valid_cells):
            raise ValueError('No valid input unit cell; supply --cell')
        cell = np.asarray(args.cell if args.cell is not None else np.median(valid_cells, axis=0))
        unit_cell = gemmi.UnitCell(*cell)
        if (not np.isfinite(cell).all() or min(cell[:3]) <= 0 or not ((cell[3:] > 0) & (cell[3:] < 180)).all()
                or not np.isfinite(unit_cell.volume) or unit_cell.volume <= 0):
            raise ValueError('Invalid reference unit cell')
        spacegroup = stream._core.spacegroup
        print(f'{name}: {stream.n_crystals:,} crystals, {stream.n_images:,} images; '
              f'{spacegroup}, cell {cell.tolist()}', flush=True)
        frame = array(stream.image_id)[array(stream.crystal_id).astype(int)]
        labels = np.full(stream.n_images, name, dtype=object)
        if lookup is not None:
            if len(stream.filenames) != stream.n_images or len(stream.events) != stream.n_images:
                raise ValueError('Custom splits require filename/event metadata')
            labels = np.array([lookup.get(k) for k in zip(stream.filenames, stream.events)], dtype=object)
            if any(v is None for v in labels):
                raise ValueError('Custom split file leaves sampled images unlabelled')
        selected = args.select or sorted(set(labels))
        for label in selected:
            if label in results or not label or Path(label).name != label or label in ('.', '..'):
                raise ValueError(f'Unsafe or duplicate output label: {label!r}')
            mask = labels[frame] == label
            if not mask.any():
                raise ValueError(f'No observations for requested label {label!r}')
            sigma = stream._core.levels['obs'].get('sigma')
            results[label] = calculate(array(stream.hkl)[mask], array(stream.I)[mask], frame[mask],
                cell, spacegroup, friedel=args.merge_friedel, sigma=None if sigma is None else array(sigma)[mask])
            results[label]['cell'] = cell
            results[label]['spacegroup'] = np.array(spacegroup)
            results[label]['merge_friedel'] = np.array(args.merge_friedel)
            results[label]['n_images'] = np.array(int((labels == label).sum()))
            results[label]['n_crystals'] = np.array(int((labels[array(stream.image_id)] == label).sum()))
        if (path.stat().st_size, path.stat().st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
            raise ValueError('Input changed during reading')
        sources.append(dict(path=str(path.resolve()), size=stamp.st_size, mtime_ns=stamp.st_mtime_ns,
                            cell=cell.tolist(), spacegroup=spacegroup,
                            sampled_crystals=stream.n_crystals, sampled_images=stream.n_images,
                            sidecar_sha256=digest(path.with_suffix('.sfx.json')) if path.suffix == '.sfx' else None))
    return results, sources


def save_cache(folder, results, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    entries = []
    for j, (label, table) in enumerate(results.items()):
        path = folder / f'{j:03d}.npz'
        np.savez_compressed(path, **table)
        entries.append(dict(label=label, file=path.name, sha256=digest(path)))
    (folder / 'manifest.json').write_text(json.dumps(dict(schema=1, entries=entries, **metadata), indent=2) + '\n')


def load_cache(folder):
    folder = Path(folder)
    meta = json.loads((folder / 'manifest.json').read_text())
    if meta['schema'] != 1:
        raise ValueError('Unsupported stats cache version')
    results = {}
    for item in meta['entries']:
        if Path(item['file']).name != item['file'] or Path(item['label']).name != item['label'] or item['label'] in ('.', '..'):
            raise ValueError('Invalid cache entry path')
        path = folder / item['file']
        if digest(path) != item['sha256']:
            raise ValueError(f'Cache checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as blob:
            results[item['label']] = {k: blob[k] for k in blob.files}
    return results


def render(results, output, *, plots=PLOTS, title='', targets=(-1, 0, 1, 2, 4, 6), min_obs=30, min_frames=20, intensity_unit='ADU'):
    """Only PNGs; no clipping. Examples use deterministic nearest unique HKLs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    output = Path(output)
    extent = np.concatenate([t['d'][np.isfinite(t['d']) & (t['d'] > 0) & ~t['absent']] for t in results.values()])
    if not len(extent):
        raise ValueError('No reflections with valid resolution')
    output.mkdir(parents=True, exist_ok=False)
    lo, hi = extent.min(), extent.max()
    lo, hi = (lo / 1.01, hi * 1.01) if lo == hi else (lo, hi)
    edges = np.geomspace(lo, hi, 21)
    width = float(np.median([t['width'] for t in results.values()]))
    limits = [min(t['I'].min() for t in results.values()), max(t['I'].max() for t in results.values())]
    transformed = np.arcsinh(np.array(limits) / width)
    if transformed[0] == transformed[1]:
        transformed += [-.1, .1]
    hist_edges = np.sinh(np.linspace(*transformed, 121)) * width
    hist_edges[0], hist_edges[-1] = min(hist_edges[0], limits[0]), max(hist_edges[-1], limits[1])
    def eligible(t):
        return (t['n'] >= min_obs) & (t['n_frames'] >= min_frames) & np.isfinite(t['skew']) & np.isfinite(t['d']) & (t['d'] > 0) & ~t['absent']
    gap_values = np.concatenate([t['gap'][eligible(t) & np.isfinite(t['gap'])] for t in results.values()])
    gap_limits = (min(0., gap_values.min()), max(0., gap_values.max())) if len(gap_values) else (0., 1.)
    pad = max(gap_limits[1] - gap_limits[0], .1) * .05
    for label, t in results.items():
        folder = output / label
        folder.mkdir()
        def finish(fig, filename, heading):
            fig.suptitle(f'{heading}\n{title + " · " if title else ""}{label}', fontsize=13)
            fig.tight_layout(rect=(0, .04, 1, .95))
            fig.savefig(folder / f'{filename}.png', dpi=200)
            plt.close(fig)
        if 'raw-intensity' in plots:
            fig, ax = plt.subplots(figsize=(9, 5))
            counts = np.histogram(t['I'], hist_edges)[0].astype(float)
            counts[counts == 0] = np.nan
            ax.stairs(counts, hist_edges)
            ax.set_xscale('asinh', linear_width=width)
            ax.set(yscale='log', xlabel=f'Observed intensity ({intensity_unit}; asinh scale)', ylabel='Observations per bin (log scale)')
            powers = 10.**np.arange(np.ceil(np.log10(2 * width)), max(np.ceil(np.log10(2 * width)), np.ceil(np.log10(max(abs(hist_edges))))) + 1)
            ticks = [v for v in np.r_[-powers[::-1], 0., powers] if hist_edges[0] <= v <= hist_edges[-1]]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.axvline(0, color='grey', lw=.7)
            ax.text(.98, .97, f'Observations: {len(t["I"]):,}\nNegative: {np.mean(t["I"] < 0):.1%}', transform=ax.transAxes, ha='right', va='top')
            fig.text(.5, .015, 'Unscaled observations; negative and zero intensities retained', ha='center', fontsize=9)
            finish(fig, 'raw_intensity_log_counts', 'Raw intensity distribution')
        good = eligible(t)
        for key, plot, heading, ylabel in [('skew', 'reflection-skew', 'Reflection skewness vs resolution', 'Sample skewness'),
                                         ('gap', 'mean-median-gap', 'Mean–median gap vs resolution', 'Mean–median gap / middle-50% width')]:
            if plot not in plots:
                continue
            shown = good & np.isfinite(t[key])
            fig, ax = plt.subplots(figsize=(10, 6))
            if shown.any():
                if key == 'skew':
                    points = ax.scatter(t['d'][shown], t[key][shown], c=t['n'][shown], s=5, alpha=.5, linewidths=0,
                                        norm=LogNorm(vmin=min_obs, vmax=max(min_obs + 1, t['n'][shown].max())))
                    fig.colorbar(points, ax=ax, label='Observations per reflection (log scale)')
                else:
                    ax.scatter(t['d'][shown], t[key][shown], s=5, alpha=.15, linewidths=0)
                    trend = []
                    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                        mask = shown & (t['d'] >= a) & ((t['d'] <= b) if j == 19 else (t['d'] < b))
                        trend.append(np.median(t[key][mask]) if mask.sum() >= 20 else np.nan)
                    ax.plot(np.sqrt(edges[:-1] * edges[1:]), trend, 'o-', color='darkorange', ms=3, label='Median per resolution bin')
                    ax.legend()
            else:
                ax.text(.5, .5, 'No eligible reflections', transform=ax.transAxes, ha='center')
            ax.set(xscale='log', xlim=(hi * 1.03, lo / 1.03), xlabel='Resolution (Å; log scale)', ylabel=ylabel)
            ticks = [v for v in [.5, .7, 1, 1.5, 2, 3, 5, 10, 20, 50, 100] if lo <= v <= hi]
            ax.xaxis.set_major_locator(FixedLocator(ticks if len(ticks) > 1 else np.geomspace(lo, hi, 3)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.axhline(0, color='black', ls='--', lw=.8)
            fig.text(.5, .015, f'{shown.sum():,} HKLs; at least {min_obs} observations and {min_frames} distinct images per HKL; systematic absences excluded',
                     ha='center', fontsize=9)
            if key == 'gap':
                ax.set_ylim(gap_limits[0] - pad, gap_limits[1] + pad)
            finish(fig, plot.replace('-', '_') + '_vs_resolution', heading)
        if 'skew-examples' in plots:
            available = good.copy()
            overview, axes = plt.subplots((len(targets) + 2) // 3, 3, figsize=(15, 4 * ((len(targets) + 2) // 3)), squeeze=False)
            for j, target in enumerate(targets):
                candidates = np.flatnonzero(available)
                if not len(candidates):
                    axes.flat[j].text(.5, .5, 'No eligible reflection', transform=axes.flat[j].transAxes, ha='center')
                    continue
                k = candidates[np.lexsort((candidates, -t['n'][candidates], abs(t['skew'][candidates] - target)))[0]]
                available[k] = False
                values = t['I'][t['start'][k]:t['start'][k] + t['n'][k]]
                counts, bins = np.histogram(values, bins=int(np.clip(np.ceil(np.sqrt(len(values))), 10, 60)))
                fig, ax = plt.subplots(figsize=(9, 5))
                for axis in [ax, axes.flat[j]]:
                    axis.stairs(np.where(counts > 0, counts, np.nan), bins)
                    axis.axvline(t['mean'][k], color='tab:red', ls='--', label='Mean')
                    axis.axvline(t['median'][k], color='black', ls=':', label='Median')
                    axis.set(yscale='log', xlabel=f'Observed intensity ({intensity_unit})', ylabel='Observations per bin (log scale)')
                    note = ' (nearest available)' if abs(t['skew'][k] - target) > .25 else ''
                    axis.set_title(f'Skewness = {float(round(t["skew"][k], 2)) + 0.:.2f}{note} · HKL {tuple(t["hkl"][k].tolist())}\n{t["d"][k]:.2f} Å · {t["n"][k]} observations · {t["n_frames"][k]} images · target {target:g}', fontsize=9)
                    axis.legend(fontsize=8)
                fig.text(.5, .015, 'Example selected by proximity to requested skewness, not randomly', ha='center', fontsize=9)
                finish(fig, f'hkl_example_{j + 1:02d}', 'Reflection intensity distribution')
            for axis in list(axes.flat)[len(targets):]:
                axis.set_visible(False)
            overview.text(.5, .015, f'Examples selected by proximity to requested skewness; at least {min_obs} observations and {min_frames} distinct images per HKL',
                          ha='center', fontsize=9)
            finish(overview, 'skew_examples', 'Examples of reflection skewness')
        print(f'{label}: {len(t["I"]):,} finite observations; {good.sum():,} eligible HKLs; {int(t["n_nonfinite"])} nonfinite intensities excluded', flush=True)
