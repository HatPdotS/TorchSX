"""Raw statistics, cache integrity and the real CLI I/O seam; CPU only."""
import numpy as np
import pytest
import torch
from torchsx.stats import calculate, load_cache, save_cache, render
from torchsx.cli.stats import main
from torchsx.stream import SFXStream

CELL = [20, 20, 20, 90, 90, 90]


def table(values, **kwargs):
    return calculate(np.tile([1, 0, 0], (len(values), 1)), values, np.arange(len(values)), CELL, 'P 1', **kwargs)


def test_numerics_and_affine_invariance():
    x = np.random.default_rng(42).exponential(size=101) - 2
    t = table(x)
    np.testing.assert_allclose([t['q25'][0], t['median'][0], t['q75'][0]], np.quantile(x, [.25, .5, .75]))
    np.testing.assert_allclose(t['mean'], x.mean())
    np.testing.assert_allclose(t['gap'], (x.mean() - np.median(x)) / np.diff(np.quantile(x, [.25, .75])))
    np.testing.assert_allclose(table(x * 30 - 17)['gap'], t['gap'])
    scipy = pytest.importorskip('scipy.stats')
    np.testing.assert_allclose(t['skew'], scipy.skew(x, bias=False), rtol=1e-12)


def test_signed_values_invalid_sigma_and_constant_groups():
    t = table([-2, 0, 1, np.nan], sigma=[np.nan, -1, 0, 1])
    assert t['n'].tolist() == [3] and t['n_nonfinite'] == 1
    assert t['I'].tolist() == [-2, 0, 1] and np.isfinite(t['gap']).all()
    assert np.isnan(table([1, 1, 1])['gap']).all()
    assert np.isnan(table([1, 1, 1])['skew']).all()
    with pytest.raises(ValueError, match='No finite'):
        table([np.nan])


def test_symmetry_friedel_frame_counts_and_absences():
    h = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]])
    t = calculate(h, [1, 2, 3], [0, 0, 1], CELL, 'P 1')
    assert t['n'].tolist() == [3] and t['n_frames'].tolist() == [2]
    assert len(calculate(h, [1, 2, 3], [0, 0, 1], CELL, 'P 1', friedel=False)['n']) == 2
    t = calculate([[1, 2, 3], [-1, 2, -3], [0, 1, 0]], [1, 2, 3], [0, 1, 2], CELL, 'P 1 21 1')
    assert sorted(t['n'].tolist()) == [1, 2] and t['absent'].sum() == 1


def test_cache_roundtrip_and_tampering(tmp_path):
    original = table([-1, 0, 5])
    save_cache(tmp_path / 'cache', {'dark': original}, {})
    for k, value in load_cache(tmp_path / 'cache')['dark'].items():
        np.testing.assert_array_equal(value, original[k])
    with (tmp_path / 'cache/000.npz').open('ab') as f:
        f.write(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        load_cache(tmp_path / 'cache')


def test_cli_sfx_and_rerender_without_source(sample_core, tmp_path):
    core = sample_core.replace(hkl=torch.tensor([[1, 0, 0]] * 5, dtype=torch.int16))
    path = tmp_path / 'dark.sfx'
    SFXStream(core).save(path)
    main([str(path), '--spacegroup', 'P 1', '--cell', *map(str, CELL), '--full', '--output', str(tmp_path / 'png'),
          '--target-skew', '0', '--min-observations', '3', '--min-frames', '2'])
    files = list((tmp_path / 'png').rglob('*.*'))
    assert len(files) == 5 and all(p.suffix == '.png' for p in files)
    path.unlink()
    main(['--from-cache', str(tmp_path / 'png-cache'), '--output', str(tmp_path / 'rerender'), '--plots', 'raw-intensity'])
    assert len(list((tmp_path / 'rerender').rglob('*.png'))) == 1
    with pytest.raises(SystemExit):
        main(['--from-cache', str(tmp_path / 'png-cache'), '--output', str(tmp_path / 'rerender')])


def test_custom_splits_strict_and_selection(sample_core, tmp_path):
    sample_core.metadata['image'] = {'filename': ['/a.h5', '/b.h5'], 'event': ['//0', '//1']}
    path, split = tmp_path / 'data.sfx', tmp_path / 'custom.split'
    SFXStream(sample_core).save(path)
    split.write_text('/a.h5 //0 dark\n/b.h5 //1 light\n')
    argv = [str(path), '--split-file', str(split), '--select', 'dark', '--spacegroup', 'P 1', '--cell', *map(str, CELL),
            '--full', '--output', str(tmp_path / 'png'), '--plots', 'raw-intensity']
    main(argv)
    t = load_cache(tmp_path / 'png-cache')['dark']
    assert int(t['n_input']) == 3
    split.write_text('/a.h5 //0 dark\n/a.h5 //0 light\n')
    argv[argv.index(str(tmp_path / 'png'))] = str(tmp_path / 'bad')
    with pytest.raises(UserWarning, match='conflicting'):
        main(argv)


def test_no_eligible_constant_input_and_png_only(tmp_path):
    render({'constant': table([0, 0, 0])}, tmp_path / 'figures', targets=[0])
    assert len(list((tmp_path / 'figures').rglob('*.png'))) == 4


def test_input_validation(tmp_path):
    with pytest.raises(SystemExit):
        main(['--output', str(tmp_path / 'out')])
    with pytest.raises(SystemExit):
        main(['--output', str(tmp_path / 'out'), '--cache', str(tmp_path / 'out/cache')])


def test_stream_reader_preserves_signed_intensities(sample_core, tmp_path):
    core = sample_core.replace(I=torch.tensor([-2., 0., 3., 4., 5.]))
    path = tmp_path / 'signed.stream'
    SFXStream(core).write_stream(path)
    main([str(path), '--spacegroup', 'P 1', '--cell', *map(str, CELL), '--full',
          '--output', str(tmp_path / 'png'), '--plots', 'raw-intensity'])
    np.testing.assert_array_equal(np.sort(load_cache(tmp_path / 'png-cache')['signed']['I']), [-2, 0, 3, 4, 5])


def test_resolution_ticks_and_intensity_unit(tmp_path, monkeypatch):
    import matplotlib.figure
    axes = []
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', lambda fig, *a, **k: axes.extend(fig.axes))
    hkl = np.repeat(np.column_stack([np.arange(1, 21), np.zeros((20, 2), int)]), 10, axis=0)
    t = calculate(hkl, np.tile(np.arange(10), 20), np.arange(200), CELL, 'P 1')
    render({'test': t}, tmp_path / 'png', min_obs=3, min_frames=1,
           plots=['raw-intensity', 'reflection-skew', 'mean-median-gap'], intensity_unit='stream units')
    assert 'stream units' in axes[0].get_xlabel()
    assert all(len(ax.get_xticks()) >= 5 for ax in axes if ax.get_xlabel().startswith('Resolution'))
