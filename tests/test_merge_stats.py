"""Merge statistics: estimator recovery on synthetic merges, readers, and the CLI seam; CPU only."""
import csv
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torchref.io.mtz import write as write_mtz_frame
from torchref.symmetry import SpaceGroup

from torchsx.cli.merge_stats import discover, main
from torchsx.merge_stats import (d_spacing, design, difference_stats, halves_stats, load_merge, merge_from_arrays,
                                 possible_reflections, split_half_stats, summary_stats)

CELL = (40.0, 50.0, 60.0, 90.0, 90.0, 90.0)
SG = 'P 21 21 21'


def asu(dmin=1.6):
    """The unique reflections to ``dmin`` on the canonical ASU."""
    rep, _ = possible_reflections(CELL, SG, dmin)
    return np.unique(SpaceGroup(SG).canonicalize_hkl(torch.from_numpy(rep), sort=False)[0].numpy().astype(np.int64), axis=0)


def merge(hkl, F, SIGF, **kw):
    return merge_from_arrays(hkl, cell=CELL, spacegroup=SG, F=F, SIGF=SIGF, **kw)


def by_key(m):
    order = np.lexsort(m.hkl.T[::-1])
    return m.hkl[order], m.F[order], m.SIGF[order]


def write_mtz(path, hkl, columns):
    """``columns``: TorchRef label (F-obs, SIGF-obs, I-obs, SIGI-obs, N) -> values."""
    write_mtz_frame(pd.DataFrame({'H': hkl[:, 0], 'K': hkl[:, 1], 'L': hkl[:, 2], **columns}), list(CELL), SG, str(path))


def write_crystfel(path, hkl, I, SIGI, nmeas):
    rows = [f'{h:4d} {k:4d} {l:4d} {i:12.2f}        - {s:10.2f} {n:7d}' for (h, k, l), i, s, n in zip(hkl, I, SIGI, nmeas)]
    path.write_text('CrystFEL reflection list version 2.0\nSymmetry: mmm\n'
                    '   h    k    l          I    phase   sigma(I)   nmeas\n' + '\n'.join(rows) + '\nEnd of reflections\n')


def write_mmcif(path, hkl, I, SIGI):
    a, b, c, al, be, ga = CELL
    rows = '\n'.join(f'{h} {k} {l} {i:.2f} {s:.2f}' for (h, k, l), i, s in zip(hkl, I, SIGI))
    path.write_text(f"data_synthetic\n_cell.length_a {a}\n_cell.length_b {b}\n_cell.length_c {c}\n"
                    f"_cell.angle_alpha {al}\n_cell.angle_beta {be}\n_cell.angle_gamma {ga}\n"
                    f"_symmetry.space_group_name_H-M '{SG}'\nloop_\n_refln.index_h\n_refln.index_k\n_refln.index_l\n"
                    f"_refln.intensity_meas\n_refln.intensity_sigma\n{rows}\n")


def gaussian_halves(rng, n, mean=50.0, sd=10.0, noise=5.0):
    F = mean + sd * rng.standard_normal(n)
    return F, F + noise * rng.standard_normal(n), F + noise * rng.standard_normal(n)


def test_halves_recover_cc_noise_ratio_and_honest_sigmas():
    rng = np.random.default_rng(0)
    hkl = asu()
    _, FA, FB = gaussian_halves(rng, len(hkl))
    sig = np.full(len(hkl), 5.0)
    res = halves_stats(merge(hkl, FA, sig), merge(hkl, FB, sig))
    o = res['overall']
    assert o['n'] >= 0.999 * len(hkl) and res['intensities'] == 'F^2'
    assert o['cc_F'] == pytest.approx(100 / 125, abs=0.015)
    assert o['lnsr_F'] == pytest.approx(np.log(25 / 100), abs=0.06)
    assert o['rms_z'] == pytest.approx(1.0, abs=0.03)
    assert len(res['shells']) == 20 and res['shells'][0]['d_max'] > res['shells'][-1]['d_min']
    assert res['scale']['g_range'][0] > 0.95 and res['scale']['g_range'][1] < 1.05


def test_halves_invariant_to_anisotropic_scale_of_b():
    rng = np.random.default_rng(1)
    hkl = asu()
    _, FA, FB = gaussian_halves(rng, len(hkl))
    sig = np.full(len(hkl), 5.0)
    applied = np.array([0.3, 0.2, -0.1, 0.15, 0.05, 0.0, 0.0])
    g = np.exp(design(hkl) @ applied)
    plain = halves_stats(merge(hkl, FA, sig), merge(hkl, FB, sig))
    scaled = halves_stats(merge(hkl, FA, sig), merge(hkl, FB * g, sig * g))
    for key in ('cc_I', 'cc_F', 'rms_z', 'lnsr_I'):
        assert scaled['overall'][key] == pytest.approx(plain['overall'][key], abs=3e-3)
    np.testing.assert_allclose(scaled['scale']['c'][:5], -applied[:5] + np.array(plain['scale']['c'][:5]), atol=0.02)


def test_split_half_cc_and_rsplit_after_scaling():
    rng = np.random.default_rng(3)
    hkl = asu()
    _, FA, FB = gaussian_halves(rng, len(hkl))
    sig = np.full(len(hkl), 5.0)
    g = np.exp(design(hkl) @ np.array([0.3, 0.2, -0.1, 0.15, 0.05, 0.0, 0.0]))
    res = split_half_stats(merge(hkl, FA, sig), merge(hkl, FB * g, sig * g))
    IA, IB = FA**2, FB**2
    assert res['intensities'] == 'F^2' and len(res['shells']) == 20
    assert res['overall']['cc_half'] == pytest.approx(np.corrcoef(IA, IB)[0, 1], abs=3e-3)
    assert res['overall']['r_split'] == pytest.approx(np.sqrt(2) * np.abs(IA - IB).sum() / (IA + IB).sum(), rel=0.02)
    full = halves_stats(merge(hkl, FA, sig), merge(hkl, FB * g, sig * g))
    assert [(s['cc_half'], s['r_split']) for s in res['shells']] == [(s['cc_I'], s['rsplit_I']) for s in full['shells']]
    h = res['headline']
    assert (h['cc_half_outer'], h['r_split_outer']) == (res['shells'][-1]['cc_half'], res['shells'][-1]['r_split'])


def arms(rng, hkl, tau, sigma=3.0):
    F0 = 50 + 10 * rng.standard_normal(len(hkl))
    delta = tau * rng.standard_normal(len(hkl))
    s = np.full(len(hkl), sigma)
    make = lambda F: merge(hkl, F + sigma * rng.standard_normal(len(hkl)), s)    # noqa: E731
    return (make(F0), make(F0)), (make(F0 + delta), make(F0 + delta))


def test_difference_bits_match_gaussian_information():
    rng = np.random.default_rng(2)
    hkl = asu()
    ref, other = arms(rng, hkl, tau=4.0)
    res = difference_stats(ref, other)
    n = res['n_common']
    # per arm: signal variance 16 against noise variance 2 * 3^2 = 18
    assert res['headline']['bits'] == pytest.approx(n * 0.5 * np.log2(1 + 16 / 18), rel=0.08)
    assert np.mean([s['cc_dF'] for s in res['shells']]) == pytest.approx(16 / 34, abs=0.03)
    assert np.mean([s['lnsr_dF'] for s in res['shells']]) == pytest.approx(np.log(18 / 16), abs=0.08)
    assert len(res['bins']) == 20 * 5 and sum(b['n'] for b in res['bins']) == n
    null = difference_stats(*arms(rng, hkl, tau=0.0))
    assert abs(null['headline']['bits']) < 0.03 * n


def test_mtz_crystfel_and_mmcif_load_the_same_reflections(tmp_path):
    rng = np.random.default_rng(3)
    hkl = asu(2.0)
    I = np.round(rng.exponential(1000.0, len(hkl)) + 30 * rng.standard_normal(len(hkl)), 2)
    SIGI = np.full(len(hkl), 30.0)
    # symmetry equivalents and Friedel mates in place of the ASU indices
    moved = hkl.copy()
    moved[::3] *= np.array([-1, -1, 1])
    moved[1::3] *= -1
    write_mtz(tmp_path / 'i.mtz', hkl, {'I-obs': I, 'SIGI-obs': SIGI, 'N': np.full(len(hkl), 4, np.int32)})
    write_crystfel(tmp_path / 'i.hkl', moved, I, SIGI, np.full(len(hkl), 4))
    write_mmcif(tmp_path / 'i.cif', moved, I, SIGI)
    m_mtz = load_merge(tmp_path / 'i.mtz')
    m_hkl = load_merge(tmp_path / 'i.hkl', cell=CELL, spacegroup=SG)
    m_cif = load_merge(tmp_path / 'i.cif')
    ref = by_key(m_mtz)
    for m in (m_hkl, m_cif):
        h, F, S = by_key(m)
        np.testing.assert_array_equal(h, ref[0])
        np.testing.assert_allclose(F, ref[1], rtol=1e-4)
        np.testing.assert_allclose(S, ref[2], rtol=1e-4)
    assert m_mtz.amplitudes == 'french-wilson' and m_mtz.I is not None
    np.testing.assert_array_equal(m_hkl.counts, 4.0)
    np.testing.assert_array_equal(m_mtz.counts, 4.0)
    with pytest.raises(ValueError, match='no cell'):
        load_merge(tmp_path / 'i.hkl')


def test_amplitude_modes():
    rng = np.random.default_rng(4)
    hkl = asu(2.0)
    I = rng.exponential(100.0, len(hkl)) + 20 * rng.standard_normal(len(hkl))
    SIGI = np.full(len(hkl), 20.0)
    root = merge_from_arrays(hkl, cell=CELL, spacegroup=SG, I=I, SIGI=SIGI, amplitudes='sqrt')
    fw = merge_from_arrays(hkl, cell=CELL, spacegroup=SG, I=I, SIGI=SIGI)
    # every measured reflection keeps its row; only those with an amplitude enter comparisons
    assert root.amplitudes == 'sqrt' and len(root) == len(fw) == len(hkl)
    assert len(root.with_amplitudes()) == (I > 0).sum()
    order = np.lexsort(hkl.T[::-1])
    h, F, S = by_key(root.with_amplitudes())
    keep = I[order] > 0
    np.testing.assert_allclose(F, np.sqrt(I[order][keep]))
    np.testing.assert_allclose(S, SIGI[order][keep] / (2 * np.sqrt(I[order][keep])))
    amplitudes = fw.with_amplitudes()
    assert fw.amplitudes == 'french-wilson' and len(amplitudes) > (I > 0).sum() and (amplitudes.F > 0).all()


@pytest.mark.parametrize('cell, sg', [((62.0, 62.0, 110.0, 90, 90, 120), 'P 63'),
                                      ((60.0, 40.0, 50.0, 90, 105, 90), 'C 1 2 1'),
                                      ((30.0, 30.0, 30.0, 90, 90, 90), 'P 21 3')])
def test_one_representative_per_unique_reflection(cell, sg):
    rep, d = possible_reflections(cell, sg, 3.0)
    group = SpaceGroup(sg)
    canonical = group.canonicalize_hkl(torch.from_numpy(rep), sort=False)[0].numpy()
    assert len(np.unique(canonical, axis=0)) == len(rep) and (d >= 3.0 * (1 - 1e-9)).all()
    assert not group.is_absent(torch.from_numpy(rep)).any()
    # every equivalent of every index within the limit maps onto a representative
    eq = group.expand_reciprocal(torch.from_numpy(rep)).cpu().numpy().reshape(-1, 3)
    eq_canonical = group.canonicalize_hkl(torch.from_numpy(eq), sort=False)[0].numpy()
    assert len(np.unique(eq_canonical, axis=0)) == len(rep)


def test_summary_completeness_multiplicity_and_wilson_b():
    rng = np.random.default_rng(5)
    hkl = asu(1.5)
    d = d_spacing(hkl, CELL)
    eps = SpaceGroup(SG).epsilon(torch.from_numpy(hkl), friedel=False).double().numpy()
    I = 1e4 * eps * rng.exponential(1.0, len(hkl)) * np.exp(-25.0 / (2 * d**2))
    full = merge_from_arrays(hkl, cell=CELL, spacegroup=SG, I=I, SIGI=np.ones(len(hkl)), counts=np.full(len(hkl), 7))
    res = summary_stats(full)
    h = res['headline']
    assert h['completeness'] == pytest.approx(1.0) and h['n_absent'] == 0
    assert h['multiplicity'] == pytest.approx(7.0)
    assert all(abs(s['completeness'] - 1) < 0.02 for s in res['shells'])
    assert h['wilson_B'] == pytest.approx(25.0, abs=1.0)
    keep = rng.random(len(hkl)) > 0.1
    part = merge_from_arrays(hkl[keep], cell=CELL, spacegroup=SG, I=I[keep], SIGI=np.ones(keep.sum()))
    assert summary_stats(part)['headline']['completeness'] == pytest.approx(keep.mean(), abs=1e-9)
    assert np.isnan(summary_stats(part)['headline']['multiplicity'])


def rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


@pytest.fixture
def merges(tmp_path):
    rng = np.random.default_rng(6)
    hkl = asu(2.0)
    (dA, dB), (lA, lB) = arms(rng, hkl, tau=4.0)
    files = {}
    for name, m in {'dark_A': dA, 'dark_B': dB, 'light_A': lA, 'light_B': lB}.items():
        files[name] = tmp_path / f'{name}.mtz'
        write_mtz(files[name], m.hkl, {'F-obs': m.F, 'SIGF-obs': m.SIGF})
    return {k: str(v) for k, v in files.items()}


def test_cli_datasets_reference_and_outputs(merges, tmp_path, capsys):
    out = tmp_path / 'stats'
    main([merges['dark_A'], merges['dark_B'], '--name', 'dark', '--dataset', 'light', merges['light_A'],
          merges['light_B'], '--dataset', 'solo', merges['light_A'], '--format', 'tsv', 'json', 'plots',
          '-o', str(out)])
    captured = capsys.readouterr()
    assert 'skipping difference solo - dark' in captured.err
    for f in ('summary.tsv', 'shells_summary.tsv', 'shells_halves.tsv', 'shells_difference.tsv',
              'bins_difference.tsv', 'stats.json', 'summary.png', 'halves.png', 'difference.png'):
        assert (out / f).is_file(), f
    summary = rows(out / 'summary.tsv')
    assert [(r['kind'], r['name'], r['reference']) for r in summary] == [
        ('halves', 'dark', ''), ('halves', 'light', ''), ('summary', 'solo', ''), ('difference', 'light', 'dark')]
    assert float(summary[-1]['bits']) > 0
    results = json.loads((out / 'stats.json').read_text())
    assert results['settings']['reference'] == 'dark' and len(results['differences']) == 1
    assert len(rows(out / 'shells_difference.tsv')) == 20

    flipped = tmp_path / 'flipped'
    main([merges['dark_A'], merges['dark_B'], '--name', 'dark', '--dataset', 'light', merges['light_A'],
          merges['light_B'], '--reference', 'light', '-o', str(flipped)])
    back = rows(flipped / 'summary.tsv')[-1]
    assert (back['name'], back['reference']) == ('dark', 'light')
    assert float(back['bits']) == pytest.approx(float(summary[-1]['bits']), rel=0.05)
    assert not (flipped / 'stats.json').exists() and (flipped / 'difference.png').is_file()


def test_cli_crystfel_takes_symmetry_from_another_input(merges, tmp_path):
    rng = np.random.default_rng(7)
    m = load_merge(merges['dark_A'])
    for half in ('hkl1', 'hkl2'):
        I = m.F**2 + 5 * rng.standard_normal(len(m))
        write_crystfel(tmp_path / f'x.{half}', m.hkl, I, np.full(len(m), 5.0), np.full(len(m), 3))
    out = tmp_path / 'stats'
    main([str(tmp_path / 'x.hkl1'), str(tmp_path / 'x.hkl2'), '--dataset', 'mtz', merges['dark_A'],
          '--amplitudes', 'sqrt', '--format', 'tsv', '-o', str(out)])
    summary = rows(out / 'summary.tsv')
    assert [(r['kind'], r['name']) for r in summary] == [('halves', 'x'), ('summary', 'mtz')]
    assert float(summary[0]['cc_half']) > 0.9


def test_cli_rejects_bad_arguments(merges, tmp_path, monkeypatch):
    hkl_only = tmp_path / 'y.hkl'
    write_crystfel(hkl_only, asu(3.0), np.ones(len(asu(3.0))), np.ones(len(asu(3.0))), np.ones(len(asu(3.0)), int))
    (tmp_path / 'exists').mkdir()
    monkeypatch.chdir(tmp_path)
    bad = [
        ['--dataset', 'x', merges['dark_A'], merges['dark_B'], merges['light_A'], '-o', str(tmp_path / 'a')],
        [merges['dark_A'], merges['dark_B'], merges['light_A'], '-o', str(tmp_path / 'a')],
        [merges['dark_A'], '--reference', 'nope', '-o', str(tmp_path / 'b')],
        [str(hkl_only), '-o', str(tmp_path / 'c')],
        [merges['dark_A'], '-o', str(tmp_path / 'exists')],
        [merges['dark_A'], '--labels', 'X=Y', '-o', str(tmp_path / 'd')],
        [merges['dark_A']],
        ['-o', str(tmp_path / 'e')],
        [str(tmp_path / 'exists'), '-o', str(tmp_path / 'f')],
        [str(tmp_path / 'exists'), merges['dark_A'], '-o', str(tmp_path / 'g')],
    ]
    for argv in bad:
        with pytest.raises(SystemExit):
            main(argv)


def write_export(folder, states, stem='merged'):
    """A torchsx.train export: ``states`` maps a state to its (merge, half 1, half 2)."""
    folder.mkdir(parents=True, exist_ok=True)
    for state, products in states.items():
        base = f'{stem}_{state}' if len(states) > 1 else stem
        for suffix, m in zip(('', '_half1', '_half2'), products):
            write_mtz(folder / f'{base}{suffix}.mtz', m.hkl, {'F-obs': m.F, 'SIGF-obs': m.SIGF})
    (folder / f'{stem}.metrics.json').write_text(json.dumps({s: {'split': s} for s in states}) + '\n')
    return folder


def test_discover_names_states_and_finds_their_files(tmp_path):
    def export(folder, stem, states, bases):
        folder.mkdir(exist_ok=True)
        (folder / f'{stem}.metrics.json').write_text(json.dumps({s: {} for s in states}))
        for base in bases:
            for suffix in ('', '_half1', '_half2'):
                (folder / f'{base}{suffix}.mtz').touch()

    two = tmp_path / 'two'
    export(two, 'merged', ['dark', 'light', 'empty'], ['merged_dark', 'merged_light'])
    assert [(d.name, d.merge.name, tuple(f.name for f in d.halves)) for d in discover(two)] == [
        ('dark', 'merged_dark.mtz', ('merged_dark_half1.mtz', 'merged_dark_half2.mtz')),
        ('light', 'merged_light.mtz', ('merged_light_half1.mtz', 'merged_light_half2.mtz'))]
    one = tmp_path / 'one'
    export(one, 'merged', ['dark'], ['merged'])
    assert [(d.name, d.merge.name) for d in discover(one)] == [('dark', 'merged.mtz')]
    plain = tmp_path / 'plain'
    export(plain, 'merged', [''], ['merged'])
    (plain / 'merged.mtz').unlink()
    assert [(d.name, d.merge, len(d.halves)) for d in discover(plain)] == [('merged', None, 2)]
    export(plain, 'other', [''], ['other'])
    assert [d.name for d in discover(plain)] == ['merged', 'other']
    assert discover(tmp_path) == []


@pytest.fixture
def run_dir(tmp_path):
    rng = np.random.default_rng(8)
    (dA, dB), (lA, lB) = arms(rng, asu(2.0), tau=4.0)
    return write_export(tmp_path / 'run', {'dark': (dA, dA, dB), 'light': (lA, lA, lB)})


def test_cli_discovers_the_states_of_a_torchsx_export(run_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(run_dir)
    main(['--format', 'tsv', 'json'])
    assert 'dark (merge + halves), light (merge + halves)' in capsys.readouterr().out
    summary = rows(run_dir / 'merge_stats' / 'summary.tsv')
    assert [(r['kind'], r['name'], r['reference']) for r in summary] == [
        ('summary', 'dark', ''), ('halves', 'dark', ''), ('summary', 'light', ''), ('halves', 'light', ''),
        ('difference', 'light', 'dark')]
    assert summary[0]['files'].endswith('merged_dark.mtz') and float(summary[-1]['bits']) > 0
    with pytest.raises(SystemExit):
        main([])

    flipped = tmp_path / 'flipped'
    main([str(run_dir), '--reference', 'light', '--format', 'tsv', '-o', str(flipped)])
    back = rows(flipped / 'summary.tsv')[-1]
    assert (back['kind'], back['name'], back['reference']) == ('difference', 'dark', 'light')
    with pytest.raises(SystemExit):
        main([str(run_dir), '--name', 'x', '-o', str(tmp_path / 'named')])


def test_cli_single_state_export_has_no_difference(tmp_path, capsys):
    rng = np.random.default_rng(9)
    (a, b), _ = arms(rng, asu(2.0), tau=0.0)
    run = write_export(tmp_path / 'run', {'': (a, a, b)})
    main([str(run), '--format', 'tsv', 'json', 'plots'])
    assert 'difference' not in capsys.readouterr().err
    out = run / 'merge_stats'
    assert sorted(f.name for f in out.iterdir()) == [
        'halves.png', 'shells_halves.tsv', 'shells_summary.tsv', 'stats.json', 'summary.png', 'summary.tsv']
    assert [(r['kind'], r['name']) for r in rows(out / 'summary.tsv')] == [('summary', 'merged'), ('halves', 'merged')]
    assert json.loads((out / 'stats.json').read_text())['differences'] == []

    pair = tmp_path / 'pair'
    main([str(run / 'merged_half1.mtz'), str(run / 'merged_half2.mtz'), '-o', str(pair)])
    assert sorted(f.name for f in pair.iterdir()) == ['halves.png', 'shells_halves.tsv', 'summary.tsv']
    (row,) = rows(pair / 'summary.tsv')
    assert (row['kind'], row['name']) == ('halves', 'merged') and float(row['cc_half']) > 0.8
    assert set(rows(pair / 'shells_halves.tsv')[0]) == {'name', 'shell', 'd_max', 'd_min', 'inv_d2', 'n', 'cc_half',
                                                        'r_split'}
