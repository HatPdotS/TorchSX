"""Pseudo-translation classes: the phase cosines as the scale model's features, the
class columns of the tether's phase term, and a run that uses both."""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from tests.test_amortized_scale import _toy_prep
from tests.test_tether import _batch, _cli, _with
from torchsx.merging.scale import AmortizedScale
from torchsx.merging.scale.pseudo_translation import PhaseClasses
from torchsx.merging.tether import Tether, integral_step

T = [[0.5, 0.377, 0.5], [0.0, 0.2455, 0.0]]


def _grid():
    g = np.arange(-6, 7)
    hkl = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    hkl = hkl[(hkl != 0).any(axis=1)]
    cell = np.array([40.0, 50.0, 60.0])
    return hkl, ((hkl / cell) ** 2).sum(axis=1)


def test_cells_and_columns():
    """Every reflection lies in one cell; its row carries T_0 .. T_degree of its resolution
    in that cell's block and nothing elsewhere."""
    hkl, s2 = _grid()
    pc = PhaseClasses.fit(hkl, s2, T, "P 2 2 2")
    assert pc.n_cells == 16 and [len(e) for e in pc.edges] == [3, 3]
    cell = pc.cells(hkl)
    assert cell.min() >= 0 and cell.max() < pc.n_cells
    X = pc.columns(hkl, s2)
    k = pc.degree + 1
    assert X.shape == (len(hkl), pc.n_cells * k)
    np.testing.assert_allclose(X[np.arange(len(hkl)), cell * k], 1.0)        # T_0 = indicator
    blocks = X.reshape(len(hkl), pc.n_cells, k)
    other = np.ones((len(hkl), pc.n_cells), bool)
    other[np.arange(len(hkl)), cell] = False
    assert not blocks[other].any()
    assert np.abs(X).max() <= 1.0 + 1e-12


def test_one_translation_takes_all_the_cells():
    hkl, s2 = _grid()
    pc = PhaseClasses.fit(hkl, s2, T[:1], "P 2 2 2")
    assert pc.n_cells == 16


def test_definition_round_trip():
    hkl, s2 = _grid()
    pc = PhaseClasses.fit(hkl, s2, T, "P 2 2 2")
    again = PhaseClasses.from_definition(json.loads(json.dumps(pc.definition())))
    np.testing.assert_array_equal(again.cells(hkl), pc.cells(hkl))
    np.testing.assert_allclose(again.columns(hkl, s2), pc.columns(hkl, s2))


def test_features_read_the_reflections_cosines():
    prep = _toy_prep()
    pc = PhaseClasses.from_prep(prep, T, "P 2 2 2")
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, phase_classes=pc)
    f = scale.features
    assert "phase_c0" in f.self_features and "phase_c1" in f.self_features
    assert "phase_c0" in f.context_features
    raw = f._raw_columns(prep.obs_info, ("phase_c0", "phase_c1"))
    want = pc.cosines(prep.hkl_valid.long().numpy())[prep.obs_info.hkl_idx.numpy()]
    np.testing.assert_allclose(raw["phase_c0"].numpy(), want[:, 0], atol=1e-6)
    np.testing.assert_allclose(raw["phase_c1"].numpy(), want[:, 1], atol=1e-6)
    plain = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    assert not any(n.startswith("phase_c") for n in plain.features.self_features)


def test_the_phase_term_holds_a_class_brightness_the_smooth_gauge_cannot():
    """K scaled per class, as a network that reads the phase features could: the smooth
    gauge hardly sees it, the phase term does, and its steps undo it up to a constant."""
    batch = _batch(phase=True)
    both = Tether(["gauge", "phase"])
    phi, _, _ = both._columns(batch)

    def fixed_point(lk):
        lk = lk.clone()
        for _ in range(20):
            b = integral_step(phi, batch.hkl_idx, lk, batch.crystal_id, batch.n_crystals)
            lk = lk - (phi @ b).index_select(0, batch.hkl_idx)
        return lk

    base = _with(batch, fixed_point(batch.log_K))
    cls = batch.phase[:, :4].argmax(1)
    factor = torch.tensor([0.6, -0.4, 0.2, 0.0], dtype=torch.float64)[cls]
    moved = _with(batch, base.log_K + factor[batch.hkl_idx])
    assert float(both.penalty(base)[0]) < 1e-12
    assert Tether(["gauge"]).penalty(moved)[1]["tether_gauge"] < 0.3 * float(factor.std())
    assert float(both.penalty(moved)[0]) > 0.01
    diff = fixed_point(moved.log_K) - base.log_K
    assert float(diff.max() - diff.min()) < 1e-6


def _found(monkeypatch, train):
    found = [{"t": T[0], "amplitude": 1.1, "significance": 50.0, "harmonic_of": None, "order": 1}]
    monkeypatch.setattr(train, "find_used_translations", lambda prep, sg: (found, "one translation"))


def test_end_to_end(tmp_path, monkeypatch):
    """A default run on data with a translation: the checkpoint records the classes and the
    prior's modulation, the network reads the phase features, the tether holds the phase
    term, and torchsx.merge rebuilds them."""
    from torchsx.cli import merge_nn, train

    _cli(monkeypatch)
    _found(monkeypatch, train)
    run = tmp_path / 'run'
    train.main(['x.stream', '-o', str(run), '--device', 'cpu', '--steps', '3', '--seed', '3'])
    ck = torch.load(run / 'best.pt', weights_only=False)
    assert ck['scale']['tether'] == ['gauge', 'phase']
    assert ck['scale']['phase_classes']['translations'] == [T[0]]
    assert ck['intensity_modulation']['translations'] == [T[0]]
    assert 'phase_c0' in ck['scale']['self_features']
    steps = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    steps = [r for r in steps if r.get('phase') == 'train']
    assert all(math.isfinite(r['tether_phase']) for r in steps)
    metrics = json.loads((run / 'merged.metrics.json').read_text())
    merge_nn.main([str(run / 'best.pt'), 'x.stream', '-o', str(tmp_path / 'again.mtz'), '--device', 'cpu'])
    again = json.loads((tmp_path / 'again.metrics.json').read_text())
    for name, rec in metrics.items():
        assert np.isclose(again[name]['tether']['phase']['rms'], rec['tether']['phase']['rms'],
                          rtol=1e-4, atol=1e-7)


def test_off_skips_the_search_and_none_is_refused_with_a_translation(tmp_path, monkeypatch):
    import pytest

    from torchsx.cli import train

    _cli(monkeypatch)
    _found(monkeypatch, train)
    run = tmp_path / 'off'
    train.main(['x.stream', '-o', str(run), '--device', 'cpu', '--steps', '2', '--pseudo-translation', 'off'])
    ck = torch.load(run / 'best.pt', weights_only=False)
    assert ck['scale']['tether'] == ['gauge'] and ck['scale']['phase_classes'] is None
    assert ck['intensity_modulation'] is None
    with pytest.raises(SystemExit, match="pseudo-translation"):
        train.main(['x.stream', '-o', str(tmp_path / 'none'), '--device', 'cpu', '--steps', '2',
                    '--tether', 'none'])
