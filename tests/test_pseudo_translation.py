"""Translational pseudo-symmetry: the phase cosines and the translation search."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from torchsx.merging.scale.pseudo_translation import (
    MIN_REFLECTIONS, PhaseClasses, find_pseudo_translations, refine_translation, select_translations,
)


def test_quarter_translation_separates_the_k_classes():
    hkl = np.array([[1, k, 2] for k in range(6)])
    c = PhaseClasses([[0.0, 0.25, 0.0]], "P 2 2 2", [np.zeros(0)], (0.0, 1.0)).cosines(hkl)[:, 0]
    np.testing.assert_allclose(c, [1, 0, -1, 0, 1, 0], atol=1e-6)


def _planted(t, amplitude=0.8, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.array([40.0, 50.0, 60.0, 90.0, 90.0, 90.0])
    g = np.arange(-8, 9)
    hkl = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    hkl = hkl[(hkl != 0).any(axis=1)]
    hkl = hkl[np.array([next(x for x in r if x != 0) > 0 for r in hkl])]   # one per Friedel pair
    I = rng.exponential(1.0, len(hkl)) * (1 + amplitude * np.cos(2 * np.pi * hkl @ np.asarray(t))) * 100.0
    d = 1.0 / np.sqrt(((hkl / cell[:3]) ** 2).sum(axis=1))
    n = len(hkl)
    return SimpleNamespace(
        hkl_valid=torch.as_tensor(hkl, dtype=torch.int16), d_spacing=torch.as_tensor(d),
        I_obs=torch.as_tensor(I, dtype=torch.float32), n_valid=n,
        obs_info=SimpleNamespace(hkl_idx=torch.arange(n)), median_cell=torch.as_tensor(cell),
    )


def test_translation_search_finds_a_planted_translation():
    """Intensities modulated by 1 + 0.8 cos(2 pi h.t) put a Patterson peak at t, with that amplitude."""
    t = np.array([0.0, 0.25, 0.0])
    peaks = find_pseudo_translations(_planted(t), "P 1")
    assert peaks, "no translation found"
    u = np.mod(np.array(peaks[0]["t"]) + 0.5, 1.0) - 0.5
    assert np.allclose(np.abs(u), t, atol=0.01), peaks[0]
    assert 0.4 < peaks[0]["amplitude"] < 1.2


def test_translation_search_finds_nothing_in_unmodulated_data():
    peaks = find_pseudo_translations(_planted([0.0, 0.25, 0.0], amplitude=0.0, seed=3), "P 1")
    assert all(p["amplitude"] < 0.1 for p in peaks), peaks


def test_refinement_climbs_to_the_maximum_near_the_planted_translation():
    """From a start off the peak, the refined translation scores at least as well as the planted one
    (on a finite set of reflections the maximum sits slightly off it) and lies next to it."""
    rng = np.random.default_rng(1)
    hkl = rng.integers(-10, 11, size=(4000, 3)).astype(np.float64)
    t = np.array([0.13, 0.31, 0.07])
    v = np.cos(2 * np.pi * hkl @ t)
    f = lambda x: float(np.dot(v, np.cos(2 * np.pi * hkl @ x)))
    start = t + np.array([0.004, -0.003, 0.002])
    got = refine_translation(hkl, v, start)
    assert f(got) >= f(t) > f(start)
    np.testing.assert_allclose(got, t, atol=5e-4)


def _peak(amplitude, harmonic_of=None, order=1):
    return {"t": [0.0, 0.1 * order, 0.0], "amplitude": amplitude, "significance": 100 * amplitude,
            "distance_A": 10.0, "harmonic_of": harmonic_of, "order": order}


def test_a_dominant_translation_is_used_with_its_harmonics():
    peaks = [_peak(1.1), _peak(0.22, harmonic_of=0, order=2), _peak(0.075), _peak(0.05, harmonic_of=0, order=3)]
    used, _ = select_translations(peaks)
    assert used == peaks[:2]


def test_comparable_peaks_are_not_used():
    """Several peaks of similar height: the interatomic vectors of a small structure."""
    used, why = select_translations([_peak(0.55), _peak(0.54), _peak(0.48), _peak(0.28)])
    assert used == [] and "dominant" in why


def test_weak_peaks_are_not_used():
    assert select_translations([_peak(0.038)])[0] == []
    assert select_translations([])[0] == []


def test_too_few_reflections_are_not_searched():
    """A strong planted translation in fewer than MIN_REFLECTIONS reflections gives no peak."""
    prep = _planted([0.5, 0.0, 0.5], amplitude=0.9)
    keep = torch.arange(MIN_REFLECTIONS - 1)
    small = SimpleNamespace(hkl_valid=prep.hkl_valid[keep], d_spacing=prep.d_spacing[keep],
                            I_obs=prep.I_obs[keep], n_valid=len(keep),
                            obs_info=SimpleNamespace(hkl_idx=torch.arange(len(keep))),
                            median_cell=prep.median_cell)
    assert find_pseudo_translations(small, "P 1") == []
    assert find_pseudo_translations(prep, "P 1")
