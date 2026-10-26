"""Reflection profiles from translational pseudo-symmetry: the phase table, the
two-component profile, the total-area readout of the merge, the checkpoint
round trip and the translation search."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_amortized_scale import _toy_prep
from torchsx.merging.scale import AmortizedScale
from torchsx.merging.scale.pseudo_translation import (
    PhaseBroad, PhaseWidth, find_pseudo_translations, refine_translation, select_translations,
)


def _scale(prep):
    torch.manual_seed(1)
    scale = AmortizedScale.from_prep(prep, d_model=32, n_layers=2, anchor_log_k=False)
    # non-trivial heads, so the prediction depends on the offset feature
    for p in scale.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    return scale.eval()


def _grouped(prep):
    """Crystal-grouped rows, as the trainer feeds them (chunking needs that order)."""
    return prep.obs_info.gather(torch.argsort(prep.obs_info.crystal_id, stable=True))


def _install(scale, prep, translations, spacegroup):
    width = PhaseWidth.from_prep(prep, translations, spacegroup)
    broad = PhaseBroad.from_prep(prep, translations, spacegroup)
    scale.install_pseudo_translation(width, broad)
    return width, broad


def _set_log_fraction(broad, value):
    with torch.no_grad():
        broad.rho_coef.zero_()
        broad.rho_coef[0] = PhaseBroad.RHO_HALF * math.atanh((value - PhaseBroad.RHO_CENTRE) / PhaseBroad.RHO_HALF)


def test_quarter_translation_separates_the_k_classes():
    hkl = np.array([[1, k, 2] for k in range(6)])
    pw = PhaseWidth(hkl, np.full(len(hkl), 0.1), [[0.0, 0.25, 0.0]], "P 2 2 2")
    c = pw.table[:, 0].numpy()
    np.testing.assert_allclose(c, [1, 0, -1, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(pw.table[:, 2].numpy(), c ** 2, atol=1e-6)


def test_profile_on_the_sphere():
    """At zero offset both components read the same features, so
    K = K_shared (1 + rho / W) with zero width coefficients."""
    prep = _toy_prep()
    scale = _scale(prep)
    meta = dataclasses.replace(prep.obs_info, ewald_offset=torch.zeros_like(prep.obs_info.ewald_offset))
    with torch.no_grad():
        before = scale.predict(meta).log_K.clone()
        _, broad = _install(scale, prep, [[0.0, 0.25, 0.0]], "P 2 2 2")
        _set_log_fraction(broad, -1.0)
        after = scale.predict(meta).log_K
    rho, W = math.exp(-1.0), PhaseBroad.W_INIT
    torch.testing.assert_close(after - before, torch.full_like(before, math.log(1 + rho / W)),
                               rtol=1e-5, atol=1e-5)


def test_broad_part_is_linear_in_its_fraction():
    """K = K_core + rho K_broad: affine in rho."""
    prep = _toy_prep()
    scale = _scale(prep)
    meta = prep.obs_info
    _, broad = _install(scale, prep, [[0.0, 0.25, 0.0]], "P 2 2 2")
    f = []
    with torch.no_grad():
        for lr in (-3.0, -2.0, -1.0):
            _set_log_fraction(broad, lr)
            f.append(scale.predict(meta).log_K.exp())
    r = [math.exp(x) for x in (-3.0, -2.0, -1.0)]
    torch.testing.assert_close(f[2] - f[0], (r[2] - r[0]) / (r[1] - r[0]) * (f[1] - f[0]), rtol=1e-4, atol=1e-6)


def test_core_width_stretches_the_profile_at_constant_area():
    """With a negligible broad part, a uniform log w = L is the shared network at offset / e^L, minus L."""
    prep = _toy_prep()
    scale = _scale(prep)
    meta = prep.obs_info
    L = 0.7
    with torch.no_grad():
        ref = scale.predict(dataclasses.replace(meta, ewald_offset=meta.ewald_offset / math.exp(L))).log_K
        width, broad = _install(scale, prep, [[0.0, 0.0, 0.0]], "P 1")  # c = 1 for every reflection
        width.coef[0] = PhaseWidth.BOUND * math.atanh(L / PhaseWidth.BOUND)
        _set_log_fraction(broad, PhaseBroad.RHO_CENTRE - PhaseBroad.RHO_HALF + 1e-3)
        got = scale.predict(meta).log_K
    torch.testing.assert_close(got, ref - L, rtol=0, atol=2e-3)


def test_gradient_reaches_the_profile_coefficients():
    prep = _toy_prep()
    scale = _scale(prep).train()
    width, broad = _install(scale, prep, [[0.0, 0.25, 0.0]], "P 2 2 2")
    scale.predict(prep.obs_info).log_K.square().sum().backward()
    for prm in (width.coef, broad.rho_coef, broad.width_coef):
        assert prm.grad is not None and bool(torch.isfinite(prm.grad).all()) and float(prm.grad.abs().sum()) > 0


def test_chunked_prediction_matches():
    prep = _toy_prep()
    scale = _scale(prep)
    meta = _grouped(prep)
    _, broad = _install(scale, prep, [[0.0, 0.25, 0.0]], "P 2 2 2")
    _set_log_fraction(broad, -1.0)
    with torch.no_grad():
        ref = scale.predict(meta).log_K.clone()
        scale.set_encoder_chunking(5)
        got = scale.predict(meta).log_K
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_merge_reports_the_total_area(tmp_path, monkeypatch):
    """The fit measures the core's area; the merge reports 1 + rho times it:
    I0 x (1 + rho), F and its sd x sqrt(1 + rho), the moment estimators likewise."""
    from tests.test_amortized_training import _synthetic
    from tests.test_cli_pipeline import _stack

    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=25, per=20)
    scale, _obj, tr = _stack(prep, params, tmp_path)
    _, broad = _install(scale, prep, [[0.0, 0.25, 0.0]], "P 2 2 2")
    with torch.no_grad():
        broad.rho_coef.copy_(torch.linspace(-0.5, 0.8, broad.rho_coef.numel()))
    scale.eval()
    c = 1 + broad.log_fraction(torch.arange(prep.n_valid)).exp().detach()
    estimators = ("posterior", "flat", "pooled")
    total = {e: tr.merge(estimator=e) for e in estimators}
    monkeypatch.setattr(scale, "readout_log_factor", lambda hkl_idx: None)
    core = {e: tr.merge(estimator=e) for e in estimators}
    for e in ("posterior", "flat"):
        torch.testing.assert_close(total[e]["I0"], core[e]["I0"] * c, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(total[e]["F"], core[e]["F"] * c.sqrt(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(total[e]["F_sd"], core[e]["F_sd"] * c.sqrt(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(total["pooled"]["I"], core["pooled"]["I"] * c, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(total["pooled"]["I_sd"], core["pooled"]["I_sd"] * c, rtol=1e-5, atol=1e-6)


def test_widths_and_fraction_are_bounded():
    prep = _toy_prep()
    idx = torch.arange(prep.n_valid)
    width = PhaseWidth.from_prep(prep, [[0.0, 0.0, 0.0]], "P 1")
    broad = PhaseBroad.from_prep(prep, [[0.0, 0.0, 0.0]], "P 1")
    assert float(broad.log_width(idx).exp().min()) == pytest.approx(PhaseBroad.W_INIT, rel=1e-5)
    with torch.no_grad():
        width.coef.fill_(1e3)
        broad.width_coef.fill_(-1e3)
        broad.rho_coef.fill_(1e3)
    assert float(width.log_width(idx).abs().max()) <= PhaseWidth.BOUND + 1e-6
    assert float(broad.log_width(idx).min()) >= math.log(PhaseBroad.W_MIN) - 1e-6
    assert float(broad.log_fraction(idx).max()) <= PhaseBroad.RHO_CENTRE + PhaseBroad.RHO_HALF + 1e-6


def test_definition_round_trip():
    prep = _toy_prep()
    t = [[0.5, 0.377, 0.5], [0.0, 0.246, 0.0]]
    width = PhaseWidth.from_prep(prep, t, "P 2 2 2")
    broad = PhaseBroad.from_prep(prep, t, "P 2 2 2")
    with torch.no_grad():
        width.coef.copy_(torch.linspace(-0.4, 0.6, width.coef.numel()))
        broad.rho_coef.copy_(torch.linspace(-0.3, 0.5, broad.rho_coef.numel()))
        broad.width_coef.copy_(torch.linspace(0.4, -0.2, broad.width_coef.numel()))
    w2 = PhaseWidth.from_definition(width.definition(), prep); w2.load_state_dict(width.state_dict())
    b2 = PhaseBroad.from_definition(broad.definition(), prep); b2.load_state_dict(broad.state_dict())
    idx = torch.arange(prep.n_valid)
    torch.testing.assert_close(w2.log_width(idx), width.log_width(idx))
    torch.testing.assert_close(b2.log_fraction(idx), broad.log_fraction(idx))
    torch.testing.assert_close(b2.log_width(idx), broad.log_width(idx))


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
