"""``torchsx.merging.calibrate.ChebyshevScaleSurface`` -- posterior -> pooled scale calibration."""

from __future__ import annotations

import numpy as np
import pytest

from torchsx.merging.calibrate import ChebyshevScaleSurface, default_shell_edges


def _synthetic(n=6000, seed=0, bias=True):
    """Wilson-like intensities on a 0.9-20 A reflection list with a smooth planted
    posterior/pooled offset g(s^2, log I) and pooled noise."""
    rng = np.random.default_rng(seed)
    d = 1.0 / np.sqrt(rng.uniform(1.0 / 20.0 ** 2, 1.0 / 0.9 ** 2, n))
    s2 = 1.0 / (4 * d * d)
    sigma_w = 60.0 * np.exp(-2.0 * 6.0 * s2)                   # Wilson fall-off, B = 6
    I_true = rng.exponential(sigma_w)
    x1 = 2 * (s2 - s2.min()) / (s2.max() - s2.min()) - 1
    y = np.log(np.clip(I_true, 1e-6, None)); x2 = 2 * (y - np.quantile(y, 0.002)) / (np.quantile(y, 0.998) - np.quantile(y, 0.002)) - 1
    g = 1.4 - 0.5 * x1 + 0.3 * np.clip(x2, -1, 1) ** 2 if bias else np.ones(n)
    I_post = I_true / g                                        # posterior sits off the mean by 1/g
    sd_pooled = 0.05 * np.abs(I_true) + 0.1
    I_pooled = I_true + rng.normal(0, sd_pooled)
    return d, I_post, I_pooled, sd_pooled, g


def _shell_means(d, a, b, edges):
    out = []
    for hi, lo in zip(edges[:-1], edges[1:]):
        s = (d <= hi) & (d > lo); out.append(a[s].mean() / b[s].mean())
    return np.array(out)


def test_identical_readouts_give_unit_factor():
    d, I_post, _, sd, _ = _synthetic(bias=False)
    surf = ChebyshevScaleSurface()
    surf.fit(I_post, I_post, sd, d)
    f = surf.factor(d, I_post)
    assert np.abs(f - 1).max() < 1e-3


def test_recovers_a_planted_smooth_bias():
    d, I_post, I_pooled, sd, g = _synthetic()
    surf = ChebyshevScaleSurface()
    info = surf.fit(I_post, I_pooled, sd, d)
    assert info["n_coef"] == 100 and info["n_fit"] == len(d)
    f = surf.factor(d, I_post)
    edges = default_shell_edges(d)
    corrected = _shell_means(d, f * I_post, I_pooled, edges)
    raw = _shell_means(d, I_post, I_pooled, edges)
    assert np.abs(corrected - 1).max() < 0.02, corrected
    assert np.abs(raw - 1).max() > 0.1            # there was something to correct
    # weakest octile per shell, against the noise-free truth g * I_post; the
    # pooled values there carry little information, so the tolerance is wider
    for hi, lo in zip(edges[:-1], edges[1:]):
        s = (d <= hi) & (d > lo); w = s & (I_post <= np.quantile(I_post[s], 0.125))
        assert abs((f * I_post)[w].mean() / (g * I_post)[w].mean() - 1) < 0.10
    # pointwise the factor tracks g above the noise-dominated decile
    bulk = (I_post > np.quantile(I_post, 0.1))
    assert np.median(np.abs(f[bulk] / g[bulk] - 1)) < 0.03


def test_surface_transfers_to_an_independent_set():
    d, I_post, I_pooled, sd, _ = _synthetic(seed=1)
    surf = ChebyshevScaleSurface(); surf.fit(I_post, I_pooled, sd, d)
    d2, P2, Q2, _, _ = _synthetic(seed=7)
    f2 = surf.factor(d2, P2)
    edges = default_shell_edges(d2)
    assert np.abs(_shell_means(d2, f2 * P2, Q2, edges) - 1).max() < 0.03
    rows = surf.shell_check(d2, P2, Q2, edges)
    assert len(rows) == len(edges) - 1 and all(abs(r["corrected"] - 1) < 0.03 for r in rows)


def test_apply_and_round_trip():
    d, I_post, I_pooled, sd, _ = _synthetic(n=3000, seed=2)
    surf = ChebyshevScaleSurface(nx=5, ny=6); surf.fit(I_post, I_pooled, sd, d)
    F = np.sqrt(np.clip(I_post, 0, None)); F_sd = 0.1 * F + 0.01
    out = surf.apply(d, I_post, I_sd=sd, F=F, F_sd=F_sd)
    np.testing.assert_allclose(out["I"], I_post * out["factor"])
    np.testing.assert_allclose(out["F"], F * np.sqrt(out["factor"]))
    np.testing.assert_allclose(out["F_sd"], F_sd * np.sqrt(out["factor"]))
    again = ChebyshevScaleSurface.from_dict(surf.to_dict())
    assert again.n_coef == 42
    np.testing.assert_array_equal(again.factor(d, I_post), out["factor"])


def test_clamps_and_guards():
    d, I_post, I_pooled, sd, _ = _synthetic(n=2500, seed=3)
    surf = ChebyshevScaleSurface(f_min=0.5, f_max=1.2); surf.fit(I_post, I_pooled, sd, d)
    f = surf.factor(np.array([0.5, 50.0, 1.0]), np.array([1e-9, 1e9, 1.0]))   # outside the fitted ranges
    assert (f >= 0.5).all() and (f <= 1.2).all() and np.isfinite(f).all()
    with pytest.raises(RuntimeError):
        ChebyshevScaleSurface().factor(d, I_post)
    with pytest.raises(ValueError):
        ChebyshevScaleSurface().fit(I_post[:50], I_pooled[:50], sd[:50], d[:50])
    # torch inputs are accepted
    torch = pytest.importorskip("torch")
    f_t = surf.factor(torch.tensor(d[:10]), torch.tensor(I_post[:10]))
    np.testing.assert_allclose(f_t, surf.factor(d[:10], I_post[:10]))
