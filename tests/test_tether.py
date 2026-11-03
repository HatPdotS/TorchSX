"""The sum-of-spots tether: its statistics, the contract every term keeps, the trainer and the CLI.

The K-mean statistic must move exactly with a smooth shift of log K and not at
all with a crystal constant, carry the exact derivative of the solution, survive
an output drifted many nats along the gauge, ignore runaway observations, and
vanish -- with its penalty and the penalty's gradient -- at the constraint; the
closed-form training step must vanish on the same constraint.  The strength
covariate must come from the other half of the crystals.  The response slope
must read a planted bend of the likelihood and nothing of the level.  Every
registered term must keep the contract: unchanged by crystal constants (K-mean)
or by a smooth rescaling of K (response), zero where the data put it, and the
same over a whole merge as in a batch of the same observations.
"""
from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch

from torchsx.merging.tether import (
    TERMS,
    KMeanTerm,
    ResponseTerm,
    Tether,
    TetherBatch,
    integral_mode,
    integral_step,
    mode_mean_square,
    parse_terms,
    response_slope,
    smooth_columns,
    step_mean_square,
)
from torchsx.merging.tether.kmean import _Fit


def _basis(n_hkl=240, seed=0):
    from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis

    g = torch.Generator().manual_seed(seed)
    hkl = torch.randint(-12, 13, (4 * n_hkl, 3), generator=g)
    hkl = torch.unique(hkl[(hkl != 0).any(1)], dim=0)[:n_hkl]
    cell = torch.tensor([40.0, 52.0, 61.0, 90.0, 90.0, 90.0])
    basis = ReciprocalBasis.from_hkl(hkl, cell, "P 1", degree=4)
    return smooth_columns(basis.phi.to(torch.float64)), basis


def _observations(phi, n_cry=300, per=40, seed=1):
    """log K of rocking-curve samples: partiality exp(-x^2 / 2 s^2) at an excitation
    error x uniform over a fixed prediction window, a spot width s growing with the
    reflection's first basis column, and a scale per crystal."""
    g = torch.Generator().manual_seed(seed)
    n_hkl = phi.shape[0]
    cid = torch.arange(n_cry).repeat_interleave(per)
    h = torch.randint(0, n_hkl, (n_cry * per,), generator=g)
    x = torch.rand(n_cry * per, generator=g, dtype=torch.float64) * 2 - 1
    t = (phi[:, 0] - phi[:, 0].min()) / (phi[:, 0].max() - phi[:, 0].min())
    width = (0.4 + 0.5 * t)[h]
    lk = -0.5 * (x / width) ** 2 + 0.3 * torch.randn(n_cry, generator=g, dtype=torch.float64)[cid]
    lk = lk + 0.2 * torch.randn(n_cry * per, generator=g, dtype=torch.float64)
    return lk, h, cid, n_cry


def _counts(n_hkl=2000, per_hkl=20, slope=0.0, seed=2, bg=5.0, scale=400.0):
    """Counts of rocking-curve samples of Wilson-distributed intensities, as a
    :class:`TetherBatch` (the model's log K is the true one, tilted by ``slope``
    times the centred log intensity when planted), with the true K and I."""
    phi, _ = _basis(n_hkl=n_hkl)
    n_hkl = phi.shape[0]
    g = torch.Generator().manual_seed(seed)
    n_obs, per_cry = n_hkl * per_hkl, 40
    n_cry = n_obs // per_cry
    I = -torch.rand(n_hkl, generator=g, dtype=torch.float64).log() * scale
    cid = torch.arange(n_cry).repeat_interleave(per_cry)
    h = torch.randint(0, n_hkl, (n_cry * per_cry,), generator=g)
    x = torch.rand(h.numel(), generator=g, dtype=torch.float64) * 2 - 1
    lk = -0.5 * (x / 0.4) ** 2 + 0.3 * torch.randn(n_cry, generator=g, dtype=torch.float64)[cid]
    n = torch.poisson(lk.exp() * I[h] + bg, generator=g)
    logI = I.log()
    model = lk + slope * (logI - logI.mean())[h]
    q = torch.quantile(phi[:, 0], torch.linspace(0, 1, 11, dtype=torch.float64)[1:-1])
    shell = torch.bucketize(phi[:, 0], q)
    batch = TetherBatch(n - bg, n.clamp(min=1.0), model, h, cid, n_cry, phi, shell)
    return batch, {"n": n, "bg": bg, "lk": lk, "I": I}


def _poisson_curves(batch, n, bg, bend=0.0, pivot=100.0):
    """Curves of a Poisson likelihood with mean pivot (K I / pivot)^(1 + bend) + bg, the
    model's K: bent against the counts for ``bend != 0`` about a typical signal, and like
    any likelihood a function of K I only."""
    lk = batch.log_K.detach().to(torch.float64)

    def curves(requests):
        out = []
        for rows, n_rows, v in requests:
            v = v.to(torch.float64)
            lam = pivot * ((1.0 + bend) * (lk[:, None] + v.index_select(0, rows) - math.log(pivot))).exp() + bg
            ll = n[:, None] * lam.log() - lam
            out.append(torch.zeros(n_rows, v.shape[1], dtype=torch.float64).index_add_(0, rows, ll))
        return out
    return curves


def _with(batch, log_K):
    return TetherBatch(batch.n_signal, batch.var, log_K, batch.hkl_idx, batch.crystal_id,
                       batch.n_crystals, batch.phi, batch.shell)


# ── the exact K-mean fit (readout) ─────────────────────────────────────


def test_constant_columns_are_left_to_the_crystal_constants():
    phi, basis = _basis()
    assert basis.phi.shape[1] == phi.shape[1] + 1
    torch.testing.assert_close(basis.phi[:, 0], torch.ones(basis.phi.shape[0]))


def test_mode_meets_the_constraint():
    """At b every crystal's mean of K exp(-c - phi b) is one, and so is its
    phi-weighted sum: the profiled gradient vanishes."""
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    b = integral_mode(phi, h, lk, cid, n)
    e = lk - (phi @ b)[h]
    c = torch.zeros(n, dtype=e.dtype).index_add_(0, cid, e.exp()) / torch.bincount(cid, minlength=n)
    r = e.exp() / c[cid]
    assert float((phi[h].T @ (1 - r)).abs().max()) < 1e-8 * len(lk)


def test_mode_moves_with_a_smooth_shift_and_ignores_crystal_constants():
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    b0 = integral_mode(phi, h, lk, cid, n)
    g = torch.Generator().manual_seed(4)
    delta = torch.randn(phi.shape[1], generator=g, dtype=torch.float64) * 0.4
    const = torch.randn(n, generator=g, dtype=torch.float64) * 2.0
    b1 = integral_mode(phi, h, lk + (phi @ delta)[h] + const[cid], cid, n)
    torch.testing.assert_close(b1, b0 + delta, rtol=0, atol=1e-8)
    # it is not the geometric gauge: the least-squares mode of log K differs
    b_ls = _Fit(phi, h, cid, n, torch.ones_like(lk)).start(lk)
    assert float((phi @ (b_ls - b0)).abs().max()) > 0.1


def test_mode_derivative_is_the_implicit_derivative_of_the_solution():
    """Identity along the gauge, zero along crystal constants, finite differences elsewhere."""
    phi, _ = _basis(n_hkl=120)
    lk, h, cid, n = _observations(phi, n_cry=120, per=30)
    x = lk.clone().requires_grad_(True)
    b = integral_mode(phi, h, x, cid, n)
    J = torch.stack([torch.autograd.grad(b[i], x, retain_graph=True)[0] for i in range(b.numel())])
    g = torch.Generator().manual_seed(7)
    delta = torch.randn(phi.shape[1], generator=g, dtype=torch.float64)
    torch.testing.assert_close(J @ (phi @ delta)[h], delta, rtol=0, atol=1e-7)
    const = torch.randn(n, generator=g, dtype=torch.float64)
    assert float((J @ const[cid]).abs().max()) < 1e-7
    v = torch.randn(lk.numel(), generator=g, dtype=torch.float64)
    eps = 1e-5
    fd = (integral_mode(phi, h, lk + eps * v, cid, n) - integral_mode(phi, h, lk - eps * v, cid, n)) / (2 * eps)
    torch.testing.assert_close(J @ v, fd, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("drift", [3.0, 27.0])
def test_mode_recovers_a_large_drift_along_the_gauge(drift):
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    b0 = integral_mode(phi, h, lk, cid, n)
    delta = torch.randn(phi.shape[1], generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    delta = delta * (drift / float((phi @ delta).abs().max()))
    b1 = integral_mode(phi, h, lk + (phi @ delta)[h], cid, n)
    torch.testing.assert_close(b1, b0 + delta, rtol=0, atol=1e-8)


def test_runaway_observations_are_left_out():
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    g = torch.Generator().manual_seed(6)
    wild = torch.rand(lk.numel(), generator=g) < 0.01
    planted = torch.where(wild, lk + 40.0, lk)
    b, w = integral_mode(phi, h, planted, cid, n, return_weights=True)
    assert bool((w[wild] == 0).all()) and bool((w[~wild] == 1).all())
    torch.testing.assert_close(b, integral_mode(phi, h[~wild], lk[~wild], cid[~wild], n), rtol=0, atol=1e-8)
    delta = torch.randn(phi.shape[1], generator=g, dtype=torch.float64) * 0.5
    torch.testing.assert_close(integral_mode(phi, h, planted + (phi @ delta)[h], cid, n), b + delta,
                               rtol=0, atol=1e-8)


# ── the closed-form step (training) ────────────────────────────────────


def test_step_vanishes_where_the_mode_does():
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    b = integral_mode(phi, h, lk, cid, n)
    assert float(b.abs().max()) > 1e-2
    at = (lk - (phi @ b)[h]).clone().requires_grad_(True)
    pen = step_mean_square(phi, h, integral_step(phi, h, at, cid, n))
    pen.backward()
    assert float(pen.detach()) < 1e-20 and float(at.grad.abs().max()) < 1e-12
    away = lk.clone().requires_grad_(True)
    step_mean_square(phi, h, integral_step(phi, h, away, cid, n)).backward()
    assert float(away.grad.abs().max()) > 1e-6
    assert float(mode_mean_square(phi, h, b)) > 1e-4


def test_step_is_the_mode_to_first_order():
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    at = lk - (phi @ integral_mode(phi, h, lk, cid, n))[h]
    delta = 0.02 * torch.randn(phi.shape[1], generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    err = [float((integral_step(phi, h, at + (phi @ (f * delta))[h], cid, n) - f * delta).norm())
           for f in (1.0, 0.25)]
    assert err[0] < 0.2 * float(delta.norm())
    assert err[1] < err[0] / 10, err                       # second order: a quarter the shift, 1/16 the error


@pytest.mark.parametrize("drift", [0.0, 3.0, 27.0])
def test_step_penalty_alone_reaches_the_constraint(drift):
    """Minimised over a smooth shift of log K, the step's penalty reaches the full
    solve's constraint, from a start drifted ``drift`` nats along the gauge."""
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    tilt = torch.randn(phi.shape[1], generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    lk = lk + drift * (phi @ tilt)[h] / float((phi @ tilt).abs().max())
    delta = torch.zeros(phi.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([delta], lr=1.0, max_iter=200, tolerance_grad=0.0, tolerance_change=0.0,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = step_mean_square(phi, h, integral_step(phi, h, lk + (phi @ delta)[h], cid, n))
        loss.backward()
        return loss

    for _ in range(4):
        opt.step(closure)
    b = integral_mode(phi, h, (lk + (phi @ delta)[h]).detach(), cid, n)
    assert float(mode_mean_square(phi, h, b).sqrt()) < 1e-6


# ── the shared batch quantities ────────────────────────────────────────


def test_parts_alternate_in_id_order():
    """Crystal ids that are all even (states assigned in turn) still split in two."""
    batch, _ = _counts(n_hkl=200)
    even = TetherBatch(batch.n_signal, batch.var, batch.log_K, batch.hkl_idx, 2 * batch.crystal_id,
                       2 * batch.n_crystals, batch.phi, batch.shell)
    part = even.rows(2) // even.n_hkl
    assert set(part.tolist()) == {0, 1}
    torch.testing.assert_close(even.rows(4) % even.n_hkl, even.hkl_idx)
    torch.testing.assert_close(even.half_of_row(4)[even.rows(4)], even.rows(2))


def test_strength_covariate_is_free_of_the_basis():
    batch, _ = _counts(n_hkl=400)
    st = batch.strength
    n_hkl = batch.n_hkl
    A = torch.cat([torch.ones(2 * n_hkl, 1, dtype=torch.float64), batch.phi.repeat(2, 1)], 1)
    c = batch.groups(2)["count"]
    w = torch.where(st["valid"], c, torch.zeros_like(c))
    torch.testing.assert_close((A * w[:, None]).T @ st["x_row"], torch.zeros(A.shape[1], dtype=torch.float64),
                               rtol=0, atol=1e-6)
    torch.testing.assert_close(st["x"], st["x_row"][batch.rows(2)])
    assert bool((st["x_row"][~st["valid"]] == 0).all())
    assert 0.45 < float(st["bright"].sum()) / float(st["valid"].sum()) <= 0.5


def test_a_covariate_from_the_same_observations_invents_a_strength_slope():
    """With K correct the cross-fitted covariate finds no strength slope; taken from the
    very observations K is averaged over, it does: a sample of fuller spots raises a
    reflection's mean K and its plain mean together."""
    batch, _ = _counts()
    tether = Tether(["gauge", "strength"])
    beta = tether.penalty(batch)[1]["tether_strength"]
    two = TetherBatch(batch.n_signal.repeat(2), batch.var.repeat(2), batch.log_K.repeat(2),
                      batch.hkl_idx.repeat(2), torch.cat([2 * batch.crystal_id, 2 * batch.crystal_id + 1]),
                      2 * batch.n_crystals, batch.phi, batch.shell)
    x_same = two.strength["x"][: batch.hkl_idx.numel()]
    p = batch.phi.shape[1]
    beta_same = float(integral_step(batch.phi, batch.hkl_idx, batch.log_K, batch.crystal_id, batch.n_crystals,
                                    extra=x_same[:, None])[p])
    assert beta_same > 0.01
    assert abs(beta) < 0.3 * beta_same, (beta, beta_same)


def test_a_planted_strength_slope_of_k_is_read_back():
    batch, _ = _counts(slope=0.1)
    beta = Tether(["gauge", "strength"]).penalty(batch)[1]["tether_strength"]
    assert 0.08 < beta < 0.11, beta


# ── the response slope ─────────────────────────────────────────────────


def test_response_slope_reads_a_planted_slope_and_nothing_else():
    g = torch.Generator().manual_seed(11)
    n = 3000
    phi = torch.randn(n, 6, generator=g, dtype=torch.float64)
    A = torch.cat([torch.ones(n, 1, dtype=torch.float64), phi], 1)
    x = torch.randn(n, generator=g, dtype=torch.float64) + 0.3 * phi[:, 0]
    F = torch.rand(n, generator=g, dtype=torch.float64) * 100 + 1
    for level in (0.0, 0.7):
        r = 0.04 * x + level * (1 + phi @ torch.linspace(-1, 1, 6, dtype=torch.float64))
        s = (F * r).requires_grad_(True)
        kappa, C = response_slope(s, F, A, x)
        assert math.isclose(float(kappa[0]), 0.04, rel_tol=1e-9)
        x_hat = x - A @ torch.linalg.lstsq(A * F.sqrt()[:, None], x * F.sqrt()).solution
        assert math.isclose(float(C[0, 0]), float((F * x_hat**2).sum() / F.sum()), rel_tol=1e-6)
        kappa[0].backward()
        torch.testing.assert_close(s.grad, x_hat / (F * x_hat**2).sum(), rtol=1e-6, atol=1e-12)


def test_response_reads_a_bend_of_the_likelihood():
    """The counts' own Poisson likelihood holds no strength trend at the sum of spots; one
    whose mean goes as (K I)^(1 + bend) does, negative and in proportion to the bend (the
    noisy covariate dilutes the slope by a factor common to both)."""
    batch, d = _counts(n_hkl=3000, per_hkl=40)
    tether = Tether(["response"], min_groups=100)
    k = {b: tether.penalty(batch, _poisson_curves(batch, d["n"], d["bg"], bend=b))[1]["tether_response"]
         for b in (0.0, 0.02, 0.04)}
    assert abs(k[0.0]) < 0.002, k
    assert k[0.02] < -0.004, k
    assert 1.6 < (k[0.04] - k[0.0]) / (k[0.02] - k[0.0]) < 2.4, k


# ── the contract of every term ─────────────────────────────────────────


@pytest.mark.parametrize("name", [n for n, t in TERMS.items() if isinstance(t, KMeanTerm)])
def test_kmean_terms_ignore_crystal_constants_and_vanish_at_their_fixed_point(name):
    batch, _ = _counts(n_hkl=600, slope=0.05)
    tether = Tether([name])
    const = torch.randn(batch.n_crystals, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    p0, _ = tether.penalty(batch)
    p1, _ = tether.penalty(_with(batch, batch.log_K + const[batch.crystal_id]))
    assert float(p0) > 1e-6
    assert math.isclose(float(p1), float(p0), rel_tol=1e-8), (float(p0), float(p1))
    # repeated steps reach the term's constraint, where the penalty and its gradient vanish
    phi, x, _ = tether._kmean_columns(batch)
    lk = batch.log_K.clone()
    for _ in range(30):
        b = integral_step(phi, batch.hkl_idx, lk, batch.crystal_id, batch.n_crystals, extra=x)
        m = (phi @ b[:phi.shape[1]]).index_select(0, batch.hkl_idx)
        if x is not None:
            m = m + x @ b[phi.shape[1]:]
        lk = lk - m
    at = lk.clone().requires_grad_(True)
    pen, _ = tether.penalty(_with(batch, at))
    pen.backward()
    assert float(pen.detach()) < 1e-16 and float(at.grad.abs().max()) < 1e-9


@pytest.mark.parametrize("name", [n for n, t in TERMS.items() if isinstance(t, ResponseTerm)])
def test_response_terms_ignore_a_smooth_rescaling_of_k(name):
    """K -> K g(h) moves each group's sum of spots by 1 / g(h) and the nodes with it: the
    curves there, and the term, do not change."""
    batch, d = _counts(n_hkl=2000, per_hkl=30)
    tether = Tether([name], min_groups=100)
    k0 = tether.penalty(batch, _poisson_curves(batch, d["n"], d["bg"], bend=0.05))[1][f"tether_{name}"]
    delta = torch.randn(batch.phi.shape[1], generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    shift = 0.7 * (batch.phi @ delta) / float((batch.phi @ delta).abs().max())
    moved = _with(batch, batch.log_K + shift[batch.hkl_idx])
    k1 = tether.penalty(moved, _poisson_curves(moved, d["n"], d["bg"], bend=0.05))[1][f"tether_{name}"]
    assert abs(k0) > 0.01
    assert math.isclose(k1, k0, rel_tol=1e-4), (k0, k1)


def test_readout_over_a_merge_is_the_penalty_of_the_same_observations():
    """The readout sums the same per-observation quantities: on one set of observations it
    reports what the training term holds."""
    batch, d = _counts(n_hkl=2000, per_hkl=30, slope=0.05)
    tether = Tether(TERMS, min_groups=1)
    curves = _poisson_curves(batch, d["n"], d["bg"], bend=0.05)
    _, rec = tether.penalty(batch, curves)
    out = tether.readout(batch, curves)
    assert math.isclose(out["response"]["kappa"], rec["tether_response"], rel_tol=1e-9)
    assert out["response"]["n_groups"] == rec["tether_groups"]
    # the K-mean readout (exact gauge, then a step for the rest) and the training step reach
    # zero together: where repeated steps put every K-mean term at its constraint
    phi, x, _ = tether._kmean_columns(batch)
    lk = batch.log_K.clone()
    for _ in range(30):
        b = integral_step(phi, batch.hkl_idx, lk, batch.crystal_id, batch.n_crystals, extra=x)
        lk = lk - (phi @ b[:phi.shape[1]]).index_select(0, batch.hkl_idx) - x @ b[phi.shape[1]:]
    at = _with(batch, lk)
    out_at, rec_at = tether.readout(at, curves), tether.penalty(at, curves)[1]
    assert out_at["gauge"]["rms"] < 1e-8 and abs(out_at["strength"]["value"]) < 1e-8
    assert rec_at["tether_gauge"] < 1e-8 and abs(rec_at["tether_strength"]) < 1e-8


def test_terms_parse_in_registry_order():
    assert parse_terms("response,gauge") == ("gauge", "response")
    assert parse_terms("none") == () and not Tether("none")
    with pytest.raises(ValueError, match="unknown tether term"):
        parse_terms("gauge,wobble")


# ── the trainer ────────────────────────────────────────────────────────


def _trainer(tmp_path, anchor="median", **cfg_kw):
    from tests.test_amortized_training import _synthetic
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.scale import AmortizedScale
    from torchsx.nn.objective import MarginalNLL, wide_log_grid
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=48, per=12, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2, anchor_stat=anchor)
    cfg = {"device": "cpu", "outdir": str(tmp_path), "heldout_frac": 0.25, "posterior_k_coarse": 9,
           "posterior_m_fine": 33, "split_half_every": 0, "tether": ("gauge", "strength", "response")}
    cfg.update(cfg_kw)
    return AmortizedTrainer(prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
                            config=TrainConfig(**cfg))


def _state_rows(tr, k, n=8):
    return tr.index.rows(tr.train_crystals[tr.condition[tr.train_crystals] == k][:n])


@pytest.fixture
def small_groups(monkeypatch):
    monkeypatch.setattr(Tether, "MIN_OBS", 2)
    monkeypatch.setattr(Tether, "MIN_GROUPS", 5)


def test_trainer_refuses_settings_that_cannot_hold(tmp_path):
    with pytest.raises(ValueError, match="anchor_stat"):
        _trainer(tmp_path, anchor="basis")
    with pytest.raises(ValueError, match="unknown tether term"):
        _trainer(tmp_path, tether=("gauge", "wobble"))
    with pytest.raises(ValueError, match="lam_tether"):
        _trainer(tmp_path, lam_tether=0.0)


def test_trainer_records_the_tether(tmp_path, small_groups):
    tr = _trainer(tmp_path)
    try:
        for k in (0, 1, 0, 1):
            rec = tr.step(_state_rows(tr, k, n=24))
            assert not rec.get("step_skipped")
            for key in ("tether_gauge", "tether_strength", "tether_response", "tether_response_half",
                        "tether_response_quarter", "tether_penalty"):
                assert math.isfinite(rec[key]), key
            assert rec["tether_groups"] >= 5
            assert math.isclose(rec["tether_response"],
                                2 * rec["tether_response_half"] - rec["tether_response_quarter"],
                                rel_tol=1e-9, abs_tol=1e-12)
    finally:
        tr.close()


def test_trainer_response_ignores_a_smooth_rescaling_of_k(tmp_path, small_groups):
    tr = _trainer(tmp_path)
    try:
        rows = _state_rows(tr, 0, n=36)
        phi = tr._tether_phi
        delta = torch.randn(phi.shape[1], generator=torch.Generator().manual_seed(2), dtype=phi.dtype)
        shift = (0.5 * phi @ delta / float((phi @ delta).abs().max())).to(torch.float32)
        predict = tr.scale.predict

        def kappa(move):
            if move:
                tr.scale.predict = lambda meta: dataclasses.replace(
                    predict(meta), log_K=predict(meta).log_K + shift[meta.hkl_idx])
            try:
                with torch.no_grad():
                    meta, pred, _, _, n_ph, bg, sigma = tr._forward(rows, grad=False)
                    batch = tr._tether_batch(meta, pred.log_K, n_ph, bg, sigma)
                    return Tether(["response"]).penalty(batch, tr._curves(pred, n_ph, bg, meta))[1]
            finally:
                tr.scale.predict = predict
                tr.scale.clear()

        r0, r1 = kappa(False), kappa(True)
        assert r0["tether_groups"] == r1["tether_groups"] >= 5
        # float32 sums of a few observations per group: equal to rounding
        assert abs(r1["tether_response"] - r0["tether_response"]) < 2e-2 * max(abs(r0["tether_response"]), 1e-2)
    finally:
        tr.close()


def test_three_node_slope_matches_autograd(tmp_path):
    """The response's three-node slope of the summed log-likelihood is its derivative in log I."""
    from torchsx.merging.tether.response import slope_curvature

    tr = _trainer(tmp_path)
    try:
        rows = _state_rows(tr, 0, n=36)
        with torch.no_grad():
            meta, pred, _, _, n_ph, bg, sigma = tr._forward(rows, grad=False)
            batch = tr._tether_batch(meta, pred.log_K, n_ph, bg, sigma)
        g = batch.groups(2)
        v0 = torch.maximum(g["I_lin"], g["sd"]).clamp(min=1e-12).log()
        v = (v0[:, None] + Tether.DELTA * torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)).float()
        args = (tr.objective, tr.scale.scatter, tr.params, pred, n_ph, bg, g["rows"], meta)
        s, _ = slope_curvature(tr.target.accumulate(*args, v, n_rows=g["n_rows"]), v)
        node = v[:, 1:2].clone().requires_grad_(True)
        with torch.enable_grad():
            (grad,) = torch.autograd.grad(tr.target.accumulate(*args, node, n_rows=g["n_rows"]).sum(), node)
        used = g["count"] >= 2
        torch.testing.assert_close(s[used], grad[used, 0].double(), rtol=2e-2, atol=1e-2)
        tr.scale.clear()
    finally:
        tr.close()


# ── the command line ───────────────────────────────────────────────────


def _cli(monkeypatch):
    from tests.test_amortized_training import _synthetic
    from torchsx.cli import _pipeline as P

    prep, _, *_ = _synthetic(n_hkl=20, n_xtal=40, per=10, n_cond=2)
    bundle = P.PrepBundle(prep, {'spacegroup': 'P 1', 'd_min': 0.9, 'gain': 1.0, 'radii': [7, 9, 11],
                                 'merge_friedel': True}, ['dark', 'light'], 0.1)
    monkeypatch.setattr(P, 'load_prep', lambda *a, **k: bundle)
    for key, value in {"d_model": 16, "n_layers": 2, "inner_points": 32, "k_coarse": 15, "m_fine": 65}.items():
        monkeypatch.setitem(P.DEFAULTS, key, value)
    return P


def test_tether_end_to_end(tmp_path, monkeypatch, small_groups):
    """A run with every term: the record carries them, the checkpoint and run.json the
    terms, the scale pins the scalar gauge only, every state's merge reports every term,
    and torchsx.merge reproduces the readout from the checkpoint."""
    import numpy as np

    from torchsx.cli import merge_nn, train

    P = _cli(monkeypatch)
    run = tmp_path / 'run'
    train.main(['x.stream', '-o', str(run), '--device', 'cpu', '--steps', '4', '--seed', '3',
                '--tether', 'gauge,strength,response'])
    ck = torch.load(run / 'best.pt', weights_only=False)
    assert ck['scale']['anchor_stat'] == 'median'
    assert ck['scale']['tether'] == ['gauge', 'strength', 'response']
    assert json.loads((run / 'run.json').read_text())['training']['config']['tether'] == \
        ['gauge', 'strength', 'response']
    steps = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    steps = [r for r in steps if r.get('phase') == 'train']
    assert len(steps) == 4 and all(math.isfinite(r['tether_gauge']) for r in steps)
    assert all(math.isfinite(r['tether_strength']) and math.isfinite(r['tether_response']) for r in steps)
    metrics = json.loads((run / 'merged.metrics.json').read_text())
    assert set(metrics) == {'dark', 'light'}
    for rec in metrics.values():
        t = rec['tether']
        assert t['n_obs'] > 0 and len(t['gauge']['by_shell']) == 10
        assert math.isfinite(t['strength']['value']) and math.isfinite(t['response']['kappa'])
    built = {}
    original = P.build_stack

    def capture(*a, **k):
        out = original(*a, **k)
        built['anchor'] = out[2].anchor_stat
        return out

    monkeypatch.setattr(P, 'build_stack', capture)
    merge_nn.main([str(run / 'best.pt'), 'x.stream', '-o', str(tmp_path / 'again.mtz'), '--device', 'cpu'])
    assert built['anchor'] == 'median'
    again = json.loads((tmp_path / 'again.metrics.json').read_text())
    for name in ('dark', 'light'):
        for key in ('gauge', 'strength'):
            a, b = again[name]['tether'][key], metrics[name]['tether'][key]
            k = 'rms' if key == 'gauge' else 'value'
            assert np.isclose(a[k], b[k], rtol=1e-4, atol=1e-7)
        assert np.isclose(again[name]['tether']['response']['kappa'], metrics[name]['tether']['response']['kappa'],
                          rtol=1e-3, atol=1e-6)


def test_a_default_run_holds_the_gauge_only(tmp_path, monkeypatch):
    from torchsx.cli import train

    _cli(monkeypatch)
    run = tmp_path / 'run'
    train.main(['x.stream', '-o', str(run), '--device', 'cpu', '--steps', '2'])
    ck = torch.load(run / 'best.pt', weights_only=False)
    assert ck['scale']['anchor_stat'] == 'median' and ck['scale']['tether'] == ['gauge']
    steps = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    steps = [r for r in steps if r.get('phase') == 'train']
    assert all(math.isfinite(r['tether_gauge']) and 'tether_response' not in r for r in steps)
    # the readout reports every term all the same
    metrics = json.loads((run / 'merged.metrics.json').read_text())
    assert all({'gauge', 'strength', 'response'} <= set(rec['tether']) for rec in metrics.values())


def test_tether_none_restores_the_basis_gauge(tmp_path, monkeypatch):
    from torchsx.cli import train

    _cli(monkeypatch)
    run = tmp_path / 'run'
    train.main(['x.stream', '-o', str(run), '--device', 'cpu', '--steps', '2', '--tether', 'none'])
    ck = torch.load(run / 'best.pt', weights_only=False)
    assert ck['scale']['anchor_stat'] == 'basis' and ck['scale']['tether'] == []
    steps = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert not any('tether_penalty' in r for r in steps)
    # the readout still reports where the basis gauge leaves the sum of spots
    metrics = json.loads((run / 'merged.metrics.json').read_text())
    assert all(math.isfinite(rec['tether']['gauge']['rms']) for rec in metrics.values())


# ── CUDA ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mode_and_step_on_cuda_match_cpu():
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    b_cpu = integral_mode(phi, h, lk, cid, n)
    x = lk.float().cuda().requires_grad_(True)
    b = integral_mode(phi.cuda(), h.cuda(), x, cid.cuda(), n)
    mode_mean_square(phi.cuda(), h.cuda(), b).backward()
    torch.testing.assert_close(b.detach().cpu(), b_cpu, rtol=0, atol=1e-5)
    assert x.grad.dtype == torch.float32 and bool(torch.isfinite(x.grad).all())
    torch.testing.assert_close(integral_step(phi.cuda(), h.cuda(), lk.cuda(), cid.cuda(), n).cpu(),
                               integral_step(phi, h, lk, cid, n), rtol=0, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grouped_rows_on_the_kernel_match_the_node_loop(monkeypatch):
    """accumulate into (reflection, part) rows: the kernel equals the node loop, and the
    two halves add up to the reflection's curve."""
    import torchsx.nn.posterior_target as pt
    from tests.test_posterior_nodes import HAVE_TRITON, Meta, build, problem

    if not HAVE_TRITON:
        pytest.skip("needs Triton")
    n_ph, bg, mu, mix, hkl, v = problem(n=6000, n_hkl=200, K=3)
    meta = Meta(torch.arange(len(mu), device="cuda"), hkl)
    obj, scatter = build(1.16, 0.03)
    scatter.set_prediction(meta.obs_id, mix)

    class Pred:
        log_K, log_e_s = mu, torch.zeros_like(mu)

    rows = hkl + 200 * (torch.arange(len(mu), device="cuda") % 2)
    target = pt.PosteriorMergeTarget(torch.ones(200)).cuda()
    got = target.accumulate(obj, scatter, None, Pred, n_ph, bg, rows, meta, v.repeat(2, 1), n_rows=400)
    whole = target.accumulate(obj, scatter, None, Pred, n_ph, bg, hkl, meta, v)
    monkeypatch.setattr(pt, "_NODE_LOOP", True)
    want = target.accumulate(obj, scatter, None, Pred, n_ph, bg, rows, meta, v.repeat(2, 1), n_rows=400)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=2e-3)
    torch.testing.assert_close(got[:200] + got[200:], whole, rtol=1e-5, atol=4e-3)
