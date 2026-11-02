"""The sum-of-spots tether: its statistic, the contract every term keeps, the trainer and the CLI.

The K-mean statistic must move exactly with a smooth shift of log K and not at
all with a crystal constant, carry the exact derivative of the solution, survive
an output drifted many nats along the gauge, ignore runaway observations, and
vanish -- with its penalty and the penalty's gradient -- at the constraint; the
closed-form training step must vanish on the same constraint.  Every registered
term must keep the contract: unchanged by crystal constants, zero, with a zero
gradient, where the data put it, and the same over a whole merge as in a batch
of the same observations.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from torchsx.merging.tether import (
    TERMS,
    KMeanTerm,
    Tether,
    TetherBatch,
    integral_mode,
    integral_step,
    mode_mean_square,
    parse_terms,
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


def _batch(seed=1, phase=False):
    """With ``phase``, four classes of reflections, each with a linear trend in the first
    basis column, as the phase columns."""
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi, seed=seed)
    cols = None
    if phase:
        cls = torch.randint(0, 4, (phi.shape[0],), generator=torch.Generator().manual_seed(seed + 7))
        onehot = torch.nn.functional.one_hot(cls, 4).to(torch.float64)
        cols = torch.cat([onehot, onehot * phi[:, :1]], 1)
        lk = lk + 0.3 * (cls.to(torch.float64) - 1.5)[h]
    return TetherBatch(lk, h, cid, n, phi, phase=cols)


def _with(batch, log_K):
    return TetherBatch(log_K, batch.hkl_idx, batch.crystal_id, batch.n_crystals, batch.phi, phase=batch.phase)


# ── the exact fit (readout) ────────────────────────────────────────────


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
    """Minimised over a smooth shift of log K, the step's penalty reaches zero where
    the full solve's mode is zero, from a start drifted ``drift`` nats along the gauge."""
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


def test_step_reads_a_trend_along_a_column_given_per_observation():
    """Columns given per observation enter the same fit after the basis: a planted
    trend of log K along one comes back as its coefficient, the basis part unchanged."""
    phi, _ = _basis()
    lk, h, cid, n = _observations(phi)
    lk = lk - (phi @ integral_mode(phi, h, lk, cid, n))[h]
    x = torch.randn(lk.numel(), generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    x = x - x.mean()
    b = integral_step(phi, h, lk + 0.05 * x, cid, n, extra=x[:, None])
    assert abs(float(b[-1]) - 0.05) < 0.01, float(b[-1])          # one step: first order
    assert float(b[:-1].abs().max()) < 0.01


# ── the contract of every term ─────────────────────────────────────────


@pytest.mark.parametrize("name", [n for n, t in TERMS.items() if isinstance(t, KMeanTerm)])
def test_terms_ignore_crystal_constants_and_vanish_at_their_fixed_point(name):
    batch = _batch(phase=True)
    tether = Tether([name])
    const = torch.randn(batch.n_crystals, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    p0, _ = tether.penalty(batch)
    p1, _ = tether.penalty(_with(batch, batch.log_K + const[batch.crystal_id]))
    assert float(p0) > 1e-6
    assert math.isclose(float(p1), float(p0), rel_tol=1e-8), (float(p0), float(p1))
    # repeated steps reach the term's constraint, where the penalty and its gradient vanish
    phi, x, _ = tether._columns(batch)
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


def test_readout_and_penalty_agree_on_the_same_observations():
    """The readout solves exactly what the training step linearises: both report the same
    gauge to first order, and both reach zero together."""
    batch = _batch()
    tether = Tether(["gauge"])
    out, rec = tether.readout(batch), tether.penalty(batch)[1]
    assert out["n_obs"] == batch.hkl_idx.numel()
    assert math.isclose(out["gauge"]["rms"], rec["tether_gauge"], rel_tol=0.2)
    b = integral_mode(batch.phi, batch.hkl_idx, batch.log_K, batch.crystal_id, batch.n_crystals)
    at = _with(batch, batch.log_K - (batch.phi @ b)[batch.hkl_idx])
    assert tether.readout(at)["gauge"]["rms"] < 1e-8 and tether.penalty(at)[1]["tether_gauge"] < 1e-8
    d = 1.0 / (1.0 + batch.phi[:, 0] - batch.phi[:, 0].min())
    shells = tether.readout(batch, d_spacing=d)["gauge"]["by_shell"]
    assert len(shells) == 10 and sum(s["n_obs"] for s in shells) == batch.hkl_idx.numel()


def test_terms_parse_in_registry_order():
    assert parse_terms("gauge") == ("gauge",) == parse_terms(["gauge", "gauge"])
    assert parse_terms("phase,gauge") == ("gauge", "phase")
    assert parse_terms("none") == () and not Tether("none")
    with pytest.raises(ValueError, match="unknown tether term"):
        parse_terms("gauge,wobble")


# ── the trainer ────────────────────────────────────────────────────────


def _trainer(tmp_path, anchor="median", **cfg_kw):
    from tests.test_amortized_training import _synthetic
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
    from torchsx.merging.scale import AmortizedScale
    from torchsx.nn.objective import MarginalNLL
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=48, per=12, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, context_folds=2, anchor_stat=anchor)
    cfg = {"device": "cpu", "outdir": str(tmp_path), "heldout_frac": 0.25, "posterior_k_coarse": 9,
           "posterior_m_fine": 33, "tether": ("gauge",)}
    cfg.update(cfg_kw)
    return AmortizedTrainer(prep, scale, MarginalNLL(SplitLogGrid(n_points=96), Poisson()), params,
                            config=TrainConfig(**cfg))


def _state_rows(tr, k, n=8):
    return tr.index.rows(tr.train_crystals[tr.condition[tr.train_crystals] == k][:n])


def test_trainer_refuses_settings_that_cannot_hold(tmp_path):
    with pytest.raises(ValueError, match="anchor_stat"):
        _trainer(tmp_path, anchor="basis")
    with pytest.raises(ValueError, match="unknown tether term"):
        _trainer(tmp_path, tether=("gauge", "wobble"))
    with pytest.raises(ValueError, match="lam_tether"):
        _trainer(tmp_path, lam_tether=0.0)


def test_trainer_records_the_tether(tmp_path):
    tr = _trainer(tmp_path)
    try:
        for k in (0, 1, 0, 1):
            rec = tr.step(_state_rows(tr, k, n=24))
            assert not rec.get("step_skipped")
            assert math.isfinite(rec["tether_gauge"]) and math.isfinite(rec["tether_penalty"])
    finally:
        tr.close()


def test_penalty_trains_the_network_onto_the_gauge(tmp_path):
    """The penalty reaches the network's parameters, and minimising it alone moves the
    network's output onto the constraint."""
    tr = _trainer(tmp_path)
    try:
        torch.manual_seed(3)
        for p in tr.scale.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.3)       # an output with a smooth trend to remove
        tr.scale.eval()                                  # plain autograd through the encoder
        rows = _state_rows(tr, 0, n=24)
        meta, *_ = tr._batch(rows)
        opt = torch.optim.Adam(tr.scale.parameters(), lr=3e-3)
        first = None
        for _ in range(60):
            opt.zero_grad()
            pen, _ = tr.tether.penalty(tr._tether_batch(meta, tr.scale.predict(meta).log_K))
            pen.backward()
            first = float(pen.detach()) if first is None else first
            opt.step()
        last, _ = tr.tether.penalty(tr._tether_batch(meta, tr.scale.predict(meta).log_K))
        assert first > 1e-6 and float(last.detach()) < 1e-2 * first
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
    for key, value in {"d_model": 16, "inner_points": 32, "k_coarse": 15, "m_fine": 65}.items():
        monkeypatch.setitem(P.DEFAULTS, key, value)
    return P


def test_a_default_run_holds_the_gauge(tmp_path, monkeypatch):
    """A default run: the record carries the gauge, the checkpoint and run.json the terms,
    the scale pins the scalar gauge only, every state's merge reports the gauge by
    resolution shell, and torchsx.merge reproduces the readout from the checkpoint."""
    import numpy as np

    from torchsx.cli import merge_nn, train

    P = _cli(monkeypatch)
    run = tmp_path / 'run'
    train.main(['x.stream', '-o', str(run), '--device', 'cpu', '--steps', '4', '--seed', '3'])
    ck = torch.load(run / 'best.pt', weights_only=False)
    assert ck['scale']['anchor_stat'] == 'median' and ck['scale']['tether'] == ['gauge']
    assert json.loads((run / 'run.json').read_text())['training']['config']['tether'] == ['gauge']
    steps = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    steps = [r for r in steps if r.get('phase') == 'train']
    assert len(steps) == 4 and all(math.isfinite(r['tether_gauge']) for r in steps)
    metrics = json.loads((run / 'merged.metrics.json').read_text())
    assert set(metrics) == {'dark', 'light'}
    for rec in metrics.values():
        t = rec['tether']
        assert t['n_obs'] > 0 and len(t['gauge']['by_shell']) == 10 and math.isfinite(t['gauge']['rms'])
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
        assert np.isclose(again[name]['tether']['gauge']['rms'], metrics[name]['tether']['gauge']['rms'],
                          rtol=1e-4, atol=1e-7)


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
