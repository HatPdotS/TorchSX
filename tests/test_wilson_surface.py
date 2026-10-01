"""The Wilson surface: trained by the evidence, stored as a function of hkl."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.objective import MarginalNLL, wide_log_grid
from torchsx.nn.posterior_target import PosteriorMergeTarget
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
from tests.test_amortized_training import _synthetic


def _target(I_true, width=0.02):
    n = I_true.numel()
    t = PosteriorMergeTarget(I_true.clone(), k_coarse=31, m_fine=129, n_sigma=8.0)
    t.set_basis(torch.ones(n, 1))                       # log Sigma = one coefficient
    t.fit_wilson_surface()
    t._loc = (I_true.log(), torch.full((n,), 0.05))
    v = t._loc[0][:, None] + t._loc[1][:, None] * t.u_coarse[None, :]
    S = -0.5 * ((v - I_true.log()[:, None]) / width) ** 2  # sharp likelihood at the true intensity
    return t, S


def test_evidence_recovers_the_wilson_mean():
    torch.manual_seed(0)
    I_true = torch.distributions.Exponential(1 / 50.0).sample((4000,)).clamp(min=0.5)
    t, S = _target(I_true)
    p = t.enable_evidence_prior()
    mean = float(I_true.mean())
    # with the intensities known, the Wilson maximum likelihood is their mean:
    # the evidence rises in log Sigma below it and falls above it
    for x, sign in ((0.8 * mean, 1.0), (1.25 * mean, -1.0)):
        with torch.no_grad():
            p.fill_(math.log(x))
        p.grad = None
        t.integrate(S)[0].sum().backward()
        assert float(p.grad) * sign > 0
    with torch.no_grad():
        p.fill_(math.log(0.5 * mean))
    opt = torch.optim.Adam([p], lr=0.02)
    for _ in range(300):
        opt.zero_grad()
        (-t.integrate(S)[0].sum()).backward()
        opt.step()
    t.sync_evidence_prior()
    assert abs(float(t.wilson_sigma[0]) / mean - 1) < 0.03
    torch.testing.assert_close(t.wilson_coef, p.detach())


def test_no_grad_passes_read_the_mirrored_prior():
    I_true = torch.linspace(1.0, 100.0, 50)
    t, S = _target(I_true)
    p = t.enable_evidence_prior()
    t.sync_evidence_prior()
    with torch.no_grad():
        ref = t.integrate(S)[0].clone()
        p.add_(1.0)                                     # trained, not yet mirrored
        torch.testing.assert_close(t.integrate(S)[0], ref)
    assert not torch.allclose(t.integrate(S)[0].detach(), ref)   # the training pass sees it
    t.sync_evidence_prior()
    with torch.no_grad():
        assert not torch.allclose(t.integrate(S)[0], ref)


def test_evidence_prior_needs_the_smooth_basis():
    t = PosteriorMergeTarget(torch.ones(5))
    with pytest.raises(ValueError):
        t.enable_evidence_prior()
    assert t.wilson_coef_param is None and "wilson_coef_param" not in t.state_dict()


def _trainer(tmp_path, **cfg_kw):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=24, per=8, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2)
    return AmortizedTrainer(
        prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
        config=TrainConfig(device="cpu", outdir=str(tmp_path), heldout_frac=0.25,
                           posterior_k_coarse=5, posterior_m_fine=17, split_half_every=0, **cfg_kw),
    )


def _state0(tr, n=6):
    return tr.train_crystals[tr.condition[tr.train_crystals] == 0][:n]


def test_training_moves_the_trained_state_surface_only(tmp_path):
    tr = _trainer(tmp_path)
    try:
        p0, p1 = (t.wilson_coef_param for t in tr.targets)
        start = p0.detach().clone(), p1.detach().clone()
        assert not tr.step(tr.index.rows(_state0(tr))).get("step_skipped")
        assert not torch.equal(p0.detach(), start[0])
        torch.testing.assert_close(p1.detach(), start[1])            # the other state saw no data
        torch.testing.assert_close(tr.targets[0].wilson_coef, p0.detach())
        phi = tr.targets[0].wilson_phi
        torch.testing.assert_close(tr.targets[0].wilson_sigma,
                                   (phi @ p0.detach()).exp().clamp(min=tr.targets[0].floor))
    finally:
        tr.close()


def test_restore_best_brings_back_the_surface_of_the_best_evaluation(tmp_path):
    tr = _trainer(tmp_path)
    try:
        tr.step(tr.index.rows(_state0(tr)))
        tr.evaluate(tr.heldout_crystals)
        best = [t.wilson_coef.clone() for t in tr.targets]
        tr.best_heldout = -math.inf                                  # nothing later wins
        for _ in range(2):
            tr.step(tr.index.rows(_state0(tr)))
        assert not torch.equal(tr.targets[0].wilson_coef, best[0])
        assert tr.restore_best()
        for t, c in zip(tr.targets, best):
            torch.testing.assert_close(t.wilson_coef, c)
            torch.testing.assert_close(t.wilson_coef_param.detach(), c)
        ck = torch.load(tr.save_best(tmp_path / "best.pt"), weights_only=False)
        for c, saved in zip(best, ck["wilson_surface"]["coef"]):
            torch.testing.assert_close(saved, c)
    finally:
        tr.close()


def test_no_basis_no_training_under_the_wilson_prior(tmp_path):
    tr = _trainer(tmp_path)
    try:
        for t in tr.targets:
            t.wilson_coef_param = None
        with pytest.raises(RuntimeError):
            tr.step(tr.index.rows(_state0(tr)))
    finally:
        tr.close()


def test_merge_takes_each_state_surface_by_name(tmp_path):
    from torchsx.cli.merge_nn import restore_wilson_surface
    tr = _trainer(tmp_path)
    try:
        b = tr.reciprocal_basis
        coef = [t.wilson_coef.clone() + 0.1 * (k + 1) for k, t in enumerate(tr.targets)]
        ckpt = {"splits": ["dark", "light"],
                "wilson_surface": {"basis": b.definition(), "coef": coef}}
        sub = torch.tensor([5, 1, 33, 20, 8])
        other = [PosteriorMergeTarget(torch.ones(sub.numel()), prior="wilson") for _ in range(3)]
        sub_basis = ReciprocalBasis.from_hkl(tr.prep.hkl_valid[sub], tr.prep.median_cell, None, degree=4)
        for t in other:
            t.set_basis(sub_basis.phi)            # standardised on the other set
        fake = SimpleNamespace(targets=other, cfg=SimpleNamespace(posterior_prior="wilson"))
        restore_wilson_surface(fake, SimpleNamespace(hkl_valid=tr.prep.hkl_valid[sub]), ckpt,
                               names=["light", "dark", "late"])
        for t, c in zip(other, (coef[1], coef[0], coef[0])):          # a new state takes state 0's
            torch.testing.assert_close(t.wilson_sigma, (b.phi[sub] @ c).exp().clamp(min=t.floor))
    finally:
        tr.close()
