"""Training and merging with no per-reflection running state."""

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


def _basis(prep):
    return ReciprocalBasis.from_hkl(prep.hkl_valid, prep.median_cell, "P 1", degree=2)


def test_basis_definition_evaluates_like_the_basis_on_any_reflections():
    prep, *_ = _synthetic(n_hkl=40, n_xtal=4, per=8)
    b = _basis(prep)
    again = ReciprocalBasis.from_definition(b.definition())
    torch.testing.assert_close(again.evaluate(prep.hkl_valid), b.phi)
    sub = torch.tensor([3, 17, 0, 29])
    torch.testing.assert_close(again.evaluate(prep.hkl_valid[sub]), b.phi[sub])


def test_batch_centre_is_the_weighted_mean_and_floors_unseen_reflections():
    t = PosteriorMergeTarget(torch.ones(3), prior="wilson")
    t.wilson_sigma.fill_(100.0)
    hkl = torch.tensor([0, 0, 1, 1])
    K = torch.tensor([0.5, 2.0, 1.0, 1.0])
    I0 = torch.tensor([40.0, 40.0, 7.0, 7.0])
    pred = SimpleNamespace(log_K=K.log(), log_w=torch.full((4,), -20.0))
    c = t.batch_centre(pred, K * I0[hkl], torch.ones(4), hkl)
    torch.testing.assert_close(c.exp(), torch.tensor([40.0, 7.0, 100.0 * math.exp(-4.0)]))


def _trainer(tmp_path, **cfg_kw):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=24, per=8, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2, peer_scale="rank")
    tr = AmortizedTrainer(
        prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
        config=TrainConfig(
            device="cpu", outdir=str(tmp_path), heldout_frac=0.25,
            merge_target="posterior", loss="evidence", posterior_prior="wilson",
            posterior_k_coarse=5, posterior_m_fine=17, split_half_every=0,
            feature_reference="static", wilson_fit="evidence", **cfg_kw,
        ),
    )
    tr.set_reciprocal_basis(_basis(prep))
    return tr


def test_no_running_state_moves_and_the_result_ignores_the_seed(tmp_path):
    tr = _trainer(tmp_path, reference_update="none")
    try:
        assert all(t.centre_mode == "batch" and not t.track_reference for t in tr.targets)
        seeds = [(t.ema_I0.clone(), t.ema_log_sigma.clone()) for t in tr.targets]
        crystals = tr.train_crystals[tr.condition[tr.train_crystals] == 0][:6]
        for _ in range(2):
            rec = tr.step(tr.index.rows(crystals))
            assert not rec.get("step_skipped") and rec["n_used"] > 0
        for t, (ema, width) in zip(tr.targets, seeds):
            torch.testing.assert_close(t.ema_I0, ema)
            torch.testing.assert_close(t.ema_log_sigma, width)
        rows = tr.index.rows(crystals)
        nll = float(tr._forward(rows, grad=False)[3])
        g = torch.Generator().manual_seed(5)
        for t in tr.targets:          # a different seed must not change a batch's result
            t.ema_I0.mul_(torch.exp(torch.randn(t.ema_I0.shape, generator=g)))
        assert math.isclose(float(tr._forward(rows, grad=False)[3]), nll, rel_tol=1e-5)
    finally:
        tr.close()


def test_running_reference_still_moves_by_default(tmp_path):
    tr = _trainer(tmp_path)
    try:
        before = tr.targets[0].ema_I0.clone()
        crystals = tr.train_crystals[tr.condition[tr.train_crystals] == 0][:6]
        tr.step(tr.index.rows(crystals))
        assert not torch.equal(tr.targets[0].ema_I0, before)
        with pytest.raises(ValueError):
            _trainer(tmp_path, reference_update="frozen")
    finally:
        tr.close()


def test_the_cli_refuses_a_stateless_run_that_would_follow_a_fixed_seed():
    from torchsx.cli.train import check_reference_update
    ok = SimpleNamespace(reference_update="none", feature_reference="static", wilson_fit="evidence")
    check_reference_update(ok)
    for bad in (dict(feature_reference="running"), dict(wilson_fit="reference")):
        with pytest.raises(SystemExit):
            check_reference_update(SimpleNamespace(**{**vars(ok), **bad}))


def test_merge_evaluates_the_checkpoint_surface_on_another_reflection_set(tmp_path):
    from torchsx.cli.merge_nn import _restore_fitted_wilson
    tr = _trainer(tmp_path, reference_update="none")
    try:
        b = tr.reciprocal_basis
        coef = [t.wilson_coef.clone() + 0.1 * (k + 1) for k, t in enumerate(tr.targets)]
        ckpt = {"reference_update": {"wilson_fit": "evidence"},
                "wilson_surface": {"basis": b.definition(), "coef": coef}}
        sub = torch.tensor([5, 1, 33, 20, 8])
        other = [PosteriorMergeTarget(torch.ones(sub.numel()), prior="wilson") for _ in coef]
        sub_basis = ReciprocalBasis.from_hkl(tr.prep.hkl_valid[sub], tr.prep.median_cell, "P 1", degree=2)
        for t in other:
            t.set_basis(sub_basis.phi)            # standardised on the other set
        _restore_fitted_wilson(SimpleNamespace(targets=other),
                               SimpleNamespace(hkl_valid=tr.prep.hkl_valid[sub]), ckpt)
        for t, c in zip(other, coef):
            torch.testing.assert_close(t.wilson_sigma, (b.phi[sub] @ c).exp().clamp(min=t.floor))
    finally:
        tr.close()
