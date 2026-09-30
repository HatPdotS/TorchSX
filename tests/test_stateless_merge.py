"""Training and merging carry no per-reflection state from one batch to the next."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.objective import MarginalNLL, wide_log_grid
from torchsx.nn.posterior_target import PosteriorMergeTarget
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
from tests.test_amortized_training import _synthetic


def test_basis_definition_evaluates_like_the_basis_on_any_reflections():
    prep, *_ = _synthetic(n_hkl=40, n_xtal=4, per=8)
    b = ReciprocalBasis.from_hkl(prep.hkl_valid, prep.median_cell, "P 1", degree=2)
    again = ReciprocalBasis.from_definition(b.definition())
    torch.testing.assert_close(again.evaluate(prep.hkl_valid), b.phi)
    sub = torch.tensor([3, 17, 0, 29])
    torch.testing.assert_close(again.evaluate(prep.hkl_valid[sub]), b.phi[sub])


def test_batch_centre_is_the_weighted_mean_and_floors_unseen_reflections():
    t = PosteriorMergeTarget(torch.ones(3), prior="wilson")
    t.wilson_sigma.fill_(100.0)
    hkl = torch.tensor([0, 0, 1, 1])
    K = torch.tensor([0.5, 2.0, 1.0, 1.0])
    I0 = torch.tensor([40.0, 7.0])               # per reflection
    pred = SimpleNamespace(log_K=K.log(), log_w=torch.full((4,), -20.0))
    c = t.batch_centre(pred, K * I0[hkl], torch.ones(4), hkl)
    torch.testing.assert_close(c.exp(), torch.tensor([40.0, 7.0, 100.0 * math.exp(-4.0)]))


def test_the_merge_starts_from_the_unity_merge_floored_below_the_surface():
    t = PosteriorMergeTarget(torch.tensor([50.0, 1e-3, 3.0]), prior="wilson")
    t.wilson_sigma.copy_(torch.tensor([10.0, 10.0, 10.0]))
    floor = 10.0 * math.exp(-4.0)
    torch.testing.assert_close(t.start_centre().exp(), torch.tensor([50.0, floor, 3.0]))


def test_a_batch_result_depends_on_the_batch_alone(tmp_path):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=24, per=8, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2)
    tr = AmortizedTrainer(
        prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
        config=TrainConfig(device="cpu", outdir=str(tmp_path), heldout_frac=0.25,
                           posterior_k_coarse=5, posterior_m_fine=17, split_half_every=0),
    )
    try:
        crystals = tr.train_crystals[tr.condition[tr.train_crystals] == 0][:6]
        for _ in range(2):
            assert not tr.step(tr.index.rows(crystals)).get("step_skipped")
        rows = tr.index.rows(crystals)
        nll = float(tr._forward(rows, grad=False)[3])
        g = torch.Generator().manual_seed(5)
        for t in tr.targets:          # the unity merge only starts a full merge
            t.I_start.mul_(torch.exp(torch.randn(t.I_start.shape, generator=g)))
        assert math.isclose(float(tr._forward(rows, grad=False)[3]), nll, rel_tol=1e-5)
        assert not any(name.startswith("ema") for t in tr.targets for name, _ in t.named_buffers())
    finally:
        tr.close()
