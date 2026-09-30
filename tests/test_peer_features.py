"""Peer features fixed by the data: the within-reflection rank and the unity merge."""

from __future__ import annotations

import math

import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.merge_target import unity_merge
from torchsx.nn.objective import MarginalNLL, wide_log_grid
from torchsx.nn.obs_features import (
    CONTEXT_ONLY_FEATURES, ObsFeatureSpec, within_reflection_normal_score,
)
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
from tests.test_amortized_scale import _toy_prep
from tests.test_amortized_training import _synthetic


def _phi_inv(r: float) -> float:
    return math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * r - 1.0, dtype=torch.float64)))


def test_normal_score_of_the_rank_within_each_reflection():
    I = torch.tensor([3.0, -1.0, 2.0, 5.0, 5.0, -0.0, 0.0, -2.5])
    h = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    z = within_reflection_normal_score(I, h, 4)
    # reflection 0: positions 2, 0, 1 of 3
    expect = [_phi_inv(2.5 / 3), _phi_inv(0.5 / 3), _phi_inv(1.5 / 3)]
    # reflection 1: a tie shares the mean position, 0.5 of 2 -> the median
    expect += [0.0, 0.0]
    # reflection 2: -2.5 < -0.0 < 0.0
    expect += [_phi_inv(1.5 / 3), _phi_inv(2.5 / 3), _phi_inv(0.5 / 3)]
    torch.testing.assert_close(z, torch.tensor(expect), rtol=0, atol=1e-6)


def test_normal_score_ignores_a_monotone_rescaling_of_one_reflection():
    torch.manual_seed(0)
    h = torch.randint(0, 50, (5000,))
    I = torch.randn(5000) * 20 + 10
    z = within_reflection_normal_score(I, h, 50)
    J = torch.where(h == 7, 40.0 * I + 3.0, I)
    J = torch.where(h == 11, J.sign() * J.abs() ** 1.5, J)
    torch.testing.assert_close(within_reflection_normal_score(J, h, 50), z)
    # every reflection's scores are the symmetric set of normal quantiles
    tot = torch.zeros(50).index_add_(0, h, z)
    assert float(tot.abs().max()) < 1e-3


def test_normal_score_does_not_depend_on_the_row_order():
    torch.manual_seed(1)
    h = torch.randint(0, 30, (2000,))
    I = torch.randn(2000)
    perm = torch.randperm(2000)
    z = within_reflection_normal_score(I, h, 30)
    torch.testing.assert_close(within_reflection_normal_score(I[perm], h[perm], 30), z[perm])


def _raw(spec, x, name):
    k = spec._stat_names.index(name)
    return x * spec._std[k] + spec._mean[k]


def test_context_carries_the_rank_and_the_unity_merge():
    prep = _toy_prep()
    spec = ObsFeatureSpec(prep)
    assert spec.context_features == spec.self_features + CONTEXT_ONLY_FEATURES
    meta = prep.obs_info
    ctx = spec.build(meta)[1]
    rank = ctx[:, spec.context_features.index("rank_score")]
    unity = ctx[:, spec.context_features.index("log_I_unity")]
    torch.testing.assert_close(
        _raw(spec, rank, "rank_score"),
        within_reflection_normal_score(prep.I_obs, meta.hkl_idx, prep.n_valid),
        rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        _raw(spec, unity, "log_I_unity"),
        unity_merge(prep.I_obs, spec._sigma, meta.hkl_idx, prep.n_valid).log()[meta.hkl_idx],
        rtol=1e-5, atol=1e-5)


def test_training_leaves_the_features_where_the_data_put_them(tmp_path):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=24, per=8, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2)
    tr = AmortizedTrainer(
        prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
        config=TrainConfig(device="cpu", outdir=str(tmp_path), heldout_frac=0.25,
                           posterior_k_coarse=5, posterior_m_fine=17, split_half_every=0),
    )
    feats = scale.features
    before = {k: getattr(feats, k).clone() for k in ("_rank_score", "_log_I_unity", "_mean", "_std")}
    try:
        crystals = tr.train_crystals[tr.condition[tr.train_crystals] == 0][:6]
        for _ in range(2):
            assert not tr.step(tr.index.rows(crystals)).get("step_skipped")
        for k, v in before.items():
            assert torch.equal(getattr(feats, k), v), k
    finally:
        tr.close()
