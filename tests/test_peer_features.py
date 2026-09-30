"""Peer features without a running merge: the within-reflection rank and the
static feature reference."""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.objective import MarginalNLL, wide_log_grid
from torchsx.nn.obs_features import (
    RANK_CONTEXT_FEATURES, ObsFeatureSpec, within_reflection_normal_score,
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


def test_rank_is_context_only():
    prep = _toy_prep()
    with pytest.raises(ValueError, match="self features"):
        ObsFeatureSpec(prep, self_features=("s_sq", "rank_score"))


def test_rank_peer_scale_replaces_the_ratio_channels():
    prep = _toy_prep()
    ratio = ObsFeatureSpec(prep)
    rank = ObsFeatureSpec(prep, peer_scale="rank")
    assert rank.context_features == rank.self_features + RANK_CONTEXT_FEATURES
    assert "obs_scale" not in rank.context_features
    assert "obs_scale_over_sigma" not in rank.context_features
    assert rank.n_context == ratio.n_context - 1
    with pytest.raises(ValueError):
        ObsFeatureSpec(prep, peer_scale="quantile")
    # the rank column is fixed by the data: moving the reference leaves it alone
    meta = prep.obs_info
    col = rank.context_features.index("rank_score")
    before = rank.build(meta)[1][:, col].clone()
    rank.set_reference_I0(torch.rand(prep.n_valid) * 50 + 1)
    after = rank.build(meta)[1][:, col]
    raw = within_reflection_normal_score(prep.I_obs, meta.hkl_idx, prep.n_valid)
    torch.testing.assert_close(before * rank._std[rank._stat_names.index("rank_score")]
                               + rank._mean[rank._stat_names.index("rank_score")], raw,
                               rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)


def _trainer(tmp_path, **cfg_kw):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=24, per=8, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2,
                                     peer_scale=cfg_kw.pop("peer_scale", "ratio"))
    tr = AmortizedTrainer(
        prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
        config=TrainConfig(
            device="cpu", outdir=str(tmp_path), heldout_frac=0.25,
            merge_target="posterior", loss="evidence", posterior_prior="wilson",
            posterior_k_coarse=5, posterior_m_fine=17, split_half_every=0, **cfg_kw,
        ),
    )
    return tr


def _state0(tr):
    return tr.train_crystals[tr.condition[tr.train_crystals] == 0][:3]


@pytest.mark.parametrize("peer_scale", ["ratio", "rank"])
def test_static_reference_is_the_unity_merge_and_never_moves(tmp_path, peer_scale):
    from torchsx.nn.merge_target import unity_merge
    tr = _trainer(tmp_path, feature_reference="static", peer_scale=peer_scale)
    try:
        feats = tr.scale.features
        seed = unity_merge(tr.prep.I_obs, tr.sigma, tr.prep.hkl_id, tr.prep.n_valid).clamp(min=1e-3)
        torch.testing.assert_close(feats._I0_ref, seed)
        stats = feats._mean.clone(), feats._std.clone()
        for crystal in _state0(tr):
            rec = tr.step(tr.index.rows(crystal.reshape(1)))
            assert not rec.get("step_skipped")
            tr._refresh_feature_reference()
        assert not torch.equal(tr.targets[0].ema_I0, seed)       # the running merge moved
        torch.testing.assert_close(feats._I0_ref, seed)          # the features did not
        torch.testing.assert_close(feats._mean, stats[0])
        torch.testing.assert_close(feats._std, stats[1])
    finally:
        tr.close()


def test_running_reference_follows_the_running_merge(tmp_path):
    tr = _trainer(tmp_path)
    try:
        for crystal in _state0(tr):
            tr.step(tr.index.rows(crystal.reshape(1)))
        tr._refresh_feature_reference()
        torch.testing.assert_close(tr.scale.features._I0_ref,
                                   tr.targets[0].ema_I0.clamp(min=1e-3))
        with pytest.raises(ValueError):
            _trainer(tmp_path, feature_reference="merged")
    finally:
        tr.close()
