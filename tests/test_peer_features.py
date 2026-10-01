"""Peer features fixed by the data: the ratio to each reflection's plain mean."""

from __future__ import annotations

import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.merge_target import plain_mean
from torchsx.nn.objective import MarginalNLL, wide_log_grid
from torchsx.nn.obs_features import CONTEXT_ONLY_FEATURES, ObsFeatureSpec
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
from tests.test_amortized_scale import _toy_prep
from tests.test_amortized_training import _synthetic


def test_plain_mean_weights_every_observation_equally():
    I = torch.tensor([1.0, 3.0, 8.0, -4.0, -2.0])
    h = torch.tensor([0, 0, 0, 1, 1])
    torch.testing.assert_close(plain_mean(I, h, 3), torch.tensor([4.0, 1e-3, 1e-3]))


def _raw(spec, x, name):
    k = spec._stat_names.index(name)
    return x * spec._std[k] + spec._mean[k]


def test_context_carries_the_ratio_to_the_plain_mean():
    prep = _toy_prep()
    spec = ObsFeatureSpec(prep)
    assert spec.context_features == spec.self_features + CONTEXT_ONLY_FEATURES
    meta = prep.obs_info
    ctx = spec.build(meta)[1]
    I_mean = plain_mean(prep.I_obs, meta.hkl_idx, prep.n_valid)[meta.hkl_idx]
    torch.testing.assert_close(_raw(spec, ctx[:, spec.context_features.index("obs_scale")], "obs_scale"),
                               (prep.I_obs / I_mean).clamp(-20.0, 20.0), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(_raw(spec, ctx[:, spec.context_features.index("log_I_mean")], "log_I_mean"),
                               I_mean.log(), rtol=1e-5, atol=1e-5)


def test_training_leaves_the_features_where_the_data_put_them(tmp_path):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=24, per=8, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2)
    tr = AmortizedTrainer(
        prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
        config=TrainConfig(device="cpu", outdir=str(tmp_path), heldout_frac=0.25,
                           posterior_k_coarse=5, posterior_m_fine=17, split_half_every=0),
    )
    feats = scale.features
    before = {k: getattr(feats, k).clone() for k in ("_I_mean", "_mean", "_std")}
    try:
        crystals = tr.train_crystals[tr.condition[tr.train_crystals] == 0][:6]
        for _ in range(2):
            assert not tr.step(tr.index.rows(crystals)).get("step_skipped")
        for k, v in before.items():
            assert torch.equal(getattr(feats, k), v), k
    finally:
        tr.close()
