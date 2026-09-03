"""The CLI seam: the global-fit scale, ``AmortizedTrainer.merge``, free flags.

What these guard is not the argument parsing but the three places where the
shipped commands could silently disagree with the bench scripts that measured
the published numbers:

* the global scale really is global (K == 1, one scalar per quantity) and the
  encoder is never called for it;
* ``merge()`` reproduces the inline accumulate/integrate loop the bench merge
  uses, estimator by estimator;
* a free set generated from a seed is reproducible and one taken from a file is
  copied, not reinvented.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.config import MergeConfig
from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.heads import ScaleHeads
from torchsx.nn.objective import MarginalNLL
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

from tests.test_amortized_training import _synthetic


def _stack(prep, params, tmp_path, *, global_scale=False, skew=True, n_cond=1):
    kw = dict(d_model=32, n_layers=2, include_hkl=False, predict_shape=False,
              beta_init=1.0, w_init=1.65)
    if global_scale:
        kw.update(predict_width=False, predict_shape=False, unity_k=True,
                  anchor_log_k=False, global_skew=skew)
    else:
        kw.update(predict_skew=skew)
    scale = AmortizedScale.from_prep(prep, **kw)
    obj = MarginalNLL(SplitLogGrid(n_points=64), Poisson(bg_var_factor=0.1))
    cfg = TrainConfig(
        max_obs_per_batch=6000, epochs=0, lr=1e-2, lam0=0.0,
        lam_halflife_steps=0, log_every=10 ** 9, heldout_frac=0.0,
        merge_target="posterior", loss="evidence", posterior_prior="wilson",
        posterior_k_coarse=15, posterior_m_fine=65, posterior_n_sigma=8.0,
        device="cpu", outdir=str(tmp_path),
    )
    return scale, obj, AmortizedTrainer(prep, scale, obj, params, config=cfg)


# ── the global scale ───────────────────────────────────────────────────


def test_global_heads_are_constant_and_unity():
    h = ScaleHeads(8, w_init=1.65, beta_init=1.0, predict_width=False,
                   predict_shape=False, global_skew=True, unity_k=True)
    assert not h.needs_embedding
    names = {n for n, _ in h.named_parameters()}
    assert names == {"global_log_w", "global_beta_raw", "global_log_skew"}, names
    p = h(torch.zeros(7, 0))
    assert torch.equal(p.log_K, torch.zeros(7))
    for t in (p.log_w, p.beta, p.log_alpha, p.log_alpha_hi, p.log_e_s):
        assert t.shape == (7,)
        assert torch.allclose(t, t[0].expand(7))
    assert math.isclose(float(p.log_w[0].detach().exp()), 1.65, rel_tol=1e-5)


def test_default_heads_are_unchanged():
    h = ScaleHeads(8)
    assert h.needs_embedding and not h.global_skew and not h.unity_k
    assert h(torch.zeros(3, 8)).log_alpha_hi is None


def test_predict_skew_and_global_skew_are_exclusive():
    with pytest.raises(ValueError):
        ScaleHeads(8, predict_skew=True, global_skew=True)


def test_global_scale_skips_the_encoder(tmp_path):
    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    scale, _obj, _tr = _stack(prep, params, tmp_path, global_scale=True)

    called = []
    scale.encoder.forward = lambda *a, **k: called.append(1)      # type: ignore
    scale.features.build = lambda *a, **k: called.append(1)       # type: ignore
    pred = scale.predict(prep.obs_info)
    assert not called, "the encoder ran for a scale with no per-observation content"
    assert torch.equal(pred.log_K, torch.zeros_like(pred.log_K))
    assert pred.log_w.std() == 0 and pred.beta.std() == 0


def test_global_fit_moves_the_width_and_is_the_only_free_parameter(tmp_path):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=40, per=20)
    scale, _obj, tr = _stack(prep, params, tmp_path, global_scale=True)
    trainable = [n for n, p in scale.named_parameters() if p.requires_grad]
    assert trainable == ["heads.global_log_w", "heads.global_beta_raw",
                         "heads.global_log_skew"], trainable

    w0 = float(scale.heads.global_log_w.detach().exp())
    for _ in range(3):
        for _, rows in tr.index.batches(6000, generator=tr.gen,
                                        crystals=tr.train_crystals):
            tr.step(rows)
    w1 = float(scale.heads.global_log_w.detach().exp())
    assert w1 != w0 and 0.05 < w1 < 20.0, (w0, w1)
    # K stays exactly 1: there is no scale model to drift.
    with torch.no_grad():
        assert torch.equal(scale.predict(prep.obs_info).log_K,
                           torch.zeros(prep.n_obs_total))


# ── merge() ────────────────────────────────────────────────────────────


def _inline_merge(tr, scale, obj, params, prep):
    """The bench merge, written out: locate, accumulate, integrate."""
    n_hkl = prep.n_valid
    tgt = tr.target
    crystals = torch.arange(tr.index.n_crystals)
    with torch.no_grad():
        scan = tgt.scan_grid()
        acc_s = torch.zeros(n_hkl, scan.shape[1])
        for _, r in tr.index.batches(tr.cfg.max_obs_per_batch, crystals=crystals,
                                     shuffle=False):
            m = prep.obs_info.gather(r)
            pr = scale.prime(m)
            acc_s = acc_s + tgt.accumulate(obj, scale.scatter, params, pr,
                                           tr.n_ph[r], prep.bg[r], m.hkl_idx, m, scan)
            scale.clear()
        centre, sd = tgt.peak_from_scan(acc_s)
        tgt._loc = (centre, sd)
        v_nodes = centre[:, None] + sd[:, None] * tgt.u_coarse[None, :]
        acc = torch.zeros(n_hkl, v_nodes.shape[1])
        for _, r in tr.index.batches(tr.cfg.max_obs_per_batch, crystals=crystals,
                                     shuffle=False):
            m = prep.obs_info.gather(r)
            pr = scale.prime(m)
            acc = acc + tgt.accumulate(obj, scale.scatter, params, pr,
                                       tr.n_ph[r], prep.bg[r], m.hkl_idx, m, v_nodes)
            scale.clear()
        _, I0, I0_sd, w_post, v, edge = tgt.integrate(acc, v_nodes, None)
        rootI = v.exp().clamp(min=0).sqrt()
        F = (w_post * rootI).sum(1)
        F_sd = ((w_post * (rootI - F[:, None]) ** 2).sum(1)).clamp(min=0).sqrt()
    return F, F_sd, I0, edge


@pytest.mark.parametrize("skew", [False, True])
def test_merge_matches_the_inline_bench_loop(tmp_path, skew):
    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=25, per=20)
    scale, obj, tr = _stack(prep, params, tmp_path, skew=skew)
    scale.eval()
    res = tr.merge()
    F, F_sd, I0, edge = _inline_merge(tr, scale, obj, params, prep)
    assert torch.allclose(res["F"], F, rtol=1e-5, atol=1e-6)
    assert torch.allclose(res["F_sd"], F_sd, rtol=1e-5, atol=1e-6)
    assert torch.allclose(res["I0"], I0, rtol=1e-5, atol=1e-6)
    assert torch.allclose(res["edge_mass"], edge, rtol=1e-5, atol=1e-6)
    assert bool(res["seen"].all())


def test_merge_estimators(tmp_path):
    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=25, per=20)
    scale, _obj, tr = _stack(prep, params, tmp_path)
    scale.eval()
    post = tr.merge(estimator="posterior")
    flat = tr.merge(estimator="flat")
    ml = tr.merge(estimator="ml")
    # Same nodes and the same accumulated curve; only the measure differs, so
    # the Wilson posterior must sit between the prior-free maximum and infinity
    # rather than reproduce either.
    assert not torch.allclose(post["F"], flat["F"])
    assert not torch.allclose(flat["F"], ml["F"])
    for r in (post, flat, ml):
        assert torch.isfinite(r["F"]).all() and (r["F"] >= 0).all()

    pooled = tr.merge(estimator="pooled")
    mean = tr.merge(estimator="mean")
    for r in (pooled, mean):
        assert set(("I", "I_sd", "n_meas")) <= set(r)
        assert int(r["n_meas"].sum()) == prep.n_obs_total
    # K == 1 here (zero-init heads), so pooled is sum(n-bg)/n_obs per reflection.
    ref = torch.zeros(prep.n_valid).index_add_(0, prep.obs_info.hkl_idx,
                                               prep.I_obs) / prep.obs_counts
    assert torch.allclose(pooled["I"], ref, rtol=1e-4, atol=1e-4)


def test_merge_refuses_to_pool_splits_and_honours_one(tmp_path):
    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=24, per=20, n_cond=2)
    scale, _obj, tr = _stack(prep, params, tmp_path)
    scale.eval()
    assert tr.n_conditions == 2
    with pytest.raises(ValueError):
        tr.merge(crystals=torch.arange(tr.index.n_crystals))
    a, b = tr.merge(condition=0), tr.merge(condition=1)
    assert a["n_crystals"] + b["n_crystals"] == tr.index.n_crystals
    assert a["condition"] == 0 and b["condition"] == 1
    with pytest.raises(ValueError):
        tr.merge(condition=2)


def test_checkpoint_carries_the_recipe(tmp_path):
    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    scale, _obj, tr = _stack(prep, params, tmp_path, skew=True)
    tr.data_recipe = {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0,
                      "radii": [7.0, 9.0, 11.0]}
    tr.split_names = ["dark"]
    ck = torch.load(tr.save_best(tmp_path / "best.pt"), weights_only=False)
    assert ck["data"]["radii"] == [7.0, 9.0, 11.0]
    assert ck["likelihood"]["bg_var_factor"] == pytest.approx(0.1)
    assert ck["likelihood"]["bg_var_mode"] == "deviance"
    assert ck["scatter"]["predict_skew"] and not ck["scatter"]["global_skew"]
    assert ck["quadrature"] == {"kind": "SplitLogGrid", "n_points": 64}
    assert ck["scale"]["predict_shape"] is False and ck["scale"]["unity_k"] is False
    assert ck["splits"] == ["dark"]
    # no held-out set here: the current state is written rather than nothing
    assert set(ck["state_dict"]) == set(scale.state_dict())


# ── free flags ─────────────────────────────────────────────────────────


class _FreeArgs:
    def __init__(self, **kw):
        self.no_free = False
        self.free_from = None
        self.free_frac = 0.05
        self.free_seed = 20260902
        self.__dict__.update(kw)


def test_free_flags_are_reproducible_and_copied(tmp_path):
    import pickle

    import numpy as np

    from torchsx.cli import _pipeline as P

    grid = np.stack(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5),
                                np.arange(0, 5)), -1).reshape(-1, 3)
    # Friedel mates share a canonical key, so a flag file keyed on it would
    # cover both; keep one representative each so "absent -> work" is testable.
    _, first = np.unique(P.canonical_hkl(grid), axis=0, return_index=True)
    hkl = grid[np.sort(first)]
    a = P.work_mask(hkl, _FreeArgs(), verbose=False)
    b = P.work_mask(hkl, _FreeArgs(), verbose=False)
    c = P.work_mask(hkl, _FreeArgs(free_seed=1), verbose=False)
    assert (a == b).all() and not (a == c).all()
    assert abs((~a).mean() - 0.05) < 0.01
    assert P.work_mask(hkl, _FreeArgs(no_free=True), verbose=False) is None

    # a set taken from a file is copied, and absent reflections go to WORK
    src = {tuple(int(v) for v in k): 0 for k in P.canonical_hkl(hkl[:50])}
    path = tmp_path / "free.pkl"
    pickle.dump(src, open(path, "wb"))
    w = P.work_mask(hkl, _FreeArgs(free_from=path), verbose=False)
    assert (~w[:50]).all() and w[50:].all()


def test_radii_and_gain_parsing():
    from torchsx.cli import _pipeline as P

    assert P.parse_radii("1,4,7") == (1.0, 4.0, 7.0)
    with pytest.raises(ValueError):
        P.parse_radii("7,9")
    with pytest.raises(ValueError):
        P.parse_radii("9,7,11")          # r_peak < r_gap <= r_bg
    assert P.parse_gain(None) == "stream"
    assert P.parse_gain("auto") == "auto"
    assert P.parse_gain("10.01") == pytest.approx(10.01)


def test_build_stack_rebuilds_a_checkpoint_geometry(tmp_path):
    """A checkpoint's head geometry must survive the round trip.

    Rebuilding a globally-fitted checkpoint with per-observation heads would
    fail the state_dict load; rebuilding the other way would load a subset of
    the weights and merge with the rest at their init.
    """
    from torchsx.cli import _pipeline as P
    from torchsx.cli.merge_nn import _build_parser

    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    bundle = P.PrepBundle(prep, {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0,
                                 "radii": [7.0, 9.0, 11.0]}, [], 0.1)
    args = _build_parser().parse_args(
        ["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu",
         "--inner-points", "64"])

    for global_scale in (False, True):
        scale, _obj, tr = _stack(prep, params, tmp_path, global_scale=global_scale)
        tr.data_recipe = dict(bundle.recipe)
        ck = torch.load(tr.save_best(tmp_path / "g.pt"), weights_only=False)
        assert ck["scale"]["unity_k"] is global_scale
        assert ck["scatter"]["global_skew"] is global_scale
        _p, _q, rebuilt, _o, _s = P.build_stack(bundle, args, ckpt=ck, verbose=False)
        # strict: every trained weight must land, and nothing may be left at init
        rebuilt.load_state_dict(ck["state_dict"])
        assert rebuilt.heads.needs_embedding is not global_scale
