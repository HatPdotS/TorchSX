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
    # One locating window, because that is what the inline loop does.  The
    # shipped default refines over three (see
    # test_merge_locating_survives_high_multiplicity); this test is about the
    # rest of the machinery matching the bench, not about the placement.
    tr.cfg.locate_windows = (tr.target.scan_halfwidth,)
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
    # The pooled sigma must cover the crystal-to-crystal scatter, not just the
    # counting variance: with K == 1 the empirical term is the plain standard
    # error of the mean, sqrt(sum (I_i - I)^2 / (n (n - 1))), and the reported
    # sigma is at least that (and at least the counting floor).
    dev = (prep.I_obs - ref[prep.obs_info.hkl_idx]) ** 2
    ss = torch.zeros(prep.n_valid).index_add_(0, prep.obs_info.hkl_idx, dev)
    n = prep.obs_counts.double()
    se = (ss.double() / (n * (n - 1).clamp(min=1))).sqrt().float()
    assert (pooled["I_sd"] >= se * (1 - 1e-4)).all()
    assert (pooled["I_sd"] > 0).all()
    # ``mean`` uses the equal-weight standard error of the same ratios, so with
    # K == 1 the two sigmas agree wherever the empirical term dominates.
    both = pooled["I_sd"] > 1.01 * se
    # ``wls``: inverse-variance mean with the model's per-observation variance.
    # Same reflections, finite, positive sigma at least the model variance,
    # and every estimate inside the range of its own observations.
    wls = tr.merge(estimator="wls")
    assert torch.equal(wls["n_meas"], pooled["n_meas"]) and torch.isfinite(wls["I"]).all()
    assert (wls["I_sd"] > 0).all()
    lo = torch.full((prep.n_valid,), float("inf")).scatter_reduce(0, prep.obs_info.hkl_idx, prep.I_obs, "amin")
    hi = torch.full((prep.n_valid,), -float("inf")).scatter_reduce(0, prep.obs_info.hkl_idx, prep.I_obs, "amax")
    assert ((wls["I"] >= lo - 1e-4) & (wls["I"] <= hi + 1e-4)).all()
    assert torch.allclose(pooled["I_sd"][~both], mean["I_sd"][~both] * (n[~both] / (n[~both] - 1)).sqrt().float(), rtol=1e-3, atol=1e-5)


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


# ── custom splits ──────────────────────────────────────────────────────


class _DataArgs:
    """The data half of the CLI namespace, with everything defaulted."""

    def __init__(self, **kw):
        self.stream, self.split, self.split_file, self.split_names = [], [], None, None
        self.min_split_crystals = 0
        self.spacegroup, self.int_radius, self.gain = "P 1", "7,9,11", 1.0
        self.dmin, self.dmax = 2.0, 999.0
        self.cache_prep = self.sfx_cache = self.crystals_file = None
        self.n_workers, self.merge_friedel, self.max_crystals = 1, True, None
        self.__dict__.update(kw)


def test_custom_split_file_labels_every_crystal(tmp_path):
    """A partialator ``--custom-split`` file is a valid ``--split-file``.

    The unit in that file is a frame, so the label reaches a crystal through
    its image -- and the labels must line up with the crystal order
    ``prepare_observations`` ends up with, which is only true after the
    resolution filter has reindexed the image table.
    """
    from torchsx.cli import _pipeline as P
    from torchsx.stream import SFXStream

    from tests.conftest import STREAM_PATH

    cache = tmp_path / "s.sfx"
    s = SFXStream.from_stream(STREAM_PATH, spacegroup="P 1", cache=cache)
    names = ["beta", "alpha"]            # deliberately NOT sorted
    split = tmp_path / "custom.split"
    with open(split, "w") as fh:
        for i, (f, e) in enumerate(zip(s.filenames, s.events)):
            fh.write(f"{f} {e} {names[i % 2]}\n")

    b = P.load_prep(_DataArgs(stream=[cache], split_file=split), verbose=False)
    labels = b.prep.obs_info.delay_bin_idx
    # names come back sorted, so "alpha" is split 0 and takes the odd images
    assert b.split_names == ["alpha", "beta"]
    assert labels.numel() == b.prep.obs_info.n_crystals
    assert int(labels.min()) == 0 and int(labels.max()) == 1
    assert set(labels.tolist()) == {0, 1}

    # a frame the split file does not mention is refused, not silently pooled
    partial = tmp_path / "partial.split"
    with open(partial, "w") as fh:
        half = list(zip(s.filenames, s.events))[: max(1, len(s.filenames) // 2)]
        for f, e in half:
            fh.write(f"{f} {e} alpha\n")
    with pytest.raises(SystemExit, match="unlabelled"):
        P.load_prep(_DataArgs(stream=[cache], split_file=partial), verbose=False)

    # A crystal mask that empties a dataset (a dark-only subset of a light/dark
    # file): the file still defines the datasets, and --min-split-crystals
    # drops the empty ones rather than the count check refusing the file.
    mask = tmp_path / "alpha_only.pt"
    torch.save(labels == 0, mask)
    sub = P.load_prep(_DataArgs(stream=[cache], split_file=split, crystals_file=mask,
                                min_split_crystals=1), verbose=False)
    assert sub.split_names == ["alpha"]
    assert sub.prep.obs_info.n_crystals == int((labels == 0).sum())
    assert set(sub.prep.obs_info.delay_bin_idx.tolist()) == {0}


def test_stream_batches_keeps_the_observations_off_the_device(tmp_path):
    """``--stream-batches`` must leave ``prep`` exactly where it was.

    The trainer decides host-vs-device per batch by reading
    ``self.prep.bg.device``, so a ``prep.to(device)`` in the builder turns the
    flag into a device-to-device copy -- invisible on data that fit the card,
    and an out-of-memory before the first step on 794 M observations.
    """
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser

    prep, _params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    bundle = P.PrepBundle(prep, {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0,
                                 "radii": [7.0, 9.0, 11.0]}, [], 0.1)
    base = ["x.sfx", "-o", str(tmp_path), "--device", "cpu", "--inner-points", "64"]

    streamed, *_ = P.build_stack(
        bundle, _build_parser().parse_args(base + ["--stream-batches"]), verbose=False)
    assert streamed is bundle.prep

    moved, *_ = P.build_stack(bundle, _build_parser().parse_args(base), verbose=False)
    assert moved is not bundle.prep


def test_min_split_crystals_drops_the_split_from_the_data(tmp_path):
    """A split under the threshold leaves the data set, not just the merge.

    A handful of crystals still gets its own in-batch merge target during
    training, and a target with no per-reflection support is what produced the
    non-finite batch that killed a full run.  So the crystals go, the surviving
    labels renumber contiguously, and the names follow.
    """
    from torchsx.cli import _pipeline as P
    from torchsx.stream import SFXStream

    from tests.conftest import STREAM_PATH

    cache = tmp_path / "s.sfx"
    s = SFXStream.from_stream(STREAM_PATH, spacegroup="P 1", cache=cache)
    n_img = len(s.filenames)
    tiny = max(1, n_img // 20)
    split = tmp_path / "custom.split"
    with open(split, "w") as fh:
        for i, (f, e) in enumerate(zip(s.filenames, s.events)):
            fh.write(f"{f} {e} {'tiny' if i < tiny else 'big'}\n")

    kept = P.load_prep(_DataArgs(stream=[cache], split_file=split), verbose=False)
    assert kept.split_names == ["big", "tiny"]
    n_all = kept.prep.obs_info.n_crystals

    cut = int(torch.bincount(kept.prep.obs_info.delay_bin_idx)[1]) + 1
    b = P.load_prep(_DataArgs(stream=[cache], split_file=split,
                              min_split_crystals=cut), verbose=False)
    assert b.split_names == ["big"]
    labels = b.prep.obs_info.delay_bin_idx
    assert int(labels.max()) == 0
    assert b.prep.obs_info.n_crystals < n_all


def test_train_defaults_match_the_scored_recipe():
    """The CLI's training knobs are the ones bench/asym.py was scored with.

    R_free 0.1182 came out of a specific config; a CLI that quietly trains on
    a different one is not the same method.  lam_spread is the exception and
    stays 0: the scored run passed -0.25, which ``_lam_spread`` gates out at
    ``<= 0``, so it never applied a penalty.
    """
    from torchsx.cli.train import _build_parser

    a = _build_parser().parse_args(["x.sfx", "-o", "out"])
    assert (a.lr, a.max_obs, a.epochs and True) == (1e-3, 1_500_000, True)
    assert a.lam0 == 10.0 and a.lam_halflife == 400 and a.warmup == 100
    assert (a.train_k_coarse, a.train_m_fine, a.train_n_sigma) == (31, 129, 6.0)
    assert a.heldout_frac == 0.05


def test_lam_holds_when_the_halflife_is_zero(tmp_path):
    """``lam_halflife_steps <= 0`` means no decay, as the help says.

    It used to mean ``max(hl, 1)`` -- halve at EVERY step -- so lam0=10 was
    below 1e-9 by step 30 and the in-batch target ran with no prior at all.
    """
    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    _scale, _obj, tr = _stack(prep, params, tmp_path)

    tr.cfg.lam0, tr.cfg.lam_min = 10.0, 0.0
    tr.cfg.lam_halflife_steps = 0
    for tr.step_i in (0, 30, 5000):
        assert tr._lam() == 10.0
    tr.cfg.lam_halflife_steps = 400
    tr.step_i = 0
    assert tr._lam() == 10.0
    tr.step_i = 400
    assert abs(tr._lam() - 5.0) < 1e-9


def test_merge_locating_survives_high_multiplicity():
    """The merge grid must still cover the peak at 1e5 counts per reflection.

    The locating scan reads the peak off a parabola through three nodes 2 nats
    apart, but the accumulated log-likelihood ``A v - B e^v`` is exponential
    above the peak and linear below, so the fitted vertex carries a bias that
    does NOT shrink with A -- while the grid built around it is ``n_sigma * sd``
    wide with ``sd ~ 1/sqrt(A)``.  Past ~1e4 counts the grid misses the peak
    entirely (measured: all-NaN F on a 422 k-crystal split).  Shrinking windows
    square the bias away.
    """
    from torchsx.nn.posterior_target import PosteriorMergeTarget

    A, v_true = 1e5, 0.0
    B = A * math.exp(-v_true)

    def locate(windows):
        t = PosteriorMergeTarget(torch.tensor([math.exp(v_true + 0.5)]),
                                 k_coarse=63, m_fine=257, n_sigma=24.0,
                                 max_halfwidth=12.0, prior="flat_log")
        for hw in windows:
            t.scan_halfwidth = hw
            nodes = t.scan_grid().double()
            v, sd = t.peak_from_scan((A * nodes - B * nodes.exp()).float())
            t.ema_I0 = v.exp()
        return abs(float(v) - v_true), float(sd)

    one_err, one_sd = locate((8.0,))
    ref_err, ref_sd = locate((8.0, 1.0, 0.125))

    # one round: mis-centred by far more than the grid it then builds
    assert one_err > 0.3 and one_err > 24.0 * one_sd
    # refined: inside the grid with room to spare
    assert ref_err < 0.01 and ref_err < 0.1 * (24.0 * ref_sd)


def test_every_cli_can_stream_batches(tmp_path):
    """All three commands need --stream-batches, not just the trainer.

    A merge holds the same prepared observations a training run does, so a data
    set that does not fit the card cannot be merged without it either -- the
    flag being train-only killed a 38-split re-merge at argument parsing.
    """
    from torchsx.cli.merge_global import _build_parser as global_parser
    from torchsx.cli.merge_nn import _build_parser as nn_parser
    from torchsx.cli.train import _build_parser as train_parser

    for build, argv in (
        (train_parser, ["x.sfx", "-o", str(tmp_path)]),
        (nn_parser, ["ck.pt", "x.sfx", "-o", str(tmp_path / "m.mtz")]),
        (global_parser, ["x.sfx", "-o", str(tmp_path / "g.mtz")]),
    ):
        assert build().parse_args(argv).stream_batches is False
        assert build().parse_args(argv + ["--stream-batches"]).stream_batches is True


def test_merge_survives_a_non_finite_batch(tmp_path):
    """One non-finite batch must not empty a whole split.

    ``accumulate`` is additive over observations, so a NaN from one batch
    reaches every reflection that batch touched; ``seen`` is
    ``acc.abs().sum(1) > 0`` and ``NaN > 0`` is False, so the merge returns
    nothing at all.  Two splits of the full SwissFEL set were lost that way,
    one of them the largest.
    """
    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=25, per=20)
    scale, obj, tr = _stack(prep, params, tmp_path)
    scale.eval()
    good = tr.merge()
    assert bool(good["seen"].all()) and not good["n_nonfinite_rows"]

    # poison one batch's contribution, exactly as a bad observation would
    real, calls = tr.target.accumulate, {"n": 0}

    def poisoned(*a, **kw):
        out = real(*a, **kw)
        calls["n"] += 1
        return out * float("nan") if calls["n"] == 1 else out

    tr.target.accumulate = poisoned
    res = tr.merge()
    tr.target.accumulate = real

    assert res["n_nonfinite_rows"] > 0          # reported, not hidden
    assert int(res["seen"].sum()) > 0           # and the split survives


# ── smooth Wilson basis / gauge anchor round trip ──────────────────────


def _trainer_from_stack(prep, params, scale, obj, S):
    cfg = TrainConfig(
        epochs=0, merge_target="posterior", loss="evidence", posterior_prior=S["prior"],
        wilson_basis=S["wilson_basis"], wilson_degree=S["wilson_degree"],
        posterior_k_coarse=9, posterior_m_fine=33, device="cpu", heldout_frac=0.1,
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    if S.get("basis") is not None:
        tr.set_reciprocal_basis(S["basis"], smooth_prior=S["wilson_basis"] == "smooth")
    return tr


def test_smooth_basis_flags_round_trip_through_the_checkpoint(tmp_path):
    """--wilson-basis smooth / --gauge-anchor basis are recorded in best.pt, rebuilt by
    merge_nn without flags, and the fitted Sigma coefficients come back exactly."""
    from torchsx.cli import _pipeline as P
    from torchsx.cli.merge_nn import _build_parser as merge_parser
    from torchsx.cli.train import _build_parser as train_parser
    from torchsx.nn.checkpoint_context import load_checkpoint_context

    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    recipe = {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0, "radii": [7.0, 9.0, 11.0]}
    bundle = P.PrepBundle(prep, recipe, [], 0.1)
    targs = train_parser().parse_args(
        ["x.sfx", "-o", str(tmp_path), "--device", "cpu", "--inner-points", "64",
         "--wilson-basis", "smooth", "--gauge-anchor", "basis", "--wilson-degree", "2"])
    prep_, params_, scale, obj, S = P.build_stack(bundle, targs, verbose=False)
    assert S["wilson_basis"] == "smooth" and S["basis"] is not None
    assert scale.anchor_stat == "basis" and S["basis"].degree == 2
    tr = _trainer_from_stack(prep_, params_, scale, obj, S)
    t = tr.targets[0]
    assert t.has_basis and t.wilson_coef.abs().sum() > 0
    tr.data_recipe = dict(recipe)
    ck = torch.load(tr.save_best(tmp_path / "smooth.pt"), weights_only=False)
    assert ck["posterior_grid"]["wilson_basis"] == "smooth"
    assert ck["posterior_grid"]["wilson_degree"] == 2
    assert ck["scale"]["anchor_stat"] == "basis"
    assert "wilson_coef" in ck["inference_context"]["targets"][0]
    assert "gauge_phi" not in ck["state_dict"], "the basis rows are rebuilt, not stored"

    # Merge side: no flags at all, everything comes from the checkpoint.
    margs = merge_parser().parse_args(
        ["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu", "--inner-points", "64"])
    p2, q2, rebuilt, o2, S2 = P.build_stack(bundle, margs, ckpt=ck, verbose=False)
    assert S2["wilson_basis"] == "smooth" and S2["wilson_degree"] == 2
    assert rebuilt.anchor_stat == "basis" and S2["basis"] is not None
    rebuilt.load_state_dict(ck["state_dict"])
    tr2 = _trainer_from_stack(p2, q2, rebuilt, o2, S2)
    load_checkpoint_context(tr2, ck, "exact")
    assert torch.equal(tr2.targets[0].wilson_coef, t.wilson_coef)
    torch.testing.assert_close(tr2.targets[0].wilson_sigma, t.wilson_sigma)
    torch.testing.assert_close(rebuilt.gauge_phi, scale.gauge_phi)
    # And the pinned predictions are reproduced (the context's probe check).
    from torchsx.nn.checkpoint_context import verify_probes
    verify_probes(tr2, ck["inference_context"])


def test_shells_checkpoint_rebuilds_without_a_basis(tmp_path):
    """A checkpoint written before the smooth basis existed rebuilds in shells / median mode,
    whatever today's defaults are."""
    from torchsx.cli import _pipeline as P
    from torchsx.cli.merge_nn import _build_parser as merge_parser
    from torchsx.cli.train import _build_parser as train_parser

    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    recipe = {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0, "radii": [7.0, 9.0, 11.0]}
    bundle = P.PrepBundle(prep, recipe, [], 0.1)
    targs = train_parser().parse_args(["x.sfx", "-o", str(tmp_path), "--device", "cpu", "--inner-points", "64",
                                       "--wilson-basis", "shells", "--gauge-anchor", "median"])
    prep_, params_, scale, obj, S = P.build_stack(bundle, targs, verbose=False)
    assert S["basis"] is None and S["wilson_basis"] == "shells" and scale.anchor_stat == "median"
    tr = _trainer_from_stack(prep_, params_, scale, obj, S)
    assert not tr.targets[0].has_basis
    tr.data_recipe = dict(recipe)
    ck = torch.load(tr.save_best(tmp_path / "shells.pt"), weights_only=False)
    del ck["posterior_grid"]["wilson_basis"], ck["posterior_grid"]["wilson_degree"]   # pre-basis writer
    del ck["scale"]["anchor_stat"]
    margs = merge_parser().parse_args(
        ["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu", "--inner-points", "64"])
    _p, _q, rebuilt, _o, S2 = P.build_stack(bundle, margs, ckpt=ck, verbose=False)
    assert S2["basis"] is None and rebuilt.anchor_stat == "median"
    rebuilt.load_state_dict(ck["state_dict"])


def test_defaults_are_smooth_prior_and_basis_anchor(tmp_path):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser as train_parser
    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    bundle = P.PrepBundle(prep, {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0, "radii": [7.0, 9.0, 11.0]}, [], 0.1)
    targs = train_parser().parse_args(["x.sfx", "-o", str(tmp_path), "--device", "cpu", "--inner-points", "64"])
    _p, _q, scale, _o, S = P.build_stack(bundle, targs, verbose=False)
    assert S["wilson_basis"] == "smooth" and S["basis"] is not None and scale.anchor_stat == "basis"
    assert TrainConfig().wilson_basis == "smooth"


# ── merge_nn products ──────────────────────────────────────────────────


def test_merge_nn_writes_three_products_with_one_surface(tmp_path):
    """Default run: scaled posterior, raw posterior and pooled per split and half, one
    calibration surface fitted on all splits; --no-scale-calibration reproduces the raw
    posterior as the only product."""
    import json
    import gemmi
    import numpy as np
    from torchsx.cli import merge_nn
    from torchsx.cli.merge_nn import _build_parser
    from torchsx.merging.calibrate import ChebyshevScaleSurface

    prep, params, *_ = _synthetic(n_hkl=60, n_xtal=40, per=20, n_cond=2)
    scale, _obj, tr = _stack(prep, params, tmp_path, n_cond=2)
    scale.eval()
    names, wanted = ["dark", "light"], [0, 1]
    out = tmp_path / "a" / "m.mtz"
    args = _build_parser().parse_args(
        ["ck.pt", "x.sfx", "-o", str(out), "--device", "cpu", "--all-splits", "--calib-degree", "2,2"])
    summary = merge_nn.merge_products(tr, prep, args, names, wanted, spacegroup="P 1", verbose=False)
    assert set(summary) == set(names)

    d = prep.d_spacing.numpy()
    for k, lab in enumerate(names):
        side = json.loads((out.parent / f"m_{lab}.calibration.json").read_text())
        assert side["fitted_on"] == names and side["calibration"]["nx"] == 2
        assert len(side["calibration"]["beta"]) == 9 and side["shell_check"]
        surf = ChebyshevScaleSurface.from_dict(side["calibration"])
        for half in ("", "_half1", "_half2"):
            scaled = gemmi.read_mtz_file(str(out.parent / f"m_{lab}{half}.mtz"))
            raw = gemmi.read_mtz_file(str(out.parent / f"m_{lab}{half}_posterior.mtz"))
            pooled = gemmi.read_mtz_file(str(out.parent / f"m_{lab}{half}_pooled.mtz"))
            assert "F-obs" in scaled.column_labels() and "F-obs" in raw.column_labels()
            assert [c.type for c in pooled.columns if c.label in ("I-obs", "SIGI-obs")] == ["J", "Q"]
            assert not any(c.type == "F" for c in pooled.columns)
            a, b = np.array(scaled, copy=False), np.array(raw, copy=False)
            assert a.shape == b.shape and np.array_equal(a[:, :3], b[:, :3])
            ca, cb = {l: i for i, l in enumerate(scaled.column_labels())}, {l: i for i, l in enumerate(raw.column_labels())}
            ratio = (a[:, ca["F-obs"]] / b[:, cb["F-obs"]]) ** 2
            assert (ratio >= surf.f_min - 1e-4).all() and (ratio <= surf.f_max + 1e-4).all()
            # the sigma carries the same factor
            np.testing.assert_allclose(a[:, ca["SIGF-obs"]] / b[:, cb["SIGF-obs"]], np.sqrt(ratio), rtol=1e-4)
        # the factor is the surface evaluated at the raw posterior of THIS split
        res = tr.merge(condition=k, estimator="posterior")
        seen = res["seen"].numpy()
        f = surf.factor(d[seen], res["I0"].numpy()[seen])
        full = gemmi.read_mtz_file(str(out.parent / f"m_{lab}.mtz")); fr = gemmi.read_mtz_file(str(out.parent / f"m_{lab}_posterior.mtz"))
        a, b = np.array(full, copy=False), np.array(fr, copy=False)
        ca = {l: i for i, l in enumerate(full.column_labels())}
        key = {tuple(int(v) for v in row[:3]): i for i, row in enumerate(a)}
        pairs = [(key[t], j) for j, h in enumerate(prep.hkl_valid.numpy()[seen]) for t in [tuple(int(v) for v in h)] if t in key]
        rows, js = zip(*pairs)
        got = (a[list(rows), ca["F-obs"]] / b[list(rows), ca["F-obs"]]) ** 2
        np.testing.assert_allclose(got, f[list(js)], rtol=2e-3)

    # the opt-out reproduces the raw posterior as the only product, bit-exactly
    out2 = tmp_path / "b" / "m.mtz"
    args2 = _build_parser().parse_args(
        ["ck.pt", "x.sfx", "-o", str(out2), "--device", "cpu", "--all-splits", "--no-scale-calibration"])
    merge_nn.merge_products(tr, prep, args2, names, wanted, spacegroup="P 1", verbose=False)
    for lab in names:
        assert not (out2.parent / f"m_{lab}_pooled.mtz").exists() and not (out2.parent / f"m_{lab}.calibration.json").exists()
        x = np.array(gemmi.read_mtz_file(str(out2.parent / f"m_{lab}.mtz")), copy=False)
        y = np.array(gemmi.read_mtz_file(str(out.parent / f"m_{lab}_posterior.mtz")), copy=False)
        np.testing.assert_array_equal(x, y)
