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


def _stack(prep, params, tmp_path, *, skew=True, n_cond=1):
    kw = dict(d_model=32, n_layers=2, include_hkl=False, predict_shape=False,
              beta_init=1.0, w_init=1.65, predict_skew=skew)
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


def test_default_heads_are_unchanged():
    h = ScaleHeads(8)
    assert h(torch.zeros(3, 8)).log_alpha_hi is None


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
    assert ck["scatter"]["predict_skew"]
    assert ck["quadrature"] == {"kind": "SplitLogGrid", "n_points": 64}
    assert ck["scale"]["predict_shape"] is False
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


def test_free_flags_never_split_a_bijvoet_pair():
    """Mates share a flag, and that grouping leaves the merged draw alone."""
    import numpy as np

    from torchsx.cli import _pipeline as P

    grid = np.stack(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5),
                                np.arange(0, 5)), -1).reshape(-1, 3)
    _, first = np.unique(P.canonical_hkl(grid), axis=0, return_index=True)
    merged = grid[np.sort(first)]
    n = len(merged)

    # A signed column carries both mates; they must land on the same side.
    w = P.work_mask(np.concatenate([merged, -merged]), _FreeArgs(), verbose=False)
    assert (w[:n] == w[n:]).all()
    assert abs((~w).mean() - 0.05) < 0.02

    # One row per reflection: the draw is what it was before mates could split.
    g = torch.Generator().manual_seed(_FreeArgs().free_seed)
    free = torch.zeros(n, dtype=torch.bool)
    free[torch.randperm(n, generator=g)[: int(round(0.05 * n))]] = True
    assert (P.work_mask(merged, _FreeArgs(), verbose=False) == (~free).numpy()).all()


def test_merge_friedel_flag_and_recipe_resolution():
    """The flag overrides the checkpoint recipe; absent, the recipe decides."""
    import argparse

    from torchsx.cli import _pipeline as P

    p = argparse.ArgumentParser()
    P.add_data_args(p, require_recipe=False)
    assert p.parse_args(["x.sfx"]).merge_friedel is None
    assert p.parse_args(["x.sfx", "--merge-friedel"]).merge_friedel is True
    assert p.parse_args(["x.sfx", "--no-merge-friedel"]).merge_friedel is False

    resolve = lambda flag, recipe: (bool(flag) if flag is not None
                                    else bool(recipe.get("merge_friedel", True)))
    assert resolve(None, {}) is True
    assert resolve(None, {"merge_friedel": False}) is False
    assert resolve(True, {"merge_friedel": False}) is True


def test_anomalous_products_unstack_onto_asu_rows(tmp_path):
    """Signed indices become (+)/(-) columns; mates share one row and one flag."""
    import gemmi
    import numpy as np

    from torchsx.cli import _pipeline as P
    from torchsx.stream import _is_centric
    from torchref.symmetry import SpaceGroup

    sg = "P 63"
    grid = torch.tensor([[h, k, l] for h in range(-3, 4) for k in range(-3, 4)
                         for l in range(-3, 4)], dtype=torch.int32)
    grid = grid[grid.abs().sum(1) > 0]
    group = SpaceGroup(sg)
    canon, _, flags, order = group.canonicalize_hkl(grid, include_friedel=True)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel())
    canon, minus = canon[inv], flags[inv].clone()
    idx = minus.nonzero(as_tuple=True)[0]
    minus[idx] = ~_is_centric(group, canon[idx])
    signed = torch.unique(torch.where(minus[:, None], -canon, canon), dim=0).numpy()

    n = len(signed)
    val = torch.arange(1.0, n + 1)
    cell = [50.0, 50.0, 80.0, 90.0, 90.0, 120.0]
    keep = np.ones(n, dtype=bool)
    n_asu = len(np.unique(np.abs(signed), axis=0))

    out = tmp_path / "anom.mtz"
    P.write_mtz(out, signed, val, val * 0.1, cell=cell, spacegroup=sg,
                work=P.work_mask(signed, _FreeArgs(), verbose=False), keep=keep,
                merge_friedel=False, verbose=False)
    mtz = gemmi.read_mtz_file(str(out))
    labels = [c.label for c in mtz.columns]
    assert "F-obs(+)" in labels and "F-obs(-)" in labels
    assert mtz.nreflections == n_asu
    flag = np.array(mtz.column_with_label("R-free-flags"), dtype=int)
    assert set(np.unique(flag)) <= {0, 1}          # one flag per ASU row, not per mate

    out_i = tmp_path / "anom_i.mtz"
    counts = torch.arange(n) % 5 + 1
    P.write_intensity_mtz(out_i, signed, val, val * 0.1, cell=cell, spacegroup=sg,
                          keep=keep, n_meas=counts, merge_friedel=False, verbose=False)
    mtz_i = gemmi.read_mtz_file(str(out_i))
    labels_i = [c.label for c in mtz_i.columns]
    assert {"I-obs(+)", "SIGI-obs(+)", "I-obs(-)", "SIGI-obs(-)",
            "N(+)", "N(-)"} <= set(labels_i)
    assert {c.type for c in mtz_i.columns if c.label.startswith("N(")} == {"I"}
    assert mtz_i.nreflections == n_asu
    asu = gemmi.ReciprocalAsu(gemmi.SpaceGroup(sg))
    written = np.array(mtz_i, dtype=float)[:, :3].astype(int)
    assert all(asu.is_in(row.tolist()) for row in written)
    # Every measurement is filed under exactly one branch, none invented.
    n_plus = np.array(mtz_i.column_with_label("N(+)"))
    n_minus = np.array(mtz_i.column_with_label("N(-)"))
    assert (n_plus + n_minus).sum() == counts.sum()

    out_m = tmp_path / "merged_i.mtz"
    P.write_intensity_mtz(out_m, np.abs(signed), val, val * 0.1, cell=cell,
                          spacegroup=sg, keep=keep, n_meas=counts, verbose=False)
    mtz_m = gemmi.read_mtz_file(str(out_m))
    assert next(c.type for c in mtz_m.columns if c.label == "N") == "I"


def test_radii_and_gain_parsing():
    from torchsx.cli import _pipeline as P

    assert P.parse_radii("1,4,7") == (1.0, 4.0, 7.0)
    with pytest.raises(ValueError):
        P.parse_radii("7,9")
    with pytest.raises(ValueError):
        P.parse_radii("9,7,11")          # r_peak < r_gap <= r_bg
    assert P.parse_gain(None) == "auto"
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
        ["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu"])

    scale, _obj, tr = _stack(prep, params, tmp_path)
    tr.data_recipe = dict(bundle.recipe)
    ck = torch.load(tr.save_best(tmp_path / "g.pt"), weights_only=False)
    _p, _q, rebuilt, _o, _s = P.build_stack(bundle, args, ckpt=ck, verbose=False)
    # strict: every trained weight must land, and nothing may be left at init
    rebuilt.load_state_dict(ck["state_dict"])


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


def test_custom_split_file_labels_every_crystal(tmp_path, sample_core):
    """A partialator ``--custom-split`` file is a valid ``--split-file``.

    The unit in that file is a frame, so the label reaches a crystal through
    its image -- and the labels must line up with the crystal order
    ``prepare_observations`` ends up with, which is only true after the
    resolution filter has reindexed the image table.
    """
    from torchsx.cli import _pipeline as P
    from torchsx.stream import SFXStream

    from torchsx.io._metadata import StreamMetadata

    cache = tmp_path / "s.sfx"
    metadata = StreamMetadata(image={"filename": ["/raw/a.h5", "/raw/b.h5"],
                                     "event": ["//0", "//1"]})
    s = SFXStream(sample_core.replace(metadata=metadata, friedel_merged=False))
    s = SFXStream.concat([s, s])
    s = SFXStream(s._core.replace(image_id=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
                                  crystal_in_image=torch.tensor([0, 0, 1, 1], dtype=torch.int32)))
    s.save(cache)
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
    assert labels[0] == labels[2] and labels[1] == labels[3]

    # Both crystals of frame a belong to three bins. Repeating the same
    # membership must not count either crystal twice within that bin.
    overlapping = tmp_path / "overlapping.split"
    overlapping.write_text(
        "/raw/a.h5 //0 beta\n/raw/a.h5 //0 alpha\n"
        "/raw/a.h5 //0 gamma\n/raw/a.h5 //0 alpha\n"
        "/raw/b.h5 //1 alpha\n"
    )
    overlap_args = _DataArgs(stream=[cache], split_file=overlapping,
                             cache_prep=tmp_path / "overlapping.pt")
    expanded = P.load_prep(overlap_args, verbose=False)
    assert expanded.split_names == ["alpha", "beta", "gamma"]
    expanded_labels = expanded.prep.obs_info.delay_bin_idx
    assert torch.bincount(expanded_labels).tolist() == [4, 2, 2]
    obs_labels = expanded_labels[expanded.prep.obs_info.crystal_id]
    original_labels = labels[b.prep.obs_info.crystal_id]
    for k in range(3):
        expected = b.prep.I_obs if k == 0 else b.prep.I_obs[original_labels == 1]
        torch.testing.assert_close(expanded.prep.I_obs[obs_labels == k].sort().values,
                                   expected.sort().values)
    cached = P.load_prep(overlap_args, verbose=False)
    assert torch.equal(cached.prep.obs_info.delay_bin_idx, expanded_labels)
    torch.testing.assert_close(cached.prep.I_obs, expanded.prep.I_obs)

    # a frame the split file does not mention is refused, not silently pooled
    partial = tmp_path / "partial.split"
    with open(partial, "w") as fh:
        frames = list(dict.fromkeys(zip(s.filenames, s.events)))
        half = frames[: max(1, len(frames) // 2)]
        for f, e in half:
            fh.write(f"{f} {e} alpha\n")
    with pytest.raises(SystemExit, match="unlabelled"):
        P.load_prep(_DataArgs(stream=[cache], split_file=partial), verbose=False)









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
        ["x.sfx", "-o", str(tmp_path), "--device", "cpu"])
    prep_, params_, scale, obj, S = P.build_stack(bundle, targs, verbose=False)
    assert S["wilson_basis"] == "smooth" and S["basis"] is not None
    assert scale.anchor_stat == "basis" and S["basis"].degree == 4
    tr = _trainer_from_stack(prep_, params_, scale, obj, S)
    t = tr.targets[0]
    assert t.has_basis and t.wilson_coef.abs().sum() > 0
    tr.data_recipe = dict(recipe)
    ck = torch.load(tr.save_best(tmp_path / "smooth.pt"), weights_only=False)
    assert ck["posterior_grid"]["wilson_basis"] == "smooth"
    assert ck["posterior_grid"]["wilson_degree"] == 4
    assert ck["scale"]["anchor_stat"] == "basis"
    assert "wilson_coef" in ck["inference_context"]["targets"][0]
    assert "gauge_phi" not in ck["state_dict"], "the basis rows are rebuilt, not stored"

    # Merge side: no flags at all, everything comes from the checkpoint.
    margs = merge_parser().parse_args(
        ["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu"])
    p2, q2, rebuilt, o2, S2 = P.build_stack(bundle, margs, ckpt=ck, verbose=False)
    assert S2["wilson_basis"] == "smooth" and S2["wilson_degree"] == 4
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
    targs = train_parser().parse_args(["x.sfx", "-o", str(tmp_path), "--device", "cpu"])
    prep_, params_, scale, obj, S = P.build_stack(bundle, targs, verbose=False)
    tr = _trainer_from_stack(prep_, params_, scale, obj, S)
    tr.data_recipe = dict(recipe)
    ck = torch.load(tr.save_best(tmp_path / "shells.pt"), weights_only=False)
    # A checkpoint written before the smooth basis existed: no wilson_basis key
    # and no anchor_stat.  That is now the only way to select the shells prior.
    del ck["posterior_grid"]["wilson_basis"], ck["posterior_grid"]["wilson_degree"]
    del ck["scale"]["anchor_stat"]
    margs = merge_parser().parse_args(
        ["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu"])
    _p, _q, rebuilt, _o, S2 = P.build_stack(bundle, margs, ckpt=ck, verbose=False)
    assert S2["basis"] is None and rebuilt.anchor_stat == "median"
    rebuilt.load_state_dict(ck["state_dict"])


def test_defaults_are_smooth_prior_and_basis_anchor(tmp_path):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser as train_parser
    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    bundle = P.PrepBundle(prep, {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0, "radii": [7.0, 9.0, 11.0]}, [], 0.1)
    targs = train_parser().parse_args(["x.sfx", "-o", str(tmp_path), "--device", "cpu"])
    _p, _q, scale, _o, S = P.build_stack(bundle, targs, verbose=False)
    assert S["wilson_basis"] == "smooth" and S["basis"] is not None and scale.anchor_stat == "basis"
    assert TrainConfig().wilson_basis == "smooth"


# ── detector noise under the gain policy ───────────────────────────────


def _gain_bundle(prep, selected):
    from torchsx.cli import _pipeline as P
    recipe = {"d_min": 0.9, "spacegroup": "P 1", "gain": "auto", "radii": [7.0, 9.0, 11.0],
              "measurement": {"gain_selection": {"selected": selected}}}
    return P.PrepBundle(prep, recipe, [], 0.1)


@pytest.mark.parametrize("mode,selected,expected", [
    (None, "calibration", "off"), ("auto", "calibration", "global_B"),
    ("auto", "input", "off"), ("off", "calibration", "off"), ("panel", "input", "panel")])
def test_detector_noise_auto_fits_global_B_only_for_a_calibrated_gain(tmp_path, mode, selected, expected):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser as train_parser
    prep, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    argv = ["x.sfx", "-o", str(tmp_path), "--device", "cpu"]
    if mode is not None:
        argv += ["--detector-noise", mode]
    _p, _q, _s, obj, S = P.build_stack(_gain_bundle(prep, selected), train_parser().parse_args(argv),
                                       verbose=False)
    dn = obj.detector_noise
    assert S["detector_noise"] == expected
    if expected == "off":
        assert dn is None
    elif expected == "panel":
        assert dn.terms == "AB"
    else:
        assert (dn.n_groups, dn.terms) == (1, "B")
        assert dn.log_B.requires_grad and not dn.log_A.requires_grad


def test_detector_noise_checkpoint_record_wins_over_the_gain_selection(tmp_path):
    """A merge restores the trained global B, whatever the rebuilt preparation selected."""
    from torchsx.cli import _pipeline as P
    from torchsx.cli.merge_nn import _build_parser as merge_parser
    from torchsx.cli.train import _build_parser as train_parser
    prep, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    bundle = _gain_bundle(prep, "calibration")
    targs = train_parser().parse_args(["x.sfx", "-o", str(tmp_path), "--device", "cpu", "--detector-noise", "auto"])
    prep_, params_, scale, obj, S = P.build_stack(bundle, targs, verbose=False)
    with torch.no_grad():
        obj.detector_noise.log_B.fill_(math.log(1.2))
    tr = _trainer_from_stack(prep_, params_, scale, obj, S)
    tr.data_recipe = dict(bundle.recipe)
    ck = torch.load(tr.save_best(tmp_path / "b.pt"), weights_only=False)
    assert ck["detector_noise"]["config"] == {"n_groups": 1, "terms": "B"}
    margs = merge_parser().parse_args(["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu"])
    _p, _q, _r, restored, S2 = P.build_stack(_gain_bundle(prep, "input"), margs, ckpt=ck, verbose=False)
    assert S2["detector_noise"] == "checkpoint"
    assert (restored.detector_noise.n_groups, restored.detector_noise.terms) == (1, "B")
    torch.testing.assert_close(restored.detector_noise.log_B.exp(), torch.tensor([1.2]))


# ── merge_nn products ──────────────────────────────────────────────────
