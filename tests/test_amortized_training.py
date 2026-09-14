"""End-to-end: can the leave-one-out context recover a scale it was not told?

Synthetic data carries a hidden per-crystal brightness ``G``.  The network is
given **no feature that encodes G** — not the crystal id, not a per-crystal
aggregate.  The only route to it is the leave-one-out context: a crystal with
G = 2 has all its *other* reflections twice as bright, and averaged over ~45
random reflections that shows up as a shift in mean log intensity.

This is the design's central claim reduced to something falsifiable in seconds.
If it ever regresses, the architecture is broken regardless of what the loss
curve says.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.config import MergeConfig
from torchsx.merging.hkl import build_adjacency
from torchsx.merging.likelihood import Poisson
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.nn.objective import wide_log_grid
from torchsx.merging.scale import AmortizedScale
from torchsx.merging.scatter_shape import FreePerHKLShape
from torchsx.nn.metrics import scale_metrics, spearman
from torchsx.nn.objective import MarginalNLL
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig


def _synthetic(n_hkl=60, n_xtal=150, per=40, seed=0, n_cond=1, shift=1.5):
    """Poisson data with a hidden per-crystal scale.

    With ``n_cond > 1`` crystals are labelled round-robin with a condition and
    the first quarter of the reflections is ``shift`` times brighter in every
    condition but the first -- a light/dark difference the merge must keep.
    """
    torch.manual_seed(seed)
    log_I0 = torch.randn(n_hkl) * 0.8 + math.log(60.0)
    log_G = torch.randn(n_xtal) * 0.45

    cid = torch.arange(n_xtal).repeat_interleave(per)
    hkl = torch.randint(0, n_hkl, (n_xtal * per,))
    cond = torch.arange(n_xtal) % n_cond
    shifted = torch.arange(n_hkl) < n_hkl // 4
    log_shift = (cond[cid] > 0) & shifted[hkl]
    mu = (log_I0[hkl] + log_G[cid] + log_shift.float() * math.log(shift)).exp()
    bg = torch.full_like(mu, 4.0)
    n_ph = torch.poisson(mu + bg)
    I = n_ph - bg
    sig = (mu + 2 * bg).sqrt()

    order = hkl.argsort()
    hkl, cid, I, bg, sig = (t[order] for t in (hkl, cid, I, bg, sig))

    d = torch.linspace(3.0, 0.9, n_hkl)
    hkl_valid = torch.stack(
        [torch.arange(n_hkl) % 7 - 3, torch.arange(n_hkl) // 7 - 3,
         torch.arange(n_hkl) % 5 - 2], 1
    ).to(torch.int16)
    oi = ObsInfo(
        crystal_id=cid.long(), s_sq=1 / (4 * d[hkl] ** 2), hkl_idx=hkl,
        n_crystals=n_xtal, panel_id=torch.randint(0, 4, hkl.shape),
        fs=torch.rand(hkl.shape) * 1000, ss=torch.rand(hkl.shape) * 1000,
        peak=I + bg, background=bg,
        ewald_offset=torch.randn(hkl.shape) * 1e-4, sigma=sig,
        obs_id=torch.arange(hkl.numel()),
        delay_bin_idx=cond if n_cond > 1 else None,
        unique_timepoints=(torch.arange(n_cond, dtype=torch.float64) - 1.0
                           if n_cond > 1 else None),
    )
    prep = PreparedObs(
        hkl_valid=hkl_valid, d_spacing=d, n_valid=n_hkl,
        median_cell=torch.tensor([10., 11., 12., 90., 90., 90.]),
        I_obs=I, hkl_id=hkl, bg=bg,
        obs_counts=torch.bincount(hkl, minlength=n_hkl),
        n_obs_total=hkl.numel(),
        mu_init=torch.full((n_hkl,), math.log(60.0)),
        alpha_init=torch.full((n_hkl,), 1.65),
        beta_perhkl=torch.full((n_hkl,), 2.0),
        adjacency=build_adjacency(hkl_valid), obs_info=oi, partiality=None,
        shell_id_fw=torch.arange(n_hkl) % 4,
        is_centric=torch.zeros(n_hkl, dtype=torch.bool),
    )
    params = Parameters(
        log_I0_init=prep.mu_init.clone(),
        scatter_shape=FreePerHKLShape(prep.alpha_init.log(), prep.beta_perhkl.log()),
    )
    return prep, params, log_G, cid, per


@pytest.mark.slow
def test_leave_one_out_context_recovers_a_hidden_per_crystal_scale(tmp_path):
    prep, params, log_G, cid, per = _synthetic()
    n_xtal = log_G.numel()

    scale = AmortizedScale.from_prep(
        prep, d_model=64, n_layers=3, include_hkl=False
    )
    obj = MarginalNLL(wide_log_grid(), Poisson())
    # ~30 epochs to converge from zero-init.  The scale ramps up from 0 and
    # OVERSHOOTS around epoch 10 (slope of predicted-on-true peaks near 1.6
    # before settling at ~0.94), so a short run measures the overshoot, not the
    # answer.  Same endpoint at lr 5e-4, just slower.
    cfg = TrainConfig(
        max_obs_per_batch=6000, epochs=35, lr=3e-3, lam0=3.0,
        lam_halflife_steps=60, log_every=5, heldout_frac=0.1,
        device="cpu", outdir=str(tmp_path),
    )
    trainer = AmortizedTrainer(prep, scale, obj, params, config=cfg)

    def per_crystal_log_k():
        scale.eval()
        with torch.no_grad():
            lk = scale.predict(prep.obs_info).log_K
        return torch.zeros(n_xtal).index_add_(0, cid.long(), lk) / per

    before = spearman(per_crystal_log_k(), log_G)
    trainer.run()
    after_pc = per_crystal_log_k()
    after = spearman(after_pc, log_G)

    assert abs(before) < 0.25, f"untrained model already correlated ({before})"
    assert after > 0.85, f"failed to recover the hidden scale (rho={after})"

    # Magnitude, not just ranking: the recovered spread should match the truth.
    # Converged slope is ~0.94 — mild shrinkage, which is what a regularised
    # predictor should do.
    scale.eval()
    with torch.no_grad():
        lk = scale.predict(prep.obs_info).log_K
    assert 0.6 < lk.std() / log_G.std() < 1.4

    # The truth here is purely per-crystal, so the variance decomposition must
    # say so.  On real data this same number is what distinguishes "learned
    # partiality" from "rediscovered G".
    m = scale_metrics(lk, prep.obs_info.crystal_id, n_xtal)
    assert m["between_frac"] > 0.8


@pytest.mark.slow
def test_hkl_permutation_stays_flat_when_hkl_is_not_a_feature(tmp_path):
    """Control for the degeneracy monitor: with hkl excluded it must read 0."""
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=60, per=30)
    scale = AmortizedScale.from_prep(prep, d_model=32, n_layers=2, include_hkl=False)
    obj = MarginalNLL(wide_log_grid(), Poisson())
    trainer = AmortizedTrainer(
        prep, scale, obj, params,
        config=TrainConfig(max_obs_per_batch=3000, epochs=2, device="cpu",
                           outdir=str(tmp_path)),
    )
    trainer.run()
    rec = trainer.audit(trainer.index.rows(trainer.train_crystals[:32]))
    assert rec["hkl_permutation_frac"] < 1e-6


def test_crystals_are_never_split_across_batches():
    """A truncated crystal gives a biased leave-one-out context."""
    prep, *_ = _synthetic(n_hkl=20, n_xtal=40, per=25)
    from torchsx.nn.train_amortized import CrystalIndex

    idx = CrystalIndex(prep.obs_info.crystal_id, prep.obs_info.n_crystals)
    seen: dict[int, int] = {}
    for sel, rows in idx.batches(300, shuffle=False):
        for c in prep.obs_info.crystal_id[rows].unique().tolist():
            seen[c] = seen.get(c, 0) + 1
        got = torch.bincount(
            prep.obs_info.crystal_id[rows], minlength=prep.obs_info.n_crystals
        )
        present = got > 0
        assert torch.equal(got[present], idx.counts[present]), "crystal split"
    assert all(v == 1 for v in seen.values()), "crystal appeared in two batches"


# ── condition-pure batching ─────────────────────────────────────────────────


def test_unconditioned_batches_are_unchanged():
    """No label, or one condition, must reproduce the plain greedy fill exactly."""
    prep, *_ = _synthetic(n_hkl=20, n_xtal=40, per=25)
    from torchsx.nn.train_amortized import CrystalIndex
    oi = prep.obs_info
    idx0 = CrystalIndex(oi.crystal_id, oi.n_crystals)
    idx1 = CrystalIndex(oi.crystal_id, oi.n_crystals,
                        condition=torch.zeros(oi.n_crystals, dtype=torch.long))
    assert idx0.n_conditions == 1 and idx1.n_conditions == 1

    def reference(max_obs, seed):
        # the pre-condition batcher, verbatim
        g = torch.Generator().manual_seed(seed)
        pool = idx0.nonempty[torch.randperm(idx0.nonempty.numel(), generator=g)]
        c = idx0.counts[pool]
        out, start, running = [], 0, 0
        for i in range(pool.numel()):
            ci = int(c[i])
            if running + ci > max_obs and i > start:
                out.append(pool[start:i]); start, running = i, 0
            running += ci
        if start < pool.numel():
            out.append(pool[start:])
        return out

    for seed in (0, 7):
        ref = reference(300, seed)
        for idx in (idx0, idx1):
            got = [sel for sel, _ in idx.batches(
                300, generator=torch.Generator().manual_seed(seed))]
            assert len(got) == len(ref)
            assert all(torch.equal(a, b) for a, b in zip(got, ref))


def test_batches_are_condition_pure_and_complete():
    prep, *_ = _synthetic(n_hkl=20, n_xtal=60, per=25, n_cond=3)
    from torchsx.nn.train_amortized import CrystalIndex
    oi = prep.obs_info
    idx = CrystalIndex(oi.crystal_id, oi.n_crystals, condition=oi.delay_bin_idx)
    assert idx.n_conditions == 3
    seen: dict[int, int] = {}
    conds = []
    for sel, rows in idx.batches(300, generator=torch.Generator().manual_seed(1)):
        cb = oi.delay_bin_idx[sel].unique()
        assert cb.numel() == 1, "mixed conditions in one batch"
        conds.append(int(cb))
        assert torch.equal(oi.delay_bin_idx[oi.crystal_id[rows]].unique(), cb)
        got = torch.bincount(oi.crystal_id[rows], minlength=oi.n_crystals)
        present = got > 0
        assert torch.equal(got[present], idx.counts[present]), "crystal split"
        for c in sel.tolist():
            seen[c] = seen.get(c, 0) + 1
    assert sorted(seen) == idx.nonempty.tolist()
    assert all(v == 1 for v in seen.values())
    assert set(conds) == {0, 1, 2}
    # shuffled as a list: not all of one condition first
    assert conds != sorted(conds)
    # unshuffled (evaluate / merge) is still pure and still complete
    n = sum(sel.numel() for sel, _ in idx.batches(300, shuffle=False,
                                                    crystals=idx.nonempty[:30]))
    assert n == 30
    # an unlabelled crystal is refused
    bad = oi.delay_bin_idx.clone(); bad[3] = -1
    with pytest.raises(ValueError):
        CrystalIndex(oi.crystal_id, oi.n_crystals, condition=bad)


def _two_condition_trainer(tmp_path, **cfg_kw):
    prep, params, log_G, cid, per = _synthetic(n_hkl=40, n_xtal=120, per=40, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=32, n_layers=2, include_hkl=False)
    obj = MarginalNLL(wide_log_grid(), Poisson())
    cfg = TrainConfig(
        max_obs_per_batch=1000, epochs=1, lr=1e-3, lam0=3.0, log_every=1,
        heldout_frac=0.2, device="cpu", outdir=str(tmp_path),
        merge_target="posterior", loss="evidence", posterior_prior="wilson",
        posterior_k_coarse=9, posterior_m_fine=33, posterior_n_sigma=8.0,
        split_half_every=0, **cfg_kw,
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    return tr, prep


def test_heldout_and_merge_sets_come_from_the_first_condition(tmp_path):
    tr, prep = _two_condition_trainer(tmp_path)
    cond = prep.obs_info.delay_bin_idx
    assert tr.n_conditions == 2 and len(tr.targets) == 2
    assert tr.heldout_crystals.numel() > 0
    assert torch.all(cond[tr.heldout_crystals] == 0)
    assert set(cond[tr.train_crystals].unique().tolist()) == {0, 1}
    # every crystal is either held out or trained on, exactly once
    allc = torch.cat([tr.heldout_crystals, tr.train_crystals]).sort().values
    assert torch.equal(allc, tr.index.nonempty)
    for k, v in tr.merge_sets.items():
        assert torch.all(cond[v] == 0), k
    # is_light follows the label without being passed in
    assert torch.equal(tr.is_light.cpu(), cond != 0)
    # pooled validation is refused with two conditions
    with pytest.raises(ValueError):
        _two_condition_trainer(tmp_path, heldout_condition=None)
    with pytest.raises(ValueError):
        _two_condition_trainer(tmp_path, heldout_condition=2)


def test_per_condition_targets_keep_the_difference(tmp_path):
    tr, prep = _two_condition_trainer(tmp_path)
    cond = prep.obs_info.delay_bin_idx
    n_hkl = prep.n_valid
    shifted = torch.arange(n_hkl) < n_hkl // 4

    # (a) the in-batch posterior merge of a light batch sees light only
    with torch.no_grad():
        I0 = {}
        for b in (0, 1):
            rows = tr.index.rows(tr.index.nonempty[cond[tr.index.nonempty] == b])
            _, _, tgt, *_ = tr._forward(rows, grad=False)
            assert tr._cond == b and tr.target is tr.targets[b]
            v = torch.zeros(n_hkl).index_add_(0, prep.hkl_id[rows], tgt.I0)
            c = torch.zeros(n_hkl).index_add_(0, prep.hkl_id[rows], torch.ones_like(tgt.I0))
            I0[b] = v / c.clamp(min=1)
            tr.scale.clear()
    ratio = I0[1] / I0[0]
    assert 1.3 < ratio[shifted].median() < 1.7, ratio[shifted].median()
    assert 0.85 < ratio[~shifted].median() < 1.15, ratio[~shifted].median()

    # (b) the running merges converge to their own condition, not to the pool
    for t in tr.targets:
        t.ema_decay = 0.8
    tr.cfg.epochs = 6
    tr.run()
    ema = torch.stack([t.ema_I0.cpu() for t in tr.targets])
    r = ema[1] / ema[0]
    assert 1.3 < r[shifted].median() < 1.7, r[shifted].median()
    assert 0.85 < r[~shifted].median() < 1.15, r[~shifted].median()
    # condition AUC is logged and the (condition-independent) G gives ~0.5
    rec = tr.step(tr.index.rows(tr.train_crystals[:10]))
    assert "condition_auc" in rec
    assert 0.3 < rec["condition_auc"] < 0.7, rec["condition_auc"]
    # checkpoint carries one merge per condition
    tr.evaluate(tr.heldout_crystals)
    ck = torch.load(tr.save_best(), weights_only=False)
    assert tuple(ck["ema_I0"].shape) == (2, n_hkl)
    assert ck["n_conditions"] == 2 and ck["timepoints"] == [-1.0, 0.0]
    tr.close()


def test_posterior_integrate_is_finite_for_flat_curves():
    """A flat curve at n_sigma=24 puts nodes ~96 nats above the centre.

    ``exp`` of that overflows float32; the posterior weight there is exactly
    zero, and 0 * inf = NaN took CC1/2 with it on the SACLA light merge.
    """
    from torchsx.nn.posterior_target import PosteriorMergeTarget
    n_hkl = 8
    tgt = PosteriorMergeTarget(torch.full((n_hkl,), 0.1), k_coarse=31, m_fine=129,
                               n_sigma=24.0, prior="wilson")
    tgt.set_wilson_sigma(torch.zeros(n_hkl, dtype=torch.long), 1)
    centre = torch.linspace(-7.0, 6.0, n_hkl)
    sd = torch.full((n_hkl,), 4.0)                      # the flat-curve fallback
    tgt._loc = (centre, sd)
    v = centre[:, None] + sd[:, None] * tgt.u_coarse[None, :]
    S = torch.full((n_hkl, 31), -92.5)                  # flat: no information
    log_Z, I0, I0_sd, w_post, vf, edge = tgt.integrate(S, v, None)
    assert torch.isfinite(log_Z).all() and torch.isfinite(I0).all() and torch.isfinite(I0_sd).all()
    # with a flat likelihood the posterior IS the Wilson prior: mean ~ Sigma
    # (129 fine nodes over 192 nats: the prior mean is resolved to ~6 %)
    assert torch.allclose(I0, torch.full_like(I0, float(tgt.wilson_sigma[0])), rtol=0.1)
    # and a peaked curve still gives the peak
    S2 = -0.5 * ((v - centre[:, None]) / 0.3) ** 2
    _, I0b, sdb, *_ = tgt.integrate(S2, v, None)
    assert torch.isfinite(I0b).all() and torch.all(sdb > 0)


@pytest.mark.parametrize("k_coarse, m_fine, tol", [(31, 129, 0.08), (63, 257, 0.04)])
def test_one_sided_curve_is_integrated_on_its_support(k_coarse, m_fine, tol):
    """A flat-then-cliff likelihood must not be interpolated across the cliff.

    Reference: the same curve integrated brute-force on a dense grid.  The old
    placement (parabola width ~4 nats x n_sigma 24) put coarse nodes 6.4 nats
    apart across the cliff and the posterior mean came out nats low.  With the
    knee-anchored support the remaining error is the linear interpolation of
    the cliff over one coarse spacing (0.4 nats at 31 nodes, 0.2 at 63): a few
    percent, low-biased, and halving with the node count.
    """
    from torchsx.nn.posterior_target import PosteriorMergeTarget
    n_hkl, bound = 6, torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    tgt = PosteriorMergeTarget(torch.full((n_hkl,), 1.0), k_coarse=k_coarse, m_fine=m_fine,
                               n_sigma=24.0, prior="wilson")
    tgt.set_wilson_sigma(torch.zeros(n_hkl, dtype=torch.long), 1)     # Sigma = 1
    # log-likelihood: 0 below the bound, then a steep cliff (an upper bound on I0)
    def curve(v):
        return -0.5 * (torch.relu(v - bound[:, None]) / 0.15) ** 2
    scan = tgt.scan_grid()
    c, sd = tgt.peak_from_scan(curve(scan))
    assert torch.all(sd <= tgt.max_halfwidth / tgt.n_sigma + 1e-6)
    tgt._loc = (c, sd)
    v = c[:, None] + sd[:, None] * tgt.u_coarse[None, :]
    _, I0, _, w_post, *_ = tgt.integrate(curve(v), v, None)
    # brute force on a dense grid with the same prior
    vd = torch.linspace(-30.0, 8.0, 20001)[None, :].expand(n_hkl, -1)
    logp = curve(vd) + vd - vd.exp()                                    # likelihood x Wilson (dI = I dv)
    I_ref = (torch.softmax(logp, dim=1) * vd.exp()).sum(1)
    rel = (I0 - I_ref) / I_ref
    assert float(rel.abs().max()) < tol, (I0.tolist(), I_ref.tolist())
    assert float(rel.max()) < 0.01                                      # any error is low-biased, never high
    # coverage: nothing sits at the grid edge
    n_edge = max(1, m_fine // 20)
    assert float((w_post[:, :n_edge].sum(1) + w_post[:, -n_edge:].sum(1)).max()) < 0.01
