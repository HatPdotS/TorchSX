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
from torchsx.merging.quadrature import TrapezoidalLogGrid
from torchsx.merging.scale import AmortizedScale
from torchsx.merging.scatter_shape import FreePerHKLShape
from torchsx.nn.metrics import scale_metrics, spearman
from torchsx.nn.objective import MarginalNLL
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig


def _synthetic(n_hkl=60, n_xtal=150, per=40, seed=0):
    """Poisson data with a hidden per-crystal scale."""
    torch.manual_seed(seed)
    log_I0 = torch.randn(n_hkl) * 0.8 + math.log(60.0)
    log_G = torch.randn(n_xtal) * 0.45

    cid = torch.arange(n_xtal).repeat_interleave(per)
    hkl = torch.randint(0, n_hkl, (n_xtal * per,))
    mu = (log_I0[hkl] + log_G[cid]).exp()
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
    obj = MarginalNLL(TrapezoidalLogGrid.from_prep(prep, MergeConfig()), Poisson())
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
    obj = MarginalNLL(TrapezoidalLogGrid.from_prep(prep, MergeConfig()), Poisson())
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
