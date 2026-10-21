"""Grid tooling: an external held-out set, early stopping, width flags."""
from types import SimpleNamespace

import pytest
import torch

from tests.test_amortized_training import _synthetic
from torchsx.merging.likelihood import Poisson
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.objective import MarginalNLL, wide_log_grid
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig


def _trainer(tmp_path, n_cond=2, **kw):
    prep, params, *_ = _synthetic(n_hkl=12, n_xtal=40, per=10, n_cond=n_cond)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, include_hkl=False, context_folds=2)
    return prep, AmortizedTrainer(prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
                                  config=TrainConfig(max_obs_per_batch=40, device="cpu", outdir=str(tmp_path),
                                                     posterior_k_coarse=5, posterior_m_fine=17,
                                                     split_half_every=0, **kw))


def test_explicit_heldout_crystals_are_scored_and_never_trained(tmp_path):
    prep, tr0 = _trainer(tmp_path)
    cond = tr0.condition
    tr0.close()
    ids = (cond == 0).nonzero(as_tuple=True)[0][:5]
    _, tr = _trainer(tmp_path, heldout_crystal_ids=ids.tolist())
    try:
        assert torch.equal(torch.sort(tr.heldout_crystals).values, torch.sort(ids).values)
        assert not torch.isin(tr.train_crystals, ids).any()
        assert torch.equal(tr.external_heldout, ids)
    finally:
        tr.close()
    wrong = (cond == 1).nonzero(as_tuple=True)[0][:2]
    with pytest.raises(ValueError):
        _trainer(tmp_path, heldout_crystal_ids=wrong.tolist())


def test_export_leaves_the_external_heldout_out(tmp_path):
    from torchsx.cli.merge_nn import _export_crystals, _split_halves
    _, tr0 = _trainer(tmp_path)
    ids = (tr0.condition == 0).nonzero(as_tuple=True)[0][:5]
    tr0.close()
    _, tr = _trainer(tmp_path, heldout_crystal_ids=ids.tolist())
    try:
        keep = _export_crystals(tr, 0)
        assert not torch.isin(keep, ids).any()
        assert keep.numel() == int((tr.condition == 0).sum()) - 5
        h1, h2 = _split_halves(tr, 0, 7)
        assert not torch.isin(torch.cat([h1, h2]), ids).any()
        assert _export_crystals(tr, 1).numel() == int((tr.condition == 1).sum())
    finally:
        tr.close()


def test_attach_heldout_stream_folds_the_split_into_the_first():
    from torchsx.cli.train import attach_heldout_stream
    lab = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    oi = SimpleNamespace(delay_bin_idx=lab.clone(), unique_timepoints=torch.arange(3, dtype=torch.float64))
    bundle = SimpleNamespace(prep=SimpleNamespace(obs_info=oi), split_names=["dark", "light", "heldout"])
    ids = attach_heldout_stream(bundle)
    assert ids.tolist() == [4, 5, 6]
    assert oi.delay_bin_idx.tolist() == [0, 0, 1, 1, 0, 0, 0]
    assert bundle.split_names == ["dark", "light"] and oi.unique_timepoints.numel() == 2
    with pytest.raises(ValueError):
        attach_heldout_stream(bundle)


def test_run_updates_stops_after_patience(tmp_path, monkeypatch):
    from torchsx.cli import train
    _, tr = _trainer(tmp_path, n_cond=1, heldout_frac=0.2)
    calls = []

    def fake_evaluate(crystals, phase="heldout"):
        calls.append(tr.step_i)
        if len(calls) == 2:                                   # one improvement at the first real evaluation
            tr.best_step, tr.best_heldout = tr.step_i, 1.0
        return {"nll_per_obs": 1.0 + len(calls)}
    monkeypatch.setattr(tr, "evaluate", fake_evaluate)
    monkeypatch.setattr(tr, "save_best", lambda *a, **k: None)
    monkeypatch.setattr(train, "_save_current", lambda *a, **k: None)
    try:
        updates = train.run_updates(tr, 50, eval_every=2, patience=2)
        # evaluations at 2 (best), 4, 6 -> stop after the second stale one
        assert sum(updates) == 6 and calls == [0, 2, 4, 6]
    finally:
        tr.close()


def test_width_flags_reach_the_encoder():
    from torchsx.cli import train
    a = train._build_parser().parse_args(["x.stream", "-o", "run", "--d-model", "64", "--n-layers", "3",
                                          "--eval-every", "50", "--patience", "3", "--heldout-stream", "h.stream"])
    assert (a.d_model, a.n_layers, a.eval_every, a.patience, a.heldout_stream.name) == (64, 3, 50, 3, "h.stream")
