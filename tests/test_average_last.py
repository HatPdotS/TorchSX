"""Evaluations and checkpoints hold the mean of the last updates; training
continues from the latest one."""
import torch

from tests.test_amortized_training import _synthetic
from torchsx.cli import train
from torchsx.cli.train import run_updates
from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.objective import MarginalNLL
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig


def _trainer(tmp_path, **cfg_kw):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=24, per=8)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2)
    return AmortizedTrainer(
        prep, scale, MarginalNLL(SplitLogGrid(n_points=96), Poisson()), params,
        config=TrainConfig(device="cpu", outdir=str(tmp_path), heldout_frac=0.25,
                           max_obs_per_batch=48, posterior_k_coarse=5, posterior_m_fine=17,
                           **cfg_kw),
    )


def _params(tr):
    return [p.detach().clone() for group in tr.opt.param_groups for p in group["params"]]


def _mean(states):
    return [torch.stack(vs).mean(0) for vs in zip(*states)]


def test_averaged_holds_the_mean_of_the_last_updates(tmp_path):
    tr = _trainer(tmp_path, average_last=3)
    try:
        t = tr.targets[0]
        assert t.wilson_coef_param is not None             # the surface is trained, so averaged
        after = []
        for i in range(5):
            rec = tr.step(tr.index.rows(tr.train_crystals[3 * i:3 * i + 6]))
            assert not rec.get("step_skipped")
            after.append(_params(tr))
        assert any(not torch.equal(a, b) for a, b in zip(after[-1], after[-2]))
        with tr.averaged():
            for p, m in zip(_params(tr), _mean(after[-3:])):
                torch.testing.assert_close(p, m)
            # the Wilson prior the merge reads follows the averaged coefficients
            torch.testing.assert_close(t.wilson_coef, t.wilson_coef_param.detach())
            torch.testing.assert_close(
                t.wilson_sigma,
                (t.wilson_phi.double() @ t.wilson_coef.double()).exp().float().clamp(min=t.floor))
        for p, latest in zip(_params(tr), after[-1]):
            assert torch.equal(p, latest)
    finally:
        tr.close()


def test_without_averaging_the_latest_update_stays(tmp_path):
    tr = _trainer(tmp_path)
    try:
        for i in range(3):
            tr.step(tr.index.rows(tr.train_crystals[3 * i:3 * i + 6]))
        latest = _params(tr)
        with tr.averaged():
            assert all(torch.equal(p, q) for p, q in zip(_params(tr), latest))
        assert not tr.load_average()
        assert all(torch.equal(p, q) for p, q in zip(_params(tr), latest))
    finally:
        tr.close()


def test_run_updates_checkpoints_and_ends_on_the_mean(tmp_path):
    tr = _trainer(tmp_path, average_last=4)
    try:
        after, step = [], tr.step

        def recorded(rows):
            rec = step(rows)
            after.append(_params(tr))
            return rec

        tr.step = recorded
        run_updates(tr, 6, eval_every=3)
        assert len(after) == 6
        for p, m in zip(_params(tr), _mean(after[-4:])):
            torch.testing.assert_close(p, m)
        last = torch.load(tmp_path / "last.pt", weights_only=False)
        assert last["average_last"] == 4
        for k, v in tr.scale.state_dict().items():
            torch.testing.assert_close(last["state_dict"][k], v.cpu())
    finally:
        tr.close()


def test_train_cli_averages_the_last_five_updates_per_state_by_default():
    a = train._build_parser().parse_args(["x.stream", "-o", "run"])
    assert a.average_last == 5
    a = train._build_parser().parse_args(["x.stream", "-o", "run", "--average-last", "0"])
    assert a.average_last == 0
