"""When the last updates oscillate, evaluations and checkpoints hold their
mean; training continues from the latest one."""
import math

import torch

from tests.test_amortized_training import _synthetic
from torchsx.cli import train
from torchsx.cli.train import run_updates
from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.objective import MarginalNLL
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig, step_reversal


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


def _adamw_path(sign, n=6, beta1=0.9, lr=1e-3):
    """Parameters along n AdamW updates (second moment 1) with gradient
    g_t = sign**t * g."""
    g = torch.tensor([1.0, -2.0, 0.5])
    theta, m, states = torch.zeros(3), torch.zeros(3), []
    for t in range(n):
        m = beta1 * m + (1 - beta1) * g * sign**t
        theta = theta - lr * m
        states.append([theta.clone()])
    return states, [[(lr, beta1, 0.0)]] * n


def test_step_reversal_tells_back_and_forth_from_steady_progress():
    steady, settings = _adamw_path(+1)
    assert step_reversal(steady, settings) > 0.99
    back_and_forth, settings = _adamw_path(-1)
    assert step_reversal(back_and_forth, settings) < -0.99
    assert math.isnan(step_reversal(steady[:2], settings[:2]))


def test_averaged_holds_the_mean_of_the_last_updates_when_they_oscillate(tmp_path):
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
        tr._oscillating = lambda: True
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
        tr._oscillating = lambda: False
        with tr.averaged():
            assert all(torch.equal(p, q) for p, q in zip(_params(tr), after[-1]))
    finally:
        tr.close()


def test_the_decision_follows_the_step_reversal_of_the_kept_updates(tmp_path):
    tr = _trainer(tmp_path, average_last=4)
    try:
        for i in range(2):
            tr.step(tr.index.rows(tr.train_crystals[3 * i:3 * i + 6]))
        assert not tr._oscillating() and tr.averaging is None         # too few updates
        for i in range(2, 6):
            tr.step(tr.index.rows(tr.train_crystals[3 * i:3 * i + 6]))
        decided = tr._oscillating()
        r = step_reversal([s for s, _ in tr._recent], [c for _, c in tr._recent])
        assert tr.averaging == {"window": 4, "step_reversal": r, "averaged": r < 0}
        assert decided == (r < 0)
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


def _recorded_run(tr, oscillating, steps=6):
    after, step = [], tr.step

    def recorded(rows):
        rec = step(rows)
        after.append(_params(tr))
        return rec

    tr.step = recorded
    tr._oscillating = lambda: oscillating
    run_updates(tr, steps, eval_every=3)
    assert len(after) == steps
    return after


def test_run_updates_checkpoints_and_ends_on_the_mean_when_oscillating(tmp_path):
    tr = _trainer(tmp_path, average_last=4)
    try:
        after = _recorded_run(tr, oscillating=True)
        for p, m in zip(_params(tr), _mean(after[-4:])):
            torch.testing.assert_close(p, m)
        last = torch.load(tmp_path / "last.pt", weights_only=False)
        assert last["average_last"] == 4
        for k, v in tr.scale.state_dict().items():
            torch.testing.assert_close(last["state_dict"][k], v.cpu())
    finally:
        tr.close()


def test_run_updates_keeps_the_latest_update_when_steady(tmp_path):
    tr = _trainer(tmp_path, average_last=4)
    try:
        after = _recorded_run(tr, oscillating=False)
        assert all(torch.equal(p, q) for p, q in zip(_params(tr), after[-1]))
        last = torch.load(tmp_path / "last.pt", weights_only=False)
        for k, v in tr.scale.state_dict().items():
            torch.testing.assert_close(last["state_dict"][k], v.cpu())
    finally:
        tr.close()


def test_train_cli_averages_the_last_five_updates_per_state_by_default():
    a = train._build_parser().parse_args(["x.stream", "-o", "run"])
    assert a.average_last == 5
    a = train._build_parser().parse_args(["x.stream", "-o", "run", "--average-last", "0"])
    assert a.average_last == 0
