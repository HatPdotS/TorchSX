"""Learning-rate schedule and the bias-corrected running reference."""
import math

import pytest
import torch

from torchsx.nn.posterior_target import PosteriorMergeTarget, ema_update_corrected
from torchsx.nn.train_amortized import lr_factor


def test_constant_schedule_is_the_old_warmup():
    assert lr_factor(0, warmup=50, total=400) == pytest.approx(1 / 50)
    assert lr_factor(24, warmup=50, total=400) == pytest.approx(0.5)
    assert all(lr_factor(s, warmup=50, total=400) == 1.0 for s in (49, 50, 200, 399, 1000))
    assert lr_factor(0, warmup=0, total=400) == 1.0


def test_cosine_decays_from_the_end_of_warmup_to_the_floor():
    f = [lr_factor(s, warmup=50, total=400, schedule="cosine") for s in range(400)]
    assert f[49] == pytest.approx(1.0) and f[50] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[50:], f[51:]))                       # monotone after warm-up
    assert f[225] == pytest.approx(0.5, abs=1e-12)                             # half-way
    assert 0 < f[-1] < 1e-4
    g = [lr_factor(s, warmup=50, total=400, schedule="cosine", min_frac=0.1) for s in (50, 399, 500)]
    assert g[0] == pytest.approx(1.0) and 0.1 < g[1] < 0.1001 and g[2] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        lr_factor(0, warmup=0, total=10, schedule="step")


def test_bias_corrected_reference_forgets_the_seed():
    d = 0.99
    value = torch.full((4,), 7.0)                                             # the seed merge
    num, wt = torch.zeros(4), torch.zeros(4)
    seen = torch.tensor([True, True, False, True])
    ema_update_corrected(value, num, wt, torch.tensor([1.0, 2.0, 3.0, 4.0]), seen, d)
    torch.testing.assert_close(value, torch.tensor([1.0, 2.0, 7.0, 4.0]))    # first batch, seed gone
    ema_update_corrected(value, num, wt, torch.tensor([3.0, 4.0, 5.0, 6.0]), torch.ones(4, dtype=torch.bool), d)
    expect = torch.tensor([(d * 1 + 3) / (d + 1), (d * 2 + 4) / (d + 1), 5.0, (d * 4 + 6) / (d + 1)])
    torch.testing.assert_close(value, expect)
    # the uncorrected average after the same two updates still holds d^2 of the seed
    plain = d * (d * 7.0 + (1 - d) * 1.0) + (1 - d) * 3.0
    assert plain > 6.5 and abs(plain - float(expect[0])) > 4.5


def test_target_keeps_the_correction_state_out_of_checkpoints():
    t = PosteriorMergeTarget(torch.ones(5), bias_correct=True)
    assert t.bias_correct and "ema_num" not in t.state_dict() and "ema_wt" not in t.state_dict()
    assert not PosteriorMergeTarget(torch.ones(5)).bias_correct


def test_train_cli_accepts_the_schedule_flags():
    from torchsx.cli import train
    a = train._build_parser().parse_args(["x.stream", "-o", "run", "--warmup", "25", "--lr-schedule", "cosine",
                                          "--reference-bias-correction"])
    assert a.warmup == 25 and a.lr_schedule == "cosine" and a.reference_bias_correction
    a = train._build_parser().parse_args(["x.stream", "-o", "run"])
    assert a.warmup is None and a.lr_schedule == "constant" and not a.reference_bias_correction
    assert math.isfinite(lr_factor(0, warmup=0, total=0, schedule="cosine"))
