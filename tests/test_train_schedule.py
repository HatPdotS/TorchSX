"""Learning-rate schedule."""
import math

import pytest
import torch

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


def test_train_cli_accepts_the_schedule_flags():
    from torchsx.cli import train
    a = train._build_parser().parse_args(["x.stream", "-o", "run", "--warmup", "25", "--lr-schedule", "cosine"])
    assert a.warmup == 25 and a.lr_schedule == "cosine"
    a = train._build_parser().parse_args(["x.stream", "-o", "run"])
    assert a.warmup is None and a.lr_schedule == "constant"
    assert math.isfinite(lr_factor(0, warmup=0, total=0, schedule="cosine"))


def test_warmup_does_not_grow_with_the_number_of_states():
    from torchsx.cli import train
    parse = train._build_parser().parse_args
    default = parse(["x.stream", "-o", "run"])
    for n_states in (1, 2, 16):
        assert train.schedule(default, n_states)["warmup_steps"] == 25
        assert train.schedule(parse(["x.stream", "-o", "run", "--warmup", "40"]), n_states)["warmup_steps"] == 40
    # 16 states reach the full learning rate within the default budget
    s = train.schedule(default, 16)
    assert s["total_steps"] == 400
    assert lr_factor(24, warmup=s["warmup_steps"], total=s["total_steps"]) == 1.0
    assert train.schedule(parse(["x.stream", "-o", "run", "--warmup", "0"]), 3)["warmup_steps"] == 0
    # the averaging window stays one update of each state
    assert train.schedule(default, 3)["average_last"] == 15
