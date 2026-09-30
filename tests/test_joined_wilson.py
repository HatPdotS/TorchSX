"""Shared Wilson fitting: support, state weighting and exact saved context."""
from types import SimpleNamespace

import pytest
import torch

from torchsx.nn.posterior_target import PosteriorMergeTarget
from torchsx.nn.train_amortized import AmortizedTrainer


def trainer(values, supports, smooth):
    targets = []
    for value, support in zip(values, supports):
        target = PosteriorMergeTarget(torch.tensor(value, dtype=torch.float32), prior="wilson")
        target.ema_den.copy_(torch.tensor(support, dtype=torch.float32))
        if smooth:
            target.set_basis(torch.ones(len(value), 1))
        targets.append(target)
    tr = SimpleNamespace(targets=targets, _score_targets=None, device="cpu",
        cfg=SimpleNamespace(posterior_prior="joined-wilson", wilson_fit="reference"),
        prep=SimpleNamespace(shell_id_fw=torch.zeros(len(values[0]), dtype=torch.long),
                             is_centric=torch.zeros(len(values[0]), dtype=torch.bool)),
        _all_targets=lambda: targets)
    return tr


@pytest.mark.parametrize("smooth", [False, True])
def test_equal_state_weight_and_exclusion_of_unseen_seed(smooth):
    # Dark has four observations at 2, light only one at 10. The equally
    # state-weighted mean is 6, not the observation-weighted 3.6.
    tr = trainer([[2, 2, 2, 2], [10, 10000, 10000, 10000]],
                 [[1, 1, 1, 1], [1, 0, 0, 0]], smooth)
    AmortizedTrainer._refresh_wilson_sigma(tr)
    for target in tr.targets:
        torch.testing.assert_close(target.wilson_sigma, torch.full((4,), 6.))
    tr.targets.reverse()
    AmortizedTrainer._refresh_wilson_sigma(tr)
    torch.testing.assert_close(tr.targets[0].wilson_sigma, torch.full((4,), 6.))
    assert torch.equal(tr.targets[0].wilson_sigma, tr.targets[1].wilson_sigma)
    assert torch.equal(tr.targets[0].wilson_coef, tr.targets[1].wilson_coef)


@pytest.mark.parametrize("smooth", [False, True])
def test_pooled_bootstrap_and_single_state_equivalence(smooth):
    tr = trainer([[2, 4, 6], [2, 4, 6]], [[0, 0, 0], [0, 0, 0]], smooth)
    AmortizedTrainer._refresh_wilson_sigma(tr)
    torch.testing.assert_close(tr.targets[0].wilson_sigma, torch.full((3,), 4.))
    single = trainer([[2, 4, 6]], [[1, 0, 1]], smooth)
    ordinary = trainer([[2, 4, 6]], [[1, 0, 1]], smooth)
    ordinary.cfg.posterior_prior = "wilson"
    AmortizedTrainer._refresh_wilson_sigma(single)
    AmortizedTrainer._refresh_wilson_sigma(ordinary)
    torch.testing.assert_close(single.targets[0].wilson_sigma, ordinary.targets[0].wilson_sigma)


def test_centric_weights_and_unobserved_shell_fallback():
    tr = trainer([[2, 10, 999], [2, 10, 999]], [[1, 1, 0], [1, 1, 0]], True)
    tr.prep.is_centric[1] = True
    AmortizedTrainer._refresh_wilson_sigma(tr)
    torch.testing.assert_close(tr.targets[0].wilson_sigma, torch.full((3,), 14 / 3))
    shell = trainer([[2, 10, 999], [2, 10, 999]], [[1, 1, 0], [1, 1, 0]], False)
    shell.prep.shell_id_fw[2] = 1
    for t in shell.targets:
        t.wilson_sigma.fill_(7)
    AmortizedTrainer._refresh_wilson_sigma(shell)
    torch.testing.assert_close(shell.targets[0].wilson_sigma, torch.tensor([6., 6., 7.]))
