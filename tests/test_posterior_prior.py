"""Prior selection preserves the dI measure, evidence gradients and all moments."""
import math

import pytest
import torch

from torchsx.nn.posterior_target import PosteriorMergeTarget
from torchsx.nn.train_amortized import TrainConfig


@pytest.mark.parametrize('prior', ['none', 'wilson'])
@pytest.mark.parametrize('centric', [False, True])
def test_integrated_poisson_curve_matches_analytic_gamma(prior, centric):
    # L(I) = I**6 exp(-rate*I): a group of Poisson observations, omitting
    # factors independent of I. Exact gamma moments check both the Jacobian
    # and Wilson weighting, including the gradient used by the training loss.
    target = PosteriorMergeTarget(torch.ones(1), prior=prior,
                                  k_coarse=4097, m_fine=4097, n_sigma=12).double()
    target._loc = (torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    target.wilson_sigma.fill_(4.)
    target.is_centric.fill_(centric)
    nodes = target.u_coarse[None, :]
    rate = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    log_likelihood = 6 * nodes - rate * nodes.exp()
    log_z, mean_i, sd_i, weights, v, edge = target.integrate(log_likelihood, nodes)
    shape, effective_rate, log_constant = 7., 1.5, 0.
    if prior == 'wilson':
        shape = 6.5 if centric else 7.
        effective_rate += (0.5 if centric else 1.) / 4.
        log_constant = -0.5 * math.log(2 * math.pi * 4.) if centric else -math.log(4.)
    expected_i = shape / effective_rate
    expected_f = math.exp(math.lgamma(shape + 0.5) - math.lgamma(shape)) / math.sqrt(effective_rate)
    f = (weights * (v / 2).exp()).sum()
    var_f = (weights * ((v / 2).exp() - f).square()).sum()
    assert log_z.item() == pytest.approx(math.lgamma(shape) - shape * math.log(effective_rate) + log_constant, abs=2e-6)
    assert mean_i.item() == pytest.approx(expected_i, rel=2e-6)
    assert sd_i.item() ** 2 == pytest.approx(shape / effective_rate**2, rel=2e-6)
    assert f.item() == pytest.approx(expected_f, rel=2e-6)
    assert var_f.item() == pytest.approx(expected_i - expected_f**2, rel=2e-6)
    (-log_z.sum()).backward()
    assert rate.grad.item() == pytest.approx(expected_i, rel=2e-6)
    assert edge.item() < 1e-8


def test_none_is_not_flat_in_log_intensity_and_keeps_legacy_alias():
    targets = [PosteriorMergeTarget(torch.ones(1), prior=p, k_coarse=101,
                                   m_fine=101, n_sigma=2).double()
               for p in ['none', 'flat_I0', 'flat_log']]
    means = []
    for t in targets:
        t._loc = (torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
        nodes, _ = t.nodes()
        means.append(t.integrate(torch.zeros_like(nodes), nodes)[1].item())
    assert means[0] == means[1]
    # On finite [exp(-2), exp(2)], constant likelihood under dI is uniform I.
    assert means[0] == pytest.approx((math.exp(2) + math.exp(-2)) / 2, rel=5e-4)
    assert means[0] > means[2] * 1.5
    assert targets[1].prior == 'none'
    assert TrainConfig().posterior_prior == 'wilson'


def test_wilson_importance_nodes_cannot_silently_override_none():
    with pytest.raises(ValueError, match='Wilson node placement requires'):
        PosteriorMergeTarget(torch.ones(1), prior='none', node_placement='wilson')
