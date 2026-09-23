"""The closed-form two-width Gaussian mean correction matches the quadrature one."""
from __future__ import annotations

import torch

from torchsx.merging.scatter.gengauss import log_e_s_gengauss
from torchsx.merging.scatter.mixture import mixture_log_e_s, split_gaussian_log_e_s


def _grid(lo, hi):
    return log_e_s_gengauss(lo, torch.full_like(lo, 2.0), log_alpha_hi=hi)


def test_matches_quadrature_at_beta_2():
    g = torch.Generator().manual_seed(0)
    lo = torch.rand(4000, generator=g, dtype=torch.float64) * 4.0 - 3.0   # alpha 0.05 .. 2.7
    hi = lo - torch.rand(4000, generator=g, dtype=torch.float64) * 1.5    # upper side narrower
    torch.testing.assert_close(split_gaussian_log_e_s(lo, hi), _grid(lo, hi),
                               rtol=0, atol=2e-4)


def test_symmetric_case_is_alpha_squared_over_four():
    la = torch.linspace(-3.0, 1.0, 200, dtype=torch.float64)
    torch.testing.assert_close(split_gaussian_log_e_s(la, la), la.exp() ** 2 / 4,
                               rtol=1e-10, atol=1e-12)


def test_gradients_match_quadrature():
    g = torch.Generator().manual_seed(1)
    lo = (torch.rand(500, generator=g, dtype=torch.float64) * 3.0 - 2.0).requires_grad_()
    hi = (lo.detach() - torch.rand(500, generator=g, dtype=torch.float64)).requires_grad_()
    ga = torch.autograd.grad(split_gaussian_log_e_s(lo, hi).sum(), (lo, hi))
    gb = torch.autograd.grad(_grid(lo, hi).sum(), (lo, hi))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=0, atol=2e-3)


def test_mixture_flag_uses_the_closed_form():
    g = torch.Generator().manual_seed(2)
    n, K = 300, 4
    log_pi = torch.log_softmax(torch.randn(n, K, generator=g, dtype=torch.float64), dim=1)
    offset = torch.randn(n, K, generator=g, dtype=torch.float64) * 0.5
    lo = torch.randn(n, K, generator=g, dtype=torch.float64) * 0.5 - 0.5
    hi = lo - torch.rand(n, K, generator=g, dtype=torch.float64)
    beta = torch.full_like(lo, 2.0)
    torch.testing.assert_close(
        mixture_log_e_s(log_pi, offset, lo, hi, beta, gaussian=True),
        mixture_log_e_s(log_pi, offset, lo, hi, beta), rtol=0, atol=2e-4)
