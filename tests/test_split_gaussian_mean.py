"""The closed-form two-width Gaussian mean correction.

Pinned against adaptive numerical integration of the two-width density
(GenGauss at ``beta = 2``: ``2 / ((a_lo + a_hi) sqrt(pi)) exp(-(x / a_side)^2)``),
each half integrated separately so the width jump at zero is exact.  The
production 257-node grid of ``log_e_s_gengauss`` is only bounded.
"""
from __future__ import annotations

import math

import torch
from scipy import integrate

from torchsx.merging.scatter.gengauss import log_e_s_gengauss
from torchsx.merging.scatter.mixture import mixture_log_e_s, split_gaussian_log_e_s


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    lo = torch.rand(n, generator=g, dtype=torch.float64) * 4.0 - 3.0     # alpha 0.05 .. 2.7
    hi = lo - torch.rand(n, generator=g, dtype=torch.float64) * 1.5      # upper side narrower
    return lo, hi


def _quad(lo: float, hi: float) -> float:
    a_lo, a_hi = math.exp(lo), math.exp(hi)
    c = 2.0 / ((a_lo + a_hi) * math.sqrt(math.pi))
    below = integrate.quad(lambda x: c * math.exp(x - (x / a_lo) ** 2), -math.inf, 0.0)[0]
    above = integrate.quad(lambda x: c * math.exp(x - (x / a_hi) ** 2), 0.0, math.inf)[0]
    return math.log(below + above)


def test_matches_adaptive_integration():
    lo, hi = _inputs(60, 0)
    want = torch.tensor([_quad(a, b) for a, b in zip(lo.tolist(), hi.tolist())],
                        dtype=torch.float64)
    torch.testing.assert_close(split_gaussian_log_e_s(lo, hi), want, rtol=0, atol=1e-9)


def test_gradients():
    lo, hi = _inputs(40, 1)
    assert torch.autograd.gradcheck(split_gaussian_log_e_s,
                                    (lo.requires_grad_(), hi.requires_grad_()))


def test_production_grid_error_is_bounded():
    lo, hi = _inputs(4000, 0)
    grid = log_e_s_gengauss(lo, torch.full_like(lo, 2.0), log_alpha_hi=hi)
    assert float((split_gaussian_log_e_s(lo, hi) - grid).abs().max()) < 5e-3


def test_symmetric_case_is_alpha_squared_over_four():
    la = torch.linspace(-3.0, 1.0, 200, dtype=torch.float64)
    torch.testing.assert_close(split_gaussian_log_e_s(la, la), la.exp() ** 2 / 4,
                               rtol=1e-10, atol=1e-12)


def test_mixture_flag_combines_the_closed_form_per_component():
    g = torch.Generator().manual_seed(2)
    n, K = 300, 4
    log_pi = torch.log_softmax(torch.randn(n, K, generator=g, dtype=torch.float64), dim=1)
    offset = torch.randn(n, K, generator=g, dtype=torch.float64) * 0.5
    lo = torch.randn(n, K, generator=g, dtype=torch.float64) * 0.5 - 0.5
    hi = lo - torch.rand(n, K, generator=g, dtype=torch.float64)
    beta = torch.full_like(lo, 2.0)
    want = torch.logsumexp(torch.stack(
        [log_pi[:, k] + offset[:, k] + split_gaussian_log_e_s(lo[:, k], hi[:, k])
         for k in range(K)], 1), 1)
    got = mixture_log_e_s(log_pi, offset, lo, hi, beta, gaussian=True)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    assert float((got - mixture_log_e_s(log_pi, offset, lo, hi, beta)).abs().max()) < 5e-3
