"""Two-width (asymmetric) GenGauss scatter and the miss component.

Contract: with ``log_alpha_hi == log_alpha`` (or ``None``) every path is the
symmetric family, bit for bit where the code path is shared and to float32
rounding where the Triton kernel takes the ASYM branch; with different widths
the eager path, the Triton kernel and float64 autograd agree on value and
gradient; the mixture likelihood stays normalised.
"""
import math

import pytest
import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid, HAVE_TRITON
from torchsx.merging.scatter.gengauss import (
    GenGaussLogScatter, gengauss_log_cdf_from_obs, gengauss_log_prob_from_obs,
    log_alpha_from_log_width, log_e_s_gengauss,
)
from tests.test_split_log_grid import _batch, _scatter

CUDA = torch.cuda.is_available()


def _scatter_asym(la, lb, la_hi):
    s = _scatter(la, lb)
    s.log_alpha_hi_obs = lambda params, meta: la_hi  # noqa: E731
    return s


def _asym_widths(la, seed=0, spread=0.8):
    g = torch.Generator().manual_seed(seed)
    skew = (torch.rand(la.numel(), generator=g) * 2 - 1).to(la.device) * spread
    return la - 0.5 * skew, la + 0.5 * skew


# ---------------------------------------------------------------- density

def test_asym_density_reduces_to_symmetric_and_is_normalised():
    torch.manual_seed(0)
    la = torch.randn(64) * 0.4
    lb = torch.rand(64) * 0.6 + 0.05
    eps = torch.linspace(-30, 30, 20001)[None, :].expand(64, -1)
    sym = gengauss_log_prob_from_obs(eps, la, lb)
    same = gengauss_log_prob_from_obs(eps, la, lb, la.clone())
    assert torch.allclose(sym, same, atol=1e-6)
    lo, hi = _asym_widths(la)
    asym = gengauss_log_prob_from_obs(eps, lo, lb, hi)
    mass = torch.logsumexp(asym, dim=1).exp() * float(eps[0, 1] - eps[0, 0])
    assert torch.allclose(mass, torch.ones(64), atol=2e-3), mass
    # the two sides carry alpha_side / (alpha_lo + alpha_hi) of the mass
    left = (asym[:, eps[0] <= 0].exp().sum(1) * float(eps[0, 1] - eps[0, 0]))
    assert torch.allclose(left, lo.exp() / (lo.exp() + hi.exp()), atol=3e-3)


def test_asym_log_e_s_matches_brute_force():
    # E[exp(eps)] of a GenGauss with beta near 1 and alpha > 1 is carried by a
    # tail hundreds of nats out (the exponent eps - (eps/alpha)^beta peaks at
    # eps = (alpha^beta / beta)^(1/(beta-1))), which no finite grid holds and
    # which the production u_max = 12 truncates.  The formula is tested where
    # the integral exists on the grid: beta >= 1.3, alpha <= 1.1.
    torch.manual_seed(1)
    la = torch.rand(32) * 1.1 - 1.0
    lb = (torch.rand(32) * 0.7 + 1.3).log()
    lo, hi = _asym_widths(la, spread=0.6)
    eps = torch.linspace(-60, 60, 240001, dtype=torch.float64)[None, :]
    lp = gengauss_log_prob_from_obs(eps.expand(32, -1), lo.double(), lb.double(), hi.double())
    ref = torch.logsumexp(lp + eps, dim=1) + math.log(float(eps[0, 1] - eps[0, 0]))
    got = log_e_s_gengauss(lo, lb.exp(), log_alpha_hi=hi, n_u=1025, u_max=20.0)
    assert torch.allclose(got.double(), ref, atol=2e-3), (got - ref).abs().max()
    # the production grid (257 nodes, +-12) agrees with itself in both families
    got_p = log_e_s_gengauss(lo, lb.exp(), log_alpha_hi=hi)
    assert torch.allclose(got_p.double(), ref, atol=5e-3), (got_p - ref).abs().max()
    # symmetric call unchanged
    assert torch.allclose(log_e_s_gengauss(la, lb.exp(), log_alpha_hi=la.clone()),
                          log_e_s_gengauss(la, lb.exp()), atol=1e-6)


# ---------------------------------------------------------------- tail CDF

@pytest.mark.parametrize("beta", [1.07, 1.5, 2.0])
def test_asym_tail_cdf_value_and_gradients(beta):
    torch.manual_seed(2)
    M = 512
    la = torch.randn(M, dtype=torch.float64) * 0.4
    lb = torch.full((M,), math.log(beta), dtype=torch.float64)
    lo, hi = _asym_widths(la, spread=0.9)
    t = torch.randn(M, dtype=torch.float64) * 4.0
    # value: brute-force integral of the density up to t
    grid = torch.linspace(-60, 60, 240001, dtype=torch.float64)
    lp = gengauss_log_prob_from_obs(grid[None, :].expand(M, -1), lo, lb, hi)
    cdf_ref = ((grid[None, :] <= t[:, None]) * lp.exp()).sum(1) * float(grid[1] - grid[0])
    log_F = gengauss_log_cdf_from_obs(t, lo, lb, log_alpha_hi_obs=hi)
    ok = cdf_ref > 1e-12
    # the Riemann reference loses relative accuracy deep in the tail (log F ~ -26)
    assert torch.allclose(log_F[ok], cdf_ref[ok].log(), atol=2e-3, rtol=1e-3), (log_F[ok] - cdf_ref[ok].log()).abs().max()
    # gradients: closed form vs autograd through the plain value function
    from torchsx.merging.scatter.gengauss import _log_cdf_value
    t_, lo_, lb_, hi_ = [x.clone().requires_grad_(True) for x in (t, lo, lb, hi)]
    g_closed = torch.autograd.grad(gengauss_log_cdf_from_obs(t_, lo_, lb_, log_alpha_hi_obs=hi_).sum(),
                                   (t_, lo_, lb_, hi_))
    # torch has no shape derivative of the incomplete gamma, so autograd covers
    # (t, lo, hi) and the beta gradient is checked against a central difference
    t2, lo2, hi2 = [x.clone().requires_grad_(True) for x in (t, lo, hi)]
    g_auto = torch.autograd.grad(_log_cdf_value(t2, lo2, lb, 25.0, hi2).sum(), (t2, lo2, hi2))
    d_b = 1e-4
    g_beta_fd = (_log_cdf_value(t, lo, lb + d_b, 25.0, hi) - _log_cdf_value(t, lo, lb - d_b, 25.0, hi)) / (2 * d_b)
    for gc, ga, name in zip((g_closed[0], g_closed[1], g_closed[3], g_closed[2]),
                            g_auto + (g_beta_fd,), ("t", "lo", "hi", "beta")):
        d = (gc - ga).abs().max().item(); sc = ga.abs().max().item()
        assert d < 2e-3 * sc + 1e-4, (name, d, sc)
    # symmetric reduction
    sym = gengauss_log_cdf_from_obs(t, la, lb)
    same = gengauss_log_cdf_from_obs(t, la, lb, log_alpha_hi_obs=la.clone())
    assert torch.allclose(sym, same, atol=1e-9)


# ---------------------------------------------------------------- SplitLogGrid

def _run_split(la, lb, mu, n_ph, bg, lik, la_hi=None, eager=True):
    import torchsx.merging.quadrature.split_log_grid as slg
    slg._DISABLE_TRITON = eager
    m = mu.clone().requires_grad_(True)
    a = la.clone().requires_grad_(True)
    b = lb.clone().requires_grad_(True)
    h = None if la_hi is None else la_hi.clone().requires_grad_(True)
    sc = _scatter(a, b) if h is None else _scatter_asym(a, b, h)
    lp = SplitLogGrid(n_points=192).marginalize_signal(sc, lik, m, n_ph, bg, None, None)
    g = torch.autograd.grad(lp.sum(), (m, a, b) + (() if h is None else (h,)))
    slg._DISABLE_TRITON = False
    return lp.detach(), g


def test_split_eager_asym_equal_widths_is_symmetric():
    n_ph, bg, la, lb, mu = _batch("cpu", seed=3, M=1024)
    lik = Poisson(bg_var_factor=1.1641)
    lp_s, g_s = _run_split(la, lb, mu, n_ph, bg, lik)
    lp_a, g_a = _run_split(la, lb, mu, n_ph, bg, lik, la_hi=la.clone())
    assert torch.allclose(lp_s, lp_a, atol=1e-5), (lp_s - lp_a).abs().max()
    assert torch.allclose(g_s[0], g_a[0], atol=1e-4)
    assert torch.allclose(g_s[2], g_a[2], atol=1e-4)
    # width gradient splits across the two sides and sums to the symmetric one
    assert torch.allclose(g_s[1], g_a[1] + g_a[3], atol=1e-4), (g_s[1] - g_a[1] - g_a[3]).abs().max()


def test_split_eager_asym_matches_float64_autograd():
    n_ph, bg, la, lb, mu = _batch("cpu", seed=4, M=512)
    lo, hi = _asym_widths(la, seed=4)
    lik = Poisson(bg_var_factor=1.1641)
    lp32, g32 = _run_split(lo, lb, mu, n_ph, bg, lik, la_hi=hi)
    lp64, g64 = _run_split(lo.double(), lb.double(), mu.double(), n_ph.double(), bg.double(), lik, la_hi=hi.double())
    assert float((lp32 - lp64.float()).abs().median()) < 1e-4
    for a, b in zip(g32, g64):
        assert float((a - b.float()).abs().max()) < 1e-2 * float(b.abs().max()) + 1e-3


@pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA + triton")
@pytest.mark.parametrize("bvf", [0.0, 1.1641])
def test_split_triton_asym_matches_eager(bvf):
    n_ph, bg, la, lb, mu = _batch("cuda", seed=5)
    lo, hi = _asym_widths(la, seed=5)
    lik = Poisson(bg_var_factor=bvf)
    lp_e, g_e = _run_split(lo, lb, mu, n_ph, bg, lik, la_hi=hi, eager=True)
    lp_t, g_t = _run_split(lo, lb, mu, n_ph, bg, lik, la_hi=hi, eager=False)
    d = (lp_e - lp_t).abs()
    assert float(d.median()) < 2e-5, float(d.median())
    assert float(d.quantile(0.99)) < 1e-2, float(d.quantile(0.99))
    for ge, gt in zip(g_e, g_t):
        assert float((ge - gt).abs().max()) < 2e-2 * float(ge.abs().max().clamp(min=1e-6)) + 1e-3
    # the ASYM kernel branch with equal widths reproduces the symmetric kernel
    lp_s, g_s = _run_split(la, lb, mu, n_ph, bg, lik, eager=False)
    lp_a, g_a = _run_split(la, lb, mu, n_ph, bg, lik, la_hi=la.clone(), eager=False)
    assert float((lp_s - lp_a).abs().max()) < 1e-4
    assert float((g_s[1] - g_a[1] - g_a[3]).abs().max()) < 1e-3 * float(g_s[1].abs().max()) + 1e-4


# ---------------------------------------------------------------- heads + miss

def test_heads_skew_zero_init_is_symmetric_and_prime_passes_widths():
    from torchsx.nn.heads import ScaleHeads
    torch.manual_seed(6)
    z = torch.randn(50, 16)
    sym = ScaleHeads(16, w_init=1.65, beta_init=1.0, predict_skew=False)(z)
    skw = ScaleHeads(16, w_init=1.65, beta_init=1.0, predict_skew=True)(z)
    assert skw.log_alpha_hi is not None
    assert torch.allclose(skw.log_alpha, sym.log_alpha) and torch.allclose(skw.log_alpha_hi, sym.log_alpha)
    assert torch.allclose(skw.log_e_s, sym.log_e_s, atol=1e-6)
    from torchsx.merging.scatter.predicted import PredictedGenGaussScatter
    sc = PredictedGenGaussScatter()
    ids = torch.arange(50)
    sc.set_prediction(ids, skw.log_alpha, skw.beta.log(), skw.log_alpha_hi + 0.3)

    class M:  # minimal meta
        obs_id = ids[::2]
    assert torch.allclose(sc.log_alpha_hi_obs(None, M), skw.log_alpha[::2] + 0.3)
    sc.set_prediction(ids, skw.log_alpha, skw.beta.log())
    assert sc.log_alpha_hi_obs(None, M) is None


def test_miss_mixture_is_normalised_and_off_by_default():
    from torchsx.nn.objective import MarginalNLL
    import torchsx.merging.quadrature.split_log_grid as slg
    slg._DISABLE_TRITON = True
    torch.manual_seed(7)
    M = 6
    bg = torch.tensor([0.5, 3.0, 20.0, 100.0, 400.0, 2.0])
    la = torch.full((M,), math.log(1.0)); lb = torch.zeros(M)
    mu = torch.tensor([0.0, 1.0, 3.0, 4.0, 5.0, -3.0])   # e^mu well inside the n grid, Laplace tail included
    # plain Poisson: the kernel's peak-relative form plus logpeak IS log p(n), so
    # summing exp(lp) over n must give 1 with and without the miss component
    lik = Poisson(bg_var_factor=0.0)
    n_grid = torch.arange(0, 6000).float()
    for miss in (0.0, 0.07):
        obj = MarginalNLL(SplitLogGrid(192), lik, miss_frac=miss)
        assert (obj.miss_logit is None) == (miss == 0.0)
        tot = torch.zeros(M, dtype=torch.float64)
        for n in n_grid:
            lp = obj.log_prob(_scatter(la, lb), None, mu, n.expand(M), bg, None)
            tot += lp.double().exp()
        assert torch.allclose(tot, torch.ones(M, dtype=torch.float64), atol=2e-2), tot
    lik = Poisson(bg_var_factor=0.5)
    slg._DISABLE_TRITON = False
    obj0 = MarginalNLL(SplitLogGrid(192), lik)
    obj1 = MarginalNLL(SplitLogGrid(192), lik, miss_frac=0.1)
    n = torch.tensor([0.0, 4.0, 25.0, 90.0, 380.0, 1.0])
    lp0 = obj0.log_prob(_scatter(la, lb), None, mu, n, bg, None)
    lp1 = obj1.log_prob(_scatter(la, lb), None, mu, n, bg, None)
    # the miss term can only add probability where the scatter predicted a signal
    assert (lp1 >= lp0 + math.log(0.9) - 1e-5).all()
    assert list(obj1.parameters()) and not list(obj0.parameters())
