"""SplitLogGrid: analytic tail + data-driven window against a dense reference."""
import math

import pytest
import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid, eager_only
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid, HAVE_TRITON
from torchsx.merging.scatter.gengauss import (
    GenGaussLogScatter, gengauss_log_cdf_from_obs, log_alpha_from_log_width,
)

CUDA = torch.cuda.is_available()


def _scatter(la, lb):
    s = GenGaussLogScatter()
    s.log_alpha_beta_obs = lambda params, meta: (la, lb)  # noqa: E731
    return s


def _batch(device, seed=0, M=4096):
    """Observations spanning every regime: zero counts to 2e5 photons,
    background 0.3 .. 800, widths 0.37 .. 3, beta 1.07 .. 2."""
    g = torch.Generator().manual_seed(seed)
    bg = (torch.rand(M, generator=g) * (math.log(800) - math.log(0.3)) + math.log(0.3)).exp()
    log_sig = torch.rand(M, generator=g) * (math.log(2e5) - math.log(0.01)) + math.log(0.01)
    sig = log_sig.exp() * (torch.rand(M, generator=g) < 0.7).float()      # 30 % signal-free
    n_ph = torch.poisson(sig + bg, generator=g)
    n_ph[:64] = 0.0                                                          # exact zeros
    w = (torch.rand(M, generator=g) * (math.log(3.0) - math.log(0.37)) + math.log(0.37)).exp()
    beta = torch.rand(M, generator=g) * 0.93 + 1.07
    la = log_alpha_from_log_width(w.log(), beta)
    lb = beta.log()
    # log_mean spread widely around the counting estimate, including far off
    # within +-6 nats of the counting estimate: the outer grid spans +-6 sd
    # of a located peak, and an observation whose model scale is e^8 off its
    # own counts is an e^-100 outlier whose log p level carries no weight
    mu = (sig + 0.5 * bg.sqrt()).log() + (torch.randn(M, generator=g) * 2.0).clamp(-6, 6)
    t = lambda x: x.to(device)  # noqa: E731
    return t(n_ph), t(bg), t(la), t(lb), t(mu)


def _reference(n_ph, bg, la, lb, mu, lik, n=80001, half=45.0, chunk=256):
    """Dense uniform fixed grid, eager, float64 -- the same integrand, brute force.

    float64 because the eager Poisson forms ``n log lam - lam - log n!`` and
    loses ~0.2 nats at n = 2e5 in float32.  Chunked to bound the (M, K) tensors.
    """
    q = FixedLogGrid(S_min=math.exp(-half), S_max=math.exp(half), n_points=n).to(n_ph.device).double()
    out = []
    with eager_only():
        for i in range(0, n_ph.numel(), chunk):
            sl = slice(i, i + chunk)
            out.append(q.marginalize_signal(
                _scatter(la[sl].double(), lb[sl].double()), lik, mu[sl].double(),
                n_ph[sl].double(), bg[sl].double(), None, None))
    return torch.cat(out).to(mu.dtype)


@pytest.mark.parametrize("beta", [1.07, 1.225, 2.0])
def test_tail_log_cdf_matches_float64(beta):
    t = torch.linspace(-25.0, 6.0, 2001, dtype=torch.float64)
    la = torch.full_like(t, math.log(0.7))
    lb = torch.full_like(t, math.log(beta))
    ref = gengauss_log_cdf_from_obs(t, la, lb, x_switch=1e9)     # float64, direct gamma only
    got = gengauss_log_cdf_from_obs(t.float(), la.float(), lb.float()).double()
    # float64 direct gammaincc is itself reliable only until Q underflows
    ok = ref > -80          # direct gammaincc is clamped at 1e-38 below this
    err = (got - ref).abs()[ok]
    assert float(err.max()) < 2e-3, float(err.max())
    # far tail: the asymptotic branch must agree with the exact exponent to
    # a few parts in 1e3 relative in log space
    t_far = torch.tensor([-30.0, -60.0], dtype=torch.float64)
    la2, lb2 = la[:2], lb[:2]
    far = gengauss_log_cdf_from_obs(t_far.float(), la2.float(), lb2.float()).double()
    x = (t_far.abs() / 0.7) ** beta
    lead = (1 / beta - 1) * x.log() - x - torch.lgamma(torch.tensor(1 / beta, dtype=torch.float64)) + math.log(0.5)
    assert torch.allclose(far, lead, atol=0.05, rtol=0.0), (far, lead)


@pytest.mark.parametrize("bvf", [0.0, 1.1641])
def test_split_eager_matches_dense_reference(bvf):
    n_ph, bg, la, lb, mu = _batch("cpu")
    lik = Poisson(bg_var_factor=bvf)
    ref = _reference(n_ph, bg, la, lb, mu, lik)
    q = SplitLogGrid(n_points=192)
    got = q.marginalize_signal(_scatter(la, lb), lik, mu, n_ph, bg, None, None)
    d = (got - ref).abs()
    assert float(d.median()) < 1e-4, float(d.median())
    assert float(d.quantile(0.99)) < 2e-3, float(d.quantile(0.99))
    assert float(d.max()) < 1e-2, (float(d.max()), int(d.argmax()))


def test_split_gradients_no_worse_than_production_grid():
    """Per-observation gradients against the float64 dense reference.

    The GenGauss slope changes sign at eps = 0, and for beta near 1 that is a
    kink; the trapezoid rule's error on d log p / d log_mean is then FIRST order
    in the node spacing, for the production FixedLogGrid (h = 0.051) as much as
    for this one.  So the bar is the production grid's own error, not zero.
    """
    n_ph, bg, la, lb, mu = _batch("cpu", seed=1, M=1024)
    lik = Poisson(bg_var_factor=1.1641)

    def grads(fn):
        m = mu.clone().requires_grad_(True)
        a = la.clone().requires_grad_(True)
        b = lb.clone().requires_grad_(True)
        lp = fn(m, a, b)
        g = torch.autograd.grad(lp.sum(), (m, a, b))
        return lp.detach(), g

    _, g_ref = grads(lambda m, a, b: _reference(n_ph, bg, a, b, m, lik))
    qs = SplitLogGrid(n_points=192)
    _, g_split = grads(lambda m, a, b: qs.marginalize_signal(_scatter(a, b), lik, m, n_ph, bg, None, None))
    qf = FixedLogGrid(S_min=math.exp(-34.5), S_max=math.exp(34.5), n_points=1355)
    with eager_only():
        _, g_fixed = grads(lambda m, a, b: qf.marginalize_signal(_scatter(a, b), lik, m, n_ph, bg, None, None))
    for name, gs, gf, gr in zip(("mu", "log_alpha", "log_beta"), g_split, g_fixed, g_ref):
        es = (gs - gr).abs()
        ef = (gf - gr).abs()
        # 192 nodes over a ~12-nat window is a ~0.065 spacing against the fixed
        # grid's 0.051, so allow that ratio (1.3) plus margin on the tail; the
        # fixed grid's eager float32 Poisson also adds its own noise at bright n
        assert float(es.quantile(0.99)) < 1.6 * float(ef.quantile(0.99)) + 1e-3, (
            name, float(es.quantile(0.99)), float(ef.quantile(0.99)))
        assert float(es.median()) < 1.6 * float(ef.median()) + 1e-4, (
            name, float(es.median()), float(ef.median()))
        cs = torch.nn.functional.cosine_similarity(gs, gr, dim=0)
        cf = torch.nn.functional.cosine_similarity(gf, gr, dim=0)
        assert float(1 - cs) < 2.0 * float(1 - cf) + 1e-5, (name, float(cs), float(cf))


def test_grid_is_data_only_and_cached():
    n_ph, bg, la, lb, mu = _batch("cpu", M=512)
    q = SplitLogGrid(n_points=192)
    s2 = 1.1641 * bg.clamp(min=0)
    q._last_bvf = 1.1641
    p1 = q.grid_params(n_ph, bg, s2)
    p2 = q.grid_params(n_ph, bg, 1.1641 * bg.clamp(min=0))
    assert p1 is p2                         # same tensors -> cache hit
    x, log_w = SplitLogGrid.nodes(p1, 192)
    assert torch.all(x[:, 1:] >= x[:, :-1])           # monotone
    assert torch.allclose(x[:, 0], p1["x_lo"]) and torch.all(x[:, -1] <= p1["x_hi"] + 1e-5)
    h = (x[:, 1:] - x[:, :-1])
    assert float(h.max()) < 0.13, float(h.max())       # coarse spacing: >= 2.7 nodes per alpha_min
    assert torch.all(log_w.exp().sum(1) <= (p1["x_hi"] - p1["x_lo"]) + 1e-4)
    # narrow scatter placed anywhere in the window is integrated: the node
    # spacing resolves alpha = 0.35 at 3 nodes per alpha
    assert float((h.max(1).values / 0.35).max()) < 0.38


@pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA + triton")
@pytest.mark.parametrize("bvf", [0.0, 1.1641])
def test_split_triton_matches_eager(bvf):
    import os
    n_ph, bg, la, lb, mu = _batch("cuda", seed=2)
    lik = Poisson(bg_var_factor=bvf)
    q = SplitLogGrid(n_points=192)
    sc = _scatter(la, lb)

    def run(eager):
        os.environ["TORCHSX_FIXEDLOGGRID_NO_TRITON"] = "1" if eager else ""
        import importlib
        import torchsx.merging.quadrature.split_log_grid as slg
        slg._DISABLE_TRITON = eager
        m = mu.clone().requires_grad_(True)
        a = la.clone().requires_grad_(True)
        b = lb.clone().requires_grad_(True)
        qq = SplitLogGrid(n_points=192)
        lp = qq.marginalize_signal(sc.__class__() if False else _scatter(a, b), lik, m, n_ph, bg, None, None)
        g = torch.autograd.grad(lp.sum(), (m, a, b))
        return lp.detach(), g

    lp_e, g_e = run(True)
    lp_t, g_t = run(False)
    import torchsx.merging.quadrature.split_log_grid as slg
    slg._DISABLE_TRITON = False
    # float32 rounding: the kernel's lg2.approx enters as n * log(lam/n), so at
    # n ~ 2e5 a 1e-7 relative log error is 0.02 nats away from the peak -- the
    # same level the fixed-grid kernel has.  Bulk agreement is what is tested.
    d = (lp_e - lp_t).abs()
    assert float(d.median()) < 2e-5, float(d.median())
    assert float(d.quantile(0.99)) < 1e-2, float(d.quantile(0.99))
    assert float(d.max()) < 3e-2, float(d.max())
    for ge, gt in zip(g_e, g_t):
        assert float((ge - gt).abs().max()) < 2e-2 * float(ge.abs().max().clamp(min=1e-6)) + 1e-3


@pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA + triton")
def test_split_triton_speed_vs_fixed():
    """Not an assertion on speed; prints the ratio for the record."""
    import time
    M = 1_000_000
    g = torch.Generator().manual_seed(3)
    bg = (torch.rand(M, generator=g) * 4 + 3).exp().cuda()
    n_ph = torch.poisson(bg * 1.1).cuda()
    la = torch.full((M,), math.log(1.0), device="cuda")
    lb = torch.full((M,), math.log(1.225), device="cuda")
    mu = (n_ph - bg).clamp(min=1).log()
    lik = Poisson(bg_var_factor=1.1641)
    qs = SplitLogGrid(n_points=192)
    qf = FixedLogGrid(S_min=math.exp(-34.5), S_max=math.exp(34.5), n_points=1355).cuda()
    for q in (qs, qf):
        ts = []
        for i in range(4):
            torch.cuda.synchronize(); t0 = time.time()
            q.marginalize_signal(_scatter(la, lb), lik, mu, n_ph, bg, None, None)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        print(f"{type(q).__name__}: {1e3 * sorted(ts[1:])[1]:.1f} ms")
