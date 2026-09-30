"""Fixed Gaussian mixture fusion: values, derivatives, dispatch and addressing."""
import math

import pytest
import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid, HAVE_TRITON
from torchsx.merging.scatter.mixture import PredictedMixtureScatter
from torchsx.nn.objective import MarginalNLL


class Meta:
    def __init__(self, ids):
        self.obs_id = ids

    def slice(self, start, end):
        return Meta(self.obs_id[start:end])


def problem(n=257, c=4, device="cpu", dtype=torch.float64):
    g = torch.Generator().manual_seed(971)
    rand = lambda *shape: torch.rand(*shape, generator=g, dtype=torch.float64)
    bg = (rand(n) * math.log(800 / 0.3) + math.log(0.3)).exp()
    signal = (rand(n) * math.log(2e5 / 0.01) + math.log(0.01)).exp() * (rand(n) < 0.7)
    counts = torch.poisson(signal + bg, generator=g)
    counts[:8] = 0
    mu = (signal + 0.5 * bg.sqrt()).log() + (rand(n) - 0.5) * 12
    pi = torch.log_softmax((rand(n, c) - 0.5) * 10, 1)
    offset = (rand(n, c) - 0.5) * 4
    lo = math.log(0.25) + rand(n, c) * math.log(3 / 0.25)
    hi = lo + (rand(n, c) - 0.5) * 2
    mix = torch.cat((pi, offset, lo, hi, torch.full_like(lo, math.log(2))), 1)
    return tuple(x.to(device=device, dtype=dtype) for x in (counts, bg, mu, mix))


def oracle(mu, mix, counts, bg, quad, lik):
    """Float64 autograd oracle, including exact Gaussian tails via log_ndtr.

    Independently differentiates the integrand and CDF, without the custom
    gradient implementations in either Triton or GenGaussLogCDF.
    """
    p = quad.grid_params(counts, bg, quad._sig_b2(lik, bg), lik.bg_mean_offset)
    x, log_w = quad.nodes(p, quad.n_points)
    lam = x.exp() + p["bg_model"][:, None]
    nn = counts.clamp(min=1e-12)[:, None]
    dev = torch.where(counts[:, None] > 0, nn * (lam / nn).log() - (lam - nn), -lam)
    if lik.bg_var_factor:
        iphi = lam / (lam + (lik.bg_var_factor * bg)[:, None])
        dev = dev * iphi + 0.5 * iphi.log()
    log_lik = dev + p["logpeak"][:, None]
    terms = []
    c = mix.shape[1] // 5
    pi, offset, lo, hi, _ = mix.split(c, 1)
    for k in range(c):
        eps = x - mu[:, None] - offset[:, k, None]
        log_sum = torch.logaddexp(lo[:, k], hi[:, k])
        norm = math.log(2 / math.sqrt(math.pi)) - log_sum
        a = torch.where(eps > 0, hi[:, k, None], lo[:, k, None]).exp()
        window = torch.logsumexp(log_lik + norm[:, None] - (eps / a).square() + log_w, 1)
        t = p["x_lo"] - mu - offset[:, k]
        la = torch.where(t > 0, hi[:, k], lo[:, k])
        log_q = math.log(2) + torch.special.log_ndtr(-math.sqrt(2) * t.abs() / la.exp())
        cdf = torch.where(t > 0, torch.log1p(-(hi[:, k] - log_sum + log_q).exp()),
                          lo[:, k] - log_sum + log_q)
        terms.append(pi[:, k] + torch.logaddexp(window, p["log_L0"] + cdf))
    return torch.logsumexp(torch.stack(terms), 0)


def test_fixed_gaussian_contract_and_packed_addressing():
    _, _, _, mix = problem(9)
    sc = PredictedMixtureScatter(4, fixed_gaussian=True)
    ids = torch.tensor([45, 7, 99, 8, 3, 76, 6, 0, 22])
    sc.set_prediction(ids, mix)
    torch.testing.assert_close(sc.mixture_obs(Meta(ids[[4, 2, 4, 7]])), mix[[4, 2, 4, 7]])
    with pytest.raises(KeyError):
        sc.mixture_obs(Meta(torch.tensor([100])))
    with pytest.raises(ValueError, match="fixed_gaussian"):
        sc.set_prediction(ids, mix + 0.1)
    sc.clear_prediction()
    with pytest.raises(RuntimeError):
        sc.mixture_obs(Meta(ids))


@pytest.mark.parametrize("shape,fixed", [("gaussian", True), ("partiality", True),
                                        ("shared", False), ("free", False)])
def test_scale_selects_fusion_only_for_fixed_gaussian_heads(shape, fixed):
    from torchsx.nn.heads import ScaleHeads
    from torchsx.merging.scale import AmortizedScale
    heads = ScaleHeads(16, scatter_family="mixture", n_components=4, component_shape=shape)
    scale = AmortizedScale(torch.nn.Identity(), torch.nn.Identity(), heads, n_crystals=2)
    assert scale.scatter.fixed_gaussian is fixed
    prediction = heads(torch.randn(7, 16))
    scale.scatter.set_prediction(torch.arange(7), prediction.mix)


CUDA = pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_TRITON), reason="CUDA + Triton")


@CUDA
@pytest.mark.parametrize("c,nodes,bvf", [(1, 96, 0.0), (3, 192, 0.2788), (4, 192, 1.1641), (8, 96, 0.1477)])
def test_fused_values_and_gradients_against_float64_oracle(c, nodes, bvf):
    counts, bg, mu, mix = problem(c=c, device="cuda", dtype=torch.float32)
    mu.requires_grad_(); mix.requires_grad_()
    ids = torch.arange(mu.numel(), device="cuda")
    sc = PredictedMixtureScatter(c, fixed_gaussian=True)
    sc.set_prediction(ids, mix)
    lik = Poisson(bg_var_factor=bvf, bg_mean_offset=bvf)
    q = SplitLogGrid(nodes)
    got = q.marginalize_mixture(sc, lik, mu, counts, bg, None, Meta(ids))
    weights = torch.linspace(-0.7, 1.3, mu.numel(), device="cuda")
    grads = torch.autograd.grad((got * weights).sum(), (mu, mix))
    rd = [x.detach().double().requires_grad_(x.requires_grad) for x in (mu, mix)]
    ref = oracle(*rd, counts.double(), bg.double(), SplitLogGrid(nodes), lik)
    ref_grads = torch.autograd.grad((ref * weights.double()).sum(), rd)
    torch.testing.assert_close(got.double(), ref, atol=8e-4, rtol=2e-5)
    for actual, expected in zip(grads, ref_grads):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected, atol=2e-3, rtol=3e-4)


@CUDA
def test_chunked_miss_component_and_gradients():
    n, c = 121, 4
    counts, bg, mu, mix = problem(n, c, "cuda", torch.float32)
    mu.requires_grad_(); mix.requires_grad_()
    ids = torch.randperm(n, device="cuda") + 30
    # Duplicate and permuted queries exercise the gradient scatter back to
    # the installed prediction, rather than assuming row identity.
    selected = torch.arange(n, device="cuda").roll(17)
    selected[:3] = selected[3]
    meta = Meta(ids[selected])
    sc = PredictedMixtureScatter(c, fixed_gaussian=True)
    sc.set_prediction(ids, mix)
    obj = MarginalNLL(SplitLogGrid(96), Poisson(1.1641, bg_mean_offset=1.1641),
                      miss_frac=0.05).cuda()
    tensors = (mu, mix, obj.miss_logit)
    full = obj.log_prob(sc, None, mu, counts, bg, meta)
    gfull = torch.autograd.grad(full.sum(), tensors)
    obj.chunk_size = 23
    chunked = obj.log_prob(sc, None, mu, counts, bg, meta)
    gchunked = torch.autograd.grad(chunked.sum(), tensors)
    torch.testing.assert_close(chunked, full, atol=5e-5, rtol=2e-6)
    for got, ref in zip(gchunked, gfull):
        torch.testing.assert_close(got, ref, atol=5e-5, rtol=2e-5)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eager_override_and_free_shape_fallback(monkeypatch, dtype):
    import torchsx.merging.quadrature.split_log_grid as mod
    counts, bg, mu, mix = problem(17, 3, "cuda", dtype)
    sc = PredictedMixtureScatter(3, fixed_gaussian=True)
    ids = torch.arange(17, device="cuda")
    sc.set_prediction(ids, mix)
    q, lik = SplitLogGrid(96), Poisson()
    monkeypatch.setattr(mod, "_DISABLE_TRITON", True)
    def unexpected(*args, **kwargs):
        raise AssertionError("fused path must be bypassed")
    monkeypatch.setattr(mod, "marginalize_gaussian_mixture", unexpected)
    got = q.marginalize_mixture(sc, lik, mu, counts, bg, None, Meta(ids))
    ref = Quadrature.marginalize_mixture(q, sc, lik, mu, counts, bg, None, Meta(ids))
    torch.testing.assert_close(got, ref)
    sc.fixed_gaussian = False
    mix[:, -3:] = math.log(1.3)
    sc.set_prediction(ids, mix)
    monkeypatch.setattr(mod, "_DISABLE_TRITON", False)
    q.marginalize_mixture(sc, lik, mu, counts, bg, None, Meta(ids))


@CUDA
def test_gaussian_network_gradients_through_checkpointed_posterior(monkeypatch):
    from torchsx.nn.heads import ScaleHeads
    from torchsx.merging.scale import AmortizedScale
    from torchsx.nn.posterior_target import PosteriorMergeTarget
    torch.manual_seed(17)
    n, h = 120, 12
    heads = ScaleHeads(16, scatter_family="mixture", n_components=4,
                       component_shape="gaussian", predict_skew=True).cuda()
    scale = AmortizedScale(torch.nn.Identity(), torch.nn.Identity(), heads, n_crystals=2)
    pred = heads(torch.randn(n, 16, device="cuda"))
    ids = torch.arange(n, device="cuda")
    meta = Meta(ids)
    scale.scatter.set_prediction(ids, pred.mix)
    bg = torch.full((n,), 5.0, device="cuda")
    counts = torch.poisson(torch.full_like(bg, 65.0))
    q = SplitLogGrid(96)
    obj = MarginalNLL(q, Poisson(0.3, bg_mean_offset=0.3), chunk_size=37)
    target = PosteriorMergeTarget(torch.full((h,), 75.0, device="cuda"),
                                   k_coarse=9, m_fine=33, prior="flat_log").cuda()
    target.checkpoint_nodes = True
    target._loc = (torch.full((h,), math.log(75.0), device="cuda"),
                   torch.full((h,), 4.0, device="cuda"))
    nodes = target._loc[0][:, None] + target._loc[1][:, None] * target.u_coarse[None, :]
    hid = ids % h

    def evidence():
        curve = target.accumulate(obj, scale.scatter, None, pred, counts, bg, hid, meta, nodes)
        return target.integrate(curve)[0]

    fast = evidence()
    parameters = tuple(heads.parameters())
    grads = torch.autograd.grad(fast.sum(), parameters, retain_graph=True)
    monkeypatch.setattr(q, "marginalize_mixture",
                        lambda *a, **kw: Quadrature.marginalize_mixture(q, *a, **kw))
    ref = evidence()
    reference = torch.autograd.grad(ref.sum(), parameters)
    torch.testing.assert_close(fast, ref, atol=1e-4, rtol=1e-5)
    assert any(bool(g.abs().max() > 0) for g in grads)
    for actual, expected in zip(grads, reference):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=5e-4)


@CUDA
@pytest.mark.parametrize("mu_grad", [False, True])
def test_zero_background_tail_and_partial_gradient_requests(mu_grad):
    counts, bg, mu, mix = problem(13, 4, "cuda", torch.float32)
    counts[:4] = 0; bg[:2] = 0
    mu[:3] = torch.tensor([-35.0, -5.0, 20.0], device="cuda")
    mu.requires_grad_(mu_grad); mix.requires_grad_(not mu_grad)
    ids = torch.arange(len(mu), device="cuda")
    sc = PredictedMixtureScatter(4, fixed_gaussian=True)
    sc.set_prediction(ids, mix)
    got = SplitLogGrid(192).marginalize_mixture(sc, Poisson(), mu, counts, bg, None, Meta(ids))
    leaf = mu if mu_grad else mix
    (grad,) = torch.autograd.grad(got.sum(), leaf)
    rd = [x.detach().double().requires_grad_(x.requires_grad) for x in (mu, mix)]
    ref = oracle(*rd, counts.double(), bg.double(), SplitLogGrid(192), Poisson())
    (expected,) = torch.autograd.grad(ref.sum(), rd[0] if mu_grad else rd[1])
    torch.testing.assert_close(got.double(), ref, atol=8e-4, rtol=2e-5)
    torch.testing.assert_close(grad.double(), expected, atol=2e-3, rtol=3e-4)
