"""All-nodes posterior kernel: S and its gradients against the node-by-node path."""
import math

import pytest
import torch

import torchsx.nn.posterior_target as pt
from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.split_log_grid import HAVE_TRITON, SplitLogGrid
from torchsx.merging.scatter.mixture import PredictedMixtureScatter
from torchsx.nn.objective import MarginalNLL
from torchsx.nn.pooling import group_sum

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_TRITON),
                                reason="needs CUDA and Triton")


class Meta:
    def __init__(self, ids, hkl):
        self.obs_id, self.hkl_idx = ids, hkl

    def slice(self, start, end):
        return Meta(self.obs_id[start:end], self.hkl_idx[start:end])


def problem(n=20000, n_hkl=700, K=31, c=4, seed=5):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    bg = (rand(n) * math.log(800 / 0.3) + math.log(0.3)).exp()
    signal = (rand(n) * math.log(2e4 / 0.01) + math.log(0.01)).exp() * (rand(n) < 0.8)
    counts = torch.poisson(signal + bg, generator=g)
    counts[:8] = 0
    mu = (signal + 0.5 * bg.sqrt()).log() + (rand(n) - 0.5) * 2
    pi = torch.log_softmax((rand(n, c) - 0.5) * 6, 1)
    offset = (rand(n, c) - 0.5) * 2
    offset[:, 0] = 0
    lo = math.log(0.25) + rand(n, c) * math.log(3 / 0.25)
    hi = lo - rand(n, c)
    mix = torch.cat((pi, offset, lo, hi, torch.full_like(lo, math.log(2))), 1)
    hkl = torch.randint(0, n_hkl, (n,), generator=g)
    v = (rand(n_hkl)[:, None] - 0.5) * 2 + torch.linspace(-3, 3, K, dtype=torch.float64)[None, :]
    f = lambda x: x.to("cuda", torch.float32)
    return f(counts), f(bg), f(mu), f(mix), hkl.cuda(), f(v)


def build(bvf: float, miss: float, outlier: float = 0.0):
    lik = Poisson(bg_var_factor=bvf, bg_mean_offset=bvf)
    obj = MarginalNLL(SplitLogGrid(n_points=192), lik, miss_frac=miss, outlier_frac=outlier).cuda()
    return obj, PredictedMixtureScatter(4, fixed_gaussian=True)


def node_loop(obj, scatter, mu, n_ph, bg, meta, v, n_hkl):
    cols = [group_sum(obj.log_prob(scatter, None, v[:, c].index_select(0, meta.hkl_idx) + mu,
                                   n_ph, bg, meta), meta.hkl_idx, n_hkl)
            for c in range(v.shape[1])]
    return torch.stack(cols, 1)


def fused(obj, scatter, mu, n_ph, bg, meta, v, n_hkl):
    from torchsx.merging.quadrature._posterior_nodes_triton import posterior_node_sums
    q, lik = obj.quadrature, obj.likelihood
    sig_b2 = q._sig_b2(lik, bg)
    p = q.grid_params(n_ph, bg, sig_b2, q._bg_mean_offset(lik))
    olog = obj.outlier_logit
    return posterior_node_sums(mu, scatter.mixture_obs(meta), obj.miss_logit, v, meta.hkl_idx,
                               n_ph, p, q.n_points, sig_b2=sig_b2, n_hkl=n_hkl, outlier_logit=olog,
                               log_out=obj.log_outlier(n_ph, bg) if olog is not None else None)


@pytest.mark.parametrize("bvf,miss,outlier", [(1.16, 0.03, 0.0), (0.0, 0.0, 0.0), (1.16, 0.03, 0.01),
                                               (1.16, 0.0, 0.01)])
def test_values_and_gradients_match_node_loop(bvf, miss, outlier):
    n_ph, bg, mu0, mix0, hkl, v = problem()
    if outlier:
        # a few gross count errors, both signs, so the component carries weight
        n_ph = n_ph.clone()
        n_ph[8:208:2] += 4000.0
        n_ph[9:209:2] = (n_ph[9:209:2] - 3000.0).clamp(min=0.0)
    n_hkl = v.shape[0]
    meta = Meta(torch.arange(len(mu0), device="cuda"), hkl)
    out = {}
    for name, fn in (("loop", node_loop), ("fused", fused)):
        obj, scatter = build(bvf, miss, outlier)
        mu = mu0.clone().requires_grad_()
        mix = mix0.clone().requires_grad_()
        scatter.set_prediction(meta.obs_id, mix)
        S = fn(obj, scatter, mu, n_ph, bg, meta, v, n_hkl)
        W = torch.randn(S.shape, generator=torch.Generator().manual_seed(3)).cuda()
        params = ([mu, mix] + ([obj.miss_logit] if obj.miss_logit is not None else [])
                  + ([obj.outlier_logit] if obj.outlier_logit is not None else []))
        out[name] = (S.detach(), torch.autograd.grad((S * W).sum(), params))
    S_ref, g_ref = out["loop"]
    S_new, g_new = out["fused"]
    torch.testing.assert_close(S_new, S_ref, rtol=1e-5, atol=2e-3)
    for a, b in zip(g_new, g_ref):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4 * float(b.abs().max()))


def test_accumulate_dispatches_to_the_kernel(monkeypatch):
    n_ph, bg, mu, mix, hkl, v = problem(n=5000, n_hkl=200, K=9)
    meta = Meta(torch.arange(len(mu), device="cuda"), hkl)
    obj, scatter = build(1.16, 0.03)
    scatter.set_prediction(meta.obs_id, mix)

    class Pred:
        log_K, log_e_s = mu, torch.zeros_like(mu)

    target = pt.PosteriorMergeTarget(torch.ones(200)).cuda()
    got = target.accumulate(obj, scatter, None, Pred, n_ph, bg, hkl, meta, v)
    monkeypatch.setattr(pt, "_NODE_LOOP", True)
    want = target.accumulate(obj, scatter, None, Pred, n_ph, bg, hkl, meta, v)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=2e-3)


def test_outlier_component_absorbs_a_gross_count_error():
    # with the component one gross count error barely moves the reflection's curve; without it,
    # it dominates
    n_ph, bg, mu, mix, hkl, v = problem(n=4000, n_hkl=100, K=9)
    meta = Meta(torch.arange(len(mu), device="cuda"), hkl)
    bad = n_ph.clone()
    j0 = int((hkl == 0).nonzero()[0])
    bad[j0] = bad[j0] + 20000.0                     # one zinger among the reflection's observations
    shift = {}
    for outlier in (0.0, 0.01):
        obj, scatter = build(1.16, 0.03, outlier)
        scatter.set_prediction(meta.obs_id, mix)
        a = fused(obj, scatter, mu, n_ph, bg, meta, v, 100)
        b = fused(obj, scatter, mu, bad, bg, meta, v, 100)
        # where the corrupted reflection's posterior sits on its nodes (flat prior)
        mean = lambda c: float((torch.softmax(c, 0) * v[0]).sum())
        shift[outlier] = abs(mean(b[0]) - mean(a[0]))
    assert shift[0.01] < 0.2 * shift[0.0], shift
