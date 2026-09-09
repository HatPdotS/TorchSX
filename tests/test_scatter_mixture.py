"""Mixture of two-width GenGauss components: density, mean-one centring, moments, addressing."""
import math

import pytest
import torch

from torchsx.merging.scatter.gengauss import gengauss_log_prob_from_obs, log_e_s_gengauss
from torchsx.merging.scatter.mixture import (
    PredictedMixtureScatter, mixture_log_e_s, mixture_log_sd, split_mix, two_width_moments,
)


class _Meta:
    def __init__(self, obs_id):
        self.obs_id = obs_id


def _params(n, K, seed=0, spread=True):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, K, generator=g) if spread else torch.zeros(n, K)
    log_pi = torch.log_softmax(logits, dim=1)
    offset = (torch.randn(n, K, generator=g) * 0.6) if spread else torch.zeros(n, K)
    la_lo = -0.6 + 0.4 * torch.randn(n, K, generator=g)
    la_hi = la_lo + 0.5 * torch.randn(n, K, generator=g)
    beta = 1.4 + 1.2 * torch.rand(n, generator=g)      # tails light enough for the u_max grid of log_e_s
    return log_pi, offset, la_lo, la_hi, beta


def _mix(log_pi, offset, la_lo, la_hi):
    return torch.cat([log_pi, offset, la_lo, la_hi], dim=1)


def test_single_component_reduces_to_the_two_width_family():
    n = 40
    log_pi, offset, la_lo, la_hi, beta = _params(n, 1, spread=False)
    sc = PredictedMixtureScatter(1)
    ids = torch.arange(n)
    sc.set_prediction(ids, beta.log(), _mix(log_pi, offset, la_lo, la_hi))
    eps = torch.linspace(-12, 12, 4001)[None, :].expand(n, -1)
    got = sc.log_prob(eps, None, _Meta(ids))
    ref = gengauss_log_prob_from_obs(eps, la_lo[:, 0], beta.log(), la_hi[:, 0])
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        mixture_log_e_s(log_pi, offset, la_lo, la_hi, beta),
        log_e_s_gengauss(la_lo[:, 0], beta, log_alpha_hi=la_hi[:, 0]), atol=1e-6, rtol=0)


def test_negligible_extra_components_leave_the_density_unchanged():
    n, K = 30, 3
    log_pi, offset, la_lo, la_hi, beta = _params(n, K)
    log_pi = torch.log_softmax(torch.tensor([[0.0, -40.0, -40.0]]).expand(n, K), dim=1)
    sc = PredictedMixtureScatter(K); ids = torch.arange(n)
    sc.set_prediction(ids, beta.log(), _mix(log_pi, offset, la_lo, la_hi))
    eps = torch.linspace(-8, 8, 2001)[None, :].expand(n, -1)
    got = sc.log_prob(eps, None, _Meta(ids))
    ref = gengauss_log_prob_from_obs(eps - offset[:, :1], la_lo[:, 0], beta.log(), la_hi[:, 0])
    # identical wherever the dominant component carries any mass; far out in its
    # tail the negligible components take over, as a mixture must
    m = ref > -25.0
    assert m.float().mean() > 0.3
    torch.testing.assert_close(got[m], ref[m], atol=1e-5, rtol=0)
    assert (got >= ref - 1e-6).all()


def test_mixture_density_is_normalised_and_mean_one_after_centring():
    n, K = 24, 3
    log_pi, offset, la_lo, la_hi, beta = _params(n, K, seed=3)
    sc = PredictedMixtureScatter(K); ids = torch.arange(n)
    sc.set_prediction(ids, beta.log(), _mix(log_pi, offset, la_lo, la_hi))
    eps = torch.linspace(-25, 25, 50001, dtype=torch.float64)
    lp = sc.log_prob(eps[None, :].expand(n, -1).float(), None, _Meta(ids)).double()
    d = float(eps[1] - eps[0])
    mass = torch.logsumexp(lp, dim=1).exp() * d
    torch.testing.assert_close(mass, torch.ones(n, dtype=torch.float64), atol=2e-3, rtol=0)
    # E[exp(eps)] by brute force equals the closed-form correction, and centring
    # the mixture by it makes the mean multiplicative factor one
    log_E = mixture_log_e_s(log_pi, offset, la_lo, la_hi, beta).double()
    brute = torch.logsumexp(lp + eps[None, :], dim=1) + math.log(d)
    torch.testing.assert_close(brute, log_E, atol=5e-3, rtol=0)
    centred = torch.logsumexp(lp + eps[None, :] - log_E[:, None], dim=1) + math.log(d)
    torch.testing.assert_close(centred.exp(), torch.ones(n, dtype=torch.float64), atol=5e-3, rtol=0)


def _sample_two_width(la_lo, la_hi, beta, n_samples, g):
    """Draw from a two-width GenGauss centred at zero: pick the side by its mass,
    then |eps| = alpha_side * G^(1/beta) with G ~ Gamma(1/beta, 1)."""
    a_lo, a_hi = la_lo.exp(), la_hi.exp()
    p_hi = a_hi / (a_lo + a_hi)
    side_hi = torch.rand(n_samples, generator=g) < p_hi
    gam = torch.distributions.Gamma(torch.full((n_samples,), 1.0 / float(beta)), torch.ones(n_samples))
    mag = gam.sample() ** (1.0 / float(beta))
    return torch.where(side_hi, a_hi * mag, -a_lo * mag)


def test_moments_and_mixture_sd_match_monte_carlo():
    g = torch.Generator().manual_seed(11)
    torch.manual_seed(11)
    la_lo, la_hi, beta = torch.tensor([-0.3]), torch.tensor([0.4]), torch.tensor([1.7])
    x = _sample_two_width(la_lo, la_hi, beta, 2_000_000, g)
    m1, m2 = two_width_moments(la_lo, la_hi, beta)
    assert abs(float(m1) - float(x.mean())) < 0.01
    assert abs(float(m2) - float((x * x).mean())) < 0.02 * float(m2)
    # mixture SD: three components with weights, offsets and widths
    log_pi = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    offset = torch.tensor([[0.0, -1.2, 0.5]])
    lo = torch.tensor([[-0.3, 0.1, -0.8]]); hi = torch.tensor([[0.4, 0.1, -0.5]])
    comp = torch.multinomial(log_pi.exp()[0], 3_000_000, replacement=True, generator=g)
    draws = torch.empty(3_000_000)
    for k in range(3):
        m = comp == k
        draws[m] = _sample_two_width(lo[:, k], hi[:, k], beta, int(m.sum()), g) + float(offset[0, k])
    sd = mixture_log_sd(log_pi, offset, lo, hi, beta).exp()
    assert abs(float(sd) - float(draws.std())) < 0.01 * float(draws.std())


def test_addressing_dominant_component_and_split():
    n, K = 20, 3
    log_pi, offset, la_lo, la_hi, beta = _params(n, K, seed=5)
    mix = _mix(log_pi, offset, la_lo, la_hi)
    parts = split_mix(mix, K)
    assert all(p.shape == (n, K) for p in parts) and torch.equal(parts[1], offset)
    sc = PredictedMixtureScatter(K); ids = torch.arange(100, 100 + n)
    sc.set_prediction(ids, beta.log(), mix)
    sub = ids[::3]
    comps = sc.components(_Meta(sub))
    assert len(comps) == K
    torch.testing.assert_close(comps[1][0], log_pi[::3, 1]); torch.testing.assert_close(comps[2][1], offset[::3, 2])
    la, lb = sc.log_alpha_beta_obs(None, _Meta(sub))
    k = log_pi[::3].argmax(dim=1)
    torch.testing.assert_close(la, la_lo[::3][torch.arange(len(k)), k]); torch.testing.assert_close(lb, beta.log()[::3])
    torch.testing.assert_close(sc.log_alpha_hi_obs(None, _Meta(sub)), la_hi[::3][torch.arange(len(k)), k])
    # the child views resolve the same observations
    a, b = comps[0][2].log_alpha_beta_obs(None, _Meta(sub))
    torch.testing.assert_close(a, la_lo[::3, 0]); torch.testing.assert_close(b, beta.log()[::3])
    with pytest.raises(KeyError):
        sc.components(_Meta(torch.tensor([5])))
    sc.clear_prediction()
    assert not sc.has_prediction and not comps[0][2].has_prediction
    with pytest.raises(ValueError):
        split_mix(mix, 2)


# ---------------------------------------------------------------- heads and scale model

def test_mixture_heads_start_at_the_single_family_and_expose_components():
    from torchsx.nn.heads import MixtureConditioner, ScaleHeads
    torch.manual_seed(2)
    z = torch.randn(64, 16)
    base = ScaleHeads(16, w_init=1.65, beta_init=1.0, predict_skew=True)(z)
    heads = ScaleHeads(16, w_init=1.65, beta_init=1.0, predict_skew=True, scatter_family="mixture", n_components=3)
    p = heads(z)
    assert p.mix is not None and p.mix.shape == (64, 12) and heads.n_components == 3
    log_pi, offset, la_lo, la_hi = split_mix(p.mix, 3)
    torch.testing.assert_close(log_pi.exp().sum(1), torch.ones(64), atol=1e-5, rtol=0)
    assert (log_pi[:, 0] > math.log(0.99)).all()                   # component 0 carries the mass at init
    torch.testing.assert_close(la_lo[:, 0], base.log_alpha); torch.testing.assert_close(la_hi[:, 0], base.log_alpha_hi)
    torch.testing.assert_close(p.log_alpha, base.log_alpha); torch.testing.assert_close(p.log_alpha_hi, base.log_alpha_hi)
    # mean correction and reported width within a few thousandths of the single family
    assert (p.log_e_s - base.log_e_s).abs().max() < 5e-3
    assert (p.log_w - base.log_w).abs().max() < 2e-2
    # zero-init conditioner: the raw output is exactly the bias, independent of the input
    cond = MixtureConditioner(3)
    logits, offset, d_lo, d_hi = cond(torch.randn(10), torch.rand(10) + 1, torch.randn(10))
    bias = MixtureConditioner.init_bias(3)
    torch.testing.assert_close(logits, bias[:3].expand(10, -1)); torch.testing.assert_close(offset, bias[3:6].expand(10, -1))
    torch.testing.assert_close(d_lo, bias[6:9].expand(10, -1))
    assert (d_hi[:, 0] == 0).all() and (d_hi[:, 1:] < 0).all()      # extra components only narrow the upper side
    # the single family is untouched by the new field
    assert base.mix is None and base.detach().mix is None and p.detach().mix is not None


def test_mixture_scale_model_primes_the_mixture_scatter_and_chunks_exactly():
    from tests.test_amortized_training import _synthetic
    from torchsx.merging.scale import AmortizedScale
    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=24, per=12)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, include_hkl=False, predict_shape=False,
                                     beta_init=1.0, w_init=1.65, predict_skew=True, scatter_family="mixture", n_components=3)
    assert isinstance(scale.scatter, PredictedMixtureScatter) and scale.scatter.n_components == 3
    # break the zero init so chunking has something non-trivial to reproduce
    with torch.no_grad():
        for prm in scale.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    # the trainer feeds crystal-grouped rows and chunking requires that order
    order = torch.argsort(prep.obs_info.crystal_id, stable=True)
    meta = prep.obs_info.gather(order)
    pred = scale.prime(meta)
    assert scale.scatter.has_prediction and pred.mix is not None
    comps = scale.scatter.components(meta)
    assert len(comps) == 3 and all(v.has_prediction for _, _, v in comps)
    torch.testing.assert_close(comps[0][0], split_mix(pred.mix, 3)[0][:, 0])
    scale.clear()
    assert not scale.scatter.has_prediction
    full = scale.predict(meta)
    scale.set_encoder_chunking(5)
    chunked = scale.predict(meta)
    for name in ("log_K", "log_w", "beta", "log_alpha", "log_e_s", "log_alpha_hi", "mix"):
        torch.testing.assert_close(getattr(chunked, name), getattr(full, name), atol=1e-5, rtol=1e-5)
    # gradients reach the conditioner
    scale.set_encoder_chunking(0)
    out = scale.predict(meta)
    (out.log_e_s.sum() + out.mix.sum()).backward()
    assert scale.heads.mixture.l2.weight.grad is not None and scale.heads.mixture.l2.weight.grad.abs().sum() > 0
