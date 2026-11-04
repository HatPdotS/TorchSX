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

    def slice(self, s, e):
        return _Meta(self.obs_id[s:e])


def _params(n, K, seed=0, spread=True):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, K, generator=g) if spread else torch.zeros(n, K)
    log_pi = torch.log_softmax(logits, dim=1)
    offset = (torch.randn(n, K, generator=g) * 0.6) if spread else torch.zeros(n, K)
    la_lo = -0.6 + 0.4 * torch.randn(n, K, generator=g)
    la_hi = la_lo + 0.5 * torch.randn(n, K, generator=g)
    beta = 1.4 + 1.2 * torch.rand(n, generator=g)      # tails light enough for the u_max grid of log_e_s
    return log_pi, offset, la_lo, la_hi, beta


def _mix(log_pi, offset, la_lo, la_hi, beta=None):
    """5-block mixture parameters; ``beta`` (n,) shared or (n, K) per component."""
    if beta is None:
        lb = torch.zeros_like(log_pi)
    else:
        lb = (beta[:, None].expand_as(log_pi) if beta.dim() == 1 else beta).log()
    return torch.cat([log_pi, offset, la_lo, la_hi, lb], dim=1)


def test_single_component_reduces_to_the_two_width_family():
    n = 40
    log_pi, offset, la_lo, la_hi, beta = _params(n, 1, spread=False)
    sc = PredictedMixtureScatter(1)
    ids = torch.arange(n)
    sc.set_prediction(ids, _mix(log_pi, offset, la_lo, la_hi, beta))
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
    sc.set_prediction(ids, _mix(log_pi, offset, la_lo, la_hi, beta))
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
    sc.set_prediction(ids, _mix(log_pi, offset, la_lo, la_hi, beta))
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
    mix = _mix(log_pi, offset, la_lo, la_hi, beta)
    parts = split_mix(mix, K)
    assert len(parts) == 5 and all(p.shape == (n, K) for p in parts) and torch.equal(parts[1], offset)
    sc = PredictedMixtureScatter(K); ids = torch.arange(100, 100 + n)
    sc.set_prediction(ids, mix)
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

def test_mixture_heads_start_at_their_biases():
    from torchsx.nn.heads import ScaleHeads
    torch.manual_seed(2)
    z = torch.randn(64, 16)
    heads = ScaleHeads(16, w_init=1.65, n_components=3)
    assert heads.mixture.out_features == 3 + 3 * 2                 # weights, then offsets and widths of 1..K-1
    p = heads(z)
    assert p.mix.shape == (64, 15)
    log_pi, offset, la_lo, la_hi, lb = split_mix(p.mix, 3)
    torch.testing.assert_close(log_pi.exp().sum(1), torch.ones(64), atol=1e-5, rtol=0)
    torch.testing.assert_close(lb.exp(), torch.full((64, 3), 2.0))   # Gaussian components
    assert (log_pi[:, 0] > math.log(0.85)).all()                     # component 0 carries most of the mass
    assert (offset[:, 0] == 0).all()
    la0 = math.log(1.65) + 0.5 * math.log(2.0)                       # alpha = w sqrt(2)
    torch.testing.assert_close(la_lo[:, 0], torch.full((64,), la0), atol=1e-5, rtol=0)
    torch.testing.assert_close(la_hi[:, 0], la_lo[:, 0])             # component 0 is symmetric
    assert (la_hi[:, 1:] < la_hi[:, :1]).all()                       # the others start narrower above
    # zero-init weights: every observation gets the biases, whatever its embedding
    torch.testing.assert_close(p.mix, p.mix[:1].expand_as(p.mix))
    assert p.detach().mix is not None


def test_extra_components_may_be_wider_above_than_component_0():
    from torchsx.nn.heads import ScaleHeads
    torch.manual_seed(0)
    heads = ScaleHeads(16, n_components=3)
    with torch.no_grad():
        heads.mixture.weight.normal_(0.0, 1.0)
    la_hi = split_mix(heads(torch.randn(200, 16)).mix, 3)[3]
    assert bool((la_hi[:, 1:] > la_hi[:, :1]).any())


def test_mixture_scale_model_primes_the_mixture_scatter():
    from tests.test_amortized_training import _synthetic
    from torchsx.merging.scale import AmortizedScale
    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=24, per=12)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_components=3)
    assert isinstance(scale.scatter, PredictedMixtureScatter) and scale.scatter.n_components == 3
    # break the zero init
    with torch.no_grad():
        for prm in scale.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    # crystal-grouped rows, as the trainer feeds them
    order = torch.argsort(prep.obs_info.crystal_id, stable=True)
    meta = prep.obs_info.gather(order)
    pred = scale.prime(meta)
    assert scale.scatter.has_prediction and pred.mix is not None
    comps = scale.scatter.components(meta)
    assert len(comps) == 3 and all(v.has_prediction for _, _, v in comps)
    torch.testing.assert_close(comps[0][0], split_mix(pred.mix, 3)[0][:, 0])
    assert pred.mix.shape[1] == 15
    scale.clear()
    assert not scale.scatter.has_prediction
    # gradients reach the conditioner
    out = scale.predict(meta)
    (out.log_e_s.sum() + out.mix.sum()).backward()
    assert scale.heads.mixture.weight.grad is not None and scale.heads.mixture.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------- objective

def _split_grid_setup(n, K, seed):
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
    from torchsx.nn.objective import MarginalNLL
    from tests.test_split_log_grid import _batch
    n_ph, bg, la, lb, _mu = _batch("cpu", seed=seed, M=n)[:5]
    log_pi, offset, la_lo, la_hi, _ = _params(n, K, seed=seed)
    la_lo = la[:, None] + 0.3 * torch.randn(n, K, generator=torch.Generator().manual_seed(seed + 1))
    la_hi = la_lo - 0.4 * torch.rand(n, K, generator=torch.Generator().manual_seed(seed + 2))
    obj = MarginalNLL(SplitLogGrid(96), Poisson(bg_var_factor=0.2), miss_frac=0.03)
    return obj, n_ph, bg, lb, (log_pi, offset, la_lo, la_hi)


def test_objective_mixture_marginal_reduces_and_combines():
    import torchsx.merging.quadrature.split_log_grid as slg
    from torchsx.merging.scatter.predicted import PredictedGenGaussScatter
    slg._DISABLE_TRITON = True
    try:
        n, K = 512, 3
        obj, n_ph, bg, lb, (log_pi, offset, la_lo, la_hi) = _split_grid_setup(n, K, seed=4)
        ids = torch.arange(n); meta = _Meta(ids)
        log_mean = torch.log(n_ph.clamp(min=1.0)) + 0.2 * torch.randn(n, generator=torch.Generator().manual_seed(9))
        mix_sc = PredictedMixtureScatter(K)
        # one component at full weight: identical to the single two-width family shifted by its offset
        one = torch.log_softmax(torch.tensor([[0.0, -60.0, -60.0]]).expand(n, K), dim=1)
        mix_sc.set_prediction(ids, _mix(one, offset, la_lo, la_hi, lb.exp()))
        got = obj.log_prob(mix_sc, None, log_mean, n_ph, bg, meta)
        single = PredictedGenGaussScatter(); single.set_prediction(ids, la_lo[:, 0], lb, la_hi[:, 0])
        ref = obj.log_prob(single, None, log_mean + offset[:, 0], n_ph, bg, meta)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-5)
        # general weights: the mixture marginal is the weighted combination of per-component marginals
        mix_sc.set_prediction(ids, _mix(log_pi, offset, la_lo, la_hi, lb.exp()))
        got = obj.log_prob(mix_sc, None, log_mean, n_ph, bg, meta)
        parts = []
        for k in range(K):
            sk = PredictedGenGaussScatter(); sk.set_prediction(ids, la_lo[:, k], lb, la_hi[:, k])
            lp_k = obj._marginal(sk, None, log_mean + offset[:, k], n_ph, bg, meta, (0, n))
            parts.append(log_pi[:, k] + lp_k)
        ref = obj._mix_miss(torch.logsumexp(torch.stack(parts), dim=0), n_ph, bg)
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-6)
        assert torch.isfinite(got).all()
        # chunked evaluation agrees
        obj.chunk_size = 100
        torch.testing.assert_close(obj.log_prob(mix_sc, None, log_mean, n_ph, bg, meta), ref, atol=1e-5, rtol=1e-6)
    finally:
        slg._DISABLE_TRITON = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton path needs CUDA")
def test_objective_mixture_triton_matches_eager():
    import torchsx.merging.quadrature.split_log_grid as slg
    n, K = 2048, 3
    obj, n_ph, bg, lb, (log_pi, offset, la_lo, la_hi) = _split_grid_setup(n, K, seed=5)
    dev = torch.device("cuda")
    ids = torch.arange(n, device=dev); meta = _Meta(ids)
    log_mean = (torch.log(n_ph.clamp(min=1.0)) + 0.1).to(dev)
    sc = PredictedMixtureScatter(K)
    mix = _mix(log_pi, offset, la_lo, la_hi).to(dev).requires_grad_(True)
    sc.set_prediction(ids, mix)
    obj = obj.to(dev)
    fast = obj.log_prob(sc, None, log_mean, n_ph.to(dev), bg.to(dev), meta)
    g_fast, = torch.autograd.grad(fast.sum(), mix)
    slg._DISABLE_TRITON = True
    try:
        sc.set_prediction(ids, mix)
        slow = obj.log_prob(sc, None, log_mean, n_ph.to(dev), bg.to(dev), meta)
        g_slow, = torch.autograd.grad(slow.sum(), mix)
    finally:
        slg._DISABLE_TRITON = False
    torch.testing.assert_close(fast, slow, atol=3e-2, rtol=2e-3)
    torch.testing.assert_close(g_fast, g_slow, atol=5e-2, rtol=5e-2)


def test_mixture_trainer_step_runs_and_trains_the_mixture_head(tmp_path):
    from tests.test_amortized_training import _synthetic
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
    from torchsx.merging.scale import AmortizedScale
    from torchsx.nn.objective import MarginalNLL
    from torchsx.cli.train import run_updates
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
    torch.manual_seed(0)
    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=24, per=16)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_components=3)
    obj = MarginalNLL(SplitLogGrid(n_points=64), Poisson(bg_var_factor=0.1), miss_frac=0.03)
    cfg = TrainConfig(max_obs_per_batch=6000, lr=1e-2, log_every=10 ** 9,
                      heldout_frac=0.0, posterior_k_coarse=9, posterior_m_fine=33,
                      device="cpu", outdir=str(tmp_path))
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    before = scale.heads.mixture.weight.detach().clone()
    run_updates(tr, 1)
    assert not torch.equal(scale.heads.mixture.weight.detach(), before), "mixture head did not move"
    res = tr.merge(estimator="posterior")
    assert torch.isfinite(res["F"]).all() and torch.isfinite(res["I0"]).all()
    pooled = tr.merge(estimator="pooled")
    assert torch.isfinite(pooled["I"]).all()



# ---------------------------------------------------------------- CLI and checkpoint

def test_mixture_round_trips_through_the_checkpoint(tmp_path):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.merge_nn import _build_parser as merge_parser
    from torchsx.cli.train import _build_parser as train_parser
    from tests.test_amortized_training import _synthetic
    from tests.test_cli_pipeline import _trainer_from_stack

    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    recipe = {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0, "radii": [7.0, 9.0, 11.0]}
    bundle = P.PrepBundle(prep, recipe, [], 0.1)
    targs = train_parser().parse_args(["x.sfx", "-o", str(tmp_path), "--device", "cpu"])
    prep_, params_, scale, obj, S = P.build_stack(bundle, targs, verbose=False)
    # the production model: four Gaussian components
    assert S["scatter_components"] == 4 and scale.heads.n_components == 4
    assert isinstance(scale.scatter, PredictedMixtureScatter) and scale.scatter.fixed_gaussian
    tr = _trainer_from_stack(prep_, params_, scale, obj, S)
    tr.data_recipe = dict(recipe)
    ck = torch.load(tr.save_best(tmp_path / "mix.pt"), weights_only=False)
    assert ck["scatter"]["n_components"] == 4
    # merge side rebuilds the model from the checkpoint with no flags
    margs = merge_parser().parse_args(["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu"])
    p2, q2, rebuilt, o2, S2 = P.build_stack(bundle, margs, ckpt=ck, verbose=False)
    assert rebuilt.heads.n_components == 4
    rebuilt.load_state_dict(ck["state_dict"])
    _trainer_from_stack(p2, q2, rebuilt, o2, S2)      # installs the gauge basis, as merge_nn does
    scale.eval(), rebuilt.eval()
    rows = tr.index.rows(tr.index.nonempty)          # crystal-grouped, as the encoder needs
    with torch.no_grad():
        a, b = scale.predict(prep_.obs_info.gather(rows)), rebuilt.predict(p2.obs_info.gather(rows))
    torch.testing.assert_close(b.log_K, a.log_K)
    torch.testing.assert_close(b.mix, a.mix)
    # a checkpoint that does not record its model is refused, not guessed
    del ck["scatter"]["n_components"]
    with pytest.raises(SystemExit, match="does not record 'n_components'"):
        P.build_stack(bundle, margs, ckpt=ck, verbose=False)


def test_heads_give_a_normalised_mixture_with_the_closed_form_mean():
    from torchsx.nn.heads import ScaleHeads
    torch.manual_seed(3)
    z = torch.randn(40, 16)
    heads = ScaleHeads(16, w_init=1.65, n_components=3)
    with torch.no_grad():            # moderate widths: the brute-force grid must hold the tails
        heads.mixture.weight.normal_(0.0, 0.05)
    p = heads(z)
    sc = PredictedMixtureScatter(3, fixed_gaussian=True); ids = torch.arange(40); sc.set_prediction(ids, p.mix.detach())
    eps = torch.linspace(-25, 25, 40001, dtype=torch.float64)
    lp = sc.log_prob(eps[None, :].expand(40, -1).float(), None, _Meta(ids)).double(); d = float(eps[1] - eps[0])
    torch.testing.assert_close((torch.logsumexp(lp, 1).exp() * d), torch.ones(40, dtype=torch.float64), atol=3e-3, rtol=0)
    brute = torch.logsumexp(lp + eps[None, :], dim=1) + math.log(d)
    torch.testing.assert_close(brute, p.log_e_s.detach().double(), atol=5e-3, rtol=0)


