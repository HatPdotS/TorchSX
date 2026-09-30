"""The hierarchical scatter-shape prior: a KL between densities, pulling toward the population."""
import math

import torch

from torchsx.merging.scatter.mixture import mixture_log_e_s
from torchsx.nn.shape_prior import U_HI, U_LO, mixture_log_density, pack_single, shape_prior_kl


def _gauss_mix(n, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    log_pi = torch.log_softmax(torch.randn(n, K, generator=g, dtype=torch.float64), 1)
    offset = torch.randn(n, K, generator=g, dtype=torch.float64) * 0.5
    offset[:, 0] = 0
    lo = torch.rand(n, K, generator=g, dtype=torch.float64) * 1.5
    hi = lo - torch.rand(n, K, generator=g, dtype=torch.float64)
    beta = torch.full_like(lo, 2.0)
    return torch.cat([log_pi, offset, lo, hi, beta.log()], 1)


def _log_e_s(mix, K):
    b = K
    return mixture_log_e_s(mix[:, :b], mix[:, b:2 * b], mix[:, 2 * b:3 * b], mix[:, 3 * b:4 * b],
                           mix[:, 4 * b:].exp(), gaussian=True)


def test_density_integrates_to_one():
    mix = _gauss_mix(50, 3)
    u = torch.linspace(U_LO, U_HI, 5501, dtype=torch.float64)
    p = mixture_log_density(mix, u[None, :].expand(50, -1)).exp()
    torch.testing.assert_close(torch.trapezoid(p, u, dim=1), torch.ones(50, dtype=torch.float64), atol=1e-6, rtol=0)


def test_zero_when_every_observation_shares_one_shape():
    one = _gauss_mix(1, 3)
    mix = one.expand(200, -1).clone()
    kl, _, _ = shape_prior_kl(mix, _log_e_s(mix, 3))
    assert abs(float(kl)) < 1e-9


def test_independent_of_the_parametrisation():
    # the same densities written with one component, and with that component
    # split into two identical halves, give the same prior
    g = torch.Generator().manual_seed(1)
    la = torch.rand(300, generator=g, dtype=torch.float64) * 1.2
    one = pack_single(la, torch.full_like(la, 2.0), la - 0.4)
    two = torch.cat([torch.full((300, 2), -math.log(2.0), dtype=torch.float64),
                     torch.zeros(300, 2, dtype=torch.float64),
                     one[:, 2:3].expand(-1, 2), one[:, 3:4].expand(-1, 2), one[:, 4:5].expand(-1, 2)], 1)
    kl1, _, _ = shape_prior_kl(one, _log_e_s(one, 1))
    kl2, _, _ = shape_prior_kl(two, _log_e_s(two, 2))
    torch.testing.assert_close(kl1, kl2, atol=1e-10, rtol=0)
    assert float(kl1) > 1e-3


def test_pulls_an_outlier_toward_the_population():
    la = torch.full((100,), math.log(1.2), dtype=torch.float64)
    la[0] = math.log(4.0)                              # one very wide observation
    la = la.requires_grad_()
    mix = pack_single(la, torch.full_like(la, 2.0))
    kl, sd_bar, _ = shape_prior_kl(mix, _log_e_s(mix, 1))
    kl.backward()
    assert la.grad[0] > 0                              # descending narrows the outlier
    assert (la.grad[1:] < 0).all()                     # and widens the rest toward p_bar, which holds it
    # p_bar: 0.99 N(-0.36, 0.72) + 0.01 N(-4, 8) in u, SD 0.96
    assert abs(sd_bar - 0.961) < 0.01



def test_mass_moved_off_the_grid_is_penalised():
    # the loophole: one component far wider than the grid, carrying 25 % of the weight
    base = pack_single(torch.full((200,), math.log(1.5), dtype=torch.float64),
                       torch.full((200,), 2.0, dtype=torch.float64), torch.full((200,), math.log(1.0), dtype=torch.float64))
    two = torch.cat([torch.zeros(200, 2, dtype=torch.float64), torch.zeros(200, 2, dtype=torch.float64),
                     base[:, 2:3].expand(-1, 2).clone(), base[:, 3:4].expand(-1, 2).clone(),
                     base[:, 4:5].expand(-1, 2).clone()], 1)
    two[:, :2] = torch.log(torch.tensor([1.0, 1e-9], dtype=torch.float64))
    parked = two.clone()
    parked[:100, :2] = torch.log(torch.tensor([0.75, 0.25], dtype=torch.float64))
    parked[:100, 3] = -4.0                                  # offset of component 1
    parked[:100, 5] = math.log(1000.0)                      # lower width of component 1
    kl, _, off = shape_prior_kl(parked, _log_e_s(parked, 2))
    assert 0.08 < off < 0.14                                # half the observations park ~0.24 off the grid
    assert float(kl) > 0.05
    # narrow components are resolved: nothing leaks into the overflow bin
    narrow = pack_single(torch.full((50,), math.log(0.25), dtype=torch.float64), torch.full((50,), 2.0, dtype=torch.float64))
    _, _, off = shape_prior_kl(narrow, _log_e_s(narrow, 1))
    assert off < 1e-6

def test_float32_is_finite_for_extreme_widths():
    mix = _gauss_mix(64, 4).float()
    mix[:4, 8:12] = 12.0                               # lower widths ~ 1.6e5
    mix[4:8, 12:16] = -8.0                             # upper widths ~ 3e-4
    mix = mix.requires_grad_()
    kl, _, _ = shape_prior_kl(mix, _log_e_s(mix, 4))
    kl.backward()
    assert torch.isfinite(kl) and torch.isfinite(mix.grad).all()


def test_trainer_adds_the_term_and_logs_it(tmp_path):
    from tests.test_amortized_training import _synthetic
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
    from torchsx.merging.scale import AmortizedScale
    from torchsx.nn.objective import MarginalNLL
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
    torch.manual_seed(0)
    prep, params, *_ = _synthetic(n_hkl=30, n_xtal=24, per=16)
    for family in ("mixture", "gengauss"):
        scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, include_hkl=False, predict_shape=False,
                                         beta_init=1.0, w_init=1.65, predict_skew=True, scatter_family=family,
                                         n_components=3)
        obj = MarginalNLL(SplitLogGrid(n_points=64), Poisson(bg_var_factor=0.1), miss_frac=0.03)
        cfg = TrainConfig(max_obs_per_batch=6000, epochs=1, lr=1e-2,
                          log_every=10 ** 9, heldout_frac=0.0,
                          posterior_k_coarse=9, posterior_m_fine=33, device="cpu", outdir=str(tmp_path / family),
                          split_half_every=0, lam_shape_prior=0.3, shape_prior_sample=256)
        tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
        rec = tr.step(tr.index.rows(tr.train_crystals[:8]))
        # zero-initialised heads: every observation starts at one shape, no prior cost
        assert rec["shape_prior_kl"] < 1e-9 and rec["loss_total"] == rec["nll_data"]
        with torch.no_grad():
            for name, prm in scale.heads.named_parameters():
                if name.endswith("weight"):
                    prm.normal_(0.0, 0.3)
        rec = tr.step(tr.index.rows(tr.train_crystals[:8]))
        assert rec["shape_prior_kl"] > 1e-4 and rec["shape_prior_sd_bar"] > 0
        assert rec["loss_total"] > rec["nll_data"]
