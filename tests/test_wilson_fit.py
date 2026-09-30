"""Wilson surfaces fitted without the running reference: by the evidence, by an
EM step on a batch's posterior means, and by a batch's counts at the network's scale."""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis, fit_log_sigma_to_counts
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.objective import MarginalNLL, wide_log_grid
from torchsx.nn.posterior_target import PosteriorMergeTarget
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
from tests.test_amortized_training import _synthetic


def _phi(n=600, p=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(n, p, generator=g)
    phi[:, 0] = 1.0
    return phi


def test_counts_fit_recovers_the_surface_exactly():
    phi = _phi()
    c_true = torch.tensor([3.0, -0.4, 0.25, 0.1])
    scale = torch.rand(phi.shape[0], generator=torch.Generator().manual_seed(1)) * 50 + 1
    counts = scale * (phi @ c_true).exp()
    c, used = fit_log_sigma_to_counts(phi, counts, scale)
    torch.testing.assert_close(c, c_true, rtol=0, atol=1e-4)
    assert used <= 8


def test_counts_fit_from_noisy_counts_with_negatives_and_empty_rows():
    g = torch.Generator().manual_seed(2)
    phi = _phi(n=4000)
    c_true = torch.tensor([2.0, 0.3, -0.2, 0.15])
    sigma = (phi @ c_true).exp()
    n_obs = torch.randint(0, 40, (phi.shape[0],), generator=g)
    counts = torch.zeros(phi.shape[0])
    scale = torch.zeros(phi.shape[0])
    I = torch.distributions.Exponential(1.0 / sigma).sample()       # Wilson intensities
    for h in range(phi.shape[0]):
        K = torch.rand(int(n_obs[h]), generator=g) * 2
        counts[h] = (K * I[h] + torch.randn(int(n_obs[h]), generator=g) * 20).sum()
        scale[h] = K.sum()
    assert bool((counts < 0).any()) and bool((scale == 0).any())
    c, _ = fit_log_sigma_to_counts(phi, counts, scale)
    torch.testing.assert_close(c, c_true, rtol=0, atol=0.08)


def test_counts_fit_in_shells_is_the_ratio_of_sums():
    t = PosteriorMergeTarget(torch.ones(6))
    shell = torch.tensor([0, 0, 1, 1, 2, 2])
    counts = torch.tensor([10.0, 30.0, 5.0, -1.0, 0.0, 0.0])
    scale = torch.tensor([1.0, 3.0, 2.0, 2.0, 0.0, 0.0])
    t.set_wilson_sigma_from_counts(counts, scale, shell, 3)
    # the empty shell keeps its value
    torch.testing.assert_close(t.wilson_sigma, torch.tensor([10.0, 10.0, 1.0, 1.0, 1.0, 1.0]))


def _target(I_true, width=0.02):
    n = I_true.numel()
    t = PosteriorMergeTarget(I_true.clone(), k_coarse=31, m_fine=129, n_sigma=8.0)
    t.set_basis(torch.ones(n, 1))                       # log Sigma = one coefficient
    t._loc = (I_true.log(), torch.full((n,), 0.05))
    v = t._loc[0][:, None] + t._loc[1][:, None] * t.u_coarse[None, :]
    S = -0.5 * ((v - I_true.log()[:, None]) / width) ** 2  # sharp likelihood at the true intensity
    return t, S, v


def test_evidence_recovers_the_wilson_mean():
    torch.manual_seed(0)
    I_true = torch.distributions.Exponential(1 / 50.0).sample((4000,)).clamp(min=0.5)
    t, S, v = _target(I_true)
    p = t.enable_evidence_prior()
    mean = float(I_true.mean())
    for x, sign in ((0.8 * mean, 1.0), (1.25 * mean, -1.0)):
        with torch.no_grad():
            p.fill_(math.log(x))
        p.grad = None
        t.integrate(S, v)[0].sum().backward()
        assert float(p.grad) * sign > 0
    with torch.no_grad():
        p.fill_(math.log(0.5 * mean))
    opt = torch.optim.Adam([p], lr=0.02)
    for _ in range(300):
        opt.zero_grad()
        (-t.integrate(S, v)[0].sum()).backward()
        opt.step()
    t.sync_evidence_prior()
    assert abs(float(t.wilson_sigma[0]) / mean - 1) < 0.03
    torch.testing.assert_close(t.wilson_coef, p.detach())


def test_no_grad_passes_read_the_mirrored_prior():
    I_true = torch.linspace(1.0, 100.0, 50)
    t, S, v = _target(I_true)
    p = t.enable_evidence_prior()
    t.sync_evidence_prior()
    with torch.no_grad():
        ref = t.integrate(S, v)[0].clone()
        p.add_(1.0)                                     # trained, not yet mirrored
        torch.testing.assert_close(t.integrate(S, v)[0], ref)
    assert not torch.allclose(t.integrate(S, v)[0].detach(), ref)
    t.sync_evidence_prior()
    torch.testing.assert_close(t.wilson_sigma, torch.full((50,), math.e))


def test_evidence_prior_needs_the_smooth_basis():
    t = PosteriorMergeTarget(torch.ones(5))
    with pytest.raises(ValueError):
        t.enable_evidence_prior()
    assert t.wilson_coef_param is None and "wilson_coef_param" not in t.state_dict()


def test_curve_fit_recovers_the_wilson_mean_from_sharp_curves():
    torch.manual_seed(3)
    I_true = torch.distributions.Exponential(1 / 50.0).sample((4000,)).clamp(min=0.5)
    t, S, v = _target(I_true)
    t.set_wilson_coef(torch.tensor([math.log(10.0)]))          # far from the answer
    used = t.fit_wilson_to_curves(S, t._loc, torch.ones(4000, dtype=torch.bool),
                                  torch.zeros(4000, dtype=torch.long), 1)
    assert 1 <= used < 50
    assert abs(float(t.wilson_sigma[0]) / float(I_true.mean()) - 1) < 0.02


def test_curve_fit_is_the_maximum_of_the_marginal_likelihood():
    """Broad curves: the prior matters, and EM must land on the evidence maximum."""
    torch.manual_seed(4)
    n = 3000
    I_true = torch.distributions.Exponential(1 / 40.0).sample((n,)).clamp(min=0.5)
    I_obs = I_true * torch.exp(0.8 * torch.randn(n))
    t = PosteriorMergeTarget(I_true.clone(), k_coarse=31, m_fine=129, n_sigma=8.0)
    t.set_basis(torch.ones(n, 1))
    loc = (I_obs.log(), torch.full((n,), 0.6))
    v = loc[0][:, None] + loc[1][:, None] * t.u_coarse[None, :]
    S = -0.5 * ((v - I_obs.log()[:, None]) / 0.8) ** 2
    seen = torch.ones(n, dtype=torch.bool)
    t.set_wilson_coef(torch.tensor([math.log(5.0)]))
    t.fit_wilson_to_curves(S, loc, seen, torch.zeros(n, dtype=torch.long), 1, n_iter=2000, tol=1e-7)
    c_em = float(t.wilson_coef[0])

    def evidence(c):
        t.set_wilson_coef(torch.tensor([c]))
        t._loc = loc
        return float(t.integrate(S, None)[0].sum())

    grid = [c_em + d for d in (-0.02, -0.005, 0.0, 0.005, 0.02)]
    values = [evidence(c) for c in grid]
    assert max(range(5), key=values.__getitem__) == 2
    # and the answer differs from the plain mean of the noisy point estimates
    assert abs(math.exp(c_em) / float(I_obs.mean()) - 1) > 0.05


def _trainer(tmp_path, basis=False, **cfg_kw):
    prep, params, *_ = _synthetic(n_hkl=40, n_xtal=24, per=8, n_cond=2)
    scale = AmortizedScale.from_prep(prep, d_model=8, n_layers=1, context_folds=2)
    tr = AmortizedTrainer(
        prep, scale, MarginalNLL(wide_log_grid(n_points=257), Poisson()), params,
        config=TrainConfig(
            device="cpu", outdir=str(tmp_path), heldout_frac=0.25,
            merge_target="posterior", loss="evidence", posterior_prior="wilson",
            posterior_k_coarse=5, posterior_m_fine=17, split_half_every=0, **cfg_kw,
        ),
    )
    if basis:
        tr.set_reciprocal_basis(ReciprocalBasis.from_hkl(prep.hkl_valid, prep.median_cell, "P 1", degree=2))
    else:
        tr._refresh_wilson_sigma()
    return tr


def _step_state0(tr):
    crystals = tr.train_crystals[tr.condition[tr.train_crystals] == 0][:6]
    rec = tr.step(tr.index.rows(crystals))
    assert not rec.get("step_skipped") and rec["n_used"] > 0


@pytest.mark.parametrize("fit,basis", [("batch-em", False), ("batch-em", True),
                                       ("batch-ml", False), ("batch-ml", True),
                                       ("batch-counts", False), ("batch-counts", True)])
def test_batch_fits_move_the_trained_state_only_and_survive_a_refresh(tmp_path, fit, basis):
    tr = _trainer(tmp_path, basis=basis, wilson_fit=fit)
    try:
        before = [t.wilson_sigma.clone() for t in tr.targets]
        _step_state0(tr)
        after = tr.targets[0].wilson_sigma.clone()
        assert not torch.allclose(after, before[0])
        torch.testing.assert_close(tr.targets[1].wilson_sigma, before[1])
        tr._refresh_wilson_sigma()                         # no refit to the running reference
        torch.testing.assert_close(tr.targets[0].wilson_sigma, after)
        if fit == "batch-ml":
            assert tr.targets[0].wilson_em_iters >= 1 and tr.targets[0].last_curves is None
    finally:
        tr.close()


def test_reference_fit_follows_the_running_reference(tmp_path):
    tr = _trainer(tmp_path)
    try:
        _step_state0(tr)
        stepped = tr.targets[0].wilson_sigma.clone()
        tr._refresh_wilson_sigma()
        assert not torch.allclose(tr.targets[0].wilson_sigma, stepped)
    finally:
        tr.close()


def test_evidence_fit_trains_the_coefficients_and_mirrors_them(tmp_path):
    tr = _trainer(tmp_path, basis=True, wilson_fit="evidence")
    try:
        p = tr.targets[0].wilson_coef_param
        start = p.detach().clone()
        _step_state0(tr)
        assert not torch.equal(p.detach(), start)
        torch.testing.assert_close(tr.targets[0].wilson_coef, p.detach())
        mirrored = tr.targets[0].wilson_sigma.clone()
        tr._refresh_wilson_sigma()
        torch.testing.assert_close(tr.targets[0].wilson_sigma, mirrored)
    finally:
        tr.close()


def test_evidence_fit_without_a_basis_and_bad_modes_are_refused(tmp_path):
    tr = _trainer(tmp_path, wilson_fit="evidence")
    try:
        with pytest.raises(RuntimeError):
            _step_state0(tr)
    finally:
        tr.close()
    with pytest.raises(ValueError):
        _trainer(tmp_path, wilson_fit="posterior-means")


def test_merge_restores_a_fitted_surface_on_the_same_reflections(tmp_path):
    from torchsx.cli.merge_nn import _restore_fitted_wilson
    tr = _trainer(tmp_path, basis=True, wilson_fit="batch-counts")
    try:
        coef = [t.wilson_coef.clone() + 0.3 for t in tr.targets]
        ckpt = {"reference_update": {"wilson_fit": "batch-counts"},
                "inference_context": {"hkl": tr.prep.hkl_valid.clone(),
                                      "targets": [{"wilson_coef": c} for c in coef]}}
        _restore_fitted_wilson(tr, tr.prep, ckpt)
        for t, c in zip(tr.targets, coef):
            torch.testing.assert_close(t.wilson_coef, c)
            torch.testing.assert_close(t.wilson_sigma, (t.wilson_phi @ c).exp().clamp(min=t.floor))
        ckpt["inference_context"]["hkl"] = tr.prep.hkl_valid.flip(0)
        with pytest.warns(RuntimeWarning):
            _restore_fitted_wilson(tr, tr.prep, ckpt)
    finally:
        tr.close()
