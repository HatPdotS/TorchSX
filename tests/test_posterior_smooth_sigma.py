"""Smooth Wilson Sigma(h) in PosteriorMergeTarget and its installation through the trainer."""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis
from torchsx.nn.posterior_target import PosteriorMergeTarget

CELL = torch.tensor([62.23, 62.23, 110.77, 90.0, 90.0, 120.0])


def _wilson_target(seed=0, n=3000, degree=4):
    g = torch.Generator().manual_seed(seed)
    hkl = torch.unique(torch.randint(-20, 21, (n, 3), generator=g), dim=0)
    basis = ReciprocalBasis.from_hkl(hkl, CELL, "P 63", degree=degree)
    truth = torch.zeros(basis.p, dtype=torch.float64)
    truth[0], truth[1], truth[-1] = 5.0, -1.0, -0.5
    sigma = (basis.phi.double() @ truth).exp()
    I = -torch.log(torch.rand(len(sigma), generator=g, dtype=torch.float64)) * sigma
    t = PosteriorMergeTarget(I.float(), k_coarse=31, m_fine=129, n_sigma=6.0, prior="wilson")
    return t, basis, truth, sigma


def test_the_start_fit_recovers_the_smooth_sigma():
    t, basis, truth, sigma = _wilson_target()
    assert not t.has_basis
    t.set_basis(basis.phi)
    assert t.has_basis and t.wilson_coef.shape == (basis.p,)
    t.fit_wilson_surface()                                          # to the start intensities
    err = (t.wilson_sigma.double().log() - sigma.log()).abs().mean()
    assert float(err) < 0.06
    assert 1 <= t.wilson_fit_iters <= 8
    assert abs(float(t.wilson_coef[-1]) - float(truth[-1])) < 0.1
    # Sigma is a function of the basis only: symmetry mates get the same value.
    from torchref.symmetry import SpaceGroup
    sg = SpaceGroup("P 63")
    hkl = torch.unique(torch.randint(-20, 21, (3000, 3), generator=torch.Generator().manual_seed(0)), dim=0)
    mates = sg.expand_reciprocal(hkl.to(sg.device)).cpu().long()[1]
    ls_ref = basis.evaluate(hkl).double() @ t.wilson_coef.double()
    ls_mate = basis.evaluate(mates).double() @ t.wilson_coef.double()
    assert torch.allclose(ls_ref, ls_mate, atol=1e-3)
    assert torch.allclose(ls_ref.exp().float().clamp(min=t.floor), t.wilson_sigma, rtol=1e-4, atol=1e-6)


def test_the_surface_needs_the_basis():
    t, *_ = _wilson_target()
    with pytest.raises(ValueError):
        t.fit_wilson_surface()


def test_integrate_depends_on_sigma_only_through_the_buffer():
    a, basis, _, _ = _wilson_target(seed=1)
    b, _, _, _ = _wilson_target(seed=1)
    b.set_basis(basis.phi)
    b.fit_wilson_surface()
    a.wilson_sigma.copy_(b.wilson_sigma)
    loc = (a.I_start.log(), torch.full((a.n_hkl,), 0.5))
    S = -0.5 * (a.u_coarse[None, :] / 2.0) ** 2 + torch.randn(a.n_hkl, 1) * 0.1
    out_a = a.integrate(S, loc)
    out_b = b.integrate(S, loc)
    for x, y in zip(out_a, out_b):
        assert torch.allclose(x, y)


def test_state_dict_round_trip_carries_basis_buffers():
    t, basis, _, _ = _wilson_target(seed=2)
    t.set_basis(basis.phi)
    t.fit_wilson_surface()
    sd = t.state_dict()
    assert sd["wilson_phi"].shape == basis.phi.shape and sd["wilson_coef"].shape == (basis.p,)
    fresh, _, _, _ = _wilson_target(seed=2)
    fresh.set_basis(basis.phi)
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.wilson_coef, t.wilson_coef)


def test_trainer_installs_basis_on_targets_and_replaces_the_default():
    from tests.test_amortized_training import _synthetic
    from torchsx.merging.config import MergeConfig
    from torchsx.merging.likelihood import Poisson
    from torchsx.nn.objective import wide_log_grid
    from torchsx.merging.scale.amortized import AmortizedScale
    from torchsx.nn.objective import MarginalNLL
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

    prep, params, _, _, _ = _synthetic(n_hkl=60, n_xtal=60, per=30, seed=3)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=1, include_hkl=False)
    obj = MarginalNLL(wide_log_grid(), Poisson())
    cfg = TrainConfig(epochs=0, posterior_prior="wilson", device="cpu", heldout_frac=0.1)
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    basis = ReciprocalBasis.from_hkl(prep.hkl_valid.long(), prep.median_cell, "P 1", degree=2, aniso=False)
    before = tr.targets[0].wilson_sigma.clone()
    tr.set_reciprocal_basis(basis)
    t = tr.targets[0]
    assert t.has_basis and t.wilson_coef.shape == (3,)
    assert not torch.allclose(t.wilson_sigma, before)
    # the trainable coefficients follow the new basis, in one optimiser group
    groups = [g for g in tr.opt.param_groups if g.get("name") == "wilson_surface"]
    assert len(groups) == 1 and groups[0]["params"][0] is t.wilson_coef_param
    assert t.wilson_coef_param.shape == (3,)
    info = tr.wilson_fit_info()
    assert "wilson_B" in info and math.isfinite(info["wilson_B"]) and info["wilson_fit_iters"] >= 1
