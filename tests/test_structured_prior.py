"""The Wilson prior's pseudo-translation modulation: fitted to the data, fixed in training, kept by the merge."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_wilson_surface import _target, _trainer
from torchsx.merging.priors.modulation import PhaseModulation
from torchsx.merging.scale.pseudo_translation import phase_cos, point_group_rotations
from torchsx.nn.posterior_target import PosteriorMergeTarget

SG = "P 2 21 21"
T = [0.5, 0.3773, 0.5]


def _reflections():
    h = np.stack(np.meshgrid(np.arange(0, 24), np.arange(0, 32), np.arange(0, 40), indexing="ij"), -1).reshape(-1, 3)
    h = h[(h != 0).any(axis=1)]
    s2 = (h[:, 0] / 60.0) ** 2 + (h[:, 1] / 90.0) ** 2 + (h[:, 2] / 150.0) ** 2
    return h, s2


def test_fit_recovers_the_mean_modulation():
    h, s2 = _reflections()
    c = phase_cos(h, T, point_group_rotations(SG))
    rng = np.random.default_rng(0)
    E2 = (1 + 0.8 * c) * rng.exponential(1.0, len(h))           # Wilson statistics, modulated
    mod = PhaseModulation.fit(h, s2, E2, [T], SG)
    g = np.exp(mod.log_factor(h, s2))
    truth = (1 + 0.8 * c) / np.mean(1 + 0.8 * c)
    assert np.corrcoef(g, truth)[0, 1] > 0.99
    for lo, hi in ((-1.0, -0.5), (-0.2, 0.2), (0.5, 1.0)):
        m = (c >= lo) & (c <= hi)
        assert abs(g[m].mean() / truth[m].mean() - 1) < 0.08


def test_definition_round_trip_and_symmetry_factor():
    h, s2 = _reflections()
    c = phase_cos(h, T, point_group_rotations(SG))
    E2 = (1 + 0.5 * c) * np.random.default_rng(1).exponential(1.0, len(h))
    mod = PhaseModulation.fit(h, s2, E2, [T], SG)
    again = PhaseModulation.from_definition(mod.definition())
    np.testing.assert_allclose(again.log_prior_offset(h, s2), mod.log_prior_offset(h, s2))
    flat = PhaseModulation.fit(h, s2, E2, [], SG)
    np.testing.assert_array_equal(flat.log_factor(h, s2), 0.0)
    axial = np.array([[4, 0, 0], [0, 6, 0], [0, 0, 8], [1, 2, 3]])
    np.testing.assert_allclose(flat.log_prior_offset(axial, np.ones(4)), np.log([2, 2, 2, 1]))


def test_target_reads_the_offset_and_fits_the_smooth_part_under_it():
    torch.manual_seed(0)
    n = 6000
    g = torch.where(torch.arange(n) % 2 == 0, 0.2, 1.8)
    I_true = torch.distributions.Exponential(1 / 50.0).sample((n,)) * g
    t = PosteriorMergeTarget(I_true.clamp(min=0.01), k_coarse=31, m_fine=129, n_sigma=8.0)
    t.set_basis(torch.ones(n, 1))
    t.set_modulation(g.log())
    t.fit_wilson_surface()
    assert abs(float(t.wilson_coef.exp()) / 50.0 - 1) < 0.05        # the smooth part, modulation divided out
    torch.testing.assert_close(t.wilson_sigma, (t.wilson_coef.exp() * g).clamp(min=t.floor))
    coef = t.wilson_coef.clone()
    t.set_modulation(torch.zeros(n))                                  # re-evaluated, coefficients kept
    torch.testing.assert_close(t.wilson_coef, coef)
    torch.testing.assert_close(t.wilson_sigma, coef.exp().expand(n).clamp(min=t.floor))
    with pytest.raises(ValueError):
        t.set_modulation(torch.zeros(n + 1))


def test_training_pass_and_no_grad_pass_read_the_same_prior():
    I_true = torch.linspace(1.0, 100.0, 50)
    t, S = _target(I_true)
    t.set_modulation(torch.linspace(-1.0, 1.0, 50))
    t.enable_evidence_prior()
    t.sync_evidence_prior()
    with torch.no_grad():
        ref = t.integrate(S)[0].clone()
    torch.testing.assert_close(t.integrate(S)[0].detach(), ref)


def test_trainer_installs_it_and_the_merge_restores_it(tmp_path):
    from torchsx.cli.merge_nn import restore_intensity_modulation
    tr = _trainer(tmp_path)
    try:
        h = tr.prep.hkl_valid.long().numpy()
        s2 = (1.0 / tr.prep.d_spacing.double() ** 2).numpy()
        t = [0.0, 0.5, 0.0]
        c = phase_cos(h, t)
        mod = PhaseModulation.fit(h, s2, 1 + 0.6 * c, [t], None, min_per_shell=10, min_per_knot=5)
        log_mod = torch.as_tensor(mod.log_prior_offset(h, s2), dtype=torch.float32)
        tr.set_intensity_modulation(mod)
        for tg in tr.targets:
            torch.testing.assert_close(tg.wilson_log_mod, log_mod)
            torch.testing.assert_close(tg.wilson_coef_param.detach(), tg.wilson_coef)
            torch.testing.assert_close(tg.wilson_sigma, (tg.wilson_phi @ tg.wilson_coef + log_mod).exp()
                                       .clamp(min=tg.floor))
        ck = torch.load(tr.save_best(tmp_path / "best.pt"), weights_only=False)
        assert ck["intensity_modulation"] == mod.definition()
        other = _trainer(tmp_path / "other")
        try:
            restore_intensity_modulation(other, other.prep, ck)
            for tg in other.targets:
                torch.testing.assert_close(tg.wilson_log_mod, log_mod)
            restore_intensity_modulation(other, other.prep, {"intensity_modulation": None})
        finally:
            other.close()
    finally:
        tr.close()
